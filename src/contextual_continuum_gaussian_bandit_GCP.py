import numpy as np
import pandas as pd
from sklearn.gaussian_process import kernels
from continuum_gaussian_bandits import ContinuumArmedBandit, GPR
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

import json
import logging
import os
import traceback
from datetime import datetime

from google.api_core import retry
from google.cloud import bigquery
from google.cloud import firestore
from google.cloud import pubsub_v1
from google.cloud import storage
import pytz
import dill


PROJECT_ID = os.getenv('GCP_PROJECT')
BQ_DATASET = 'aa-cmab-product.OperationalScoringData'
BQ_TABLE = 'ProductionRawData'
BQ_SCORE = 'ProductionRawData'
ERROR_TOPIC = 'projects/%s/topics/%s' % (PROJECT_ID, 'streaming_error_topic')
SUCCESS_TOPIC = 'projects/%s/topics/%s' % (PROJECT_ID, 'streaming_success_topic')
DB = firestore.Client()
CS = storage.Client()
PS = pubsub_v1.PublisherClient()
BQ = bigquery.Client()


def read_from_into_bigquery(bq_dataset, bq_table_name):
    table = BQ.dataset(bq_dataset).table(bq_table_name)
    return table

def insert_into_bigquery(bucket_name, file_name):
    blob = CS.get_bucket(bucket_name).blob(file_name)
    row = json.loads(blob.download_as_string())
    table = BQ.dataset(BQ_DATASET).table(BQ_TABLE)
    errors = BQ.insert_rows_json(table,
                                 json_rows=[row],
                                 row_ids=[file_name],
                                 retry=retry.Retry(deadline=30))

def append_into_bigquery(rows, bq_dataset, bq_table_name):
    row = json_format.ParseDict(rows, Value())
    table = BQ.dataset(bq_dataset).table(bq_table_name)
    errors = BQ.insert_rows_json(table,
                                 json_rows=[row],
                                 retry=retry.Retry(deadline=30))

class VertexOracle:
    def __init__(self, 
                 location = "europe-west4",
                 api_endpoint = "eu-west4-aiplatform.googleapis.com",
                 project = 'aa-cmab-product',
                 endpoint_id = '5076928970558013440'):
        self.location = location
        self.api_endpoint = api_endpoint
        self.project=project
        self.endpoint_id = endpoint_id
        # The AI Platform services require regional API endpoints.
        self.client_options = {"api_endpoint": api_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=self.client_options)
        
        aiplatform.init(
            # your Google Cloud Project ID or number
            # environment default used is not set
            project=self.project,
        
            # the Vertex AI region you will use
            # defaults to us-central1
            location=self.location,
        
            # Googlge Cloud Stoage bucket in same region as location
            # used to stage artifacts
            staging_bucket='gs://internal-testing/',
        
            # custom google.auth.credentials.Credentials
            # environment default creds used if not set
            #credentials=my_credentials,
        
            # customer managed encryption key resource name
            # will be applied to all Vertex AI resources if set
            #encryption_spec_key_name=my_encryption_key_name,
        
            # the name of the experiment to use to track
            # logged metrics and parameters
            #experiment='my-experiment',
        
            # description of the experiment above
            #experiment_description='my experiment decsription'
        )
        # use string formating or f-strings to insert our class variables into the endpoint instantiation
        self.cmab_oracle_endpoint = aiplatform.Endpoint(f'projects/{self.project}/locations/{self.location}/endpoints/{self.endpoint_id}')
        
        
    def predict_tabular_classification_sample(self, instance_dict: Dict):
        # for more info on the instance schema, please use get_model_sample.py
        # and look at the yaml found in instance_schema_uri
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        response = self.cmab_oracle_endpoint.predict(instances=instances, parameters=parameters)
        print("response")
        print("deployed_model_id:", response.deployed_model_id)
        # See gs://google-cloud-aiplatform/schema/predict/prediction/tables_classification.yaml for the format of the predictions.
        predictions = response.predictions
        return predictions
    


class ContextualContinuumArmedBandit:
    """
    This class implements a contextual version of the ContinuumArmedBandit, which extends it to work
    in the contextual setting. We do this by augmenting the structure of the bandit to 
    incoroporate a contextual feature matrix. This identifies a unique context in which to train for 
    action/rewards. Thus, we train a ContinuumArmedBandit for each unique context encountered and use this
    as our model structure. Model API follows closely that defined by scikit-learn and also that by ContinuumArmedBandit in that 
    it merely extends that class to work for the contextual setting by specifying a context ID whenever calling similarly named methods such as fit or predict.
    
    """
        
    def __init__(self, contexts, oracle, bid_max_value=None, context_ids = None, convergence_rate=1.0, exploration_threshold=0.1, action_col_name='avg_cpc', kpi_name = 'avg_ctr'):
        self.context_dict = {} # keep stored all the trained bandit models to be called for prediction later on.
        
        if type(contexts) != pd.core.frame.DataFrame:
            print("WARNING: The contexts provided are not in the form of a Pandas DataFrame object. Attempting to convert, but no guarantees... ")
            try:
                contexts = pd.DataFrame(contexts)
            except:
                print("ERROR: Could not convert contexts into a Pandas DataFrame. Please transform it on the user end and pass as required...")
                raise
                
        if contexts.shape[0] != contexts.drop_duplicates().shape[0]:
            print("ERROR: The contexts provided are not unique. Please provide only unique lists of context information, i.e. a context matrix, to instantiate a trained ContextualContinuumArmedBandit. Returning...")
            raise
       
        if not bid_max_value:
            bid_max_value = 2.0
        
        # do not require a default list of context IDs. If none provided use a default index from 0 to N, the number of unique contexts
        if not context_ids:
            self.context_ids = np.arange(contexts.shape[0])
        else:
            self.context_ids = context_ids
        self.contexts = contexts.copy(deep=True) # keep a dictionary or dataframe containing all the contexts we wish to keep track of
        self.bid_max_value = bid_max_value # keep track of the max allowed bid value
        self.action_col_name = action_col_name # keep track of the name of action column name, by default avg_cpc
        self.kpi_name = kpi_name # keep track of the kpi name for optimization/i.e. to use as reward, default ctr
        self.exploration_threshold = exploration_threshold
        X = np.linspace(0, self.bid_max_value, 100)
        # setting X to be a 2D matrix, one data point per dimension, such that the kernel treats each point separately

        X_pred = np.atleast_2d(X).T
        self.num_contexts = len(contexts)
        # loop over the context IDs we
        for context_id in self.context_ids:
            print("INFO: Fitting continuum bandit for context: " + str(context_id + 1))
            df_context = pd.DataFrame(self.contexts.iloc[context_id, :]).transpose()
            df_context = df_context.loc[df_context.index.repeat(len(X))].reset_index(drop=True)
            df_context = pd.concat([df_context, pd.DataFrame({'avg_cpc': X})], axis = 1)
            y_pred = oracle.predict(df_context)
            # setting y to be a 2D matrix to match the dimensions of X when calculating the alpha parameter
            y_pred = np.atleast_2d(y_pred).T
            self.context_dict[self.context_ids[context_id]] = (ContinuumArmedBandit(X_pred, y_pred , convergence_rate=1.0), None, None)


    def select_action(self, context_id):
        """
        Select the next action to sample from. Can be thought of as similar to an acquisition function in Bayesian optimization
        
        Note
        ----
        context_id should preferably come from the same list of IDs as the contexts from the training data. It is very easy otherwise to predict for an incorrect context!
        
        Parameters
        ----------
        context_id : int
            This is the ID of the context in our context matrix (represented via a Pandas DataFrame object) which we want to generate a prediction for
            
        Returns
        -------
        result : float
            The result is the numeric action which will act as our next continuous action to take for this instance.
            
        """
        continuum_bandit = self.context_dict[context_id][0] # get the bandit trained algorithm for this context id
        x = continuum_bandit.select_action() # query this bandits select_action() method to retrieve the next query point based on its acquisition function.
        return x
        
    def get_x_best(self, X, context_id):
        """
        Selects the next action to take based on the acquisition function defined for the continuum gaussian bandit        

        Parameters
        ----------
        X : array(n_samples, ) 
            An array of points for which to calculate a merit score via the acquisition function of the bandit model
            thus giving rise to candidate points to choose for the next query point x_best
            
        context_id : int type
            The specific context id for which to retrieve the next best action to follow

        Returns
        -------
        x_best : float
            The next best point to try sample at as the next action
        """
        x_best = self.context_dict[context_id][0].get_x_best(X) # query the get_x_best function of the ContinuumArmedBandit associated with this context id
        
        return x_best


    def update(self, df_contexts, actions, rewards, start_from_checkpoint=True):
        """
        Updates the contextual bandit given newly observed rewards for the actions set by the model.
        
        Parameters
        ----------
        df_contexts : array(n_contexts, n_features) or pd.DataFrame of shape (n_contexts, n_features)
            Matrix of covariates for the available data.
        actions : array(n_samples, ), float type
            Arms or actions that were chosen for each observations.
        rewards : array(n_samples, ), [0,1]
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.
        start_from_checkpoint : bool
            If the policy was previously fit to data, 
            it will use previously trained models along with adding this new data to old data. In this case,
            will only refit the models that have new data according to actions and rewards specified.
        
        Returns
        -------
        self : obj
            No return type
        """
        # important checks to run, check of the object is of type dataframe
        if type(df_contexts) != pd.core.frame.DataFrame:
            print("WARNING: The contexts provided are not in the form of a Pandas DataFrame object. Attempting to convert, but no guarantees... ")
            try:
                df_contexts = pd.DataFrame(df_contexts)
            except:
                print("ERROR: Could not convert contexts into a Pandas DataFrame. Please transform it on the user end and pass as required...")
                raise
                
        # check if the data in the context matrix passed match what has been trained on historically
        if not self.contexts.columns.isin(df_contexts.columns).all():
            print("ERROR: Columns provided in df_contexts does not correspond to the column trained on for this model. Please provide a data frame matching the following column signature: " + str(self.contexts.columns))
            raise 
            
        # check the reverse holds in case we have a different set in some slight way
        if not df_contexts.columns.isin(self.contexts.columns).all():
            print("ERROR: Columns provided in df_contexts does not correspond to the column trained on for this model. Please provide a data frame matching the following column signature: " + str(self.contexts.columns))
            raise 
            
        # for each unique context, find the similar or corresponding context in our data and run this through the bandit model
        # for this context, updating the actions and rewards following this.
        for context_id in range(df_contexts.drop_duplicates().shape[0]):
            try:
                context = df_contexts.iloc[context_id, :]
                
                context_location = np.where((self.contexts == context).all(axis=1))[0][0] # getting the first possible index from the np.where results which are of the form np.array(np.array, dtype)
                continuum_bandit = self.context_dict[context_location][0]
                X = actions[context_id]
                y = rewards[context_id]
                continuum_bandit.update(X, y)
                
            except:
                print("ERROR: Unsampled or unseen context, please update or re-fit the model with the context included.")
                raise
        
    def fit(self, num_rounds):
        """
        Continues to fit the model from the last trained point in terms of parameters. In essence, 
        
        1) it samples a random context to train, 
        2) samples a next action point to query,
        3) receives a reward from the oracle environment simulation model,
        4) updates the bandit trained for this context       

        Parameters
        ----------
        num_rounds : int type 
            The number of iterations or samples which will be carried out for training of this model.
            
        Returns
        -------

        """
        np.random.seed(42)
        for round_num in num_rounds:
            sample_context_id = np.random.randint(self.num_contexts)
            sampled_context = self.contexts[sample_context_id]
            continuum_bandit = self.context_dict[sample_context_id][0]
            
            epsilon = np.random.random() # pick a random number between 0 and 1
            
            # if the number is greater than the model exploration threshold
            # pick a next query point based on select_action or using the model's acquisition function
            # or just exploit the next best point in terms of known best strategy
            if epsilon < self.exploration_threshold:
                x = continuum_bandit.predict()
            else:
                x = continuum_bandit.select_action()
            
            y_pred = self.oracle.predict(sampled_context.append(x))
            continuum_bandit.update(x, y_pred)
            
    
    def predict(self, new_contexts):
        """
        Using the currently trained-on parameters of the model, generate a series of predictions for the new contexts
        provided to the model. This follows the following steps,
        
        1) it attemps to match currently known contexts to those provided,
        2) with probability epsilon, retrieve a next action query point based on the model's acquisition function
        3) otherwise do greedy-style exploitation of best known point
        4) return the models best decision given the above.

        Parameters
        ----------
        new_contexts : pd.DataFrame of shape (n_contexts, n_features)
            The context feature matrix defining the unique contexts we would like to query for a prediction.
            
        Returns
        -------
        df_results : pd.DataFrame of shape (n_contexts, n_features + 1)
            This returns the above new_contexts dataframe object but we append a column to it with the next actions to be taken.
            
        """
        if type(new_contexts) != pd.core.frame.DataFrame:
            print("WARNING: The contexts provided are not in the form of a Pandas DataFrame object. Attempting to convert, but no guarantees... ")
            try:
                new_contexts = pd.DataFrame(new_contexts)
            except:
                print("ERROR: Could not convert contexts into a Pandas DataFrame. Please transform it on the user end and pass as required...")
                raise
                
                
        result_set = []
        for context_id in range(new_contexts.drop_duplicates().shape[0]):
            try:
                context = self.contexts.iloc[context_id, :]
                
                context_location = np.where((self.contexts == context).all(axis=1))[0][0] # getting the first possible index from the np.where results which are of the form np.array(np.array, dtype)
                continuum_bandit = self.context_dict[context_location][0] # getting the trained bandit model from the context_dict
                
                epsilon = 1.0 # random number, such that with prob. 0.1 we choose a random action to explore via select_action, and otherwise predict as usual
            
                if epsilon < self.exploration_threshold:
                    x = continuum_bandit.select_action()
                else:
                    x = continuum_bandit.predict()
            except:
                print("ERROR: Unsampled or unseen context, please update or re-fit the model with the context included.")
                raise 
            result_set.append(x)
            
        df_results = pd.DataFrame(new_contexts.copy(deep=True)) # create deepcopy
        df_results['avg_cpc'] = result_set
        return df_results
    
    def partial_fit(self, contexts_new, X_new, y_new, reward_new):
        """ TODO - for online bandit algorithm """
        for context in contexts_new:
            try:
                sampled_context = self.context_dict[context]
            except:
                print("ERROR: Unsampled or unseen context, please update or re-fit the model with the context included.")
                raise
  

if __name__ == "__main__":
    
    df_contexts = read_from_into_bigquery(BQ_DATASET, BQ_SCORE)
    df_contexts_prev = read_from_into_bigquery(BQ_DATASET, BQ_TABLE)
    
    todays_date = datetime.strftime(datetime.now(), format = "%Y-%m-%d")
    yesterdays_date = datetime.strftime(datetime.now()-datetime.timedelta(days=1), format = "%Y-%m-%d")
    
    with open('ContextualContinuumArmedBanditCloud_JOT.dill', 'rb') as inp
        gpbandit = dill.load(inp)
    
    pred, ci = gpbandit.predict(df_contexts)
    gpbandit.update(df_contexts_prev)
    
    with open("saved_ucb_model.dill", "wb") as stor
        dill.dump(m, stor)
    
    append_into_bigquery(append_into_bigquery, BQ_DATASET, BQ_SCORE)
          
            
            
            

