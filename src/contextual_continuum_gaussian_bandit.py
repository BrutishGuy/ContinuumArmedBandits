import numpy as np
import pandas as pd
from sklearn.gaussian_process import kernels
#from continuum_gaussian_bandits import ContinuumArmedBandit, GPR
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error
import dill 
import argparse


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
            
            if not type(oracle) == google.cloud.aiplatform.models.Endpoint:
                y_pred = oracle.predict(df_context)
            else:
                y_preds = oracle.predict(instances=[df_context.to_json()])
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
            result_set.append(x[0][0])
            
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
            
def train_local(dataset_name):
    """
    Example usage of models, oracle models etc.training them using a locally downloaded CSV file in the form of the JOT Google Ads
    file data schema (see ./docs/data_docs/). We preprocess the raw docs in the form that it is done to put it into the BigQuery ProductionProcessedData table
    
    Trains a catboost oracle model and then trains a bandit using this model.

    Additional fitting can be done via the fit_rounds, set to 0, or add more.
    
    Parameters
    ----------
    dataset_name : str
        Example file of the form JOT Google Ads export
        
    Returns
    -------

    """
    df = pd.read_csv(filename)
    df.columns = [client_id,
                  campaign_id,
                  group_id,
                  account_descriptive_name,
                  ad_network_type,
                  avg_position,
                  campaign_name,
                  city_criteria_id,
                  clicks,
                  cost,
                  impressions,
                  country_criteria_id,
                  date,
                  device,
                  external_customer_id,
                  metro_criteria_id,
                  region_criteria_id]

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['week_day'] = df['date'].dt.day_name()
    df['week_num'] = df['date'].dt.week
    
    id_fields = ['client_id', 'campaign_id', 'group_id', 'city_criteria_id',
           'country_criteria_id', 'external_customer_id', 'metro_criteria_id',
           'region_criteria_id']
    
    df[id_fields] = df[id_fields].astype(object)
    
    group_by_fields = ['client_id',
    'external_customer_id',
    'group_id', 
    'account_descriptive_name', 
    'ad_network_type',
    'campaign_id', 
    'campaign_name',  
    'week_day',
    'week_num']
    
    
    df_grouped = df.groupby(group_by_fields).agg({'clicks': ['mean', 'sum'], 'impressions': 'sum', 'cost': ['mean', 'sum']}).reset_index()
    df_grouped.columns = ["_".join(a) for a in df_grouped.columns.to_flat_index()]
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['avg_ctr'] = df_grouped['clicks_sum']/df_grouped['impressions_sum']
    df_grouped['avg_cpc'] = df_grouped['cost_sum']/df_grouped['clicks_sum']
    df_grouped['total_cost'] = df_grouped['cost_sum']
    df_grouped['total_clicks'] = df_grouped['clicks_sum']
    df_grouped['total_impressions'] = df_grouped['impressions_sum']
    df_grouped['avg_clicks'] = df_grouped['clicks_mean']
    df_grouped['avg_cost'] = df_grouped['cost_mean']
    
    df_grouped = df_grouped.drop(['week_num','clicks_sum', 'impressions_sum', 'cost_sum', 'cost_mean', 'clicks_mean'], axis=1)
    df_grouped = df_grouped.fillna(0)
    df_grouped = df_grouped[df_grouped.avg_cpc > 0]
    
    df_train = df_grouped.drop(['avg_cpc', 'avg_ctr'], axis=1)
    
    train_data = df_grouped[group_by_fields + ['avg_cpc']]
    
    train_labels = df_grouped['avg_ctr']
    
    cat_features = [i for i in range(len(group_by_fields))]
    model = CatBoostRegressor(iterations=1000)
    model.fit(train_data,
              train_labels,
              cat_features,
              verbose=True)
    
    train_preds = model.predict(train_data)
    print(np.sqrt(mean_squared_error(train_labels, train_preds)))
    # train the bandit example
    bandit = ContextualContinuumArmedBandit(df_train, model, bid_max_value=1.0)
    
    dill.dump(bandit, open("./data/ContextualContinuumArmedBanditCloud_JOT.dill", "wb"))
    

def example_usage(filename):
    """
    Example usage of models, oracle models etc.
    
    Parameters
    ----------
    filename : str
        Example file of the form JOT Google Ads export
        
    Returns
    -------

    """
    df = pd.read_csv(filename)
    df.columns = [client_id,
                  campaign_id,
                  group_id,
                  account_descriptive_name,
                  ad_network_type,
                  avg_position,
                  campaign_name,
                  city_criteria_id,
                  clicks,
                  cost,
                  impressions,
                  country_criteria_id,
                  date,
                  device,
                  external_customer_id,
                  metro_criteria_id,
                  region_criteria_id]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['week_day'] = df['date'].dt.day_name()
    df['week_num'] = df['date'].dt.week
    
    id_fields = ['client_id', 'campaign_id', 'group_id', 'city_criteria_id',
           'country_criteria_id', 'external_customer_id', 'metro_criteria_id',
           'region_criteria_id']
    
    df[id_fields] = df[id_fields].astype(object)
    
    group_by_fields = ['client_id',
    'external_customer_id',
    'group_id', 
    'account_descriptive_name', 
    'ad_network_type',
    'campaign_id', 
    'campaign_name',  
    'week_day',
    'week_num']
    
    
    df_grouped = df.groupby(group_by_fields).agg({'clicks': ['mean', 'sum'], 'impressions': 'sum', 'cost': ['mean', 'sum']}).reset_index()
    df_grouped.columns = ["_".join(a) for a in df_grouped.columns.to_flat_index()]
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['avg_ctr'] = df_grouped['clicks_sum']/df_grouped['impressions_sum']
    df_grouped['avg_cpc'] = df_grouped['cost_sum']/df_grouped['clicks_sum']
    df_grouped['total_cost'] = df_grouped['cost_sum']
    df_grouped['total_clicks'] = df_grouped['clicks_sum']
    df_grouped['total_impressions'] = df_grouped['impressions_sum']
    df_grouped['avg_clicks'] = df_grouped['clicks_mean']
    df_grouped['avg_cost'] = df_grouped['cost_mean']
    
    df_grouped = df_grouped.drop(['week_num','clicks_sum', 'impressions_sum', 'cost_sum', 'cost_mean', 'clicks_mean'], axis=1)
    df_grouped = df_grouped.fillna(0)
    df_grouped = df_grouped[df_grouped.avg_cpc > 0]
    
    df_train = df_grouped.drop(['avg_cpc', 'avg_ctr'], axis=1)
    
    train_data = df_grouped[group_by_fields + ['avg_cpc']]
    
    train_labels = df_grouped['avg_ctr']
    
    cat_features = [i for i in range(len(group_by_fields))]
    model = CatBoostRegressor(iterations=1000)
    model.fit(train_data,
              train_labels,
              cat_features,
              verbose=True)
    
    train_preds = model.predict(train_data)
    print(np.sqrt(mean_squared_error(train_labels, train_preds)))
    df_sample = df_train.sample(int(0.1 * df_train.shape[0]))
    
    # test the bandit example
    bandit = ContextualContinuumArmedBandit(df_sample, model, bid_max_value=1.0)
    
    y_reward_sample = 0.01 + 0.1*np.random.random(df_sample.shape[0])
    X_action_sample = 0.05 + 0.5*np.random.random(df_sample.shape[0])
    
    preds = bandit.predict(df_sample)
    bandit.update(df_sample, X_action_sample, y_reward_sample)
    
    dill.dump(bandit, open("./data/ContextualContinuumArmedBanditCloud_JOT.dill", "wb"))
    bandit_test = dill.load(open("./data/ContextualContinuumArmedBanditCloud_JOT.dill", "rb"))     

    preds = bandit_test.predict(df_sample)
    bandit_test.update(df_sample, X_action_sample, y_reward_sample)       
                           
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_filename', type=str, default='results-20210821-175504', help='Enter filename of data to be used in re-training the model')
    parser.add_argument('--model_filename', type=str, default='./data/ContextualContinuumArmedBanditCloud_JOT.dill', help='Enter filename and path of model to be used in re-training')
    opt_gen = parser.parse_args()


    df = pd.read_csv(opt_gen.data_filename)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['week_day'] = df['date'].dt.day_name()
    df['week_num'] = df['date'].dt.week
    
    id_fields = ['client_id', 'campaign_id', 'group_id', 'city_criteria_id',
           'country_criteria_id', 'external_customer_id', 'metro_criteria_id',
           'region_criteria_id']
    
    df[id_fields] = df[id_fields].astype(object)
    
    group_by_fields = ['client_id',
    'external_customer_id',
    'group_id', 
    'account_descriptive_name', 
    'ad_network_type',
    'campaign_id', 
    'campaign_name',  
    'week_day',
    'week_num']
    
    
    df_grouped = df.groupby(group_by_fields).agg({'clicks': ['mean', 'sum'], 'impressions': 'sum', 'cost': ['mean', 'sum']}).reset_index()
    df_grouped.columns = ["_".join(a) for a in df_grouped.columns.to_flat_index()]
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['cost_sum'] = df_grouped['cost_sum']/1000000
    df_grouped['avg_ctr'] = df_grouped['clicks_sum']/df_grouped['impressions_sum']
    df_grouped['avg_cpc'] = df_grouped['cost_sum']/df_grouped['clicks_sum']
    df_grouped['total_cost'] = df_grouped['cost_sum']
    df_grouped['total_clicks'] = df_grouped['clicks_sum']
    df_grouped['total_impressions'] = df_grouped['impressions_sum']
    df_grouped['avg_clicks'] = df_grouped['clicks_mean']
    df_grouped['avg_cost'] = df_grouped['cost_mean']
    
    df_grouped = df_grouped.drop(['clicks_sum', 'impressions_sum', 'cost_sum', 'cost_mean', 'clicks_mean'], axis=1)
    df_grouped = df_grouped.fillna(0)
    df_grouped = df_grouped[df_grouped.avg_cpc > 0]
    
    df_train = df_grouped.drop(['avg_cpc', 'avg_ctr'], axis=1)
    
    # test the bandit example
    #bandit = ContextualContinuumArmedBandit(df_sample, model, bid_max_value=1.0)
    bandit = dill.load(open(opt_gen.model_filepath, "rb"))     

    bandit.update(df_train, df_grouped['avg_cpc'], df_grouped['avg_ctr'])       

    dill.dump(bandit, open(opt_gen.model_filepath, "wb"))

    df_preds = bandit.predict(df_train)


    
    new_report_names = ['Action',
                     'Customer ID',
                     'Ad group',
                     'Campaign ID',
                     'Default max. CPC']
    
    new_col_names = ['Action',
                     'Customer ID',
                     'Ad group ID',
                     'Campaign ID',
                     'Default max. CPC']
    df_report = df_preds
    df_report = df_report[['client_id', 'ad_group_id', 'campaign_id', 'avg_cpc']]
    df_report['Action'] = 'Edit'
    df_report.columns = new_col_names
    df_report['Ad group ID'] = df_report['Ad group ID'].astype(np.int64)
    df_ad_groups1 = pd.read_excel('Ad Group.xlsx', sheet_name='CAN Adgroups')
    df_ad_groups2 = pd.read_excel('Ad Group.xlsx', sheet_name='USA AdGroups')
    df_ad_groups = pd.concat([df_ad_groups2, df_ad_groups1], axis = 0)
    
    df_report = df_report.merge(df_ad_groups, how='left', on=['Ad group ID'])
    df_report = df_report[new_report_names]
    df_report.to_csv('TEST_' +  datetime.datetime.today().strftime("%Y%m%d") + '_USA_CAN_AdGroupCPCAdjustments_FromModel.csv',
                     sep='/',
                     decimal=',',
                     index=False)
                    
                

