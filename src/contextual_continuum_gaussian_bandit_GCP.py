import numpy as np
import pandas as pd
from sklearn.gaussian_process import kernels
from continuum_gaussian_bandits import ContinuumArmedBandit, GPR
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_tabular_classification_sample(instance_dict: Dict):
    location= "eu-west4"
    api_endpoint= "eu-west4-aiplatform.googleapis.com"
    
    project= 'aa-cmab-product'
    endpoint_id = '5076928970558013440'
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tables_classification.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


class ContextualContinuumArmedBanditCloud:
    def __init__(self, contexts, oracle, bid_max_value, convergence_rate=1.0):
        self.context_dict = {}
        self.contexts = contexts
        self.bid_max_value = bid_max_value
        self.oracle = predict_tabular_classification_sample
        
        X = np.arange(0, self.bid_max_value, 100)
        self.num_contexts = len(contexts)
        for context in self.contexts:
            df_context = context.merge(X, how = 'right')
            y_pred = self.oracle(df_context)
            self.context_dict[context] = (ContinuumArmedBandit(X, y_pred , convergence_rate=1.0), None, None)


    def select_action(self, context):
        continuum_bandit = self.contexts[context][0]
        x = continuum_bandit.select_action()
        return x
        
    def get_x_best(self, X, context):
        x_best = self.contexts[context][0].get_x_best(X)
        return x_best

    def fit(self, num_rounds):
        np.random.seed(42)
        for round_num in num_rounds:
            sample_context_id = np.random.randint(self.num_contexts)
            sampled_context = self.contexts[sample_context_id]
            continuum_bandit = self.context_dict[sampled_context][0]
            x = continuum_bandit.select_action()
            
            y_pred = self.oracle(sampled_context.append(x))
            continuum_bandit.update(x, y_pred)
            
    
    def predict(self, contexts):
        result_set = []
        for context in contexts:
            try:
                sampled_context = self.context_dict[context]
            except:
                print("Unsampled or unseen context, please update or re-fit the model with the context included.")
                return Exception()
            result_set.append(self.get_x_best(np.arange(0, self.bid_max_value, 100), context))
            
        return np.array(result_set)
    
    def partial_fit(self, contexts_new, X_new, y_new, reward_new):
        for context in contexts_new:
            try:
                sampled_context = self.context_dict[context]
            except:
                print("Unsampled or unseen context, please update or re-fit the model with the context included.")
                return Exception()
            
            
            
            
            

