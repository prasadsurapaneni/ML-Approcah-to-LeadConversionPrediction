#!/usr/bin/env python
# coding: utf-8


from flask import Flask, jsonify, request
import flask
import pandas as pd
import numpy as np
import joblib
import time


app = Flask(__name__)


# prepare input for base learners
def prepare_input(X_q):
    """This function preprocess the data and construct the input features for base learners 
    using trained scaler for numerical features and one hot encoder for categorical features """
    
    # define the categorical & numerical features which are taken for training the model
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    num_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']
    
    # load standard scaler from pickle file
    scaler = joblib.load('num_features_scaler.pkl')
    # load trained onehotencoder for categorical features
    cat_features_ohe = joblib.load('cat_features_ohe.pkl')
    
    # tranform the numerical features with trained standard scaler and convert it into a dataframe for further use
    X_q_num = pd.DataFrame(scaler.transform(X_q[num_features]), columns=num_features)
    # print(X_q_num, type(X_q_num))
    
    # tranform the categorical features with trained onehotencoder and convert it into a dataframe for further use
    X_q_ohe = cat_features_ohe.transform(X_q[cat_features])
    # print(X_q_ohe)
    cat_feature_labels_ohe = np.concatenate(cat_features_ohe.categories_).ravel().tolist()
    X_q_ohe = pd.DataFrame(X_q_ohe.toarray(), columns=cat_feature_labels_ohe)
    # print(type(X_q_ohe), type(X_q_num))
    
    # merge both numerical feature set and categorical feature set
    X_q_final = pd.concat([X_q_ohe, X_q_num], axis=1)
    
    # print(X_q_final)
    return X_q_final



@app.route('/predict', methods=['POST'])
def final_predict(y=''):
    """This function predcts target label for query instance, the prediction is from trained meta model """    
    
    start_time = time.time()

    # get the form inputs through request
    form_inputs = request.form.to_dict()
    # print(form_inputs)
    X_q = pd.DataFrame(np.asarray(list(form_inputs.values())).reshape(1,-1), columns=form_inputs.keys())
    X_q.drop('name', axis=1, inplace=True)
    # print(X_q)
    
    # return form_inputs
    
    # data pre processing and prepare the input to predict through meta model
    X_input = prepare_input(X_q)
    
    # load the trained base models from pickle file
    base_models = joblib.load('base_learners.pkl')
    # print(base_models)
    
    # initiate list for storing the predictions from each base learner for given query point
    input_for_meta = []
    # predictions from base learners
    for model in base_models:
        input_for_meta.append(model.predict(X_input))
        
    # construct the input to meta model from base learners predictions
    input_for_meta = np.transpose(np.asarray(input_for_meta))
    # load trained meta model from pickle file 
    meta_model = joblib.load('meta_model.pkl')
    # final prediction
    final_prediction = meta_model.predict(input_for_meta)
    
    # print(final_prediction)
    if final_prediction:
        prediction = 'Positive'
    else:
        prediction = 'Negative'
    
    end_time = time.time()
    
    # computing the time taken for predicting the target label
    time_for_prediction = str(end_time - start_time) + ' seconds'
    
    # handling true label
    true_label = 'Positive' if y==1 else 'Negative'
    if y=='': true_label = '--NA--'
    
    response = jsonify({'Time taken for Prediction': time_for_prediction, 'True label': true_label, 'prediction': prediction})
    # return predicted label, time taken for predicting and the original target label if passed as input
    return response



@app.route('/index')
def index():
    return flask.render_template('index.html')



@app.route('/')
def welcome():
    return 'Welcome to the Lead Score Prediction home page'



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)


