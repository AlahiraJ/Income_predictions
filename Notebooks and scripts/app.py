# dependencies
from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import Binarizer

# API definition
app = Flask(__name__) 

#specify model path and deserialize model
model_path = r'/home/alahira/Documents/Data science projects/Income_predictions/Models/'

model_infile = open(model_path + 'logistic_regression_model_one.sav', 'rb')
model = pickle.load(model_infile)
model_infile.close()

column_data_infile = open(model_path +'column_data', 'rb' )
column_data = pickle.load(column_data_infile)
column_data_infile.close()

def preprocess(data):
    """Function to preprocess data using steps that were used during model building"""
    
    df = pd.read_json(data)

    #seperate dataframe into numerica and categorical variables 
    #for ease of handling anf processing
    numeric_vars = df.select_dtypes(exclude = ('category','object'))
    cat_vars = df.select_dtypes(include = ('category','object'))

    #scale numeric variables
    scaler = StandardScaler()
    scaled_numeric_vars = scaler.fit_transform(numeric_vars)
    dummy_vars = pd.get_dummies(data = cat_vars)

    binarize = Binarizer()
    scaled_numeric_vars['capital_gain'] = binarize.fit_transform(scaled_numeric_vars['capital_gain'].values.reshape(-1,1))
    scaled_numeric_vars['capital_loss'] = binarize.fit_transform(scaled_numeric_vars['capital_loss'].values.reshape(-1,1))

    #get dummy variables of categorical data
    dummy_vars = pd.get_dummies(data = cat_vars)

    ## merge dataframe
    new_df = pd.merge(left=scaled_numeric_vars, right=dummy_vars, left_index=True, right_index=True)

    #select features used to train model
    transformed_df = column_data.transform(new_df)
    
    return transformed_df

@app.route('/predict', methods =['POST'])
def predict():
    if model:
        
        #make request for data to be predicted
        json_ = request.json
        
        ##call data prprocessing function
        df_processed = preprocess(json_)

        ##make predictions
        model_preds = list(model.predict(df_processed))
        return jsonify({'prediction' : str(model_preds)})


if __name__ == '__main__':
    port = 12345
    app.run(port = port,debug = True) 