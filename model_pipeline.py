print("""importing modules\n""")

import time
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import SelectKBest,chi2
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# Model algoritms
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

# Metrics
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,f1_score

# Deployment
from flask import Flask, jsonify, request

class ModelPipeline():
    def __init__(self):
        self.X, self.y = self._prepare_data()
        self.train_data, self.test_data = self._split_data()
        self.model = None
        self.best_model_ = None
        self.features_column_length_ = len(self.X.columns)
    
    def _read_data(self):
        print("reading data from csv file\n")
        return pd.read_csv("data/online_shoppers_intention.csv")
    
    def _prepare_data(self):
        data = self._read_data()
        print("preparing the data\n")
        X = data.drop(['Revenue'], axis=1)
        y = data['Revenue']
        return X,y
    
    def _split_data(self):
        print("splitting the data\n")
        X_train,X_test,y_train,y_test =  train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)
        return [X_train,y_train], [X_test,y_test]
    
    def _imb_pipeline(self,model):
        num_cols = self.X.select_dtypes(exclude=['object']).columns.tolist()
        cate_cols = self.X.select_dtypes(include=['object']).columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant')),
            ('scaler', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ('imputer', SimpleImputer(strategy='constant'))
        ])

        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, num_cols),
            ('categorical', categorical_pipeline, cate_cols)
        ], remainder='passthrough')
        
        final_steps = [
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=1)),
            ('feature_selection', SelectKBest(score_func = chi2, k = int(self.features_column_length_ * 0.6))),
            ('model', model)
        ]
        
        print("pipeline is ready\n")
        return IMBPipeline(steps = final_steps)
    
    def __imb_pipeline_complex(self,model):
        return IMBPipeline(steps = [
                ('preprocessor', ColumnTransformer(
                                        [
                                            ('numeric',  Pipeline(
                                                        [
                                                            ('imputer', SimpleImputer(strategy='constant')),
                                                            ('scaler', MinMaxScaler())
                                                        ]
                                                    ), 
                                                self.train_data[0].select_dtypes(exclude=['object']).columns.tolist()
                                            ),
                                            ('categorical', Pipeline(
                                                        [
                                                            ('encoder', OneHotEncoder(handle_unknown='ignore')),
                                                            ('imputer', SimpleImputer(strategy='constant')),
                                                        ]
                                                    ),
                                                self.train_data[0].select_dtypes(include=['object']).columns.tolist()
                                            )
                                        ], remainder='passthrough'
                                    )
                ),
                
                ('smote', SMOTE(random_state=1)),
                ('feature_selection', SelectKBest(score_func = chi2, k = int(self.features_column_length_ * 0.6))),
                ('model', model)
            ]
        )
        
    def _applicable_models(self):
        classifiers = {}
    
        classifiers["RandomForestClassifier"] =  RandomForestClassifier()

        classifiers["DecisionTreeClassifier"] = DecisionTreeClassifier()

        classifiers["KNeighborsClassifier"] =  KNeighborsClassifier()
    
        classifiers["RidgeClassifier"] = RidgeClassifier()

        classifiers["BernoulliNB"] =  BernoulliNB()
        
        classifiers["SVC"] = SVC()

        classifiers["LogisticRegression"] = LogisticRegression()
        
        return classifiers
    
    def _model_cross_eval(self):
        cross_eval_scores = pd.DataFrame(columns = ['model', 'run_time', 'roc_auc'])
        classifiers = self._applicable_models()
        print("models created\n")
        X_train,y_train = self.train_data
        
        for key in classifiers:            
            start_time = time.time()
            print(f"pipeline for {key} model created")
            pipeline = self.__imb_pipeline_complex(classifiers[key])
            cv = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='roc_auc')
            
            end_time = time.time()
            
            cross_eval_scores.loc[len(cross_eval_scores)] = [key,round((end_time-start_time),2),round(cv.mean(),2)]
            print(f"evaluation for {key} completed \n")
            
        return cross_eval_scores.sort_values(by='roc_auc', ascending=False)
    
    def __mark_best_model(self, df, score_col, time_col):  
        df['score_norm'] = (df[score_col] - df[score_col].min()) / (df[score_col].max() - df[score_col].min())
        df['time_norm'] = (df[time_col] - df[time_col].min()) / (df[time_col].max() - df[time_col].min())
        df['tradeoff'] = df['score_norm'] - df['time_norm']
        best_idx = df['tradeoff'].idxmax()
        df.drop(columns=['score_norm', 'time_norm', 'tradeoff'], inplace=True)
       
        df['is_best_model'] = False
        df.loc[best_idx, 'is_best_model'] = True
        return df
    
    def select_best_model(self):
        models_df = self._model_cross_eval()
        print(f"models scores and runtime\n{models_df} \n\n")
        best_model_df = self.__mark_best_model(models_df,"roc_auc","run_time")
        best_model =  best_model_df.loc[best_model_df['is_best_model'] == True].squeeze()
        
        print(f"best model is {best_model['model']} with cross_eval_score {best_model['roc_auc']} and has runtime of {best_model['run_time']} \n")
        print(f"{best_model['model']} is selected automatically \n")
        self.best_model_ = best_model['model']
        return best_model

    def train_best_model(self):
        best_model = self.select_best_model()
        model_lst = self._applicable_models()
        
        if best_model is None:
            return "no model"
        
        print(f"training for {best_model['model']} is started \n")
        model = self._imb_pipeline(model_lst[best_model['model']])
        
        model.fit(self.train_data[0],self.train_data[1])
        print("model fitting is completed\n")
        
        self.model =  model
        return model
    
    def model_performance(self):
        model = self.train_best_model()
        y_pred = model.predict(self.test_data[0])
        
        res = {}
        res["roc_auc"] = roc_auc_score(self.test_data[1], y_pred)
        res["accuracy"] = accuracy_score(self.test_data[1], y_pred)
        res["f1_scor"] = f1_score(self.test_data[1], y_pred)
        
        classi_repo = classification_report(self.test_data[1], y_pred)
        print("here is the classification report")
        return res, classi_repo
    
    def deploy_model(self):
        model = self.train_best_model()
        
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return 'Hello, Flask!'
        
        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            features = pd.DataFrame(data['data'])
            prediction = model.predict(features)
            return jsonify({'prediction': int(prediction[0])})

        return app
    
    def serve(self):
        app = self.deploy_model()
        app.run()
        
        
if __name__ == "__main__":
    model = ModelPipeline()
    model.serve()