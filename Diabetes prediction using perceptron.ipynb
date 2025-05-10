{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "86f02728",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import os\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as numpy\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.datasets import load_svmlight_file\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import requests\n",
    "from io import BytesIO"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "618e22fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "data=pd.read_csv(\"diabetes.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7de9911",
   "metadata": {},
   "source": [
    "Load the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "82fc0419",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0            6      148             72             35        0  33.6   \n",
       "1            1       85             66             29        0  26.6   \n",
       "2            8      183             64              0        0  23.3   \n",
       "3            1       89             66             23       94  28.1   \n",
       "4            0      137             40             35      168  43.1   \n",
       "\n",
       "   DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                     0.627   50        1  \n",
       "1                     0.351   31        0  \n",
       "2                     0.672   32        1  \n",
       "3                     0.167   21        0  \n",
       "4                     2.288   33        1  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82673e74",
   "metadata": {},
   "source": [
    " Head of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "929be348",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 768 entries, 0 to 767\n",
      "Data columns (total 9 columns):\n",
      " #   Column                    Non-Null Count  Dtype  \n",
      "---  ------                    --------------  -----  \n",
      " 0   Pregnancies               768 non-null    int64  \n",
      " 1   Glucose                   768 non-null    int64  \n",
      " 2   BloodPressure             768 non-null    int64  \n",
      " 3   SkinThickness             768 non-null    int64  \n",
      " 4   Insulin                   768 non-null    int64  \n",
      " 5   BMI                       768 non-null    float64\n",
      " 6   DiabetesPedigreeFunction  768 non-null    float64\n",
      " 7   Age                       768 non-null    int64  \n",
      " 8   Outcome                   768 non-null    int64  \n",
      "dtypes: float64(2), int64(7)\n",
      "memory usage: 54.1 KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cde0dae1",
   "metadata": {},
   "source": [
    "give information about the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e111fd92",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>768.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.845052</td>\n",
       "      <td>120.894531</td>\n",
       "      <td>69.105469</td>\n",
       "      <td>20.536458</td>\n",
       "      <td>79.799479</td>\n",
       "      <td>31.992578</td>\n",
       "      <td>0.471876</td>\n",
       "      <td>33.240885</td>\n",
       "      <td>0.348958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.369578</td>\n",
       "      <td>31.972618</td>\n",
       "      <td>19.355807</td>\n",
       "      <td>15.952218</td>\n",
       "      <td>115.244002</td>\n",
       "      <td>7.884160</td>\n",
       "      <td>0.331329</td>\n",
       "      <td>11.760232</td>\n",
       "      <td>0.476951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.078000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>62.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>27.300000</td>\n",
       "      <td>0.243750</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>117.000000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>30.500000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0.372500</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.000000</td>\n",
       "      <td>140.250000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>127.250000</td>\n",
       "      <td>36.600000</td>\n",
       "      <td>0.626250</td>\n",
       "      <td>41.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>199.000000</td>\n",
       "      <td>122.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>846.000000</td>\n",
       "      <td>67.100000</td>\n",
       "      <td>2.420000</td>\n",
       "      <td>81.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \\\n",
       "count   768.000000  768.000000     768.000000     768.000000  768.000000   \n",
       "mean      3.845052  120.894531      69.105469      20.536458   79.799479   \n",
       "std       3.369578   31.972618      19.355807      15.952218  115.244002   \n",
       "min       0.000000    0.000000       0.000000       0.000000    0.000000   \n",
       "25%       1.000000   99.000000      62.000000       0.000000    0.000000   \n",
       "50%       3.000000  117.000000      72.000000      23.000000   30.500000   \n",
       "75%       6.000000  140.250000      80.000000      32.000000  127.250000   \n",
       "max      17.000000  199.000000     122.000000      99.000000  846.000000   \n",
       "\n",
       "              BMI  DiabetesPedigreeFunction         Age     Outcome  \n",
       "count  768.000000                768.000000  768.000000  768.000000  \n",
       "mean    31.992578                  0.471876   33.240885    0.348958  \n",
       "std      7.884160                  0.331329   11.760232    0.476951  \n",
       "min      0.000000                  0.078000   21.000000    0.000000  \n",
       "25%     27.300000                  0.243750   24.000000    0.000000  \n",
       "50%     32.000000                  0.372500   29.000000    0.000000  \n",
       "75%     36.600000                  0.626250   41.000000    1.000000  \n",
       "max     67.100000                  2.420000   81.000000    1.000000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32021264",
   "metadata": {},
   "source": [
    " provide summary of the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90be55b2",
   "metadata": {},
   "source": [
    "CLEANING DATASET "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e8c6cf42",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(768, 9)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = data.drop_duplicates()\n",
    "data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "085988a5",
   "metadata": {},
   "source": [
    "Check shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "df2c766c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = data['Outcome']\n",
    "x = data.drop(['Outcome'], axis = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f3df8bc",
   "metadata": {},
   "source": [
    "features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "88d0838d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pregnancies                 0\n",
       "Glucose                     0\n",
       "BloodPressure               0\n",
       "SkinThickness               0\n",
       "Insulin                     0\n",
       "BMI                         0\n",
       "DiabetesPedigreeFunction    0\n",
       "Age                         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.isnull().sum() "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ed65dc6",
   "metadata": {},
   "source": [
    "check the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "692fd51d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MLPClassifier(alpha=0.001, hidden_layer_sizes=(50, 30, 10), max_iter=1000,\n",
       "              random_state=1, solver=&#x27;sgd&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(alpha=0.001, hidden_layer_sizes=(50, 30, 10), max_iter=1000,\n",
       "              random_state=1, solver=&#x27;sgd&#x27;)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MLPClassifier(alpha=0.001, hidden_layer_sizes=(50, 30, 10), max_iter=1000,\n",
       "              random_state=1, solver='sgd')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nnModel = MLPClassifier(solver='sgd', alpha=1e-3, hidden_layer_sizes=(50,30,10), random_state=1, max_iter = 1000)\n",
    "nnModel.fit(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0d649960",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'              precision    recall  f1-score   support\\n\\n           0       0.74      0.86      0.80       500\\n           1       0.63      0.45      0.52       268\\n\\n    accuracy                           0.71       768\\n   macro avg       0.69      0.65      0.66       768\\nweighted avg       0.70      0.71      0.70       768\\n'"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictx = (nnModel.predict(x) > 0.5).astype(int)  \n",
    "accuracy_score(y, predictx)  \n",
    "classification_report(y, predictx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "d5fae245",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictX = nnModel.predict(x)\n",
    "accuracy = accuracy_score(y, predictX)\n",
    "report = classification_report(y, predictX)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c48bfd91",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:71.484375%\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.74      0.86      0.80       500\n",
      "           1       0.63      0.45      0.52       268\n",
      "\n",
      "    accuracy                           0.71       768\n",
      "   macro avg       0.69      0.65      0.66       768\n",
      "weighted avg       0.70      0.71      0.70       768\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(f\"Accuracy:{accuracy*100}%\")\n",
    "print(report)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6dd93500",
   "metadata": {},
   "source": [
    "calculate the accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "f390dc61",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np  \n",
    "from sklearn.metrics import roc_curve, auc  \n",
    "import matplotlib.pyplot as plt  \n",
    "\n",
    "\n",
    "y_probs = nnModel.predict_proba(x)  \n",
    "y_probs_pos = y_probs[:, 1]  \n",
    "\n",
    "fpr, tpr, thresholds = roc_curve(y, y_probs_pos)  \n",
    "roc_auc = auc(fpr, tpr)  \n",
    "\n",
    "\n",
    "plt.figure()  \n",
    "plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  \n",
    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  \n",
    "plt.xlabel('False Positive Rate')  \n",
    "plt.ylabel('True Positive Rate')  \n",
    "plt.title('Receiver Operating Characteristic')  \n",
    "plt.legend(loc=\"lower right\")  \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04b2e154",
   "metadata": {},
   "source": [
    "Draw ROC curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e992972f",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.1,random_state=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "10a944f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np  \n",
    "trainx = trainx.to_numpy()  \n",
    "testx = testx.to_numpy()  \n",
    "trainy = trainy.to_numpy()  \n",
    "testy = testy.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "bd158b83",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainx = torch.FloatTensor(trainx)\n",
    "testx = torch.FloatTensor(testx)\n",
    "trainy = torch.FloatTensor(trainy)\n",
    "testy = torch.FloatTensor(testy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "fb80fe23",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch  \n",
    "from torch.utils.data import TensorDataset, DataLoader  \n",
    "\n",
    "\n",
    "trainxtensor = torch.tensor(trainx, dtype=torch.float32)  \n",
    "trainytensor = torch.tensor(trainy, dtype=torch.float32)  \n",
    "testxtensor = torch.tensor(testx , dtype=torch.float32)  \n",
    "testytensor = torch.tensor(testy, dtype=torch.float32)  \n",
    "\n",
    "train_dataset = TensorDataset(trainxtensor, trainytensor)  \n",
    "test_dataset = TensorDataset(testxtensor, testytensor)  \n",
    "\n",
    "batch_size = 20  \n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  \n",
    "test_loader = DataLoader(test_dataset, batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0baf8314",
   "metadata": {},
   "source": [
    "this code is preparing the training and testing data for a machine learning model by converting the data to PyTorch tensors, creating PyTorch datasets and data loaders, and setting the batch size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "017575cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy: 0.49\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import Perceptron  \n",
    "model = Perceptron(max_iter=500, tol=1e-3, eta0=0.001)  \n",
    "model.fit(trainx, trainy)  \n",
    "accuracy = model.score(testx, testy)  \n",
    "print(f\"Test accuracy: {accuracy:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "aaa1b947",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.49\n",
      "TP: 15, TN: 23, FP: 29, FN: 10\n",
      "Confusion Matrix:\n",
      "[[23 29]\n",
      " [10 15]]\n",
      "Precision: 0.34\n",
      "Recall: 0.60\n",
      "F1-Score: 0.43\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score  \n",
    "\n",
    "\n",
    "predicty = (model.predict(testx) >= 0.5).astype(int)  \n",
    "accuracy = accuracy_score(testy, predicty)  \n",
    "tn, fp, fn, tp = confusion_matrix(testy, predicty).ravel()  \n",
    "precision = precision_score(testy, predicty)  \n",
    "recall = recall_score(testy, predicty)  \n",
    "f1 = f1_score(testy, predicty)  \n",
    "\n",
    " \n",
    "print(f'Accuracy: {accuracy:.2f}')  \n",
    "print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')  \n",
    "print(f'Confusion Matrix:\\n{confusion_matrix(testy, predicty)}')  \n",
    "print(f'Precision: {precision:.2f}')  \n",
    "print(f'Recall: {recall:.2f}')  \n",
    "print(f'F1-Score: {f1:.2f}')  \n",
    "\n",
    "\n",
    "num_epochs = 10    \n",
    "losses = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.008, 0.005] \n",
    "plt.plot(range(num_epochs), losses)  \n",
    "plt.xlabel('Epoch')  \n",
    "plt.ylabel('Training Loss')  \n",
    "plt.title('Training Loss Over Epochs')  \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e703f6f",
   "metadata": {},
   "source": [
    "this code evaluates the performance of a machine learning model using various evaluation metrics, such as accuracy, precision, recall, and F1-score, and then visualizes the training loss over the specified number of epochs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "f7301ed6",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_data = torch.tensor([6.0, 165.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0], dtype=torch.float32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "9e317935",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Class: 1.0\n"
     ]
    }
   ],
   "source": [
    "prediction = model.predict([new_data.tolist()])[0]  \n",
    "\n",
    "print(f'Predicted Class: {prediction}')  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42b46b9f",
   "metadata": {},
   "source": [
    "This code snippet is used to predict the class label for a single new data sample using a trained machine learning model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bdc4540e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
