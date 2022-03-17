import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit.components.v1 as components
import shap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelEncoder, \
    LabelBinarizer
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, classification_report, ConfusionMatrixDisplay

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from xgboost import plot_importance

import lightgbm as lgbm

from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV, \
    RandomizedSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
import lightgbm as lgbm
from IPython.core.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_percentage_error

from PIL import Image

@st.cache(suppress_st_warning=True)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
# --------------------------------------------------------------------------------------------------------------------
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
@st.cache(suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
# --------------------------------------------------------------------------------------------------------------------
def caract_entree():
    SK_ID_CURR = st.text_input("Entrer le code client", 100007)

    data = {
        'SK_ID_CURR': SK_ID_CURR,
    }

    df = pd.DataFrame(data, index=[0])
    return df


# --------------------------------------------------------------------------------------------------------------------

def path_to_image_html(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<img src="' + path + '" style=max-height:60px;margin-left:auto;margin-right:auto;display:block;"/>'


# ----------------------------------------------------------------------------------------------------------------
def path_to_image_url(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<div class ="image" ><img class="url" src="' + path + '""/></div>'


# ----------------------------------------------------------------------------------------------------------------
@st.cache(suppress_st_warning=True)
def get_explainer(df, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.iloc[:, 2:])
    return explainer, shap_values

# ----------------------------------------------------------------------------------------------------------------
#@st.cache(hash_funcs={XGBClassifier: id})
def get_explainer1():
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(df_train1.iloc[:, 2:])
    return explainer, shap_values
# ----------------------------------------------------------------------------------------------------------------
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
# ----------------------------------------------------------------------------------------------------------------
# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var, id_client, nrow=2):
    i = 0
    t1 = df_train.loc[df_train['TARGET'] != 0]
    t0 = df_train.loc[df_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6 * nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw_adjust=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw_adjust=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
        client = df_train[feature][df_train['SK_ID_CURR'] == id_client].values[0]
        var = (df_train[feature][df_train[feature] == id_client].count()) / ((df_train[feature].count()))
        plt.text(client*1.01, var, 'Client %s' % feature, fontsize=11)
        plt.axvline(client, c='red')
        plt.legend()
    st.pyplot(fig)
# ----------------------------------------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def try_read_df(file1, file2):
    return pd.read_csv(file1), pd.read_csv(file2)