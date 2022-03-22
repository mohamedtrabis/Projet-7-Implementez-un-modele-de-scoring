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

from scipy.stats import gaussian_kde

#import st_state_patch

from PIL import Image

#@st.cache(suppress_st_warning=True)
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
def get_explainer():
    explainer = shap.TreeExplainer(lgbm_clf)
    shap_values = explainer.shap_values(df_train1.iloc[:, 2:-2])
    expected_value = explainer.expected_value
    return explainer, shap_values, expected_value
# ----------------------------------------------------------------------------------------------------------------
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
# ----------------------------------------------------------------------------------------------------------------
# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var, id_client, nrow=2, ncol=2):
    i = 0
    j=0
    t1 = df_train1.loc[df_train1['TARGET'] != 0]
    t0 = df_train1.loc[df_train1['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 1, figsize=(12, 5 * nrow))

    for feature in var:
        sns.set_style("dark")
        i += 1
        plt.subplot(nrow, ncol, i)
        sns.kdeplot(t1[feature], bw_adjust=0.5, label="In default", color='red', shade=True)
        sns.kdeplot(t0[feature], bw_adjust=0.5, label="No default", color='blue', shade=True)
        plt.ylabel('Density plot', fontsize=8)
        plt.xlabel(feature, fontsize=8)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=10)
        client = df_train1[feature][df_train1['SK_ID_CURR'] == str(id_client)].values[0]
        var = (df_train1[feature][df_train1[feature] == str(id_client)].count()) / ((df_train1[feature].count()))
        #plt.text(client, var, int(client), fontsize=8)
        #plt.axvline(client, c='black')
        #plt.title(feature, fontsize=9)

        plt.legend(fontsize=8)
        if col_selected[j] in description['Row'].values:
            title = description['Description'][description['Row'] == col_selected[j]].head(1).values[0]
            #st.sidebar.write(col_selected[j]+' : '+title)
            plt.title(title,fontsize=11)
        j=j+1

        density = gaussian_kde(df_train1[feature])
        max = density(df_train1[feature]).max()
        max_features = df_train1[feature].max()
        x = client
        y = density(x)

        plt.annotate(var_code+'\n'+str(np.ceil(x)), xy=(x, y), xytext=(max_features/1.2, max* 0.5), fontsize=8,
                     #arrowprops=dict(facecolor='green', shrink=0.01),
                     #bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                     #arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=80,rad=20")
                     bbox=dict(boxstyle ="round", fc ="0.8"),
                     arrowprops=dict(arrowstyle = "->",connectionstyle = "angle,angleA=90,angleB=180,rad=0",color='black')
                     )

    st.pyplot(fig)
# ----------------------------------------------------------------------------------------------------------------
def gauge(col):
    fig = go.Figure()
    if (float(risk) > 30):
        fig.add_trace(go.Indicator(

            value=y_pred[0][1] * 100,
            delta={'reference': 30, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={

                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue",'visible': True},
                'bar': {'color': "red"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30}
            },
            domain={'row': 0, 'column': 0}))

    else:
        fig.add_trace(go.Indicator(

            value=y_pred[0][1] * 100,
            delta={'reference': 30, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={

                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue",'visible': True},
                'bar': {'color': "green"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30}
            },
            domain={'row': 0, 'column': 0}))

    fig.update_layout(
        # paper_bgcolor = "lightgray",
        # font = {'color': "darkblue", 'family': "Arial"},
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        autosize=True,
        # width=1000,
        template={'data': {'indicator': [{
            'title': {'text': "Defaut de paiement (> 30)", 'font': {'size': 20}},
            'mode': "number+delta+gauge",
            'delta': {'reference': 100}}]
        }})

    col.plotly_chart(fig, use_container_width=True)
# ----------------------------------------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def try_read_df(file):
    return pd.read_csv(file)

# ----------------------------------------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def try_read_desc(file):
    return pd.read_csv(file, encoding= 'unicode_escape', usecols=['Row','Description'])