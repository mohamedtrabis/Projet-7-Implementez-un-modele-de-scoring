import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import streamlit.components.v1 as components

from sklearn.feature_extraction.text import TfidfVectorizer

import shap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelEncoder, LabelBinarizer
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, classification_report, ConfusionMatrixDisplay

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance

import lightgbm as lgbm

from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV, RandomizedSearchCV
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
from sklearn.compose import make_column_transformer, make_column_selector,ColumnTransformer




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_percentage_error

import lightgbm as lgbm

from PIL import Image
from IPython.core.display import display,HTML



#image = Image.open("images/da.png")
#newsize = (212, 116)
#image = image.resize(newsize)
#st.image(image,'')

st.subheader("Dashbord App")

# Create a page dropdown
page = st.sidebar.radio("Choisissez votre Application",
                        ["LightGBM", "XGBoost"])


#Style CSS
st.write("""
<style>

table {
font-size:13px !important;
border:3px solid #6495ed;
border-collapse:collapse;
margin:auto;
width: auto;
height: auto;
}

th {
font-family:monospace bold;
border:1px dotted #6495ed;
background-color:#EFF6FF;
text-align:center;
}

td {
font-family:sans-serif;
font-size:95%;
border:1px solid #6495ed;
text-align:left;
width:auto;
height:60px;
}

.url {
  height:60px;
  margin-left:auto;
  margin-right:auto;
  display:block;
  -webkit-transform: scale(1.05);
  -moz-transform: scale(1.05);
  -o-transform: scale(1.05);
  transform: scale(1.05);

  -webkit-transition: all 700ms ease-in-out;
  -moz-transition: all 700ms ease-in-out;
  -o-transition: all 700ms ease-in-out;
  transition: all 700ms ease-in-out;
}

td:hover {
  font-family:sans-serif;
  /*font-weight: bolder; */
  font-size:120%;
  background-color: #f4f4f4;
  }


td:hover .url {
  /*-ms-transform: scale(1) translate(0px);*/ /* IE 9 */
  /*-webkit-transform: scale(1) translate(0px);*/ /* Safari 3-8 */
  /*transform: scale(1);*/ /* (200% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
  width:auto;
  height:250px;
}

</style>
""", unsafe_allow_html=True)


def food_caract_entree():
    code = st.text_input("Entrer le code client", 100007)


    data={
        'code':code,
    }

    food_features = pd.DataFrame(data,index=[0])
    return food_features
#--------------------------------------------------------------------------------------------------------------------

input_df=food_caract_entree()


def path_to_image_html(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<img src="'+ path + '" style=max-height:60px;margin-left:auto;margin-right:auto;display:block;"/>'

#----------------------------------------------------------------------------------------------------------------
def path_to_image_url(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<div class ="image" ><img class="url" src="'+ path + '""/></div>'
#----------------------------------------------------------------------------------------------------------------

#Transformer les données d'entrée en données adaptées à notre modèle
#importer la base de données
df_train1=pd.read_csv('../df_test.csv')

columns = ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

donnee_entree=pd.concat([input_df, df_train1[columns]])

donnee_entree=donnee_entree[:1]


columns_result = ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
donnee_sortie=pd.DataFrame(df_train1[columns_result])

var_code = donnee_entree['SK_ID_CURR'][0]

#-------------------------------------------------------------------------------------------------------------
if page == "LightGBM":
    #Méthode Undersampling

    # nombre de classes
    target_count_0, target_count_1 = df_train1['TARGET'].value_counts()

    # Classe séparée
    target_0 = df_train1[df_train1['TARGET'] == 0]
    target_1 = df_train1[df_train1['TARGET'] == 1] # affiche la forme de la classe
    print('target 0 :', target_0.shape)
    print('target 1 :', target_1.shape)


    # Undersample 0-class and concat the DataFrames of both class
    target_0_under = target_0.sample(target_count_1)
    test_under = pd.concat([target_0_under, target_1], axis=0)

    print('Random under-sampling:')
    print(test_under.TARGET.value_counts())

    #Définir X et y
    X = test_under.iloc[ :, 2:]
    y = test_under.iloc[ :,1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


    #Importer le modèle entrainé lightGBM
    lgbm_clf = pickle.load(open('../lgbm_clf.pkl', 'rb'))

    #Prédire le résultat sur les données X_test
    y_pred=lgbm_clf.predict(X_test)


    # view accuracy
    accuracy=accuracy_score(y_pred, y_test)
    st.write('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    #Prediction resultat
    donnee_entree['code'] = donnee_entree['code'].apply(str)
    df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)
    var_code = donnee_entree['code'][0]

    pred_client = df_train1[df_train1['SK_ID_CURR']==var_code]

    tab = ['No Default', 'Default']
    y_pred=lgbm_clf.predict(pred_client.iloc[ :, 2:])
    st.write('TARGET du client prédit: ',tab[y_pred[0]])

    # Initialize SHAP Tree explainer
    explainer = shap.TreeExplainer(lgbm_clf, model_output='margin')
    shap_values = explainer.shap_values(X_test)

    # Baseline value
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    print(f"Explainer expected value: {expected_value}")

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

    # force_plot
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][index_client],X_train.iloc[0,:]))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # decision_plot
    decision = shap.decision_plot(base_value = expected_value,
                       shap_values=shap_values[0][index_client],
                       features = X_test,
                       feature_names=X_test.columns.tolist(),
                       link='logit')

    st.pyplot(decision)
    #st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

    #Summarize the effects of all the features
    st.pyplot(shap.summary_plot(shap_values, pred_client.iloc[ :, 2:]))
#--------------------------------------------------------------------------------------------------------------------
if page == "XGBoost":
    #Importer le modèle entrainé lightGBM
    xgb_clf = pickle.load(open('../xgb_clf.pkl', 'rb'))

    # Définir X et y
    X = df_train1.iloc[:, 2:]
    y = df_train1.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

    # Prédire le résultat sur les données X_test
    y_pred = xgb_clf.predict(X_test)

    # view accuracy
    accuracy = accuracy_score(y_pred, y_test)
    st.write('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    # Prediction resultat
    donnee_entree['code'] = donnee_entree['code'].apply(str)
    df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)
    var_code = donnee_entree['code'][0]

    pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

    tab = ['No Default', 'Default']
    y_pred = xgb_clf.predict(pred_client.iloc[:, 2:])
    st.write('TARGET du client prédit: ', tab[y_pred[0]])

    # Initialize SHAP Tree explainer
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(df_train1.iloc[:, 2:])

    # Baseline value
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    print(f"Explainer expected value: {expected_value}")

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

    # force_plot
    st_shap(shap.force_plot(explainer.expected_value,
                            shap_values[index_client], features=df_train1.iloc[index_client, 2:],
                            feature_names=X_test.columns[0:20],
                            show=False,
                            #plot_cmap=['#77dd77', '#f99191']
                            ))

    #st.set_option('deprecation.showPyplotGlobalUse', False)

    # decision_plot
    #ecision = shap.decision_plot(base_value=expected_value,
                                  #shap_values=shap_values[0][index_client],
                                  #features=X_test,
                                  #feature_names=X_test.columns.tolist(),
                                  #link='logit')

    #st.pyplot(decision)
    # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

    # Summarize the effects of all the features
    #st.pyplot(shap.summary_plot(shap_values[4], pred_client.iloc[:, 2:]))
# --------------------------------------------------------------------------------------------------------------------


#clf=RandomForestClassifier()
#clf.fit(iris.data,iris.target)

#prediction=clf.predict(df)

#st.subheader("La catégorie de la fleur d'iris est:")
#st.write(iris.target_names[prediction])
#----------------------------------------------------------------------------------------------------------------







#importer le modèle
#load_model=pickle.load(open('prevision_credit.pkl','rb'))


#appliquer le modèle sur le profil d'entrée
#prevision=load_model.predict(donnee_entree)

#st.subheader('Résultat de la prévision')
#st.write(prevision)
