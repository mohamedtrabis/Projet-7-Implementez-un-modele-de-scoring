import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components
import shap

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

st.set_page_config(
    page_title="Home Credit",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon = 'Image/logo_home_credit.gif',
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")


image_logo = Image.open("Image/home credit.jpg")
newsize = (300, 168)

left, mid ,right = st.columns([1,1, 1])

with mid:
    image_logo = image_logo.resize(newsize)
    st.image(image_logo, '')


# image = Image.open("images/da.png")
# newsize = (212, 116)
# image = image.resize(newsize)
# st.image(image,'')

st.subheader("Dashbord Application 📈")

# Create a page dropdown
page = st.sidebar.radio("Choisissez votre Application",
                        ["LightGBM", "XGBoost"])


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

# ----------------------------------------------------------------------------------------------------------------
#@st.cache(hash_funcs={XGBClassifier: id})
def get_explainer1():
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(df_train1.iloc[:, 2:])
    return explainer, shap_values


# ----------------------------------------------------------------------------------------------------------------
left_, mid_ ,right_ = st.columns([1,1, 1])

with mid_:
    input_df = caract_entree()

# Transformer les données d'entrée en données adaptées à notre modèle
# importer la base de données
df_train1 = pd.read_csv('db/df_train1_2000.csv')
df_train = pd.read_csv('db/df_train_2000.csv')

donnee_entree = pd.concat([input_df, df_train1])

donnee_entree = donnee_entree[:1]

donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]

df_train['AGE'] = abs(np.around(df_train['DAYS_BIRTH']/365,2))
df_train['YEARS_EMPLOYED'] = abs(np.around(df_train['DAYS_EMPLOYED']/365,2))

col = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER','AGE',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_EMPLOYED',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']

donnee_sortie = df_train[col].copy()

var_code = donnee_entree['SK_ID_CURR'][0]

donnee_sortie['SK_ID_CURR'] = donnee_sortie['SK_ID_CURR'].apply(str)
donnee_sortie = donnee_sortie[(donnee_sortie['SK_ID_CURR'] == var_code)]

#st.write(HTML(donnee_sortie.to_html(index=False, escape=False, )))
st.table(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie.assign(hack='').set_index('hack'))

#st.dataframe(donnee_sortie)
# -------------------------------------------------------------------------------------------------------------
if page == "LightGBM":
    # Méthode Undersampling

    # nombre de classes
    target_count_0, target_count_1 = df_train1['TARGET'].value_counts()

    # Classe séparée
    target_0 = df_train1[df_train1['TARGET'] == 0]
    target_1 = df_train1[df_train1['TARGET'] == 1]  # affiche la forme de la classe
    print('target 0 :', target_0.shape)
    print('target 1 :', target_1.shape)

    # Undersample 0-class and concat the DataFrames of both class
    target_0_under = target_0.sample(target_count_1)
    test_under = pd.concat([target_0_under, target_1], axis=0)

    # Définir X et y
    X = test_under.iloc[:, 2:]
    y = test_under.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

    # Importer le modèle entrainé lightGBM
    lgbm_clf = pickle.load(open('lgbm_clf.pkl', 'rb'))

    # Prédire le résultat sur les données X_test
    y_pred = lgbm_clf.predict(X_test)

    # view accuracy
    accuracy = accuracy_score(y_pred, y_test)
    #st.write('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    # Prediction resultat
    donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
    df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)
    var_code = donnee_entree['SK_ID_CURR'][0]

    pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

    tab = ['No Default', 'Default']
    y_pred = lgbm_clf.predict_proba(pred_client.iloc[:, 2:])
    risk = "{:,.0f}".format(y_pred[0][1]*100)
    pred_0 = "{:,.0f}".format(y_pred[0][0]*100)
    #st.write('TARGET du client prédit: ', tab[y_pred[0]])
    st.write(y_pred)


    prediction_0, prediction_1, prediction_3 = st.columns(3)

    prediction_0.metric(label='Prédiction remboursement (min 80%)',value=pred_0+str('%'),delta=int(pred_0)-80)
    prediction_1.metric(label='Risque de défaut de paiement (min 30%)',value=risk+str('%'),delta=30-int(risk))

    #st.write('Risque de défaut de paiement : ', risk + str('%'))
    if((y_pred[0][1])<0.5):
        st.markdown('''<div class='box'>'''+'Risque de défaut de paiement : </div><div class="box" id="flag_green">'''+risk+str('%')+'''</div><div class='box'><img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQy1EjMg5nWNx1f7Fq8OwyXIc_Zgcw7ho90tA91zVAfEST6UnwggqOicekl4gxwuvOLk_M&usqp=CAU', height=40/></div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class='box'>''' + 'Risque de défaut de paieme   nt : </div><div class="box" id="flag_red">''' + risk + str('%') + '''</div><div class='box'><img src='https://americanmigrainefoundation.org/wp-content/uploads/2017/03/iStock_45096094_SMALL.jpg', height=40/></div>''',unsafe_allow_html=True)


    # Initialize SHAP Tree explainer
    explainer, shap_values = get_explainer(df_train1, lgbm_clf)

    # explainer = shap.TreeExplainer(lgbm_clf, model_output='raw')
    # shap_values = explainer.shap_values(df_train1.iloc[ :, 2:])

    # Baseline value
    expected_value = explainer.expected_value

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

    col1, col2 = st.columns([1, 1])

    st.markdown("<h2 style='text-align: center; color: black;'>Force plot</h2>", unsafe_allow_html=True)
    # force_plot
    st_shap(
        shap.force_plot(explainer.expected_value[1], shap_values[1][index_client], df_train1.iloc[index_client, 2:]))

    with col1:
        st.markdown("<h2 style='text-align: center; color: black;'>Decision plot</h2>", unsafe_allow_html=True)
        # decision_plot
        decision = shap.decision_plot(base_value=explainer.expected_value[1],
                                      shap_values=shap_values[1][index_client],
                                      features=df_train1.iloc[index_client, 2:],
                                      feature_names=X_test.columns.tolist(),
                                      link='logit')

        st.pyplot(decision, unsafe_allow_html=False)
        # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)
    with col2:
        st.markdown("<h2 style='text-align: center; color: black;'>Summary plot</h2>", unsafe_allow_html=True)
        # Summarize the effects of all the features
        st.pyplot(shap.summary_plot(shap_values, pred_client.iloc[:, 2:]))

    st.set_option('deprecation.showPyplotGlobalUse', False)

    #explainer = shap.Explainer(xgb_clf, X_train)
    #st.pyplot(shap.plots.waterfall(shap_values[0]))
# --------------------------------------------------------------------------------------------------------------------
if page == "XGBoost":
    # Importer le modèle entrainé lightGBM
    xgb_clf = pickle.load(open('xgb_clf.pkl', 'rb'))

    # Définir X et y
    X = df_train1.iloc[:, 2:]
    y = df_train1.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

    # Prédire le résultat sur les données X_test
    y_pred = xgb_clf.predict(X_test)

    # view accuracy
    accuracy = accuracy_score(y_pred, y_test)
    #st.write('XGBoost Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    # Prediction resultat
    donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
    df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)
    var_code = donnee_entree['SK_ID_CURR'][0]

    pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

    tab = ['No Default', 'Default']
    y_pred = xgb_clf.predict_proba(pred_client.iloc[:, 2:])
    risk = "{:,.0f}".format(y_pred[0][1] * 100)
    pred_0 = "{:,.0f}".format(y_pred[0][0]*100)
    # st.write('TARGET du client prédit: ', tab[y_pred[0]])
    st.write(y_pred)

    prediction_0, prediction_1, prediction_3 = st.columns(3)

    prediction_0.metric(label='Prédiction remboursement (min 80%)',value=pred_0+str('%'),delta=int(pred_0)-80)
    prediction_1.metric(label='Risque de défaut de paiement (min 30%)',value=risk+str('%'),delta=30-int(risk))

    # st.write('Risque de défaut de paiement : ', risk + str('%'))
    if((y_pred[0][1]*100)<50):
        st.markdown('''<div class='box'>'''+'Risque de défaut de paiement : </div><div class="box" id="flag_green">'''+risk+str('%')+'''</div><div class='box'><img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQy1EjMg5nWNx1f7Fq8OwyXIc_Zgcw7ho90tA91zVAfEST6UnwggqOicekl4gxwuvOLk_M&usqp=CAU', height=40/></div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class='box'>''' + 'Risque de défaut de paiement : </div><div class="box" id="flag_red">''' + risk + str('%') + '''</div><div class='box'><img src='https://americanmigrainefoundation.org/wp-content/uploads/2017/03/iStock_45096094_SMALL.jpg', height=40/></div>''',unsafe_allow_html=True)

    # Initialize SHAP Tree explainer
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(df_train1.iloc[:, 2:])
    #explainer, shap_values = get_explainer1()

    # Baseline value
    expected_value = explainer.expected_value

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]
    st.markdown("<h2 style='text-align: center; color: black;'>Force plot</h2>", unsafe_allow_html=True)
    # force_plot
    st_shap(shap.force_plot(explainer.expected_value,
                            shap_values[index_client], features=df_train1.iloc[index_client, 2:],
                            #feature_names=df_train1.columns[0:20],
                            show=False,
                            # plot_cmap=['#77dd77', '#f99191']
                            ))

    col1, col2 = st.columns([1, 1])

    with col1:
        #st.pyplot(shap.plots.waterfall(shap_values1[0]))

        st.markdown("<h2 style='text-align: center; color: black;'>Summary plot</h2>", unsafe_allow_html=True)
        # Summarize the effects of all the features
        st.pyplot(shap.summary_plot(shap_values, df_train1.iloc[:, 2:]))
    with col2:
        st.markdown("<h2 style='text-align: center; color: black;'>Waterfall legacy</h2>", unsafe_allow_html=True)
        st.pyplot(shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[index_client]))

    # st.set_option('deprecation.showPyplotGlobalUse', False)

    # decision_plot
    # ecision = shap.decision_plot(base_value=expected_value,
    # shap_values=shap_values[0][index_client],
    # features=X_test,
    # feature_names=X_test.columns.tolist(),
    # link='logit')

    # st.pyplot(decision)
    # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

    # Summarize the effects of all the features
    # st.pyplot(shap.summary_plot(shap_values[4], pred_client.iloc[:, 2:]))
# --------------------------------------------------------------------------------------------------------------------


# clf=RandomForestClassifier()
# clf.fit(iris.data,iris.target)

# prediction=clf.predict(df)

# st.subheader("La catégorie de la fleur d'iris est:")
# st.write(iris.target_names[prediction])
# ----------------------------------------------------------------------------------------------------------------


# importer le modèle
# load_model=pickle.load(open('prevision_credit.pkl','rb'))


# appliquer le modèle sur le profil d'entrée
# prevision=load_model.predict(donnee_entree)

# st.subheader('Résultat de la prévision')
# st.write(prevision)
