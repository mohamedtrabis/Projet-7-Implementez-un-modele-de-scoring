#dirname = "../"
dirname = ""

exec(open(dirname+"function.py").read())

from PIL import Image
st.set_page_config(
    page_title="Home Credit",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon = 'Image/logo_home_credit.gif',
)

#s = st.session_state

local_css(dirname+"style.css")
pd.options.display.float_format = '{:,.2f}'.format

image_logo = Image.open("Image/home credit1.jpg")
newsize = (580, 354)

left, mid ,right = st.columns([1,2, 1])

with mid:
    image_logo = image_logo.resize(newsize)
    st.image(image_logo, '')


# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1
selected = streamlit_menu(example=EXAMPLE_NO)



st.header("Dashboard 📈")

# Create a page dropdown
#page = st.sidebar.radio("Choisissez votre Application",["LightGBM", "XGBoost"])

st.set_option('deprecation.showPyplotGlobalUse', False)
#-----------------------------------------------------------------------------------------------------------------
# Transformer les données d'entrée en données adaptées à notre modèle
# importer la base de données
file = dirname+'db/df_train1_2000.csv'
file_desc = dirname+'db/HomeCredit_columns_description.csv'

description = try_read_desc(file_desc)
df_train1 =try_read_df(file)

description = description.drop_duplicates(subset='Row', keep='last')


left_, mid_ ,right_ = st.columns([1,1, 1])

with mid_:
    #SK_ID_CURR = st.sidebar.text_input("Entrer le code client", 100007)
    SK_ID_CURR = st.sidebar.selectbox('Sélectionner un client', df_train1['SK_ID_CURR'].head(1000))
    data = {
        'SK_ID_CURR': SK_ID_CURR,
    }

    input_df = pd.DataFrame(data, index=[0])


donnee_entree = pd.concat([input_df, df_train1])

donnee_entree = donnee_entree[:1]

donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]

df_train1['AGE'] = abs(np.around(df_train1['DAYS_BIRTH']/365,2))
df_train1['YEARS_EMPLOYED'] = abs(np.around(df_train1['DAYS_EMPLOYED']/365,2))

col = ['SK_ID_CURR', 'CODE_GENDER','AGE',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_EMPLOYED']

donnee_sortie = df_train1[col].copy()

var_code = donnee_entree['SK_ID_CURR'][0]

donnee_sortie['SK_ID_CURR'] = donnee_sortie['SK_ID_CURR'].apply(str)
donnee_sortie = donnee_sortie[(donnee_sortie['SK_ID_CURR'] == var_code)]

#st.table(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie)

#extraire le code client et les informations relatives au client
donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]
pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

if len(pred_client)==0:
    st.write('No data found')
    st.stop()


#plot_client = st.sidebar.checkbox('Analyse données client')
#shapley = st.sidebar.checkbox('Analyse Shapley')
#colonne_descr = st.sidebar.checkbox('Description des variables')

def clear_form():
    st.session_state["select_1"] = col_selected[0]

#initialize session state
if "select_1" not in st.session_state :
    st.session_state.select_1 = False

#initialize session state
if "load_state" not in st.session_state :
    st.session_state.load_state = False

# -------------------------------------------------------------------------------------------------------------
st.set_option('deprecation.showPyplotGlobalUse', False)

# Importer le modèle entrainé lightGBM
lgbm_clf = pickle.load(open(dirname+'lgbm_clf.pkl', 'rb'))

# Prediction resultat
tab = ['No Default', 'Default']
y_pred = lgbm_clf.predict_proba(pred_client.iloc[:, 2:-2])
risk = "{:,.0f}".format(y_pred[0][1]*100)
pred_0 = "{:,.0f}".format(y_pred[0][0]*100)
if selected!='Description':
    with st.expander("Informations Client", expanded=True):
        col_0, col_1, col_3, col_4  = st.columns([1,3,4,1])

        #Working with checkbox
        if len(pred_client)!=0 and selected!='Description':
            donnee_sortie['CODE_GENDER'] = donnee_sortie['CODE_GENDER'].map({0: 'M', 1: 'F'})
            donnee_sortie['FLAG_OWN_CAR'] = donnee_sortie['FLAG_OWN_CAR'].map({0: 'N', 1: 'Y'})
            donnee_sortie['FLAG_OWN_REALTY'] = donnee_sortie['FLAG_OWN_REALTY'].map({0: 'Y', 1: 'N'})
            df = donnee_sortie.T
            df.columns = ['Data']
            col_1.write(HTML(df.to_html(escape=False)))
            gauge(col_3)


if selected == "Description" or selected == "Data Client":
#if colonne_descr:
    select_desc = st.sidebar.selectbox('Sélectionner une variables', description['Row'])

    st.markdown("<div id='graph_shap'><h3>Description des variables</h3></div></br>", unsafe_allow_html=True)
    #t.sidebar.write(description['Description'][description['Row']==select_desc].head(1).values[0])
    st.write(HTML(description[['Row','Description']][description['Row']==select_desc].head(1).to_html(index = False, escape=False)))


# Create a list of possible values and multiselect menu with them in it.
# first we let streamlit know that we will be making a form
my_form = st.form(key="form")


if selected == "Analyse":
#if plot_client and len(pred_client)!=0:

    my_form.markdown("<div id='shapley'><h2>Analyse Client : "+var_code+"</h2></div>", unsafe_allow_html=True)


    var = ['All','EXT_SOURCE_MEAN', 'AMT_CREDIT', 'DAYS_BIRTH', 'INS_DPD_MEAN',
           'AMT_ANNUITY', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'AMT_GOODS_PRICE',
           'df_POS_CASH_balance_COUNT', 'PREV_CNT_PAYMENT_MEAN',
           'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN', 'DAYS_EMPLOYED',
           'APPROVED_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN',
           'DAYS_ID_PUBLISH', 'ACTIVE_DAYS_CREDIT_MEAN',
           'INS_AMT_PAYMENT_MEAN', 'INS_PAYMENT_PERC_MEAN',
           'INS_PAYMENT_DIFF_MEAN', 'POS_SK_DPD_DEF_MEAN', 'CODE_GENDER',
           'PREV_NAME_YIELD_GROUP_high_MEAN',
           'PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN', 'DAYS_LAST_PHONE_CHANGE',
           'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
           'INS_DAYS_ENTRY_PAYMENT_MEAN', 'INS_DBD_MEAN', 'FLAG_OWN_CAR',
           'BUREAU_AMT_CREDIT_SUM_MEAN', 'INS_DAYS_INSTALMENT_MEAN',
           'PREV_AMT_DOWN_PAYMENT_MEAN']

    col_selected = my_form.multiselect('Select Features', var)


    # all forms end with a submit button, that is how the user can trigger
    submit = my_form.form_submit_button(label="submit")
    #clear = my_form.form_submit_button(label="Clear", on_click=clear_form)

    if submit or st.session_state.load_state:

        st.session_state.load_state = True
        #st.session_state.select_1 = col_selected[0]

        if (len(col_selected) == 0):
            my_form.write('Please Select Features')

        elif (col_selected[0] == 'All'):
            col_selected = var[1:]
            plot_distribution_comp(col_selected, int(var_code), nrow=int(np.ceil(len(col_selected) / 2)), ncol=2)

        elif (len(col_selected)<3) :
            plot_distribution_comp(col_selected, int(var_code), nrow=1, ncol=len(col_selected))

        elif (len(col_selected)>0) :
            plot_distribution_comp(col_selected, int(var_code), nrow=int(np.ceil(len(col_selected)/2)), ncol=2)

if selected == "Shapley":
#if shapley:
    # Initialize SHAP Tree explainer
    explainer, shap_values, expected_value = get_explainer()


    st.markdown("<div id='shapley'><h2>Analyse Shapley</h2></div></br>", unsafe_allow_html=True)
    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

    shap1, shap2 = st.columns(2)
    shap1.markdown("<div id='graph_shap'><h3>Decision plot</h3></div>", unsafe_allow_html=True)
    # decision_plot
    decision = shap.decision_plot(base_value=explainer.expected_value[1],
                                  shap_values=shap_values[1][index_client],
                                  features=df_train1.iloc[index_client:, 2:-2],
                                  feature_names=df_train1.iloc[:, 2:-2].columns.tolist())

    shap1.pyplot(decision)
    # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

    shap2.markdown("<div id='graph_shap'><h3>Waterfall Plot</h3></div>", unsafe_allow_html=True)
    # Summarize the effects of all the features
    #shap2.pyplot(shap.summary_plot(shap_values, pred_client.iloc[:, 2:-2]))

    shap2.pyplot(shap.plots._waterfall.waterfall_legacy(expected_value[1],
                                                        shap_values[1][index_client],
                                                        feature_names = df_train1.iloc[:, 2:-2].columns,
                                                        max_display = 20))

    st.markdown("<div id='graph_shap'><h3>Force plot</h3></div>", unsafe_allow_html=True)
    # force_plot
    st_shap(shap.force_plot(expected_value[1], shap_values[1][index_client], df_train1.iloc[index_client, 2:-2]))

    #fig = px.density_contour(df_train[['AGE']], x='AGE')
    #fig = px.ecdf(df_train, x="AGE", color="TARGET", markers=True, lines=False, marginal="histogram", title='Life expectancy in Canada')
    #st.write(fig)

    #explainer = shap.Explainer(xgb_clf, X_train)
    #st.pyplot(shap.plots.waterfall(shap_values[0]))
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
