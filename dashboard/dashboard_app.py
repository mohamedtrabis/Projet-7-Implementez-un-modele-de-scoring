exec(open("function.py").read())

#import st_state_patch
#s = st.session_state


from PIL import Image
st.set_page_config(
    page_title="Home Credit",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon = 'Image/logo_home_credit.gif',
)

local_css("style.css")
pd.options.display.float_format = '{:,.2f}'.format

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
#page = st.sidebar.radio("Choisissez votre Application",["LightGBM", "XGBoost"])


#-----------------------------------------------------------------------------------------------------------------
# Transformer les données d'entrée en données adaptées à notre modèle
# importer la base de données
#df_train1 = pd.read_csv('../db/df_train1_2000.csv')
#df_train = pd.read_csv('../db/df_train_2000.csv')

#file = '../df_train1.csv'
#file2 = '../Data/application_train.csv'
file = '../db/df_train1_2000.csv'
#file2 = '../db/df_train_2000.csv'
st.set_option('deprecation.showPyplotGlobalUse', False)

df_train1 =try_read_df(file)

left_, mid_ ,right_ = st.columns([1,1, 1])

with mid_:
    SK_ID_CURR = st.text_input("Entrer le code client", 100007)
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

col = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER','AGE',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_EMPLOYED']

donnee_sortie = df_train1[col].copy()

var_code = donnee_entree['SK_ID_CURR'][0]

donnee_sortie['SK_ID_CURR'] = donnee_sortie['SK_ID_CURR'].apply(str)
donnee_sortie = donnee_sortie[(donnee_sortie['SK_ID_CURR'] == var_code)]

#st.write(HTML(donnee_sortie.to_html(index=False, escape=False, )))
st.table(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie)

#extraire le code client et les informations relatives au client
donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]
pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

# -------------------------------------------------------------------------------------------------------------
if len(pred_client)!=0:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Méthode Undersampling
    # nombre de classes
    #target_count_0, target_count_1 = df_train1['TARGET'].value_counts()

    # Classe séparée
    #target_0 = df_train1[df_train1['TARGET'] == 0]
    #target_1 = df_train1[df_train1['TARGET'] == 1]  # affiche la forme de la classe

    # Undersample 0-class and concat the DataFrames of both class
    #target_0_under = target_0.sample(target_count_1)
    #test_under = pd.concat([target_0_under, target_1], axis=0)

    # Définir X et y
    X = df_train1.iloc[:, 2:-2]
    y = df_train1.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
    st.write(df_train1.shape)

    # Importer le modèle entrainé lightGBM
    lgbm_clf = pickle.load(open('../lgbm_clf.pkl', 'rb'))

    # Prédire le résultat sur les données X_test
    y_pred = lgbm_clf.predict(X_test)

    # view accuracy
    accuracy = accuracy_score(y_pred, y_test)
    #st.write('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    # Prediction resultat
    tab = ['No Default', 'Default']
    y_pred = lgbm_clf.predict_proba(pred_client.iloc[:, 2:-2])
    risk = "{:,.0f}".format(y_pred[0][1]*100)
    pred_0 = "{:,.0f}".format(y_pred[0][0]*100)
    #st.write('TARGET du client prédit: ', tab[y_pred[0]])
    st.write(y_pred)


    prediction_0, prediction_1, prediction_3 = st.columns(3)

    prediction_0.metric(label='Prédiction remboursement (min 80%)',value=pred_0+str('%'),delta=int(pred_0)-80)
    prediction_1.metric(label='Risque de défaut de paiement (max 30%)',value=risk+str('%'),delta=30-int(risk))

    #st.write('Risque de défaut de paiement : ', risk + str('%'))
    if((y_pred[0][1])<=0.30):
        st.markdown('''<div class='box'>'''+'Risque de défaut de paiement : </div><div class="box" id="flag_green">'''+risk+str('%')+'''</div><div class='box'><img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQy1EjMg5nWNx1f7Fq8OwyXIc_Zgcw7ho90tA91zVAfEST6UnwggqOicekl4gxwuvOLk_M&usqp=CAU', height=40/></div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class='box'>''' + 'Risque de défaut de paieme   nt : </div><div class="box" id="flag_red">''' + risk + str('%') + '''</div><div class='box'><img src='https://americanmigrainefoundation.org/wp-content/uploads/2017/03/iStock_45096094_SMALL.jpg', height=40/></div>''',unsafe_allow_html=True)


    # Initialize SHAP Tree explainer
    explainer, shap_values = get_explainer(df_train1.iloc[:, :-2], lgbm_clf)

    # explainer = shap.TreeExplainer(lgbm_clf, model_output='raw')
    # shap_values = explainer.shap_values(df_train1.iloc[ :, 2:])

    # Baseline value
    expected_value = explainer.expected_value

    index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

    st.markdown("<h2 style='text-align: center; color: black;'>Force plot</h2>", unsafe_allow_html=True)
    # force_plot
    #st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][index_client], df_train1.iloc[index_client, 2:]))


    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<h2 style='text-align: center; color: black;'>Decision plot</h2>", unsafe_allow_html=True)
        # decision_plot
        decision = shap.decision_plot(base_value=explainer.expected_value[1],
                                      shap_values=shap_values[1][index_client],
                                      features=df_train1.iloc[index_client, 2:],
                                      feature_names=X_test.columns.tolist(),
                                      link='logit')

        #st.pyplot(decision, unsafe_allow_html=False)
        # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)
    with col2:
        st.markdown("<h2 style='text-align: center; color: black;'>Summary plot</h2>", unsafe_allow_html=True)
        # Summarize the effects of all the features
        #st.pyplot(shap.summary_plot(shap_values, pred_client.iloc[:, 2:]))

    st.markdown("<h2 style='text-align: center; color: black;'>Analyse Client : "+var_code+"</h2>", unsafe_allow_html=True)
    # Create a list of possible values and multiselect menu with them in it.
    # first we let streamlit know that we will be making a form
    my_form = st.form(key="test_form")
    #var = ['AGE']
    #var = ['AGE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_CHILDREN', 'PAYMENT_RATE', 'NEW_GOODS_CREDIT']
    var = ['All','NEW_EXT_MEAN', 'CODE_GENDER', 'YEARS_EMPLOYED', 'EXT_SOURCE_3',
       'NEW_GOODS_CREDIT', 'NAME_EDUCATION_TYPE_Higher education',
       'EXT_SOURCE_2', 'PAYMENT_RATE', 'AMT_ANNUITY',
       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN', 'INS_AMT_PAYMENT_MIN',
       'INS_DPD_STD', 'INS_DPD_MEAN', 'INS_AMT_PAYMENT_SUM',
       'INS_DAYS_INSTALMENT_STD', 'PREV_CNT_PAYMENT_STD',
       'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX', 'AGE',
       'NAME_INCOME_TYPE_Working', 'INS_PAYMENT_PERC_MEAN']


    col_selected = my_form.multiselect('Select Features', var)

    if (len(col_selected) == 0):
        col_selected = st.write('Please select features')
    elif (col_selected[0] == 'All'):
        col_selected = var[1:]



    # all forms end with a submit button, that is how the user can trigger
    submit = my_form.form_submit_button(label="submit")
    #st.write(var_code)

    if submit :
        if (len(col_selected) == 0):
            st.write('Please Select Features')
        elif (len(col_selected)<4) :
            plot_distribution_comp(col_selected, int(var_code), nrow=1, ncol=len(col_selected))
        elif (len(col_selected)>0) :
            plot_distribution_comp(col_selected, int(var_code), nrow=int(np.ceil(len(col_selected)/3)), ncol=3)

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
