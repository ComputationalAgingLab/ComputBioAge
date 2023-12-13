import streamlit as st
from streamlit import session_state as ses
import numpy as np
import pandas as pd
import base64
import textwrap

#tmp 
import sys
import os
# SCRIPT_DIR = os.path.dirname(os.path.abspath('/home/shappiron/Desktop/CAL/univariate_inversed_ensembler/kdm'))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from kdm.kdm import KlemeraDoubalEstimator

#tmp - move to a separate file
feature_include = {
    'fev':'Forced expiratory volume (mL)', #~2000 руб в москве
    'sbp':'Systolic blood pressure', # ---  
    'bup':'Serum blood urea nitrogen (mg/dL)', # 380 rub gemotest,
    'tcp':'Serum cholesterol (mg/dL)', # 255 rub helix
    'crp':'Serum C-reactive protein (mg/dL)', # 510 rub helix
    'cep':'Serum creatinine (mg/dL)', # 255 rub helix
    # 'appsi':'Serum alkaline phosphatase: SI (U/L)', # 620 rub invitro
    'amp':'Serum albumin (g/dL)', # 350 руб helix, 
    # 'ghp':'Glycated hemoglobin (%)', # 840 руб gemotest
    'wbc':'White blood cell count', # 650 руб invitro
    #'cmvod':'Cytomegalovirus optical density',
    ### NOT FEATURES in MODEL, but METAINFO
    #'female': 'sex binary identificator',
    #'samp_wt': 'Weights which are fucking mysterious.',
    #'age': 'Age',
}


# Function to calculate biological age based on physiological parameters
def calculate_biological_age(biomarkers, gender, age=None):
    biological_age = 0
    return biological_age

def predict_average_feature(model, key, age=None):
    if age is None:
        age = 40
    w = model.model.loc[key, 'slope']
    b = model.model.loc[key, 'intercept']
    return w * age + b

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="150" height="150"/>' % b64
    st.write(html, unsafe_allow_html=True)

@st.cache_data
def load_data():
    dfm = pd.read_csv('data/nhanes_warner_male.csv', index_col=0)
    dff = pd.read_csv('data/nhanes_warner_female.csv', index_col=0)
    return dfm, dff
    
# Main Streamlit app
def main():
    #on start
    dfm, dff = load_data()
    model = KlemeraDoubalEstimator()
    #ses['biomarkers'] = {}

    ### SIDEBAR, temporarily off
    # st.sidebar.title("Biological Age Calculator")
    # st.sidebar.write('developed by ComputAge')
    # st.sidebar.subheader("Choose an algorithm:")
    # algorithm_choice = st.sidebar.selectbox(
    #     "Select Algorithm",
    #     ["Klemera-Doubal", "MLR", "PCA"]  # Replace with your algorithm names
    # )

    # st.sidebar.subheader("About")
    # st.sidebar.write("""
    #                     With the help of a small number of easily accessible parameters of your body. 
    #                     Nevertheless, we tried to carefully select the parameters for the calculation 
    #                     so that they are as informative as possible.
    #                  """)

    f = open("front/logo.svg", "r")
    lines = f.readlines()
    line_string=''.join(lines)
    render_svg(line_string)  
    
    st.title("Biological Age Calculator")
    st.write("""
             Please choose and enter your physiological parameters which you want to include in 
             biological age calculation. You may use any combination of features including or excluding any of them.
             Features are ordered by their significance for biological age computation such that the first one has
             the highest significance. We recommend to use the first three features for maximal accuracy of bioage prediction.
             However, any combinations are possible. Try it yourself!
             """)
    st.divider()

    c1, c2 = st.columns([3, 3])
    with st.container():
        c1.write('') # for additional gap
        gender = c1.selectbox('Choose your gender', ('Male', 'Female'))
        use_chrono = c2.toggle('Use your chronological age for calculation?')
        if use_chrono:
            age_value = c2.number_input('Chronological age', value=40,  label_visibility="collapsed")
        else:
            age_value = None

    ### Choose reference dataset and model
    if gender == 'Male':
        X, y = dfm.drop('age', axis=1), dfm['age']
        model.load_model('models/model_m.pickle')
    elif gender == 'Female':
        X, y = dff.drop('age', axis=1), dff['age']
        model.load_model('models/model_f.pickle')
    else:
        raise NotImplementedError()
    ###
    
    #create input fields for biomarkers
    
    for bkey, bname in feature_include.items():
        c1, c2, c3 = st.columns([4, 3, 3])
        with st.container():
            usebio = c1.checkbox(bname, value=False)
            if usebio:
                avg_button = c3.button(label="Use average value", key=bkey+'_avg', use_container_width=True)
                user_button = c3.button(label="Use my value", key=bkey+'_user', use_container_width=True)
                ses[bkey] = 0. if bkey not in ses else ses[bkey]
                ses[bkey + '_button_state'] = False if bkey + '_button_state' not in ses else ses[bkey + '_button_state']

                if avg_button:
                    ses[bkey] = predict_average_feature(model, bkey, age=age_value)
                    ses[bkey + '_button_state']=True
                
                if user_button:
                    ses[bkey + '_button_state']=False
                
                bval = c2.number_input(bname, 
                                value=ses[bkey], 
                                disabled=ses[bkey + '_button_state'])
                ses[bkey] = bval
        st.write('')            
    
    st.divider()

    #tmp
    #st.write({k:ses[k] for k in ses if '_' not in k})
    

    # Calculate biological age on button click
    if st.button("Compute Biological Age"):
        #biological_age = calculate_biological_age(blood_pressure, leukocytes_percent, red_blood_count, height)
        biomarker_values = {k:[ses[k]] for k in ses if '_' not in k}

        ba = model.predict(pd.DataFrame(biomarker_values), 
                           feature_names=list(biomarker_values.keys()))
        st.write(f"Biological Age: {round(ba.item(), 1)}")

        if use_chrono:
            bac = model.predict_BAC(pd.DataFrame(biomarker_values), pd.Series(np.array([age_value])),
                                    X_base=X, y_base=y,
                                   feature_names=list(biomarker_values.keys()))
            st.write(f"Biological Age (corrected): {round(bac.item(), 1)}")
        
        
    st.divider()

    st.title("FAQ")
    # Folding text with info about the principle of biological age computation
    with st.expander("Principle of Biological Age Computation."):
        st.markdown("""
                Our approach mostly relies on the seminal work of [Klemera & Doubal, 2006](https://pubmed.ncbi.nlm.nih.gov/16318865/) with several
                changes in the original formuli accounting negative multicollinear effects on the biological age prediction (see our paper for details).
                As the train dataset we used NHANES-III used previously in [Levine, 2013](https://pubmed.ncbi.nlm.nih.gov/23213031/), gently preprocessed 
                by Elisa Warner and published in her [repository](https://github.com/elisawarner/Biological-Age-Project). 
                We introduced additional feature selection selection step excluding features which are coupled with complex diagnostic procedures. 
                The resulting 8 features provides flexibility for biological age computation based on the maximum information you can gather about yourself.
                Features are ordered by their significance for biological age computation, so the first three features are absolutely enough for the estimate.
                Use features in any combinations, but theoretically, the first three should provide the best accuracy of prediction.
                 """)

    with st.expander("Why I should put my chronological age?"):
        st.write("""
                 The idea to use chronological age (CA) as one of biomarkers might seem as a logical contradiction. 
                 It would be so in case of traditional computation of biological age by Multiple linear regression method 
                 as CA is dependent variable there. We use unsupervized model of biological age which does not assume any dependent variables.
                 This means that CA is just an additional feature in the model which provides an additional balancing factor for biological age estimate.
                 Thus, our model use chronological age for improving accuracy of biological age prediction.
                 """)

    with st.expander("Is this anonymous?"):
        st.write("Yes, absolutely! We do not store the data you input.")

    with st.expander("What is ComputAge?"):
        st.write("""
                 **ComputAge** - DeSci organization, that builds an ecosystem of computational biologists in longevity. 
                 Our mission is to onboard engineers into longevity, organize them into research groups, conduct research, 
                 and create a space for new scientific breakthroughs.
                 """)   

if __name__ == "__main__":
    main()
