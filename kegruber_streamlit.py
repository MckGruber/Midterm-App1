import streamlit as st
import pandas as pd
import pickle
import numpy as np

ADA_BOOST = "AdaBoostClassifier"
DT = "DecisionTreeClassifier"
RF = "RandomForestClassifier"
VC = "VotingClassifier"
MODELS = [ADA_BOOST, DT, RF, VC]

@st.cache_resource
def get_models() -> dict:
  model_dict = {}
  for model in MODELS:
    with open(model + ".pickle", "rb") as f:
      model_dict[model] = pickle.load(f)
  return model_dict

def highlight_prediction(s: pd.Series, props=""):
  return np.where(
    s == "Normal", "background-color:lime;color:black", 
    np.where(s == "Suspect", "background-color:yellow", "background-color:orange;color:black"))


st.write("# Fetal Health Predictor")
st.image("fetal_health_image.gif")
model = st.selectbox("Select Model to use", MODELS)
with st.expander("upload a csv file"):
    st.write("Example DataFrame: ")
    st.write(pd.read_csv("fetal_health_user.csv").head())
    csv_file = st.file_uploader("click to upload csv file", type="csv", accept_multiple_files=False)
if csv_file != None:
  user_df = pd.read_csv(csv_file)
  predictions = get_models()[model].predict(user_df)
  prediction_probability = pd.DataFrame(get_models()[model].predict_proba(user_df)).max(axis=1)
  user_df["Prediction"] = predictions
  user_df["Prediction Probability"] = prediction_probability
  user_df["Prediction Probability"] = user_df["Prediction Probability"].map(lambda x: f"{x*100:.1f}%")
  user_df["Prediction"] = user_df["Prediction"].map(lambda x: "Normal" if x == 1 else "Suspect" if x == 2 else "Pathological")
  # user_df.style.apply(highlight_prediction, props="", axis=0, subset=["Prediction"])
  styled = user_df.style.apply(highlight_prediction, props="", axis=0, subset=["Prediction"])
  st.write(styled)
  st.subheader("Model Insights")
  if model != DT:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Importance", 
                                "Histogram of Residuals", 
                                "Predicted Vs. Actual", 
                                "Confusin Matrix",
                                "Classification Report"]) 
    with tab1:
        st.write("### Feature Importance")
        st.image(f'{model}_feature_imp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Histogram of Residuals")
        st.image(f'{model}_residual_plot.svg')
        st.caption("Distribution of residuals to evaluate prediction quality.")
    with tab3:
        st.write("### Plot of Predicted Vs. Actual")
        st.image(f'{model}_pred_vs_actual.svg')
        st.caption("Visual comparison of predicted and actual values.")
    with tab4:
        st.write("### Confusin Matrix")
        st.image(f'{model}_confusion_mat.svg')
    with tab5:
        st.write(pd.read_csv(f'{model}_class_report.csv'))
  else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Feature Importance", 
                                "Histogram of Residuals", 
                                "Predicted Vs. Actual", 
                                "Confusin Matrix",
                                "Classification Report",
                                "Decision Tree"]) 
    with tab1:
        st.write("### Feature Importance")
        st.image(f'{model}_feature_imp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Histogram of Residuals")
        st.image(f'{model}_residual_plot.svg')
        st.caption("Distribution of residuals to evaluate prediction quality.")
    with tab3:
        st.write("### Plot of Predicted Vs. Actual")
        st.image(f'{model}_pred_vs_actual.svg')
        st.caption("Visual comparison of predicted and actual values.")
    with tab4:
        st.write("### Confusin Matrix")
        st.image(f'{model}_confusion_mat.svg')
    with tab5:
        st.write(pd.read_csv(f'{model}_class_report.csv'))
    with tab6:
        st.write("### Decision Tree Visualization")
        st.image(f'{model}_visual.svg')
        st.caption("Visualization of the Decision Tree used in prediction.")

  