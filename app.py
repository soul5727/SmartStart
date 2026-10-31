import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="SmartStat | Data Science & Mathematics Lab", page_icon="📊", layout="wide")
st.title("📊 SmartStat")
st.subheader("Interactive Data Science & Mathematics Lab")
st.write("Explore datasets, descriptive statistics, probability, visualisation and predictive modelling.")
st.caption("Educational prototype for MSc-level learning. Upload a CSV or use the demonstration dataset.")

@st.cache_data
def demo_data():
    rng = np.random.default_rng(42)
    n = 180
    hours = rng.uniform(1, 10, n)
    attendance = rng.uniform(55, 100, n)
    prior = rng.normal(62, 12, n).clip(20, 95)
    score = (18 + 3.6*hours + .28*attendance + .32*prior + rng.normal(0, 7, n)).clip(0, 100)
    return pd.DataFrame({"study_hours": hours.round(1), "attendance_percent": attendance.round(1),
                         "prior_score": prior.round(1), "programme": rng.choice(["Data Science","Mathematics","Statistics"], n),
                         "final_score": score.round(1)})

with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload CSV file", type=["csv"])
    st.caption("Do not upload confidential or personally identifying data to a public demo.")

if upload:
    try:
        df = pd.read_csv(upload)
        source = upload.name
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()
else:
    df, source = demo_data(), "Built-in demonstration dataset"

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

num = df.select_dtypes(include=np.number).columns.tolist()
cat = df.select_dtypes(exclude=np.number).columns.tolist()
st.caption(f"Dataset: {source} | Rows: {len(df):,} | Columns: {len(df.columns)}")
a,b,c,d = st.columns(4)
a.metric("Observations", f"{len(df):,}")
b.metric("Variables", len(df.columns))
c.metric("Numeric variables", len(num))
d.metric("Missing cells", int(df.isna().sum().sum()))

overview, desc, viz, corr, prob, model, ai = st.tabs(["Overview","Descriptive statistics","Visualisation","Correlation","Probability","Predictive modelling","AI study assistant"])

with overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    x,y = st.columns(2)
    with x:
        st.subheader("Data types")
        st.dataframe(pd.DataFrame({"Variable":df.columns.astype(str),"Type":df.dtypes.astype(str).values}), use_container_width=True, hide_index=True)
    with y:
        st.subheader("Missing values")
        st.dataframe(df.isna().sum().rename("Missing").to_frame(), use_container_width=True)
    st.download_button("Download dataset as CSV", df.to_csv(index=False).encode("utf-8"), "smartstat_dataset.csv", "text/csv")

with desc:
    st.subheader("Descriptive statistics")
    if num:
        st.dataframe(df[num].describe().T.round(4), use_container_width=True)
        col = st.selectbox("Choose a numeric variable", num)
        s = df[col].dropna()
        if len(s):
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Mean", f"{s.mean():.4f}")
            m2.metric("Median", f"{s.median():.4f}")
            m3.metric("Sample variance", f"{s.var():.4f}")
            m4.metric("Sample standard deviation", f"{s.std():.4f}")
            modes = s.mode().tolist()
            st.write("**Mode:**", ", ".join(map(str,modes[:5])) if modes else "No mode")
            st.write(f"**Range:** {s.max()-s.min():.4f} | **Skewness:** {s.skew():.4f} | **Excess kurtosis:** {s.kurt():.4f}")
            st.caption("Variance and standard deviation use the sample convention (ddof=1).")
    else:
        st.info("No numeric variables found.")

with viz:
    st.subheader("Data visualisation")
    if num:
        kind = st.selectbox("Chart type", ["Histogram","Box plot","Scatter plot","Line plot"])
        if kind in ["Histogram","Box plot"]:
            col = st.selectbox("Variable", num)
            fig, ax = plt.subplots(figsize=(8,4))
            if kind == "Histogram":
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_ylabel("Frequency")
            else:
                sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"{kind}: {col}")
            st.pyplot(fig); plt.close(fig)
        elif len(num) >= 2:
            xcol = st.selectbox("X variable", num, key="vizx")
            ycol = st.selectbox("Y variable", [v for v in num if v != xcol], key="vizy")
            fig, ax = plt.subplots(figsize=(8,4))
            if kind == "Scatter plot":
                sns.scatterplot(data=df, x=xcol, y=ycol, ax=ax)
            else:
                sns.lineplot(data=df, x=xcol, y=ycol, ax=ax)
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Choose a dataset with at least two numeric columns for scatter/line plots.")
    else:
        st.info("No numeric variables available.")

with corr:
    st.subheader("Correlation analysis")
    if len(num) >= 2:
        method = st.selectbox("Method", ["pearson","spearman","kendall"])
        matrix = df[num].corr(method=method)
        st.dataframe(matrix.round(4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        ax.set_title(f"{method.title()} correlation matrix")
        st.pyplot(fig); plt.close(fig)
        st.caption("Correlation indicates association, not causation.")
    else:
        st.info("At least two numeric variables are required.")

with prob:
    st.subheader("Probability distribution explorer")
    dist = st.selectbox("Distribution", ["Normal","Binomial","Poisson"])
    if dist == "Normal":
        c1,c2 = st.columns(2)
        mu = c1.number_input("Mean (μ)", value=0.0)
        sigma = c2.number_input("Standard deviation (σ)", min_value=.0001, value=1.0)
        lo = st.number_input("Lower bound", value=-1.0)
        hi = st.number_input("Upper bound", value=1.0)
        if lo <= hi:
            p = stats.norm.cdf(hi,mu,sigma)-stats.norm.cdf(lo,mu,sigma)
            st.metric("P(lower ≤ X ≤ upper)", f"{p:.6f}")
            grid = np.linspace(mu-4*sigma,mu+4*sigma,400)
            fig,ax = plt.subplots(figsize=(8,3))
            ax.plot(grid,stats.norm.pdf(grid,mu,sigma))
            ax.fill_between(grid,stats.norm.pdf(grid,mu,sigma),where=(grid>=lo)&(grid<=hi),alpha=.3)
            ax.set_title("Normal probability density")
            st.pyplot(fig); plt.close(fig)
        else:
            st.error("Lower bound must not exceed upper bound.")
    elif dist == "Binomial":
        n = st.number_input("Trials (n)",min_value=1,max_value=10000,value=10)
        p = st.slider("Success probability (p)",0.0,1.0,.5)
        k = st.number_input("Successes (k)",min_value=0,max_value=int(n),value=min(5,int(n)))
        st.metric(f"P(X={int(k)})",f"{stats.binom.pmf(int(k),int(n),p):.6f}")
        st.write(f"Expected value = {n*p:.4f} | Variance = {n*p*(1-p):.4f}")
    else:
        lam = st.number_input("Rate (λ)",min_value=.0001,value=3.0)
        k = st.number_input("Event count (k)",min_value=0,value=2)
        st.metric(f"P(X={int(k)})",f"{stats.poisson.pmf(int(k),lam):.6f}")
        st.write(f"Expected value = {lam:.4f} | Variance = {lam:.4f}")

with model:
    st.subheader("Predictive modelling")
    st.caption("Select a target variable and input features. Results are evaluated on a held-out test set.")
    if len(df) < 10:
        st.warning("At least 10 rows are recommended.")
    else:
        target = st.selectbox("Target variable", df.columns.tolist())
        task = st.radio("Task",["Regression (numeric target)","Classification (categorical target)"],horizontal=True)
        features = st.multiselect("Input features",[v for v in df.columns if v != target],default=[v for v in df.columns if v != target][:min(4,len(df.columns)-1)])
        test_size = st.slider("Test set proportion",.1,.4,.2,.05)
        algo = st.selectbox("Algorithm",["Linear / Logistic Regression","Random Forest"])
        if st.button("Train and evaluate",type="primary"):
            if not features:
                st.error("Select at least one feature.")
            else:
                work = df[features+[target]].dropna(subset=[target]).copy()
                X,y = work[features],work[target]
                if task.startswith("Regression"):
                    y = pd.to_numeric(y,errors="coerce")
                    valid = y.notna()
                    X,y = X.loc[valid],y.loc[valid]
                    if len(y)<10:
                        st.error("Not enough valid numeric target values.")
                    else:
                        ncols = X.select_dtypes(include=np.number).columns.tolist()
                        ccols = [v for v in features if v not in ncols]
                        transformers=[]
                        if ncols: transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),ncols))
                        if ccols: transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),ccols))
                        pre=ColumnTransformer(transformers)
                        estimator=LinearRegression() if algo.startswith("Linear") else RandomForestRegressor(n_estimators=150,random_state=42)
                        pipe=Pipeline([("preprocess",pre),("model",estimator)])
                        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42)
                        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
                        q1,q2,q3=st.columns(3)
                        q1.metric("R²",f"{r2_score(yte,pred):.4f}")
                        q2.metric("MAE",f"{mean_absolute_error(yte,pred):.4f}")
                        q3.metric("RMSE",f"{math.sqrt(mean_squared_error(yte,pred)):.4f}")
                        st.dataframe(pd.DataFrame({"Actual":yte.values,"Predicted":pred}).head(20),use_container_width=True)
                        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(yte,pred,alpha=.7)
                        low=min(yte.min(),pred.min()); high=max(yte.max(),pred.max())
                        ax.plot([low,high],[low,high],"--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                        st.pyplot(fig); plt.close(fig)
                else:
                    if y.nunique()<2:
                        st.error("Classification needs at least two target classes.")
                    else:
                        ncols=X.select_dtypes(include=np.number).columns.tolist()
                        ccols=[v for v in features if v not in ncols]
                        transformers=[]
                        if ncols: transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),ncols))
                        if ccols: transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),ccols))
                        pre=ColumnTransformer(transformers)
                        estimator=LogisticRegression(max_iter=1000) if algo.startswith("Linear") else RandomForestClassifier(n_estimators=150,random_state=42)
                        pipe=Pipeline([("preprocess",pre),("model",estimator)])
                        strat=y if y.value_counts().min()>=2 else None
                        try:
                            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=strat)
                            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
                            st.metric("Accuracy",f"{accuracy_score(yte,pred):.4f}")
                            st.code(classification_report(yte,pred,zero_division=0))
                            labels=pipe.classes_; cm=confusion_matrix(yte,pred,labels=labels)
                            fig,ax=plt.subplots(figsize=(6,4))
                            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
                            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                            st.pyplot(fig); plt.close(fig)
                        except ValueError as e:
                            st.error(f"Model could not be trained: {e}")
    st.caption("For rigorous work, discuss class imbalance, data leakage, cross-validation, baseline models and domain validation.")

with ai:
    st.subheader("AI-supported interpretation")
    st.write("The app calculates statistics. Use the generated prompt in ChatGPT or another GenAI tool to obtain an explanation, then verify it independently.")
    if num:
        col=st.selectbox("Variable to interpret",num,key="promptvar")
        s=df[col].dropna()
        summary={"variable":col,"n":int(len(s)),"mean":round(float(s.mean()),4),"median":round(float(s.median()),4),
                 "sample_sd":round(float(s.std()),4) if len(s)>1 else None,
                 "minimum":round(float(s.min()),4),"maximum":round(float(s.max()),4),
                 "skewness":round(float(s.skew()),4) if len(s)>2 else None}
        prompt=("Act as a statistics tutor for an MSc Data Science and Mathematics student. Explain these descriptive statistics. "
                "Interpret the mean, median, standard deviation, range and skewness where available. Do not claim causation. "
                "Mention limitations and avoid unsupported claims.\n\nResults:\n"+str(summary)+
                "\n\nGive a concise interpretation and suggest one suitable follow-up analysis.")
        st.json(summary)
        st.text_area("Copy this prompt into ChatGPT",value=prompt,height=180)
        st.info("Save your actual prompt and AI response for your assignment. Critically check the explanation against reliable academic sources.")
    else:
        st.info("Upload a dataset with numeric variables to create an interpretation prompt.")

st.divider()
st.caption("SmartStat is an educational prototype. Verify assumptions and results; do not use it as a substitute for expert statistical advice.")
