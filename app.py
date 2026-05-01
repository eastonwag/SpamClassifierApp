import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#Title
st.title("Spam Classifier App")
st.write("Classify emails as Spam or Ham (not spam) using machine learning.")

#Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter or paste the email text in the box below.  
2. Click "Classify" to see if it is spam or ham.  
3. Review the visuals to see the dataset distribution.  
""")

#Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv(
        "emails.csv",
        encoding='ISO-8859-1',
        usecols=[0, 1],                         #first 2 columns
        names=["label", "text"],                #rename columns
        skiprows=1,                             #skipping header
        quotechar='"'                           #handle quotes
    )
    return data

data = load_data()


#descriptive method
st.write("### Dataset Preview")
st.dataframe(data)

#Bar Chart
st.write("### Spam vs Ham Count")
st.bar_chart(data["label"].value_counts())

#Pie chart
st.write("### Spam vs Ham Distribution")
fig, ax = plt.subplots()
data["label"].value_counts().plot.pie(
    autopct="%1.1f%%",
    labels=["Ham", "Spam"],
    colors=["#2ca02c", "#d62728"],
    startangle=90,
    ax=ax
)
ax.set_ylabel("")
st.pyplot(fig)

#predictive method
X = data['text']
y = data['label']

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vect, y_train)

#Evaluation metrics
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

st.write("### Model Evaluation Metrics (on test set)")
st.write(f"- Accuracy: {accuracy:.2%}")
st.write(f"- Precision (Spam): {precision:.2%}")
st.write(f"- Recall (Spam): {recall:.2%}")
st.write(f"- F1 Score (Spam): {f1:.2%}")

#Email input and classification
st.write("### Classify a New Email")

email_input = st.text_area("Enter email text here:")

if st.button("Classify"):
    if email_input.strip() != "":
        input_vector = vectorizer.transform([email_input])
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector).max()
        st.write(f"Prediction: {prediction.upper()}")
        st.write(f"Confidence: {prob:.2%}")
    else:
        st.write("Please enter an email to classify.")