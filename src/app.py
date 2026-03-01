import streamlit as st
import csv
import os
from datetime import datetime
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Emotion Detection", layout="centered")

# =============================
# THEME TOGGLE
# =============================
theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color:#0E1117;color:white;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {background-color:white;color:black;}
        </style>
    """, unsafe_allow_html=True)

# =============================
# CONFIG
# =============================
CSV_FILE = "data/emotion_history.csv"

POSITIVE = {"joy","surprise"}
NEGATIVE = {"anger","sadness","fear","disgust"}

EMOJI_MAP = {
    "joy":"😊","sadness":"😢","anger":"😡",
    "fear":"😨","disgust":"🤢","surprise":"😲","neutral":"😐"
}

SARCASM_PHRASES = [
    "oh great","yeah right","thanks a lot",
    "love how","amazing job","fantastic job",
    "just perfect","wonderful job"
]

def detect_sarcasm(text):
    return any(p in text.lower() for p in SARCASM_PHRASES)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

emotion_classifier = load_model()

# =============================
# CSV SETUP
# =============================
os.makedirs("data", exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp","text","overall_sentiment","top_emotion","confidence"]
        )

def load_history():
    with open(CSV_FILE,"r",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_to_csv(text,sentiment,emotion,confidence):
    history = load_history()
    for r in history:
        if r["text"] == text:
            return

    with open(CSV_FILE,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,sentiment,emotion,f"{confidence:.2f}"
        ])

# =============================
# PDF REPORT
# =============================
def create_pdf(text,sentiment,emotion,confidence):
    path="analysis_report.pdf"
    c=canvas.Canvas(path)
    c.drawString(50,800,"Emotion Detection Report")
    c.drawString(50,760,f"Text: {text}")
    c.drawString(50,720,f"Sentiment: {sentiment}")
    c.drawString(50,680,f"Emotion: {emotion}")
    c.drawString(50,640,f"Confidence: {confidence:.2f}%")
    c.drawString(50,600,str(datetime.now()))
    c.save()
    return path

# =============================
# HEADER
# =============================
st.title("Emotion Detection & Sentiment Analysis")
st.divider()

mode = st.radio(
    "Choose Input Method:",
    ["✍️ Enter New Text",
     "📂 Choose From Saved History",
     "Batch Analysis"]
)

# =========================================================
# LIVE ANALYSIS
# =========================================================
if mode == "✍️ Enter New Text":

    text = st.text_area("Enter text to analyze:")

    if st.button("Analyze"):

        if text.strip()=="":
            st.warning("Please enter text.")
        else:
            output = emotion_classifier(text)[0]
            emotions_sorted = sorted(output,key=lambda x:x["score"],reverse=True)

            top = emotions_sorted[0]
            label = top["label"]
            score = top["score"]

            sentiment="NEUTRAL 😐"
            if label in POSITIVE:
                sentiment="POSITIVE 😊"
            elif label in NEGATIVE:
                sentiment="NEGATIVE 😞"

            save_to_csv(text,sentiment,label,score*100)

            c1,c2,c3 = st.columns(3)
            c1.metric("Sentiment",sentiment)
            c2.metric("Emotion",label.capitalize())
            c3.metric("Confidence",f"{score*100:.2f}%")

            st.markdown(f"## {EMOJI_MAP.get(label,'😐')}")

            if detect_sarcasm(text):
                st.warning("⚠ Possible sarcasm detected")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score*100,
                title={'text':"Model Confidence"},
                gauge={'axis':{'range':[0,100]}}
            ))
            st.plotly_chart(fig,use_container_width=True)

            # Progress bars
            st.subheader("Detected Emotions")
            for e in emotions_sorted:
                st.progress(e["score"],
                            text=f"{e['label']} — {e['score']*100:.2f}%")

            # Pie chart
            df=pd.DataFrame({
                "Emotion":[e["label"] for e in emotions_sorted],
                "Percentage":[e["score"]*100 for e in emotions_sorted]
            })
            st.plotly_chart(px.pie(df,names="Emotion",values="Percentage"),
                            use_container_width=True)

            pdf=create_pdf(text,sentiment,label,score*100)
            with open(pdf,"rb") as f:
                st.download_button("⬇ Download Report (PDF)",f,
                                   file_name="emotion_report.pdf")

# =========================================================
# HISTORY + SEARCH + SORTING
# =========================================================
elif mode == "📂 Choose From Saved History":

    history = list(reversed(load_history()))

    if history:

        st.subheader("📂 Select Stored Analysis")

        selected = st.selectbox(
            "Choose a saved text:",
            history,
            format_func=lambda x:
                f"{x['timestamp']} | {x['overall_sentiment']} | {x['text'][:40]}..."
        )

        # =====================
        # SHOW DETAILS PANEL
        # =====================
        st.divider()
        st.subheader("📄 Analysis Details")

        st.write("**Text:**")
        st.info(selected["text"])

        c1, c2, c3 = st.columns(3)

        c1.metric("Sentiment", selected["overall_sentiment"])
        c2.metric("Top Emotion", selected["top_emotion"].capitalize())
        c3.metric("Confidence", f"{float(selected['confidence']):.2f}%")

        st.caption(f"Analyzed on: {selected['timestamp']}")

        st.divider()

        # =====================
        # SENTIMENT SORTING
        # =====================
        st.subheader("📊 Sentiment-Based Sorting")

        choice = st.selectbox(
            "Filter sentiment",
            ["POSITIVE 😊","NEGATIVE 😞","NEUTRAL 😐"]
        )

        filtered = [
            r for r in history
            if choice.split()[0] in r["overall_sentiment"]
        ]

        sorted_list = sorted(
            filtered,
            key=lambda x: float(x["confidence"]),
            reverse=True
        )

        for i,item in enumerate(sorted_list,1):
            st.write(
                f"{i}. {item['text']} "
                f"(Confidence {float(item['confidence']):.2f}%)"
            )

        st.divider()

        # =====================
        # SEARCH
        # =====================
        st.subheader("🔎 Search Saved Feedback")

        query = st.text_input("Search keyword")

        if query:
            results = [
                h for h in history
                if query.lower() in h["text"].lower()
            ]

            if results:
                for r in results:
                    st.write(
                        f"{r['timestamp']} | "
                        f"{r['overall_sentiment']} | "
                        f"{r['text']}"
                    )
            else:
                st.info("No matching feedback found.")

    else:
        st.info("No history yet.")

# =========================================================
# BATCH ANALYSIS
# =========================================================
elif mode == "Batch Analysis":

    st.subheader("📑 Batch Emotion Analysis")

    batch_text = st.text_area(
        "Enter multiple sentences (one per line):",
        height=200
    )

    if st.button("Run Batch Analysis"):

        if batch_text.strip()=="":
            st.warning("Please enter text.")
        else:
            lines=[l.strip() for l in batch_text.split("\n") if l.strip()]
            results=[]

            for sentence in lines:
                output=emotion_classifier(sentence)[0]
                emotions_sorted=sorted(output,key=lambda x:x["score"],reverse=True)

                top=emotions_sorted[0]
                label=top["label"]
                score=top["score"]

                sentiment="NEUTRAL 😐"
                if label in POSITIVE:
                    sentiment="POSITIVE 😊"
                elif label in NEGATIVE:
                    sentiment="NEGATIVE 😞"

                save_to_csv(sentence,sentiment,label,score*100)

                results.append({
                    "Text":sentence,
                    "Sentiment":sentiment,
                    "Emotion":label.capitalize(),
                    "Confidence (%)":round(score*100,2)
                })

            df=pd.DataFrame(results)

            st.success("✅ Batch Analysis Completed")
            st.dataframe(df,use_container_width=True)
            st.bar_chart(df["Sentiment"].value_counts())

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================
st.divider()

with st.expander("📊 View Analytics Dashboard"):

    history=load_history()

    if history:
        for r in history:
            r["confidence"]=float(r["confidence"])

        st.subheader("Sentiment Distribution")
        st.bar_chart(pd.Series(
            [r["overall_sentiment"] for r in history]
        ).value_counts())

        st.subheader("Emotion Distribution")
        st.bar_chart(pd.Series(
            [r["top_emotion"] for r in history]
        ).value_counts())

        trend=[]
        for r in history:
            if "POSITIVE" in r["overall_sentiment"]:
                trend.append(1)
            elif "NEGATIVE" in r["overall_sentiment"]:
                trend.append(-1)
            else:
                trend.append(0)

        st.subheader("Sentiment Trend Over Time")
        st.line_chart(trend)