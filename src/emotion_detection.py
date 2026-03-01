from transformers import pipeline

# Load emotion detection model (return all emotions)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Emotion → Sentiment mapping
POSITIVE = {"joy", "surprise"}
NEGATIVE = {"anger", "sadness", "fear", "disgust"}

print("Emotion Detection System Ready!")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter your text: ")

    if text.lower() == "exit":
        print("Exiting program.")
        break

    raw_output = emotion_classifier(text)

    # raw_output -> [[{label, score}, ...]]
    emotions = raw_output[0]

    # Sort emotions by score (descending)
    emotions_sorted = sorted(emotions, key=lambda x: x["score"], reverse=True)

    # Top two emotions
    top_emotion = emotions_sorted[0]
    second_emotion = emotions_sorted[1]

    top_label = top_emotion["label"]
    top_score = top_emotion["score"]

    second_label = second_emotion["label"]
    second_score = second_emotion["score"]

    # ---------------------------
    # MIXED SENTIMENT LOGIC
    # ---------------------------
    sentiment = None

    if (
        top_score >= 0.50 and
        second_score >= 0.30 and
        ((top_label in POSITIVE and second_label in NEGATIVE) or
         (top_label in NEGATIVE and second_label in POSITIVE))
    ):
        sentiment = "MIXED 🤔"
    else:
        if top_label in POSITIVE:
            sentiment = "POSITIVE 😊"
        elif top_label in NEGATIVE:
            sentiment = "NEGATIVE 😞"
        else:
            sentiment = "NEUTRAL 😐"

    # ---------------------------
    # OUTPUT
    # ---------------------------
    print("\n==============================")
    print(f"Overall Sentiment : {sentiment}")
    print("==============================\n")

    print("Detected Emotions (in %):")
    for e in emotions_sorted:
        print(f"{e['label']:<10} : {e['score'] * 100:.2f}%")

    print("\nFinal Decision:")
    print(f"Top Emotion      : {top_label}")
    print(f"Confidence       : {top_score * 100:.2f}%\n")
