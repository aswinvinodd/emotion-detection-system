from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

print("Sentiment Analysis System Ready!")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter your text: ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    result = sentiment_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']

    print(f"Sentiment: {label} | Confidence: {score:.2f}\n")