import pickle
from model.preprocess import clean_text

# Load trained model
with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

print("🤖 AI Chatbot (Human-like)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    text = user_input.lower().strip()

    if text == "exit":
        print("Bot: Goodbye 👋 Take care!")
        break

    # ===============================
    # HUMAN-LIKE RULES (INTENTS)
    # ===============================

    if text in ["hi", "hello", "hey"]:
        print("Bot: Hello! 😊 How can I help you today?\n")
        continue

    if "how are you" in text:
        print("Bot: I'm doing great, thanks for asking! 😊 How about you?\n")
        continue

    if text in ["bye", "goodbye", "see you"]:
        print("Bot: Bye! 👋 Have a wonderful day!\n")
        continue

    if "thank" in text:
        print("Bot: You're welcome! 😊\n")
        continue

    if "help" in text:
        print("Bot: I can analyze your feelings. Tell me how you're feeling today.\n")
        continue

    # ===============================
    # SENTIMENT ANALYSIS (ML)
    # ===============================

    cleaned = clean_text(user_input)
    sentiment = model.predict([cleaned])[0]
    confidence = max(model.predict_proba([cleaned])[0]) * 100

    if sentiment == "positive":
        reply = "😊 That's wonderful to hear!"
    elif sentiment == "negative":
        reply = "😔 I'm sorry you're feeling this way. Want to talk about it?"
    else:
        reply = "🙂 I see. Tell me more."

    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Bot: {reply}\n")
