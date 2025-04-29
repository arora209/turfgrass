from flask import Flask, request, jsonify
from flask_cors import CORS
from ollamaresponse import askOllama

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from Vercel frontend)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    threshold = float(data.get("threshold", 0.5))

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        context = askOllama(question, threshold)
        formatted_prompt = f"""
                Use **only** the following context to answer the user's question. If the context does not contain the answer, say so.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

        chat_history = [
            {"role": "system",
             "content": "You are an expert in turfgrass and plant diseases. Answer questions based *only* on the context provided. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": formatted_prompt}
        ]

        response = ollama.chat(model="llama3.2", messages=chat_history)
        answer = response["message"]["content"]

        return jsonify({"context": context, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
