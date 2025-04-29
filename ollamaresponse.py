from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import ollama
import sqlite3

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from Vercel frontend)

_HOST = "127.0.0.1"  # Or localhost
_PORT = "19530"  # Default gRPC port for Milvus Standalone
# Connect to Milvus
def connect_to_milvus(db_name="default"):
    print(f"Connecting to Milvus...\n")
    
    # Connect using gRPC
    connections.connect(
        host=_HOST,  
        port=_PORT,        
        timeout=60  # Increased timeout
    )

    # List available databases
    db_list = db.list_database()
    print(db_list)

    # Check if the 'turf_grass' database exists
    if "turf_grass" not in db_list:
        db.create_database("turf_grass")
    
    db.using_database("turf_grass")
    print(f"Using database: turf_grass")
# Load your collection

connect_to_milvus()
collection_name = "turf_grass_data"
collection = Collection(collection_name)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
def get_embedding(text):
    """Generate an embedding for the query using Ollama."""
    response = model.encode(text).tolist()  # Adjust model as needed
    return response

def askOllama(text, distance_threshold):
    query_text = text
    query_embedding = get_embedding(query_text)
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    # Fields to check for common IDs
    fields_to_search = [
        "identifier_emb"
    ]

    all_ids = {field: [] for field in fields_to_search}  # Dictionary to store (ID, distance) pairs

    # Perform search on each field
    for field in fields_to_search:
        results = collection.search(
            data=[query_embedding], 
            anns_field=field, 
            param=search_params, 
            limit=1000,  # Getting 25 results for each field
            output_fields=["ids"]
        )

        # Collect (ID, distance) tuples from the results
        for result in results[0]:
            all_ids[field].append((int(result.ids), float(result.distance)))

    # Now, we need to count occurrences and track the best (lowest) distance for each ID

   
        # If no common IDs exist, fall back to paragraph_emb results (within threshold)
    ids_to_retrieve = [
        (id, distance) for id, distance in all_ids["identifier_emb"] 
            if distance >= distance_threshold
    ]
    print (ids_to_retrieve[:3])
    # Sort by distance (ascending, so closest matches come first)

    # Fetch the data for the selected IDs
    context = "\n".join(
        [fetch_data_from_sqlite(id_)[0][0] + " " + str(distance) for id_, distance in ids_to_retrieve[:3]]
    )
    return context



def generateresponse(text) : # Generate a response using Ollama
    print("Question:")
    print(text)
    print("Answer:")
    context = askOllama(text)
    response = ollama.chat(
        model="llama3.2",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are an expert in turfgrass and plant diseases. Answer questions based *only* on the context provided. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": f"answer {text}? , based on - {context}"}
        ]
    )

    print(response["message"]["content"])
def fetch_data_from_sqlite(ids):
    db_path = "./final_output_completed.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    query = "SELECT Identifier FROM grass WHERE id = ?"
    cursor.execute(query, (ids,))  # Tuple passed to parameterized query
    # Adjust query as needed
    row = cursor.fetchall()
    conn.close()
    return row

#ask question
chat_history = [
    {"role": "system", "content": "You are an expert in turfgrass and plant diseases. Answer questions based *only* on the context provided. If the answer is not in the context, say 'I don't know'."}
]

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

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting chat...")
#         break
#
#     context = askOllama(user_input, 0.5)
#
#     chat_history.append({"role": "user", "content": user_input})
#
#     # Instead of just appending context, structure the prompt for a better response
#     formatted_prompt = f"""
#     Use **only** the following context to answer the user's question. If the context does not contain the answer, say so.
#
#     Context:
#     {context}
#
#     Question:
#     {user_input}
#
#     Answer:
#     """
#
#     response = ollama.chat(model="llama3.2", messages=chat_history + [{"role": "user", "content": formatted_prompt}])
#
#     bot_reply = response["message"]["content"]
#     print("\nBot:", bot_reply)
#
#     chat_history.append({"role": "assistant", "content": bot_reply})