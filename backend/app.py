from flask import Flask
from flask import request, jsonify
from query import get_vector_store, get_retriever, rag_query, create_rag_chat
from flask_cors import CORS



app = Flask(__name__)

# gives react permission
CORS(app)

@app.route('/create_chat', methods=['POST'])
def create_chat():
    chat_id = create_rag_chat()
    return jsonify({"chat_id": chat_id}), 201

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')
    chat_id = data.get('chat_id', None)

    if not chat_id:
        return jsonify({"error": "Chat ID is required"}), 400

    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    vector_store = get_vector_store()

    formatted_response = rag_query(chat_id, get_retriever(vector_store), query_text)

    return jsonify(formatted_response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
