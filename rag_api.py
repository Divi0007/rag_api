from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flasgger import Swagger
import pymongo
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)

embedding_model = SentenceTransformer('thenlper/gte-large')
def get_embedding (text:str) -> list[float]:
  if not text.strip():
    print("Attempted to get embedding for empty text.") 
    return []
  embedding = embedding_model.encode(text)
  return embedding.tolist()

def get_mongo_client(mongo_uri):
    client = pymongo.MongoClient (mongo_uri)
    print("Connection to MongoDB successful")
    return client
mongo_uri ='mongodb+srv://divirao05:GOBXrbl9My4pb6eW@cluster0.oz4tn1s.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
mongo_client = get_mongo_client(mongo_uri)
# Ingest data into MongoDB
db = mongo_client['rag_judgement']
collection = db ['laws_acts']

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.
    Args:
    user_query (str): The user's query string. 
    collection (MongoCollection): The MongoDB collection to search.
    
    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,
                # Number of candidate matches to consider
                "limit": 5
                # Return top 5 matches
            }
        },
        {
            "$project": {
                "id": 1,
                "page_content": 1,
                # Include the plot field
                "score": {
                    "$meta": "vectorSearchScore"
                }
                # Include the search score
            }

        }
    ]
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result (query, collection):
    get_knowledge = vector_search (query, collection)
    search_result = ''
    for result in get_knowledge:
        search_result += f"ID:{result.get('id', 'N/A')},content:{result.get('page_content', 'N/A')}\n"
    return search_result


class UppercaseText(Resource):

    def get(self):
        """
        This method responds to the GET request for this endpoint and returns the data in uppercase.
        ---
        tags:
        - Text Processing
        parameters:
            - name: text
              in: query
              type: string
              required: true
              description: The query to search
        responses:
            200:
                description: A successful GET request
                content:
                    application/json:
                      schema:
                        type: object
                        properties:
                            text:
                                type: string
                                description: The text in uppercase
        """
        text = request.args.get('text')
        source_information = get_search_result (text,collection)
        combined_information = f"Query: {text}\nContinue to answer the query by using the Search Results: \n{source_information}."
        answer=""

        client = InferenceClient(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token="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",
        )

        for message in client.chat_completion(
            messages=[{"role": "user", "content": combined_information}],
            max_tokens=1024,
            stream=True,
        ):
            answer+=message.choices[0].delta.content

        return jsonify({"text": answer})

api.add_resource(UppercaseText, "/uppercase")

if __name__ == "__main__":
    app.run(debug=True)