import pinecone
import os
from transformers import CLIPProcessor, CLIPModel

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
API_KEY = os.environ['PINEAPI']
print("API_KEY",API_KEY)
INDEX_NAME = "videoindex250"

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


app = Flask(__name__)


# r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
CORS(app, resources=r'/*')

pinecone.init(api_key=API_KEY, environment="us-west1-gcp-free")
pinecode_index = pinecone.Index(index_name=INDEX_NAME)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search')
def search():
    query = request.args.get('q')
    embeddings = text_embeddings(query)
    
    search_response = pinecode_index.query(
        vector=embeddings.cpu().detach().numpy().tolist(),
        include_metadata=True,
        top_k=3,
    )
    print("search_response = ",search_response)
    return jsonify(search_response.to_dict())

@app.route('/api/similarity')
def similarity():
    id = request.args.get('id')
    stored_vector = pinecode_index.fetch(ids=[id])
    
    search_response = pinecode_index.query(
        vector=stored_vector.to_dict()['vectors'][id]['values'],
        top_k=10,    
        include_metadata=True,
    )

    return jsonify(search_response.to_dict())


def text_embeddings(text: str):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)    
    text_embeddings = clip_model.get_text_features(**inputs)

    return text_embeddings


if __name__ == "__main__":        
    app.run(port=8080)


