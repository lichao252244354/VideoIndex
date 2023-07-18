import pinecone
import os
from transformers import CLIPProcessor, CLIPModel

from flask import Flask, jsonify, render_template, request
import time
import ffmpeg
import os

from flask_cors import CORS
API_KEY = os.environ['PINEAPI']
print("API_KEY",API_KEY)
INDEX_NAME = "videoindex250"

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


app = Flask(__name__,static_folder='static', static_url_path='/data')


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
    movie_value = search_response["matches"]
    movie_array = []

    for var in movie_value:
        var_temp = var.to_dict()
        print("var_temp is ",var_temp)

        start = var_temp["metadata"]["start"]
        print("start is ",start)
        start_time = time.strftime('%H:%M:%S', time.gmtime(start))
        print("start_time is ",start_time)

        end = var_temp["metadata"]["end"]
        print("end is ",end)
        end_time = time.strftime('%H:%M:%S', time.gmtime(end))
        print("end_time is ",end_time)
        input_file = "\"" + "./static/" + var_temp["metadata"]["index_id"]   + ".mp4" + "\""
        out_file = "\"" + "./static/" + var_temp["id"] + ".mp4" + "\""
        
        if os.path.exists("./static/" + var_temp["id"] + ".mp4"):
            cmd = f"ffmpeg  -i  {input_file} -ss {start_time}  -to {end_time} -c copy  -f {out_file}"
        else:
            cmd = f"ffmpeg  -i  {input_file} -ss {start_time}  -to {end_time} -c copy   {out_file}"
        print("cmd is ",cmd)
        os.system(cmd)
        print("search_response is ",search_response.to_dict())
        video_out = "http://18.183.49.213:5000//data/" + var_temp["id"] + ".mp4"
        movie_array.append(video_out)

        print("movie_array is ",movie_array)

    search_response_dict = search_response.to_dict()
    search_response_dict["index_video"] = movie_array
    print(search_response_dict)
    return jsonify(search_response_dict)

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



