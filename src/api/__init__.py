from flask import Flask, jsonify, render_template, request
from bson.json_util import dumps
from collections import Counter

import pandas as pd
import numpy as np
import pymongo
import json
import model

df = pd.read_csv("dataset/movie_dataset.csv")
top = [title for title, num in Counter(np.asarray(df).flatten()).most_common()][1:6]

app = Flask(__name__, template_folder="/app/src/api/templates", static_folder="/app/src/api/static")

uri = "mongodb+srv://m220student:m220password@mflix-zhxbp.mongodb.net/test?retryWrites=true&w=majority"
columns = { "title":1, "plot":1, "poster":1, "year":1, "_id":0 }
mongo = pymongo.MongoClient(uri, maxPoolSize=50, connect=False)
db = pymongo.database.Database(mongo, "sample_mflix")
col = pymongo.collection.Collection(db, "movies")
col_results = json.loads(dumps(col.find({}, columns).sort("year", -1).limit(5)))

def find_movie(movie_title):
    data = json.loads(dumps(col.find({"title": movie_title}, columns).sort("year", -1)))
    return data[0]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            movielist=top
        )

    elif request.method == "POST":
        data = request.get_json()
        movie_info = find_movie(data["movie"])
        tit = movie_info["title"]
        des = movie_info["plot"]
        img = movie_info["poster"]
        yea = movie_info["year"]

        movie_card = render_template(
            "movie.html",
            img=img,
            tit=tit,
            des=des,
            yea=yea
        )

        return json.dumps({
            "html": movie_card
        })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    recs, confs = model.suggestion(data["movie"], data["maxrecom"])
    print("recs ", data["movie"], recs)
    print("confs ", data["movie"], confs)

    recommendations = []

    for movie, conf in zip(recs, confs):
        movie_info = find_movie(movie)

        tit = movie_info["title"]
        img = movie_info["poster"]
        des = movie_info["plot"]
        yea = movie_info["year"]
        recommendations.append((tit, img, yea, round(conf*100, 2)))

    movies_html = render_template(
        "recommendation2.html",
        recommendations=recommendations
    )

    return json.dumps({
        "html": movies_html
    })


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    srch = data["search"]

    results = [mov["title"] for mov in json.loads(dumps(col.find({'title':{'$regex':srch}}, columns)))[:5]]
    print(results)

    movies_html = render_template(
        "search.html",
        movielist=results
    )

    print(movies_html)
    return json.dumps({
        "html": movies_html
    })

@app.route("/playlist", methods=["POST"])
def playlist():
    data = request.get_json()
    retrained = model.retrain(data["playlist"])

    return json.dumps({
        "success": retrained
    })
