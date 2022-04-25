from app import app
from flask import render_template, request, redirect
import os
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from scipy import spatial

app.config["PDF_UPLOADS"] = "D:/Sem 8/NLP_Proj/app/app/static/pdfs/uploads"
import tabula
import math
import re
from collections import Counter

WORD = re.compile(r"\w+")

global sim

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def find_sim(pdf1,pdf2):
    global df1
    global df2
    dff1 = tabula.read_pdf(app.config["PDF_UPLOADS"]+'/'+pdf1.filename, pages='all')
    dff2 = tabula.read_pdf(app.config["PDF_UPLOADS"]+'/'+pdf2.filename, pages='all')
    df1 = dff1[0].dropna(axis='columns', how='all')
    df2 = dff2[0].dropna(axis='columns', how='all')
    s2 = df1.to_string()
    s1 = df2.to_string()
    vector1 = text_to_vector(s1)
    vector2 = text_to_vector(s2)
    cosine = get_cosine(vector1, vector2)
    print("Cosine:", cosine)
    return cosine

def new_sim(pdf1,pdf2):
    global df3
    global df4
    dfs1 = tabula.read_pdf(app.config["PDF_UPLOADS"]+'/'+pdf1.filename, pages='all')
    dfs2 = tabula.read_pdf(app.config["PDF_UPLOADS"]+'/'+pdf2.filename, pages='all')
    df3 = dfs1[0].dropna(axis='columns', how='all')
    df4 = dfs2[0].dropna(axis='columns', how='all')
    s3 = df3.to_string()
    s4 = df4.to_string()
    s3 = remove_stopwords(s3)
    s4 = remove_stopwords(s4)
    s3 = [s3]
    s4 = [s4]
    vectorizer = HashingVectorizer(n_features=15)
    vector_1 = vectorizer.transform(s3)
    vec1 = vector_1.toarray()
    vector_2 = vectorizer.transform(s4)
    vec2 = vector_2.toarray()
    vec1 = vec1[0]
    vec2 = vec2[0]
    percor, _ = pearsonr(vec1, vec2)
    sprcor, _ = spearmanr(vec1, vec2)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = model.encode(s3)
    embeddings2 = model.encode(s4)
    embeddings1 = embeddings1[0]
    embeddings2 = embeddings2[0]
    bertsim = 1 - spatial.distance.cosine(embeddings1, embeddings2)
    return percor,sprcor,bertsim


@app.route("/")
def index():
    return render_template("public/index.html")


@app.route("/upload-pdf", methods=["GET", "POST"])
def upload_pdf():
        if request.method == "POST":

            if request.files:

                pdf1 = request.files["pdf1"]

                pdf2 = request.files["pdf2"]

                pdf1.save(os.path.join(app.config["PDF_UPLOADS"], pdf1.filename))
                pdf2.save(os.path.join(app.config["PDF_UPLOADS"], pdf2.filename))

                print('file saved')
                global sim, percor, sprcor, bertsim
                sim = find_sim(pdf1,pdf2)
                percor,sprcor,bertsim = new_sim(pdf1,pdf2)
                print(percor,sprcor,bertsim)
                return redirect("/result")
        return render_template("public/upload_pdf.html")

@app.route("/result", methods=("POST", "GET"))
def result():
    return render_template("public/result.html", result=sim*100, result1=percor*100, result2=sprcor*100, result3=bertsim*100 ,tables=[df1.to_html(classes='data')], titles=df1.columns.values, tabless=[df2.to_html(classes='data')])

