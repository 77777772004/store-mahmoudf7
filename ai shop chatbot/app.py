from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

app = Flask(__name__, static_folder='static', template_folder='templates')

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# تحميل القاعدة
faq_data = json.load(open("faq.json", encoding="utf-8"))
all_questions, answers = [], []

for item in faq_data:
    for q in item["questions"]:
        all_questions.append(q)
        answers.append(item["answer"])

question_embeddings = model.encode(all_questions)

# دالة لتحسين الرد بأسلوب طبيعي
def personalize_reply(answer):
    templates = [
        f"of course,sir {answer}",
        f"gladly {answer}",
        f"i am pleased to inform you that {answer}",
        f"okay {answer}",
        f"thank you for your question {answer}"
    ]
    return random.choice(templates)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question", "")
    user_embedding = model.encode([user_input])
    similarity = cosine_similarity(user_embedding, question_embeddings)[0]

    best_score = similarity.max()
    best_index = similarity.argmax()

    if best_score < 0.6:
        # الحصول على أفضل 3 اقتراحات
        top_indices = similarity.argsort()[-3:][::-1]
        suggestions = [all_questions[i] for i in top_indices]
        suggestions_text = "\n- " + "\n- ".join(suggestions)
        return jsonify({
            "answer": f"عذرًا، لم أتمكن من فهم سؤالك بدقة. هل تقصد:{suggestions_text}"
        })

    # الرد الطبيعي
    response = personalize_reply(answers[best_index])
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)

