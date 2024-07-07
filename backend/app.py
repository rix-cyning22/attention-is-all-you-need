from flask import Flask, jsonify, request
from flask_cors import CORS
import run_model as rm

app = Flask(__name__)
CORS(app)


@app.route("/api", methods=["POST"])
def predict():
    data = request.json
    if data["seq"] == "":
        return jsonify({"error": "sequence cannot be empty"})
    return jsonify(
        {
            "words_gen": int(data.get("gen_seq_length", 1)),
            **rm.generate_sequence(
                text=data["seq"],
                gen_words=int(data.get("gen_seq_length", 1)),
                k=int(data.get("get_num_words", 5)),
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
