from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login

# ✅ Inicia sesión en Hugging Face con tu token personal
# (Reemplaza "tu_token_aqui" por el que generes en https://huggingface.co/settings/tokens)


# ✅ Configuración de Flask
app = Flask(__name__)

# ✅ Carga del modelo FLAN-T5-XL
model_id = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# ✅ Ruta principal
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Ruta para generar texto a partir del prompt
@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

# ✅ Corre la app en modo desarrollo
if __name__ == "__main__":
    app.run(debug=True)
