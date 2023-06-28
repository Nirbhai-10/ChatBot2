from flask import Flask, request, jsonify, render_template
# import ChatBot
import ChatBotV2


app = Flask(__name__)

@app.route('/')
def home():
    return "Working at new route"

@app.route('/Chat')
def chat():
    return render_template("index.html")

@app.route('/ChatApp', methods=['POST'])
def respond():
    data = request.get_json(force=True)  # Use force=True to parse even if Content-Type is not set to application/json
    message = data['message']
    response = ChatBotV2.classify_and_get_response(message)
    return jsonify({'response': response})

@app.route('/text_check', methods=['POST'])
def test():
    req_Json = request.json
    message = req_Json['message']
    return jsonify({'We got it': message})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
