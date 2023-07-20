ChatBot Project Readme
Introduction
This project is a ChatBot built using Flask, a Python web framework, and natural language processing (NLP) techniques. The ChatBot interacts with users through a web-based interface, processing user inputs, and providing appropriate responses based on pre-defined intents.
Features
User-friendly web interface for interacting with the ChatBot.
Ability to upload text files containing intent data and convert them to JSON format.
Option to paste text directly into the web interface and convert it to JSON intent.
Training functionality to train the ChatBot using the uploaded intent files.
Utilizes NLP techniques such as tokenization, stop word removal, lemmatization, POS tagging, and named entity recognition for processing user inputs.
Installation
Clone the repository:

git clone https://github.com/Nirbhai-10/ChatBot2.git
cd ChatBot-Project
Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

pip install -r requirements.txt
Running the Application
To run the ChatBot web application, execute the following command:


python app.py
The application will start on http://localhost:5000/ by default. Open a web browser and navigate to the URL to interact with the ChatBot.

Usage
Upload Intent Files:

Click on the "Upload" button to select text files containing intent data.
The selected files will be converted to JSON format and listed in the "Intent Files" section.
Paste Text for Intent Conversion:

Enter or paste text into the text area.
Click on the "Paste" button to convert the text to JSON intent.
The JSON intent will be listed in the "Intent Files" section.
Train the ChatBot:

Select the desired intent files from the "Intent Files" list by clicking on them.
Click on the "Train" button to train the ChatBot using the selected intent files.
The training progress will be displayed in the progress bar.
Interact with the ChatBot:

After training, you can interact with the ChatBot using the input field provided.
Enter a message and press "Enter" or click the "Send" button to receive the ChatBot's response.
Dependencies
Flask: A Python web framework for building web applications.
nltk: The Natural Language Toolkit for natural language processing.
Python (>=3.6): The programming language used to build the application.

Project Structure

- app.py             # The main Flask application file.
- ChatBotV2.py       # The ChatBot logic and training module.
- templates/         # Directory containing HTML templates.
- uploads/           # Directory to store uploaded text files and generated JSON files.
- requirements.txt   # List of required Python packages.
- README.md          # Project documentation.
  
Future Improvements
Add user authentication to control access to the ChatBot and data.
Implement more advanced NLP techniques to enhance the ChatBot's understanding and responses.
Improve error handling and user feedback for a smoother user experience.

Contributors
