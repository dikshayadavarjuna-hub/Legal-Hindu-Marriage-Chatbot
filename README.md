# ⚖️ Hindu Marriage Act Legal Chatbot

## 📌 Project Overview

The Hindu Marriage Act Legal Chatbot is an AI-based application that provides information from the Hindu Marriage Act, 1955.

Users can ask questions related to Hindu marriage law, and the chatbot retrieves relevant sections from the Act and provides an answer along with the legal source.

The application is developed using Python and Streamlit and is deployed online using Streamlit Community Cloud.

## 🎯 Objectives

- Provide easy access to information from the Hindu Marriage Act, 1955.
- Retrieve relevant legal sections based on user questions.
- Provide answers using a question-answering model.
- Display the source legal text along with the answer.
- Create a simple and user-friendly legal information interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- PyTorch
- NumPy

## 🧠 How the Chatbot Works

1. The Hindu Marriage Act text is stored in the project dataset.
2. The Act is divided into individual sections.
3. Sentence Transformers converts the sections into numerical embeddings.
4. FAISS is used to search for the most relevant sections.
5. The retrieved sections are provided as context to the question-answering model.
6. The chatbot generates an answer from the retrieved legal text.
7. The relevant legal source is displayed to the user.

## 📂 Project Structure

```text
Legal-Hindu-Marriage-Chatbot/
│
├── data/
│   ├── hindu_marriage_act.txt
│   └── hindu_marriage_act.pdf
│
├── src/
│   ├── chatbot.py
│   ├── retrieve.py
│   └── extract_text.py
│
├── requirements.txt
├── README.md
└── .gitignore
