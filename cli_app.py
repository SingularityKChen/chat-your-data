import pickle

from dotenv import load_dotenv
import faiss

from query_data import get_chain

load_dotenv("env.txt")

if __name__ == "__main__":
    # import langchain
    # langchain.debug = True
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    index = faiss.read_index("docs.index")
    vectorstore.index = index
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
