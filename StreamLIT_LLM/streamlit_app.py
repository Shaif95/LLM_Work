import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_faq(raw_text):
    # Split the raw_text into individual lines
    lines = raw_text.split('\n')

    # Initialize empty lists to store questions and answers
    questions = []
    answers = []

    # Temporary variables to store the current question and answer being processed
    current_question = ""
    current_answer = ""

    # Iterate through each line and process the FAQ
    for line in lines:
        line = line.strip()
        if line.endswith("?"):  # Check if the line is a question
            # If there was a previous question and answer, add them to the lists
            if current_question and current_answer:
                questions.append(current_question)
                answers.append(current_answer)

            # Update the current question
            current_question = line
            # Reset the current answer
            current_answer = ""
        else:
            # Update the current answer
            current_answer += line

    # Add the last question and answer to the lists
    if current_question and current_answer:
        questions.append(current_question)
        answers.append(current_answer)

    return questions, answers

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorizer = TfidfVectorizer()
    vectorstore = []
    for vector in text_chunks:
        vectorstore.append(vectorizer.fit_transform([vector]))
    return vectorstore

def get_responses_for_questions(text_chunks, vectorstore, questions, reference_text):
    llm = ChatOpenAI()

    # Initialize the memory for the conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Create a conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    responses = []

    # Loop through each question and get the GPT response
    for question in questions:
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        # Add the question as user input to the conversation_chain
        conversation_chain.memory.add_message(
            role="user", content=question
        )

        # Generate GPT response based on the updated conversation_chain
        gpt_response = conversation_chain.generate_response()

        # Add the GPT response to the list of responses
        responses.append(gpt_response)
    return responses

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def solve_vec(k,raw_text,text_chunks ,vectorstore):

    text_chunks_sentences = [' '.join(chunk_tokens) for chunk_tokens in text_chunks]

    # Combine the small question text "k" and the text chunks into a list
    all_texts = [k] + text_chunks_sentences

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the list of texts into TF-IDF vectors
    tfidf_vectors = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarities between the first vector (k) and all the other vectors (text chunks)
    cosine_similarities = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1:])

    # Find the index of the text chunk with the highest similarity score
    closest_chunk_index = cosine_similarities.argmax()

    closest_chunk  = ''.join(text_chunks[closest_chunk_index])

    context = closest_chunk 

    ques = k 
    source = raw_text 

    llm = OpenAI()

    prompt = "Here is a key text extracted from a PDF : "  + source + "\n Here is the part of the text that might have answer:" + context + "\n\nBased on this, give a good answer " + ques

    res = llm(prompt)

    return res


def cal_base(answers,baseline):

    sum = 0

    vectorizer = TfidfVectorizer()

    for i in range(len(answers)):
        vectorized_answers = vectorizer.fit_transform([answers[i], baseline[i]])
        cosine_sim = cosine_similarity(vectorized_answers[0], vectorized_answers[1])[0][0]
        sum = sum + cosine_sim

    mean = sum/ len(answers)

    return mean*100

def get_base(ques,source):

    llm = OpenAI()
    prompt = "Here is a reference text extracted from a PDF "  + source + "\n\nBased on this, give a good answer " + ques
    res = llm(prompt)

    return res

def solve(ques,source):

    prompt = "Here is a reference text extracted from a PDF "  + source + "\n\nBased on this, give a good answer " + ques
    
    results = []
    temp = .6
    for i in range(0,3):
        llm = OpenAI()
        results.append(llm(prompt))
        temp = temp + .1

    prompt = "you have to give a good answer to this question : " + ques + " . \nThese are answers from 3 other ChatGPT calls"+"\n 1. " + results[0] +"\n 2. " + results[1] +"\n 3. " + results[2]  +"\n 5. " +  "Here is a reference text extracted from a PDF "  + source + "\n\nBased on this, give the best worded, most comprehensive, most accurate answer"
    llm = OpenAI()
    ans = llm(prompt)

    return ans

def main():
    load_dotenv()
    st.set_page_config(page_title="AI Camp QnA Project",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("AI Camp QnA Project")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)



        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    if st.button("Evaluate"):

        with st.spinner("Processing"):

            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            questions, answers = process_faq(raw_text)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            baseline = []

            solution = []

            st.write(len(text_chunks), len(vectorstore))

            vector_sol = []

            for k in questions:
                vector_sol.append(solve_vec(k,raw_text,text_chunks ,vectorstore))
                baseline.append(get_base(k,raw_text))
                solution.append(solve(k,raw_text))
                


            acc = cal_base(answers,baseline)

            st.write("Baseline Cosine Similarity : ",acc)

            acc1 = cal_base(answers,solution)

            st.write("Cosine Similarity with Combined LLM Solution: ",acc1)

            acc2 = cal_base(answers,vector_sol)

            st.write("Cosine Similarity with Vector Solution: ",acc2)

            st.write("Questions : ")

            st.write(questions[:2])

            st.write("Answers : ")

            st.write(answers[:2])

            st.write("GPT Baseline Answers : ")

            #st.write(baseline[:2])

            st.write("Our Solution : ")

            #st.write(solution[:2])

            


if __name__ == '__main__':
    main()