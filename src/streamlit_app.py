import streamlit as st
import nltk_spacy
import textract
import pandas as pd
import tempfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import openai
import creds
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import whisper

# Export an OpenAI API key
openai.api_key = creds.openai_api_key

# Display the app title in Streamlit
st.title("Resume-Job Description Matcher and Interview Assistant")
# Display the sidebar title
st.sidebar.title("CV-Job Matching Methods")

# Create a dropdown menu for selecting a matching method
matching_option = st.sidebar.selectbox(
    "Choose Matching Method",
    ["SpaCy + NLTK", "AI SpaCy + NLTK", "Cosine Similarity"]
)


def extract_keywords_from_cv(cv_text):
    """
    Extracts keywords from a resume text using OpenAI API and specified prompt. 
    Change the prompt based on the job field.
    """
    # Create a prompt to ask the OpenAI API to extract keywords from the given resume text
    prompt = f"""
    Below you see a CV text. Extract the keywords for programming languages, frameworks, tools, and algorithms important for the job. Return as one Python list format, all lowercase.
    
    CV text:
    {cv_text}
    """

    try:
        # Send the prompt to the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts keywords."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
        )
        # Extract the response content from the OpenAI API's result
        response_content = response['choices'][0]['message']['content']
        # Use eval() to interpret the response content as a Python list
        keywords = eval(response_content)
        return keywords
    # If there are errors related to the OpenAI API, print the message
    except openai.error.OpenAIError as e:
        st.error(f"Error with OpenAI API: {e}")
        return []


def speech_to_text(audio_file):
    """
    Takes an mp3 file, converts speech from an audio file into text and returns a transcript.
    """
    # Read audio bytes from uploaded file. Streamlit's audio_file object from the file uploader provides 
    # an in-memory file-like object (BytesIO). Whisper cannot directly process this in-memory object. 
    # Writing the data to a temporary file ensures compatibility.
    audio_bytes = audio_file.read()
    
    # Create a temporary file for audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        # Write audio data to temporary file
        temp_audio_file.write(audio_bytes)
        # Close the temporary file
        temp_audio_file.close()
        # Load the Whisper "turbo" model for speech-to-text processing
        model = whisper.load_model("turbo")
        # Transcribe the audio content from the given audio file into text.
        result = model.transcribe(temp_audio_file.name)
        # Remove the temporary file
        os.remove(temp_audio_file.name)
    return result["text"]


def answer_sentiment_analysis(answer, model, tokenizer):
    """
    Analyzes the sentiment of a given text using a pre-trained ONNX text classification model.
    """
    # Create a text-classification pipeline using the provided ONNX model and tokenizer
    onnx_classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=3,
        function_to_apply="sigmoid",
    )
    return onnx_classifier(answer)


def interview_analysis(question, answer, description, roberta_model, tokenizer):
    """
    Analyzes a candidate's response to an interview question in the context of the job description.
    Returns interview question, answer transcript, sentiment and feedback.
    """
    # Get the answer transcript using speech_to_text function
    answer_transcript = speech_to_text(answer)
    # Create a prompt to analyze the answer based on specific criteria and its relation to the job description and question
    prompt = f"""
    The following is a candidate's response to an interview question. 
    Please analyze the answer based on the question, job description, and the following criteria:
    - Logical Alignment: Does the answer directly address the question, demonstrating a clear understanding of what was being asked? Are the response's details relevant to the problem or situation described in the question?
    - Depth of Insight: Does the candidate demonstrate a deep understanding of the problem and provide thoughtful insights? Are they able to explain why they chose particular actions or strategies?
    - Quantifiable Impact: Does the candidate provide measurable outcomes, such as metrics or specific improvements, to show the effectiveness of their actions?
    - Relevance: Does the answer relate directly to the job description, focusing on the required skills and responsibilities?
    - Fluency: Evaluate the language fluency, grammar, and clarity of the answer.

    Job Description:
    {description}

    Interview Question:
    {question}

    Candidate's Answer:
    {answer_transcript}

    Provide feedback and based on the analysis, provide any areas where the candidate can improve or refine their response (be specific to the question and answer):
    """
    
    try:
        # Send the prompt to the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that evaluates candidate answer based on interview question and job description, providing detailed feedback and improvement suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
        )
        # Extract the response content from the OpenAI API's result
        feedback = response['choices'][0]['message']['content']
        # Analyze the sentiment of the candidate's answer transcript using a sentiment analysis model.
        sentiment = answer_sentiment_analysis(answer_transcript, roberta_model, tokenizer)[0]

        return {
        "Question": question,
        "Answer Transcript": answer_transcript,
        "Sentiment": sentiment,
        "Feedback": feedback
    }
    # If there are errors related to the OpenAI API, print the message
    except openai.error.OpenAIError as e:
        st.error(f"Error with OpenAI API: {e}")
        return []


def extract_questions(text):
    """
    Get the generated questions text and parses generated questions text.
    """
    # Define regex pattern for extracting questions
    pattern = r"(?:Technical Questions|Behavioral Questions|Situational Questions):\n([\s\S]+?)(?=\n\n|$)"
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Initialize an empty list to store the questions
    questions = []
    # Iterate through matched blocks
    for block in matches:
        # Extract individual questions from the block
        block_questions = re.findall(r"\d+\.\s(.+)", block)
        # Add questions to the list
        questions.extend(block_questions)
    # Return the list of questions
    return questions


def interview_questions_generation(description):
    """
    Generates interview questions from a job description text using OpenAI API and specified prompt.
    """
    # Create a prompt to ask the OpenAI API to generate interview questions from the given job description text
    prompt = f"""
    Based on the following job description, generate a set of interview questions that assess the candidate's skills, responsibilities, and qualifications.
    The questions should be a mix of technical, behavioral, and situational questions.

    Job Description:
    {description}

    Include at least:
    - 3 technical questions related to the skills and responsibilities mentioned in the job description.
    - 3 behavioral questions based on the STAR method (Situation, Task, Action, Result).
    - 2 situational questions that assess the candidate's ability to handle specific scenarios.

    Generate question strictly in this format:
    Technical Questions:
    1.
    2.
    3.
    Behavioral Questions:
    1.
    2.
    3.
    Situational Questions:
    1.
    2.
    Only include the questions text. Each question text should be one-two sentences long. Nothing in the generated text should contain any kind of type (bold, italic, underline).
    """

    # Send the prompt to the OpenAI API to generate a response
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that generates interview questions based on the job description."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
        )
        # Extract the response content from the OpenAI API's result
        response_content = response['choices'][0]['message']['content']
        return response_content
    # If there are errors related to the OpenAI API, print the message
    except openai.error.OpenAIError as e:
        st.error(f"Error with OpenAI API: {e}")
        return []


def select_job_index(job_descriptions_df):
    """
    Returns job description based on the selected job index.
    """
    # Extract a list of job titles from the dataframe
    job_titles = job_descriptions_df['Job Title'].tolist()
    # Display a section header for the Streamlit interface
    st.write("### Generate Interview Questions")
    
    # Create a number input widget to allow the user to select a job index, with a unique key.
    selected_job_index = st.number_input(
        "Enter the Job Index (based on the list above):",
        min_value=0,
        max_value=len(job_titles) - 1,
        step=1,
        key="job_index_input"  # Unique key to avoid duplicate ID
    )
    
    # Retrieve the job description corresponding to the selected index
    selected_job_description = job_descriptions_df['Description'].iloc[selected_job_index]
    # Return the selected job description
    return selected_job_description


def q_a(job_descriptions_df, model, tokenizer):
    """
    Generates interview questions based on a selected job description, allows users to upload MP3 answers, 
    and analyzes the answers for sentiment and feedback using the provided model and tokenizer.
    Interacts with users via a Streamlit interface for question generation, file uploads, and result display.
    """
    # Check if generated_questions is in session state, initialize to None if not
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = None
    # Call select_job_index to get the selected job description
    selected_job_description = select_job_index(job_descriptions_df)
    # Call interview_questions_generation to generate interview questions based on the selected job description
    questions = interview_questions_generation(selected_job_description)
    # Create a button to trigger question generation and store the questions in session state
    if st.button("Generate Questions"):
        st.session_state.generated_questions = questions
    
    # Display the generated questions if available in session state
    if st.session_state.generated_questions:
        questions_text = st.session_state.generated_questions
        # Create a text area to display the questions with scrollable height
        st.text_area("Interview Questions", questions_text, height=300)

        # Add a download button for saving questions as a text file
        st.download_button(
            label="Download Questions as Text",
            data=questions_text.encode('utf-8'),
            file_name='interview_questions.txt',
            mime='text/plain'
        )

        # Call extract_questions to extract individual questions from the text
        questions = extract_questions(questions_text)

        # Show an error if no valid questions are extracted
        if not questions:
            st.error("No valid questions were extracted from the job description.")
            return

        # Display a section header for uploading answers in MP3 format
        st.write("### Upload MP3 Answers for Each Question")

        # Create file uploaders for each question to allow MP3 file uploads
        num_questions = len(questions)  # Number of questions generated
        uploaded_mp3_files = []

        for i in range(num_questions):
            # Create a file uploader for each question and store uploaded files in a list
            uploaded_file = st.file_uploader(f"Upload MP3 for Question {i+1}", type="mp3", key=f"file_uploader_{i}")
            uploaded_mp3_files.append(uploaded_file)

        # Proceed only if all MP3 files are uploaded
        if all(uploaded_mp3_files):  # Check if all files have been uploaded
            # Display the list of generated questions and their corresponding MP3 answers
            st.write("### Generated Interview Questions:")
            for i, (question, uploaded_file) in enumerate(zip(questions, uploaded_mp3_files), 1):
                st.write(f"**{i}. {question}**")
                st.audio(uploaded_file, format="audio/mp3")
            
            # Combine each question and corresponding MP3 file into a list of dictionaries
            interview_qa = []
            for question, uploaded_file in zip(questions, uploaded_mp3_files):
                interview_qa.append({
                    "Q": question,
                    "A": uploaded_file
                })
            
            # Display analysis results for each question-answer pair
            st.write("### Analysis Results:")
            for qa in interview_qa:
                question = qa['Q']
                answer = qa['A']
                # Call interview_analysis to perform analysis on the question and answer
                result = interview_analysis(question, answer, selected_job_description, model, tokenizer)
                # Display the question and its analyzed results including sentiment and feedback
                st.write(f"### Question: {result['Question']}")
                st.write(f"**Answer Transcript:** {result['Answer Transcript']}")
                st.write("**Sentiment:**")
                for element in result['Sentiment']:
                    label = element['label'].capitalize()
                    score = round(element['score'], 2)
                    st.write(f"{label} - {score}")
                st.write(f"**Feedback:** {result['Feedback']}")
                st.write("\n" + "="*50 + "\n")
        else:
            # Prompt the user to upload all MP3 files if any are missing
            st.write("Please upload all MP3 files to proceed with the analysis.")


# Specify the ID of the pre-trained ONNX model to be used
model_id = "SamLowe/roberta-base-go_emotions-onnx"
# Specify the path to the ONNX model file
file_name = "onnx/model_quantized.onnx"
# Load the ONNX model
model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name)
# Load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# File uploader accepting both PDF and DOCX
resume_path = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])
if resume_path is not None:
    # Get the uploaded file's extension
    file_extension = resume_path.name.split(".")[-1].lower()
    # Create a temporary file with the correct extension to store the uploaded resume
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        # Write the file content to the temporary file
        tmp_file.write(resume_path.read())
        # Store the path to the temporary file
        tmp_file_path = tmp_file.name
    try:
        # Extract the file text using textract. Use pdftotext method if it is a PDF file
        resume_text = textract.process(tmp_file_path, method='pdftotext' if file_extension == 'pdf' else None).decode('utf-8').lower()
        # Remove HTML tags
        resume_text = re.sub(r'<.*?>', '', resume_text)
        # Remove URLs
        resume_text = re.sub(r'http\S+|www\S+', '', resume_text)
        # Remove email addresses
        resume_text = re.sub(r'\S+@\S+', '', resume_text)
        # Remove hashtags
        resume_text = re.sub(r'#\w+', '', resume_text)
        # Remove non-ASCII characters
        resume_text = re.sub(r'[^\x00-\x7F]+', '', resume_text)
        # Remove percentages and special symbols
        resume_text = re.sub(r'[%â€™™]', '', resume_text)
        # Replace multiple spaces with a single space and strip leading/trailing spaces
        resume_text = re.sub(r'\s+', ' ', resume_text).strip()

    # If there is an error with text extraction, return the message
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

    if matching_option == "SpaCy + NLTK":
        # Display section heading for SpaCy + NLTK matching
        st.write("### SpaCy + NLTK")
        # Here we extract the keywords from the resume using spacy and nltk approaches
        keywords_resume_spacy = nltk_spacy.spacy_keywords(resume_text)
        keywords_resume_nltk = nltk_spacy.nltk_keywords(resume_text)

        # Display extracted keywords for SpaCy and NLTK
        st.write(f"**SpaCy keywords:**", ', '.join(keywords_resume_spacy))
        st.write(f"**NLTK keywords:**", ', '.join(keywords_resume_nltk))

        # File uploader to upload job descriptions in CSV format
        job_description_file = st.file_uploader("Upload Job Descriptions (CSV)", type="csv")
        if job_description_file is not None:
            # Read job descriptions from the uploaded CSV file into a DataFrame
            job_descriptions_df = pd.read_csv(job_description_file)
            # Convert job descriptions to lowercase and store them as a list
            job_descriptions = job_descriptions_df['Description'].fillna('').str.lower().tolist()

            # Initialize an empty list to store similarity results
            similarities = []
            # Iterate through the job descriptions
            for i, job_text in enumerate(job_descriptions):
                # Matching with SpaCy keywords
                # Extract keywords from the job description using SpaCy-based keyword extraction
                keywords_job_spacy = nltk_spacy.spacy_keywords(job_text)
                # Find the common keywords between the job description and the resume using SpaCy-extracted keywords
                keywords_matched_spacy = set(keywords_job_spacy).intersection(keywords_resume_spacy)
                # Calculate the percentage of matched keywords relative to the total keywords in the resume
                percentage_spacy = len(keywords_matched_spacy) / len(keywords_resume_spacy) if keywords_resume_spacy else 0.0

                # Matching with NLTK keywords
                # Extract keywords from the job description using NLTK-based keyword extraction
                keywords_job_nltk = nltk_spacy.nltk_keywords(job_text)
                # Find the common keywords between the job description and the resume using NLTK-extracted keywords
                keywords_matched_nltk = set(keywords_job_nltk).intersection(keywords_resume_nltk)
                # Calculate the percentage of matched keywords relative to the total keywords in the resume
                percentage_nltk = len(keywords_matched_nltk) / len(keywords_resume_nltk) if keywords_resume_nltk else 0.0

                # Append similarity results to the similarities list
                similarities.append([
                    f'Job {i + 1}',
                    f'{percentage_spacy:.2%}',
                    ' '.join(keywords_matched_spacy),
                    f'{percentage_nltk:.2%}',
                    ' '.join(keywords_matched_nltk)
                ])

            # Convert similarities list to DataFrame
            similarities_df = pd.DataFrame(similarities, columns=[
                'File',
                'SpaCy %', 
                'SpaCy Keywords', 
                'NLTK %', 
                'NLTK Keywords'
            ])

            # Merge with original job data
            full_results_df = pd.concat([job_descriptions_df, similarities_df], axis=1)

            # Sort by 'SpaCy %' and 'NLTK %' in descending order
            full_results_df['SpaCy %'] = full_results_df['SpaCy %'].str.replace('%', '').astype(float)
            full_results_df['NLTK %'] = full_results_df['NLTK %'].str.replace('%', '').astype(float)
            full_results_df = full_results_df.sort_values(by=['SpaCy %', 'NLTK %'], ascending=False)
            # Add '%' back to the 'SpaCy %' and 'NLTK %' columns
            full_results_df['SpaCy %'] = full_results_df['SpaCy %'].apply(lambda x: f"{x}%")
            full_results_df['NLTK %'] = full_results_df['NLTK %'].apply(lambda x: f"{x}%")

            # Display the final results in a table format
            st.dataframe(full_results_df[['Job Title', 'Company Name', 'SpaCy %', 'SpaCy Keywords', 'NLTK %', 'NLTK Keywords']])

            # A button to download the full results as a CSV file
            st.download_button(
                label="Download Full Results as CSV",
                data=full_results_df.to_csv(index=False).encode('utf-8'),
                file_name="matching_results.csv",
                mime="text/csv"
            )

            # Call q_a for interview questions generation and answers analysis
            q_a(job_descriptions_df, model, tokenizer)
            
    elif matching_option == "AI SpaCy + NLTK":
        # Display section heading for AI SpaCy + NLTK matching
        st.write("### AI SpaCy + NLTK")

        # Use session state to store extracted keywords if not already stored
        if "extracted_keywords" not in st.session_state:
            st.session_state.extracted_keywords = extract_keywords_from_cv(resume_text)

        # Retrieve extracted keywords from session state
        resume_keywords = st.session_state.extracted_keywords

        # Display the extracted keywords
        st.write(f"**Extracted Keywords:**", ', '.join(resume_keywords))

        # File uploader to upload job descriptions in CSV format
        job_description_file = st.file_uploader("Upload Job Descriptions (CSV)", type="csv", key="keyword_matching")
        if job_description_file is not None:
            # Read job descriptions from the uploaded CSV file into a DataFrame
            job_descriptions_df = pd.read_csv(job_description_file)
            # Convert job descriptions to lowercase and store them as a list
            job_descriptions = job_descriptions_df['Description'].fillna('').str.lower().tolist()

            # Initialize an empty list to store similarity results
            similarities = []
            # Iterate through the job descriptions
            for i, job_text in enumerate(job_descriptions):
                # Matching with SpaCy keywords
                # Extract keywords from the job description using SpaCy-based keyword extraction
                keywords_job_spacy = nltk_spacy.spacy_keywords(job_text)
                # Find the common keywords between the job description using SpaCy-extracted keywords and the resume
                keywords_matched_spacy = set(keywords_job_spacy).intersection(resume_keywords)
                # Calculate the percentage of matched keywords relative to the total keywords in the resume
                percentage_spacy = len(keywords_matched_spacy) / len(resume_keywords) if resume_keywords else 0.0

                # Matching with NLTK keywords
                # Extract keywords from the job description using NLTK-based keyword extraction
                keywords_job_nltk = nltk_spacy.nltk_keywords(job_text)
                # Find the common keywords between the job description using NLTK-extracted keywords and the resume
                keywords_matched_nltk = set(keywords_job_nltk).intersection(resume_keywords)
                # Calculate the percentage of matched keywords relative to the total keywords in the resume
                percentage_nltk = len(keywords_matched_nltk) / len(resume_keywords) if resume_keywords else 0.0

                # Append similarity results to the similarities list
                similarities.append([
                    f'Job {i + 1}',
                    f'{percentage_spacy:.2%}',
                    ' '.join(keywords_matched_spacy),
                    f'{percentage_nltk:.2%}',
                    ' '.join(keywords_matched_nltk)
                ])

            # Convert similarities list to DataFrame
            similarities_df = pd.DataFrame(similarities, columns=[
                'File',
                'SpaCy %', 
                'SpaCy Keywords', 
                'NLTK %', 
                'NLTK Keywords'
            ])

            # Merge with original job data
            full_results_df = pd.concat([job_descriptions_df, similarities_df], axis=1)

            # Sort by 'SpaCy %' and 'NLTK %' in descending order
            full_results_df['SpaCy %'] = full_results_df['SpaCy %'].str.replace('%', '').astype(float)
            full_results_df['NLTK %'] = full_results_df['NLTK %'].str.replace('%', '').astype(float)
            full_results_df = full_results_df.sort_values(by=['SpaCy %', 'NLTK %'], ascending=False)
            # Add '%' back to the 'SpaCy %' and 'NLTK %' columns
            full_results_df['SpaCy %'] = full_results_df['SpaCy %'].apply(lambda x: f"{x}%")
            full_results_df['NLTK %'] = full_results_df['NLTK %'].apply(lambda x: f"{x}%")

            # Display the final results in a table format
            st.dataframe(full_results_df[['Job Title', 'Company Name', 'SpaCy %', 'SpaCy Keywords', 'NLTK %', 'NLTK Keywords']])

            # A button to download the full results as a CSV file
            st.download_button(
                label="Download Results as CSV",
                data=full_results_df.to_csv(index=False).encode('utf-8'),
                file_name='similarity_results_openai.csv',
                mime='text/csv'
            )

            # Call q_a for interview questions generation and answers analysis
            q_a(job_descriptions_df, model, tokenizer)

    elif matching_option == "Cosine Similarity":
        # Display section heading for Cosine Similarity matching
        st.write("### Cosine Similarity")

        # File uploader to upload job descriptions in CSV format
        job_description_file = st.file_uploader("Upload Job Descriptions (CSV)", type="csv")
        if job_description_file is not None:
            # Read job descriptions from the uploaded CSV file into a DataFrame
            job_descriptions_df = pd.read_csv(job_description_file)
            # Convert job descriptions to lowercase and store them as a list
            job_descriptions = job_descriptions_df['Description'].fillna('').str.lower().tolist()

            # Initialize an empty list to store similarity results
            similarities = []
            # Iterate through the job descriptions
            for i, job_text in enumerate(job_descriptions):
                # Check if either resume_text or job_text is empty or contains only whitespace, then cosine similarity is set to 0.0
                if not resume_text.strip() or not job_text.strip():
                    cosine = 0.0
                else:
                    # Create a list containing the resume text and the current job description for comparison
                    text = [resume_text, job_text]
                    # Initializes a CountVectorizer instance to convert a collection of text documents into a matrix where the rows represent the documents and the columns represent the tokens
                    cv = CountVectorizer()
                    # Applies the CountVectorizer to the text list, generating a sparse matrix of token counts for the resume and job description
                    count_matrix = cv.fit_transform(text)
                    # Computes the cosine similarity between the resume text and the current job description
                    cosine = cosine_similarity(count_matrix)[0][1]

                # Append similarity results to the similarities list
                similarities.append([
                    f'Job {i + 1}',
                    f'{cosine:.2%}'
                ])

            # Convert similarities list to DataFrame
            similarities_df = pd.DataFrame(similarities, columns=[
                'File',
                'Cosine %'
            ])

            # Merge with original job data
            full_results_df = pd.concat([job_descriptions_df, similarities_df], axis=1)

            # Sort by 'Cosine %' in descending order
            full_results_df['Cosine %'] = full_results_df['Cosine %'].str.replace('%', '').astype(float)
            full_results_df = full_results_df.sort_values(by=['Cosine %'], ascending=False)
            # Add '%' back to the 'Cosine %' column
            full_results_df['Cosine %'] = full_results_df['Cosine %'].apply(lambda x: f"{x}%")

            # Display the final results in a table format
            st.dataframe(full_results_df[['Job Title', 'Company Name', 'Cosine %']])

            # A button to download the full results as a CSV file
            st.download_button(
                label="Download Full Results as CSV",
                data=full_results_df.to_csv(index=False).encode('utf-8'),
                file_name="matching_results.csv",
                mime="text/csv"
            )

            # Call q_a for interview questions generation and answers analysis
            q_a(job_descriptions_df, model, tokenizer)