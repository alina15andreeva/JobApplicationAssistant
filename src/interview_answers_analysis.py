from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import pandas as pd
import whisper
import re
import os
import openai
import creds

# Export an OpenAI API key
openai.api_key = creds.openai_api_key

def speech_to_text(audio_file):
    """
    Takes an mp3 file, converts speech from an audio file into text and returns a transcript.
    """
    # Load the Whisper "turbo" model for speech-to-text processing
    model = whisper.load_model("turbo")
    # Transcribe the audio content from the given audio file into text.
    result = model.transcribe(audio_file)
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
        print(f"Error with OpenAI API: {e}")
        return []


# Here we specify the path to the matched resume-jobs file and read the top-matched job description
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
job_description_file = os.path.join(output_dir, 'similarity_results_openai.csv') # By default we use OpenAI API keyword matching. Change to 'similarity_results_spacy_nltk.csv' or 'similarity_results_cosine_similarity.csv' if needed
job_description = pd.read_csv(job_description_file)
description = job_description['Description'].iloc[0]

# Here we specify the path to the generated interview questions for the chosen job position and read them
generated_questions_file = os.path.join(output_dir, 'generated_interview_questions.csv')
questions_file = pd.read_csv(generated_questions_file).iloc[:, -1]
# Convert all questions to strings, joining them into a single string separated by spaces
questions_text = ' '.join(questions_file.astype(str).values)
# Extract questions using a regular expression that matches numbered items
questions = re.findall(r'^\d+\.\s(.*?)(?=\n\d+\.|\n[A-Za-z]|$)', questions_text, re.DOTALL | re.MULTILINE)
# Strip leading or trailing whitespace from each question
questions = [q.strip() for q in questions]

# Here we specify the path to the interview answers directory
data_dir = os.path.join(os.path.dirname(input_dir), 'data')
answer_dir = os.path.join(data_dir, 'interview_answers')
# Create a list of audio filenames for answers numbered 1 to 8
answer_files = [f"answer{i}.mp3" for i in range(1, 9)]

# Specify the ID of the pre-trained ONNX model to be used
model_id = "SamLowe/roberta-base-go_emotions-onnx"
# Specify the path to the ONNX model file
file_name = "onnx/model_quantized.onnx"
# Load the ONNX model
model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name)
# Load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize an empty list to store question-answer pairs
interview_qa = []
# Iterate through each question and corresponding answer file 
# and append a dictionary containing the question and its corresponding answer path to the interview_qa list
for question, answer_file in zip(questions, answer_files):
    answer_file_path = os.path.join(answer_dir, answer_file)
    interview_qa.append({
        "Q": question,
        "A": answer_file_path
    })

# Specify the output file path to save interview analysis results
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
interview_analysis_file = os.path.join(output_dir, 'interview_analysis_results.csv')

# Initializes an empty list to store analysis results
results_data = []

# Iterate through each question-answer pair
for qa in interview_qa:
    question = qa['Q']
    answer = qa['A']
    # Analyze the answer using the interview_analysis function
    result = interview_analysis(question, answer, description, model, tokenizer)
    
    # Join the sentiment details into a string with each on a new line
    sentiment_details = '\n'.join([f"{element['label'].capitalize()} - {round(element['score'], 2)}" for element in result['Sentiment']])
    
    # Append the results to the results_data list
    results_data.append({
        "Question": result['Question'],
        "Answer Transcript": result['Answer Transcript'],
        "Sentiment": sentiment_details,
        "Feedback": result['Feedback']
    })

# Convert results_df list to a DataFrame and save df to a CSV file
results_df = pd.DataFrame(results_data)
results_df.to_csv(interview_analysis_file, index=False)
print(f"Results saved to {interview_analysis_file}")