import openai
import pandas as pd
import creds
import os

# Export an OpenAI API key
openai.api_key = creds.openai_api_key

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
    
    try:
        # Send the prompt to the OpenAI API to generate a response
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
        print(f"Error with OpenAI API: {e}")
        return []


# Here we specify the path to the matched resume-jobs file and read the top-matched job description
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
job_description_file = os.path.join(output_dir, 'similarity_results_openai.csv') # By default we use OpenAI API keyword matching. Change to 'similarity_results_spacy_nltk.csv' or 'similarity_results_cosine_similarity.csv' if needed
job_description = pd.read_csv(job_description_file)
description = job_description['Description'].iloc[0]

# Extract the first row of the first 11 columns and convert it into a DataFrame
job_questions = pd.DataFrame([job_description.iloc[0, 0:11]])
#  Generate interview questions from the description and add them as a new column
job_questions['Generated Questions'] = interview_questions_generation(description)

# Save df to a CSV file
output_file = os.path.join(output_dir, 'generated_interview_questions.csv')
job_questions.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")