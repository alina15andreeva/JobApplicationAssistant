import os
import textract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk_spacy
import openai
import creds
import re

# Export an OpenAI API key
openai.api_key = creds.openai_api_key

def process_file(file_path):
    """
    Processes a PDF or DOCX file using textract and returns the extracted text in lowercase.
    """
    # Extract file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    # Check if file extension is in the list
    if file_extension in ['.pdf', '.docx']:
        # Extract the file text using textract. Use pdftotext method if it is a PDF file
        extracted_text = textract.process(file_path, method='pdftotext' if file_extension == '.pdf' else None)
        return extracted_text.decode('utf-8').lower()
    # If file format is not in the list, return the message
    else:
        return "Unsupported file format. Input a PDF or DOCX file."


def extract_keywords_from_cv(resume_text):
    """
    Extracts keywords from a resume text using OpenAI API and specified prompt. 
    Change the prompt based on the job field.
    """
    # Create a prompt to ask the OpenAI API to extract keywords from the given resume text
    prompt = f"""
    Below you see a CV text. Extract the keywords for programming languages, frameworks, tools and algorithms important for the job. Return as one Python list format, all lowercase.
    
    CV text:
    {resume_text}
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
        print(f"Error with OpenAI API: {e}")
        return []


# Here we specify the path to the resume and call our function process_file to extract the resume text.
# The uploaded file can be either a PDF or DOCX format.
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
data_dir = os.path.join(os.path.dirname(input_dir), 'data')
resume_path = os.path.join(data_dir, 'resumes/DATA SCIENCE/Resume_8.pdf') # Change the file path here if needed (the file should be in data folder)
resume_text = process_file(resume_path)
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

# Extract keywords from resume using extract_keywords_from_cv function that uses OpenAI's API approach
resume_keywords = extract_keywords_from_cv(resume_text)

# Here we specify the path to the parsed jobs file and read the job descriptions
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
job_description_file = os.path.join(output_dir, 'indeed_jobs_data.csv')
job_descriptions_df = pd.read_csv(job_description_file)
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

# Save df to a CSV file
output_file = os.path.join(output_dir, 'similarity_results_openai.csv')
full_results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")