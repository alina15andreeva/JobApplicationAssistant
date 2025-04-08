import os
import textract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

# Here we specify the path to the parsed jobs file and read the job descriptions
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
job_description_file = os.path.join(output_dir, 'indeed_jobs_data.csv')
job_descriptions_df = pd.read_csv(job_description_file)
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

# Save df to a CSV file
output_file = os.path.join(output_dir, 'similarity_results_cosine_similarity.csv')
full_results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")