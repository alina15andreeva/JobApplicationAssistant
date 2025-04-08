import json
import pandas as pd
import re
import os

# Load parsed data from JSON files
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
jobs_file = os.path.join(input_dir, 'indeed_scraper/results/' + 'jobs' + '.json')
jobs_keys_file = os.path.join(input_dir, 'indeed_scraper/results/' + 'job_keys' + '.json')
search_file = os.path.join(input_dir, 'indeed_scraper/results/' + 'search' + '.json')

with open(jobs_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(jobs_keys_file, 'r', encoding='utf-8') as f:
    job_keys = json.load(f)

with open(search_file, 'r', encoding='utf-8') as f:
    search_data = json.load(f)

# Initialize an empty jobs_data list to store jobs data
jobs_data = []

# Iterate through every job in data
for job in data:
    # Initialize an empty job_dict dictionary to store data on every job on each iteration
    job_dict = {}
    # Find relevant job information and add it to job_dict
    job_dict["Job Title"] = job.get("jobTitle", "N/A")
    job_dict["Company Name"] = job.get("companyName", "N/A")
    job_dict["Location"] = job.get("formattedLocation", "N/A")
    job_dict["Description"] = job.get("description", "N/A")
    job_dict["Company Rating"] = job.get("companyReviewModel", {}).get("ratingsModel", {}).get("rating", "N/A") if job.get("companyReviewModel") else "N/A"
    job_dict["Company Reviews"] = job.get("companyReviewModel", {}).get("ratingsModel", {}).get("count", "N/A") if job.get("companyReviewModel") else "N/A"
    job_dict["Job Type"] = job.get("jobType", "N/A") or "N/A"

    # Extract salary information from search_data using job key
    job_key = job.get("jobkey")
    # Check if the job_key is in job_keys
    if job_key in job_keys:
        # Search for the first job in the search_data that has a "jobkey" that matches the job_key from the current job
        matching_job = next((item for item in search_data if item.get("jobkey") == job_key), None)
        
        if matching_job:
            estimated_salary = matching_job.get("estimatedSalary", {})
            job_dict["Salary Range"] = estimated_salary.get("formattedRange", "N/A")
            job_dict["Min Salary"] = estimated_salary.get("min", "N/A")
            job_dict["Max Salary"] = estimated_salary.get("max", "N/A")
            job_dict["Salary Type"] = estimated_salary.get("type", "N/A")

    # Append the specified job information to the jobs_data list
    jobs_data.append(job_dict)

# Convert jobs_data to DataFrame
df = pd.DataFrame(jobs_data)

# Remove any HTML tags from the 'Description' column by using a regular expression but only for rows where the description is a string
df["Description"] = df["Description"].apply(lambda x: re.sub(r'<.*?>', '', x) if isinstance(x, str) else x)
# Remove any URLs from the 'Description' column using a regular expression
df["Description"] = df["Description"].str.replace(r'http\S+|www\S+', '', regex=True)
# Remove any email addresses from the 'Description' column using a regular expression
df["Description"] = df["Description"].str.replace(r'\S+@\S+', '', regex=True)
# Remove any hashtags from the 'Description' column using a regular expression
df["Description"] = df["Description"].str.replace(r'#\w+', '', regex=True)
# Replace multiple consecutive spaces with a single space using a regular expression and remove leading and trailing spaces from the 'Description' column
df["Description"] = df["Description"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Define the output directory
output_dir = os.path.join(os.path.dirname(input_dir), 'output')
# Define the output file path
output_file = os.path.join(output_dir, 'indeed_jobs_data.csv')
# Save df to a CSV file
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")