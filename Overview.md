# Project Description

Create a local Streamlit application that allows a user to input job listing text (e.g., copied plain text from sites like LinkedIn, Indeed, Handshake)**. Each listing is given a score for semantic fit against multiple pre-tailored resumes (Data Scientist, Automation Engineer, Data Engineer, etc.) and routed to the best-matching resume. Then the resume is further edited via LLM to create a more optimized resume and cover letter. The generated content is then rated by the user, further refined, and embedded. Once the process is finished, a PDF is generated via function-call for manual submission.

**Intended to do this all with Playwright automation, but will have this all done manually to keep the project within Linkedin, Handshake, etc Terms of Service**

# Why

Widespread adoption of GenAI in the job market has created an arms race where every online application is shotgunned by LLM slop. There now exists a cottage industry of services where every apply is now “easy-apply”. The market is flooded with AI-powered resumes, cover letters, linkedin profiles, and even github repos. (A list of these start-ups are at the bottom)

Aside from competing with these services by sheer volume, the alternative is to either through the backdoor or develop a smarter, more personalized automation. If tailoring a resume has gone from signal to standard, then perhaps tailored resume generation can still be a source of edge. I'd like to use this project as a springboard for learning data science best practices and to build the bespoke tool I need to compete in this current market.

# Key Skills / Learning Objectives

*   Building interactive UIs with Streamlit
*   Information Extraction from Unstructured Text
*   Vector Embeddings for semantic search (Sentence-BERT → pgvector)
*   Relevance Ranking via cosine-similarity threshold tuning
*   Prompt Engineering w/ OpenAI API over Azure for Content Personalization

# High-Level Flow

1.  **Job Input (Manual via Streamlit)**
    *   Users manually copy plain text from job listings (e.g., from LinkedIn or other sites) and paste it into a Streamlit app running locally in their browser.
    *   Launch the Streamlit app locally (e.g., via `streamlit run app.py`).
    *   Provide a text area for pasting the full job listing text.
    *   Parse the plain text to extract key details (title, company, location, description, timestamp if available, apply URL if provided) using regex or NLP techniques (e.g., splitting on common patterns like "Job Title:", "Company:", or using spaCy for entity recognition).
    *   If I get stuck on the section above, I will either use a local LLM or OpenAI API call for structured outputs.
    *   Deduplicate against a PostgreSQL/pgvector DB by checking semantic similarity or hashed text to avoid reprocessing similar / slightly reworded listings.
    *   Enqueue processed listings in Redis for workflow orchestration (This is likely overkill for single-user application and end up doing direct function calls)

2.  **Encode (Job Vectorization)**
    *   Job descriptions are transformed into numerical representations (embeddings) for semantic comparison.
    *   Text is cleaned to remove boilerplate language (e.g., using regex to strip footers or repeated phrases). If I hit a wall with a regex-based approach, I will change to a locally run LLM or an OpenAI API call instead.
    *   Pre-trained model (`sentence-transformers/all-MiniLM-L6-v2`) converts the cleaned text into vector embeddings.
    *   These embeddings are stored in a PostgreSQL database with a pgvector extension for efficient searching.

3.  **Rank (Fit Scoring)**
    *   The job-listing embedding is compared against pre-computed embeddings of different pre-written resumes (one for Data Scientist, Data Engineer, Automation Engineer, etc..)
    *   Embeddings for my resumes are loaded from the database.
    *   Cosine similarity is calculated to determine the best match between the job and each persona.
    *   A similarity threshold (>= 0.8) is applied. If a match is found, the top rated resume is selected (Will make this threshold a slider in Streamlit).
    *   Tie-breaking: Random pick one. (Keeping it simple for now)
    *   Track usage stats for each resume in the DB; if only 2-3 are frequently selected, suggest rewrites with ATS keywords via the app's feedback loop

4.  **Route (OpenAI API Calls for Cover Letter / Resume Generation)**
    *   Rewriting the top-matching resume and generating a tailored cover letter with quantifiable quality control.
    *   I will need to generate a prompt that has the LLM rewrite the top-matching resume with the job listing’s keywords and create a new optimized resume specific to that listing (pylatex).
    *   If the LLM’s are too finicky with pylatex, backup is to try and generate a styled HTML document and convert to PDF. Anticipating a lot of issues on this one.
    *   Prepare skeleton cover letters that get edited via LLM API calls to achieve a better semantic fit
    *   Vectorize the generated documents and calculate their cosine similarity against the original job description embedding. Rescore the output to ensure it meets a minimum semantic threshold, triggering a regeneration with a different prompt if the score is too low.
    *   Generated documents are saved locally (e.g., to a downloads folder). If I end up scaling this (like with Redis, etc) I’ll consider using private S3 buckets.
    *   Depending on the cost of the API calls I might batch the Resume/Cover Letter generation tasks to save money on API calls or try and group similar requests like calculating semantic similarity between each job description and do a smarter batch

# Start-Ups / Companies already doing this:

*   **Sorce:** Job discovery platform that allows users to swipe through job opportunities in a "Tinder-like" interface and automates the application process
*   **JobCopilot:** Tool that automates the entire job search process, from generating customized resumes to tracking applications.
*   **BulkApply:** Service that automates and sends out a high volume of job applications daily to save users time.
*   **Enhancv:** AI-powered resume builder that provides design templates and content suggestions to create resumes that are friendly to applicant tracking systems (ATS).
*   **Kickresume:** AI-powered tool for creating resumes and cover letters, offering various templates and AI-driven features to help users build their application documents.
*   **AI Apply:** AI career assistant that automates the job search by creating tailored resumes and cover letters, submitting applications, and helping with interview preparation.
*   **ResumeWorded:** AI platform that reviews and provides feedback on resumes and LinkedIn profiles to help improve their effectiveness in job applications.
*   **FinalRound AI:** Tool designed for interview preparation, offering realistic AI-powered mock interviews and providing real-time feedback and coaching.
*   **LazyApply:** Job application automation tool that can apply to thousands of jobs with a single click, automatically filling out application forms on various platforms.
