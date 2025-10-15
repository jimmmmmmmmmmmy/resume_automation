#### **Phase 1: Job Ingestion and Processing Pipeline**

**1. Overview & Objectives**

*   **1.1. Vision:** To create a robust local application that allows a user to input unstructured job listing text, have it intelligently parsed into a structured format, and stored in a persistent database, ensuring no duplicate entries are processed.
*   **1.2. Key Objectives for Phase 1:**
    *   Establish a scalable project structure and environment.
    *   Develop a user-friendly Streamlit interface for manual data entry.
    *   Implement a resilient, multi-layered information extraction service.
    *   Ensure data integrity through a robust deduplication mechanism.
    *   Persist structured data securely and provide clear user feedback.

**2. System Architecture & Design**

A high-level overview of how the components will interact.

*   **Frontend:** A `Streamlit` web application (`app.py`) will serve as the user interface for inputting text and viewing results.
*   **Backend Logic:** A core Python module (e.g., `processing.py`) will contain the business logic for parsing, extraction, hashing, and database communication. This separates the logic from the UI code.
*   **Data Store:** A `PostgreSQL` database with the `pgvector` extension will serve as the persistent data layer.
*   **Configuration:** A `.env` file will manage environment variables (e.g., database credentials) and will be loaded by the `python-dotenv` library.

**3. Detailed Task Breakdown & Implementation Steps**

Note each task has a clear `Definition of Done`.

### **Task 1: Foundational Setup — Step-by-Step Implementation**

---

**Objective:** To establish the project's version control, directory structure, isolated Python environment, dependencies, and database schema.

Executing Task 1 correctly ensures a clean, organized, and reproducible development environment.

---

#### **Step 1.1: Initialize the Git Repository**

*   **Goal:** To enable version control from the very beginning. This allows you to track every change, revert mistakes, and collaborate effectively in the future.

*   **Instructions:**
    1.  Open your terminal or command prompt.
    2.  Navigate to the location where you want to create your project.
    3.  Create the main project folder and move into it.

    ```bash
    mkdir resume_optimizer
    cd resume_optimizer
    ```

    4.  Initialize a new Git repository.

    ```bash
    git init
    ```

*   **Outcome:** You will see a message like `Initialized empty Git repository in .../.git/`. A hidden `.git` folder is created, which will track all your project's history.

---

#### **Step 1.2: Create the Project Directory Structure**

*   **Goal:** To organize your project files logically. This practice is known as "separation of concerns" and makes the codebase easier to navigate, maintain, and debug.

*   **Instructions:**
    1.  From the root of your `resume_optimizer` directory, create the necessary folders and empty files.

    ```bash
    # Create a directory for scripts (like database setup)
    mkdir scripts

    # Create the main application file and other necessary files
    touch app.py requirements.txt .gitignore .env
    ```

*   **Outcome:** Your project structure should now look like this:

    ```
    resume_optimizer/
    ├── .git/
    ├── scripts/
    ├── app.py
    ├── .env
    ├── .gitignore
    └── requirements.txt
    ```

---

#### **Step 1.3: Set Up the Python Virtual Environment**

*   **Goal:** To create an isolated environment for your project's Python dependencies. This prevents conflicts with other projects on your system and ensures that your application runs with the exact package versions it requires.

*   **Instructions:**
    1.  From the project's root directory, run the following command to create a virtual environment named `venv`.

    ```bash
    # For Python 3
    python3 -m venv venv
    ```

    2.  Activate the virtual environment. The command differs based on your operating system.

    *   **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate
        ```

*   **Outcome:** Your terminal prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active. Any Python packages you install will now be confined to this environment.

---

#### **Step 1.4: Populate the `.gitignore` File**

*   **Goal:** To tell Git which files and folders it should intentionally ignore. This keeps your repository clean by excluding environment-specific files, sensitive credentials, and auto-generated code.

*   **Instructions:**
    1.  Open the `.gitignore` file you created earlier in a text editor.
    2.  Add the following content.

    ```gitignore
    # Virtual Environment
    venv/
    .venv/

    # Python cache
    __pycache__/
    *.pyc

    # Environment variables - DO NOT COMMIT SENSITIVE DATA
    .env

    # OS-specific files
    .DS_Store
    Thumbs.db
    ```

*   **Outcome:** Git will now ignore these files. If you run `git status`, you will not see the `venv` directory or the `.env` file listed as untracked files.

---

#### **Step 1.5: Install Dependencies and Generate `requirements.txt`**

*   **Goal:** To install the initial set of Python libraries and lock their versions in a file for easy replication by others (or your future self).

*   **Instructions:**
    1.  Ensure your virtual environment is still active (your prompt should show `(venv)`).
    2.  Install the core packages using `pip`.

    ```bash
    pip install streamlit psycopg2-binary spacy python-dotenv sentence-transformers
    ```

    3.  Download the English language model for spaCy.

    ```bash
    python -m spacy download en_core_web_sm
    ```

    4.  Generate the `requirements.txt` file. This command "freezes" the current state of all installed packages and their exact versions into the file.

    ```bash
    pip freeze > requirements.txt
    ```

*   **Outcome:** Your `requirements.txt` file will be populated with a list of packages (e.g., `streamlit==1.28.0`, `spacy==3.7.2`, etc.). Anyone can now perfectly replicate your environment by running `pip install -r requirements.txt`.

---

#### **Step 1.6: Define the Database Schema in `init_db.sql`**

*   **Goal:** Create a reusable SQL script that defines the structure of your `job_listings` table. This makes your database setup automated and consistent.

*   **Instructions:**
    1.  Create a new file named `init_db.sql` inside the `scripts/` directory.
    2.  Open `scripts/init_db.sql` and add the following SQL code.

    ```sql
    -- Enable the pgvector extension to handle vector data types
    -- This must be run by a superuser on the database first.
    -- CREATE EXTENSION IF NOT EXISTS vector;

    -- Drop the table if it already exists to ensure a clean setup
    DROP TABLE IF EXISTS job_listings;

    -- Create the main table for storing job listings
    CREATE TABLE job_listings (
        id SERIAL PRIMARY KEY,                          -- Unique identifier for each listing
        job_title VARCHAR(255) NOT NULL,                -- The title of the job
        company VARCHAR(255),                           -- The name of the company
        location VARCHAR(255),                          -- Job location (e.g., "Remote", "New York, NY")
        description TEXT NOT NULL,                      -- The full job description text
        apply_url TEXT,                                 -- URL to the application page
        description_hash VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256 hash for exact deduplication
        source_platform VARCHAR(100),                   -- Where the listing was found (e.g., 'LinkedIn')
        metadata JSONB,                                 -- Flexible field for extra data (e.g., extraction method)
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()   -- Timestamp when the record was created
    );

    -- Create an index on the hash for fast duplicate lookups
    CREATE INDEX idx_description_hash ON job_listings(description_hash);

    -- !!Optional: The following code adds comments to the table and columns!!
    COMMENT ON TABLE job_listings IS 'Stores processed job listings from various sources.';
    COMMENT ON COLUMN job_listings.description_hash IS 'SHA-256 hash of the normalized job description for deduplication.';
    COMMENT ON COLUMN job_listings.metadata IS 'Stores metadata like extraction method, confidence scores, etc.';
    ```

*   **Outcome:** You now have a script that can be executed on any PostgreSQL database to instantly create the correctly structured table for your application.

---

### **Verification and Definition of Done for Task 1**

Task 1 is completed when:

1.  Your project is a Git repository (`git status` runs without error).
2.  The directory structure matches the one outlined in Step 1.2.
3.  You can activate your virtual environment (`source venv/bin/activate` or Windows equivalent).
4.  Running `pip install -r requirements.txt` in a fresh environment completes without errors.
5.  The `scripts/init_db.sql` file contains the complete `CREATE TABLE` statement.
6.  You can run the SQL script against a local PostgreSQL database (using `psql -f scripts/init_db.sql -d your_db_name`) and it creates the `job_listings` table successfully.

---

Of course. Let's expand on **Task 2: Streamlit UI and User Workflow** with a detailed, step-by-step guide.

This task focuses on creating the user-facing part of your application. The key challenge here is not just displaying widgets, but managing the application's *state* as the user moves from inputting data to verifying it. We will use Streamlit's `st.session_state` to create a robust and intuitive multi-step workflow within a single page.

---

### **Task 2: Streamlit UI and User Workflow — Step-by-Step Implementation**

**Objective:** To build a user interface that guides the user through pasting job text, processing it, and then verifying the extracted information before final submission.

---

#### **Step 2.1: Initialize the Streamlit App and Session State (app.py)**

*   **Goal:** To set up the basic structure of `app.py` and initialize the `session_state`, which will act as the application's short-term memory. This is the most critical step for building a multi-step workflow.

*   **Instructions & Pseudocode:**
    1.  At the very top of your `app.py`, import Streamlit.
    2.  Immediately after, add a block to initialize `st.session_state`. We will create a key, for instance `processed_data`, and set it to `None`. This ensures that the first time the script runs, this variable exists. Streamlit re-runs the entire script on every user interaction, so this check prevents our stored data from being erased.

    ```python
    # In app.py
    import streamlit as st

    # --- 1. SESSION STATE INITIALIZATION ---
    # This must be at the top of the script. It creates a dictionary-like object
    # that persists across script reruns for a single user session.

    # 'processed_data' will hold the dictionary of extracted job details after Step 1.
    # 'form_submitted' can be a flag to help control UI flow.
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # This is a placeholder for the actual extraction logic you'll build in Task 3.
    # It helps us build the UI without needing the backend to be complete.
    def placeholder_extraction_function(raw_text):
        # In the real implementation, this will call Regex, spaCy, etc.
        # For now, it just returns a hardcoded dictionary for testing the UI.
        return {
            'job_title': "Data Scientist (Placeholder)",
            'company': "FutureTech Inc. (Placeholder)",
            'location': "Remote (Placeholder)",
            'apply_url': "https://example.com/apply (Placeholder)",
            'description': raw_text[:200] + "..." # Truncated description
        }
    ```

---

#### **Step 2.2: Build the Initial UI for Data Input**

*   **Goal:** To create the first part of the user experience: a clear title, instructions, a large text box for input, and the initial processing button.

*   **Instructions & Pseudocode:**
    1.  Use `st.title()` and `st.markdown()` or `st.write()` to set up the page header and provide guidance to the user.
    2.  Use `st.text_area()` to create the input box. Give it a descriptive label and help text.
    3.  Use `st.button()` to create the "Process Job Listing" button.

    ```python
    # --- 2. UI - STEP 1: DATA INPUT ---
    # This section is always visible to the user.

    st.title("AI-Powered Resume Optimizer")
    st.markdown("### Phase 1: Job Ingestion")
    st.info("Paste the full, unformatted text of a job listing below and click 'Process' to begin.")

    # Create the text area for user input
    job_listing_text = st.text_area("Paste Job Listing Text Here", height=300, key="job_text_input")

    # Create the initial button that triggers the extraction process
    process_button = st.button("Process Job Listing")
    ```

---

#### **Step 2.3: Handle the "Process" Button Click and State Update**

*   **Goal:** To define what happens when the user clicks the "Process" button. The core logic is to call our (currently placeholder) extraction function and, crucially, save its output to `st.session_state.processed_data`.

*   **Instructions & Pseudocode:**
    1.  Create an `if` block that checks if the `process_button` was clicked.
    2.  Inside the `if` block, add a check to ensure the user has actually entered some text.
    3.  Use `st.spinner()` to provide feedback to the user that something is happening in the background.
    4.  Call the `placeholder_extraction_function()` and assign its return value to `st.session_state.processed_data`.

    ```python
    # --- 3. LOGIC: HANDLE INITIAL PROCESSING ---
    # This block executes only when the 'Process' button is clicked.

    if process_button and job_listing_text:
        # Provide visual feedback during processing
        with st.spinner("Analyzing job description..."):
            # Call the (placeholder) function to extract information
            extracted_info = placeholder_extraction_function(job_listing_text)

        # IMPORTANT: Store the extracted dictionary in the session state.
        # This makes the data available for the next step in the workflow,
        # even after Streamlit re-runs the script.
        st.session_state.processed_data = extracted_info

        # Optional: Force a script rerun to immediately display the confirmation form.
        # st.experimental_rerun() is now st.rerun() in newer versions.
        st.rerun()
    ```

---

#### **Step 2.4: Implement the Conditional "Edit and Confirm" Form**

*   **Goal:** To display the second part of the workflow *only if* data has been processed and stored in the session state. We will use an `st.form` to group the editable fields and the final submission button together.

*   **Instructions & Pseudocode:**
    1.  Create a main `if` block that checks if `st.session_state.processed_data` is not `None`. The entire form will live inside this block.
    2.  Inside, use `st.form()` to create the form object.
    3.  Within the `with st.form(...)` block, create a series of `st.text_input` and `st.text_area` widgets.
    4.  Crucially, set the `value` of each widget to the corresponding data from `st.session_state.processed_data`. This pre-fills the form with the extracted information.
    5.  At the end of the form block, create the final submission button using `st.form_submit_button()`.

    ```python
    # --- 4. UI - STEP 2: VERIFICATION FORM ---
    # This entire section is conditional. It will only appear on screen
    # if `st.session_state.processed_data` contains data.

    if st.session_state.processed_data:
        st.header("Step 2: Verify Extracted Information")
        st.warning("Please review and correct any extracted details below before saving.")

        # `st.form` groups multiple widgets. Their values are sent only when
        # the `st.form_submit_button` inside the form is clicked.
        with st.form(key="confirmation_form"):
            data = st.session_state.processed_data

            # Create editable fields, pre-filled with the extracted data
            edited_title = st.text_input("Job Title", value=data.get('job_title'))
            edited_company = st.text_input("Company", value=data.get('company'))
            edited_location = st.text_input("Location", value=data.get('location'))
            edited_url = st.text_input("Apply URL", value=data.get('apply_url'))
            
            # The final submission button for the form
            confirm_button = st.form_submit_button("Confirm and Save to Database")

            # --- 5. LOGIC: HANDLE FINAL SUBMISSION ---
            if confirm_button:
                # In Tasks 3 & 4, this is where you'll call the real deduplication
                # and database insertion functions using the 'edited_' variables.
                
                # For now, just show a success message as a placeholder.
                st.success("Confirmed! (This will save to the DB in a future step).")

                # CRITICAL: Reset the session state to None to hide the form
                # and return the app to its initial state for the next job listing.
                st.session_state.processed_data = None
                
                # Clear the initial text area as well for a clean slate.
                st.session_state.job_text_input = ""
                st.rerun() # Rerun the script to reflect the changes immediately.
    ```

---

### **Verification and Definition of Done for Task 2**

Task 2 is complete when:

1.  The app loads showing only the title, instructions, and the main text area/button.
2.  Pasting text and clicking "Process Job Listing" makes the "Verify Extracted Information" form appear below.
3.  The form fields are correctly pre-populated with the (placeholder) extracted data.
4.  The user can edit the text within these fields.
5.  Clicking "Confirm and Save to Database" displays a success message, and then the entire form disappears, returning the app to its initial clean state, ready for the next job listing.

---

Excellent. Let's dive deep into **Task 3: Information Extraction Service**.

This task is the intellectual core of your Phase 1. It's where raw, messy text is transformed into clean, structured data. The hybrid "fill-in-the-gaps" approach is crucial for creating a system that is both efficient and robust. We'll design this as a self-contained service in `processing.py` that the Streamlit app can use as a black box.

---

### **Task 3: Information Extraction Service — Step-by-Step Implementation**

**Objective:** Develop a modular service that takes raw text as input and returns a structured `JobListing` object, progressively populating its fields using a tiered extraction strategy (Regex, spaCy, LLM).

---

#### **Step 3.1: Create `processing.py` and Define the Data Structure**

*   **Goal:** To establish a dedicated file for all backend logic and define a clear "contract" for what a structured job listing looks like using a Python `dataclass`. This is far superior to passing raw dictionaries around as it provides type hinting, auto-completion, and a single source of truth for your data's structure.

*   **Instructions & Pseudocode:**
    1.  Create a new file in your project's root directory named `processing.py`.
    2.  At the top of this file, import the necessary libraries. We'll use the `dataclasses` module and `typing` for type hints.
    3.  Define a `dataclass` named `JobListing`. The fields should mirror the columns in your database table. Using `Optional[str]` and `default=None` is key, as it allows us to create an instance that is initially empty and fill it in as we go.

    ```python
    # In processing.py
    import re
    import spacy
    from dataclasses import dataclass, field
    from typing import Optional, List

    # --- 1. DATA STRUCTURE DEFINITION ---
    # This dataclass acts as a structured container for our data.
    # It ensures consistency throughout the extraction pipeline.

    @dataclass
    class JobListing:
        job_title: Optional[str] = None
        company: Optional[str] = None
        location: Optional[str] = None
        apply_url: Optional[str] = None
        # The full description is not optional; it's the source of truth.
        description: str = ""
        # We can add other fields as needed later.
    ```

---

#### **Step 3.2: Implement the Main Orchestrator Function (`extract_job_details`)**

*   **Goal:** To create the single public function that `app.py` will call. This function acts as a manager, coordinating the calls to the different extraction layers in the correct order.

*   **Instructions & Pseudocode:**
    1.  In `processing.py`, define the main function `extract_job_details`.
    2.  It takes the raw text as input and is type-hinted to return a `JobListing` object.
    3.  Inside, it first initializes an empty `JobListing` object, passing the raw text to the `description` field.
    4.  It then calls each private helper function sequentially, passing the `JobListing` object through each one. The object gets progressively enriched at each step.

    ```python
    # --- 2. MAIN ORCHESTRATOR ---
    # This is the only function that app.py will need to import and call.

    def extract_job_details(raw_text: str) -> JobListing:
        """
        Orchestrates the extraction process using a tiered approach.
        """
        # Initialize the data container with the raw description.
        job = JobListing(description=raw_text)

        # 1. First pass: Use fast and cheap Regex for well-structured data.
        job = _extract_with_regex(raw_text, job)

        # 2. Second pass: Use NLP (spaCy) to find entities if fields are still missing.
        job = _extract_with_spacy(raw_text, job)

        # 3. Final fallback: Use an LLM for very difficult cases (optional, stubbed for now).
        # job = _extract_with_llm(raw_text, job)

        return job
    ```

---

#### **Step 3.3: Implement the Regex Extraction Layer (`_extract_with_regex`)**

*   **Goal:** To perform a fast, first-pass extraction targeting explicitly labeled information (e.g., "Company: Acme Corp"). This is highly effective for job postings copied from professional sites.

*   **Instructions & Pseudocode:**
    1.  Define a private helper function `_extract_with_regex` that takes the text and the current `JobListing` object.
    2.  Create a dictionary where keys are the field names in your `JobListing` and values are the compiled regex patterns. This is a clean and maintainable way to manage your patterns.
    3.  Iterate through this dictionary. For each field, **first check if the field in the `job` object is already filled**. This is the core of the "fill-in-the-gaps" logic.
    4.  If the field is empty, run the regex search. If a match is found, clean the result (e.g., `.strip()`) and update the `job` object.

    ```python
    # --- 3. EXTRACTION LAYER 1: REGULAR EXPRESSIONS ---

    # Define patterns in a structured way.
    REGEX_PATTERNS = {
        'job_title': re.compile(r"Job Title:\s*(.*)", re.IGNORECASE),
        'company': re.compile(r"Company:\s*(.*)", re.IGNORECASE),
        'location': re.compile(r"Location:\s*(.*)", re.IGNORECASE),
        'apply_url': re.compile(r"(https?://\S+apply\S*)", re.IGNORECASE)
        # Add more patterns as you discover them.
    }

    def _extract_with_regex(raw_text: str, job: JobListing) -> JobListing:
        """
        Fills JobListing fields using predefined regex patterns.
        """
        for field, pattern in REGEX_PATTERNS.items():
            # **FILL-IN-THE-GAPS LOGIC**: Only search if the field is not already populated.
            if getattr(job, field) is None:
                match = pattern.search(raw_text)
                if match:
                    # Update the dataclass field with the captured group.
                    setattr(job, field, match.group(1).strip())
        return job
    ```

---

#### **Step 3.4: Implement the spaCy NER Layer (`_extract_with_spacy`)**

*   **Goal:** To use a pre-trained Natural Language Processing model to identify entities like organizations (`ORG`) and locations (`GPE`) when the text is less structured and lacks explicit labels.

*   **Instructions & Pseudocode:**
    1.  At the top of `processing.py`, load the spaCy model. This is a heavy object, so you should load it only once when the module is imported to avoid slow performance.
    2.  Define the private helper `_extract_with_spacy`.
    3.  Just like with regex, **first check if the relevant fields (`company`, `location`) are empty**.
    4.  If they are, process the text with the `nlp` object to create a `doc`.
    5.  Iterate through the named entities (`doc.ents`). If you find an `ORG` entity and `job.company` is still `None`, assign it. Do the same for `GPE` and `job.location`.
    6.  For simplicity, you can just take the first entity of each type that you find.

    ```python
    # --- 4. EXTRACTION LAYER 2: SPACY NER ---

    # Load the model once at the module level for efficiency.
    try:
        NLP_MODEL = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
        NLP_MODEL = None

    def _extract_with_spacy(raw_text: str, job: JobListing) -> JobListing:
        """
        Uses spaCy's Named Entity Recognition (NER) to find missing details.
        """
        if NLP_MODEL is None:
            return job # Don't proceed if the model failed to load.
            
        # Only process the text if there's something to look for.
        if job.company is None or job.location is None:
            doc = NLP_MODEL(raw_text)
            for ent in doc.ents:
                # If company is missing and we find an Organization...
                if job.company is None and ent.label_ == "ORG":
                    job.company = ent.text.strip()
                # If location is missing and we find a Geo-Political Entity...
                if job.location is None and ent.label_ == "GPE":
                    job.location = ent.text.strip()

        return job
    ```

---

### **Verification and Definition of Done for Task 3**

Task 3 completion when:

1.  The `processing.py` file contains the `JobListing` dataclass and the `extract_job_details` function.
2.  Calling `extract_job_details` with a well-formatted job description (containing labels like "Company:") returns a `JobListing` object correctly populated by the regex layer.
3.  Calling it with a poorly-formatted job description (e.g., a simple paragraph) returns an object correctly populated by the spaCy layer.
4.  **Unit Tests are Written:**
    *   You have a separate test file (e.g., `test_processing.py`).
    *   One test asserts that a sample text with "Company: Test Inc." correctly extracts "Test Inc." via regex.
    *   Another test asserts that a sample text like "We are FutureTech, a leading innovator..." correctly extracts "FutureTech" via spaCy when the regex fails.
5.  Your `app.py` is updated to import and call `extract_job_details` instead of the placeholder function. The "Edit and Confirm" form now populates with real extracted data.

---
Of course. Let's break down **Task 4: Deduplication and Data Persistence** into a detailed implementation guide.

This task is about connecting your application's logic to its memory—the database. We will focus on two critical principles: **safety** (preventing bad data and duplicates) and **modularity** (keeping database code separate from application logic). Creating a dedicated `db_utils.py` module is a professional standard that makes your code cleaner, easier to test, and more maintainable.

---

### **Task 4: Deduplication and Data Persistence — Step-by-Step Implementation**

**Objective:** To create a robust data persistence layer that first hashes and checks for duplicate job listings before safely inserting new, unique records into the PostgreSQL database.

---

#### **Step 4.1: Implement the Hashing Function in `processing.py`**

*   **Goal:** To create a reliable, deterministic function that converts the core job description text into a unique signature (a hash) for exact duplicate detection.

*   **Instructions & Pseudocode:**
    1.  Open the `processing.py` file you created in Task 3.
    2.  Import the `hashlib` library.
    3.  Create a new public function `hash_description`.
    4.  **Normalization is key:** Inside the function, before hashing, you must normalize the text. A good starting point is to convert it to lowercase and remove all whitespace. This ensures that minor formatting changes don't result in a different hash.
    5.  Hash functions operate on bytes, so encode the normalized string to `utf-8`.
    6.  Use the SHA-256 algorithm to generate the hash and return its hexadecimal representation.

    ```python
    # In processing.py
    import hashlib
    # ... other imports from Task 3 ...

    # --- HASHING UTILITY ---
    # This function creates a unique, consistent fingerprint for a job description.

    def hash_description(text: str) -> str:
        """
        Normalizes and hashes a string using SHA-256 to create a unique identifier.
        """
        # 1. Normalize: Lowercase and remove all whitespace characters.
        normalized_text = "".join(text.lower().split())

        # 2. Encode: Convert the string to bytes, as required by hashlib.
        encoded_text = normalized_text.encode('utf-8')

        # 3. Hash: Compute the SHA-256 hash.
        hasher = hashlib.sha256(encoded_text)

        # 4. Return: Get the hexadecimal string representation of the hash.
        return hasher.hexdigest()
    ```

---

#### **Step 4.2: Create the Database Utility Module (`db_utils.py`)**

*   **Goal:** To create a centralized, self-contained module for all database interactions. This module will be responsible for handling connections, credentials, and executing queries, completely abstracting these details from `app.py`.

*   **Instructions & Pseudocode:**
    1.  Create a new file in your project's root directory named `db_utils.py`.
    2.  Import `psycopg2`, `os`, and `dotenv`.
    3.  Create a private helper function, `_get_db_connection`, that reads database credentials from your `.env` file and establishes a connection. This centralizes the connection logic.
    4.  Your `.env` file should look like this:
        ```env
        DB_NAME="your_db"
        DB_USER="your_user"
        DB_PASSWORD="your_password"
        DB_HOST="localhost"
        DB_PORT="5432"
        ```

    ```python
    # In db_utils.py
    import os
    import psycopg2
    from dotenv import load_dotenv
    from typing import Optional

    # Import the JobListing dataclass to use for type hinting.
    # Note: You might need to adjust the import path depending on your structure.
    from processing import JobListing

    # --- 1. SETUP AND CONNECTION ---
    load_dotenv() # Load variables from the .env file

    def _get_db_connection():
        """
        Establishes and returns a connection to the PostgreSQL database.
        Returns None if connection fails.
        """
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
            )
            return conn
        except psycopg2.OperationalError as e:
            # In a real app, you would log this error.
            print(f"Error: Could not connect to the database. {e}")
            return None
    ```

---

#### **Step 4.3: Implement the Duplicate Check and Insertion Functions in `db_utils.py`**

*   **Goal:** To create the two main public functions of this module: one to check if a hash exists and one to insert a new record. We will use best practices like context managers (`with`) and parameterized queries to prevent SQL injection.

*   **Instructions & Pseudocode:**
    1.  In `db_utils.py`, define the `check_for_duplicate` function. It takes the hash and returns a boolean.
    2.  Use `_get_db_connection()` to get a connection.
    3.  Use `with conn.cursor() as cur:` to ensure the cursor is properly closed.
    4.  Execute a `SELECT 1 ...` query, which is more efficient than `SELECT *`.
    5.  **Crucially, use placeholders (`%s`) for the hash value.** This lets the database driver safely handle the input, preventing SQL injection attacks.
    6.  Define the `insert_job_listing` function. It should accept the `JobListing` object from Task 3 and the generated hash.
    7.  Follow a similar pattern, but use an `INSERT` statement. The order of columns in your `INSERT` statement *must* match the order of values in the tuple you provide to `cur.execute()`.
    8.  After executing the `INSERT`, you must call `conn.commit()` to save the transaction to the database.

    ```python
    # In db_utils.py (continued)

    # --- 2. PUBLIC DATABASE FUNCTIONS ---

    def check_for_duplicate(description_hash: str) -> bool:
        """
        Checks if a job listing with the given hash already exists.
        """
        conn = _get_db_connection()
        if not conn: return False # Or raise an exception

        is_duplicate = False
        try:
            # Using a context manager ensures the cursor and connection are closed.
            with conn.cursor() as cur:
                # Use parameterized query to prevent SQL injection.
                sql = "SELECT 1 FROM job_listings WHERE description_hash = %s;"
                cur.execute(sql, (description_hash,))
                # fetchone() returns a tuple if a row is found, otherwise None.
                is_duplicate = cur.fetchone() is not None
        finally:
            conn.close()
        return is_duplicate

    def insert_job_listing(job: JobListing, job_hash: str):
        """
        Inserts a new job listing record into the database.
        """
        conn = _get_db_connection()
        if not conn: return

        try:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO job_listings (job_title, company, location, apply_url, description, description_hash)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
                # The tuple of values MUST match the order of columns in the SQL statement.
                values = (job.job_title, job.company, job.location, job.apply_url, job.description, job_hash)
                cur.execute(sql, values)
                # Commit the transaction to make the changes permanent.
                conn.commit()
        except Exception as e:
            # If anything goes wrong, roll back the transaction.
            print(f"Database insert failed: {e}")
            conn.rollback()
        finally:
            conn.close()
    ```

---

#### **Step 4.4: Integrate Persistence Logic into the Streamlit App (`app.py`)**

*   **Goal:** To call our new hashing and database functions from the Streamlit UI when the user clicks the "Confirm and Save" button.

*   **Instructions & Pseudocode:**
    1.  Open `app.py`. Import the necessary functions: `hash_description` from `processing` and the functions from `db_utils`.
    2.  Navigate to the section for the "Edit and Confirm" form, inside the `if confirm_button:` block.
    3.  First, gather the potentially edited data from the form fields.
    4.  Use this data to create a final `JobListing` object.
    5.  Generate the hash using the *original, full description text* to ensure consistency.
    6.  Call `db_utils.check_for_duplicate()`.
    7.  Based on the boolean result, either call `db_utils.insert_job_listing()` and show a success message, or show a warning message for duplicates.

    ```python
    # In app.py
    # ... inside the "if st.session_state.processed_data:" block ...
    # ... inside the "with st.form(...)" block ...

    # from processing import hash_description, JobListing
    # import db_utils

    if confirm_button:
        # 1. Gather the final, user-verified data from the form fields.
        final_job_data = JobListing(
            job_title=edited_title,
            company=edited_company,
            location=edited_location,
            apply_url=edited_url,
            # Use the original full description from the session state.
            description=st.session_state.processed_data['description']
        )

        # 2. Generate the hash from the full description.
        description_hash = hash_description(final_job_data.description)

        # 3. Check for duplicates before attempting to insert.
        if db_utils.check_for_duplicate(description_hash):
            st.warning("⚠️ This job listing is already in the database.")
        else:
            # 4. If not a duplicate, insert the data.
            try:
                db_utils.insert_job_listing(final_job_data, description_hash)
                st.success("✅ Job listing saved successfully!")
            except Exception as e:
                st.error(f"An error occurred while saving: {e}")

        # 5. Reset the state to return to the initial UI.
        st.session_state.processed_data = None
        st.session_state.job_text_input = ""
        st.rerun()
    ```

---

### **Verification and Definition of Done for Task 4**

You have successfully completed Task 4 when:

1.  The `check_for_duplicate` function correctly queries the database and returns `True` for a hash that exists and `False` for one that does not.
2.  The `insert_job_listing` function successfully writes a new row to the `job_listings` table, and the data can be verified using a SQL client.
3.  In the Streamlit app, clicking "Confirm and Save" for a new job results in a success message and a new database entry.
4.  Submitting the exact same job description a second time results in the duplicate warning message and does *not* create a new database entry.

---

**Task 5: Testing Strategy**

*   **Unit Tests:** Use a framework like `pytest` to test individual functions in isolation.
    *   Test the `hash_description` function with known inputs.
    *   Test the regex patterns with sample text snippets.
    *   Test the database connection utility.
*   **Integration Tests:** Test the interaction between components.
    *   Test the full `extract_job_details` function to ensure the hybrid logic works.
    *   Test the flow from the "Confirm and Save" button click to a successful database write (can be done with a temporary test database).
*   **Manual User Acceptance Testing (UAT):**
    *   Create a simple checklist.
    *   **Test Case 1:** Copy-paste a job from LinkedIn. Verify data is extracted correctly. Save. Verify success message and database entry.
    *   **Test Case 2:** Paste the same job again. Verify the duplicate warning is shown and no new entry is created.

