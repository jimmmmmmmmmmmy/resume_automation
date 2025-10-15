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

*   **Concepts Introduced:**
    *   **Version Control (Git):** The practice of tracking and managing changes to software code. `git init` creates a new repository.
    *   **Dependency Management (`venv`, `pip`, `requirements.txt`):** Creating isolated environments (`venv`) to manage project-specific libraries, installed via `pip`, and locking their versions in a `requirements.txt` file for reproducibility.
    *   **Environment Configuration (`.env`, `.gitignore`):** Separating sensitive information (like database passwords) into a local `.env` file and telling Git to ignore it with `.gitignore` to prevent committing secrets to version control.
    *   **Database Schema Design (SQL DDL):** Using Data Definition Language (DDL) in SQL to define the structure of your database tables, columns, and data types.

*   **Things to Research Further:**
    *   "Git branching strategies" (like GitFlow) to understand how teams collaborate.
    *   "Python virtual environments explained" to grasp why they are crucial.
    *   "SQL data types" (e.g., `VARCHAR`, `TEXT`, `TIMESTAMPTZ`, `JSONB`) to understand the choices made in the schema.
    *   "What is an idempotent SQL script?" (The use of `DROP TABLE IF EXISTS` makes our script safely re-runnable).

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

### **Task 2: Streamlit UI and User Workflow — Step-by-Step Implementation**

**Objective:** To build a user interface that guides the user through pasting job text, processing it, and then verifying the extracted information before final submission.

*   **Concepts Introduced:**
    *   **Declarative UI Frameworks (Streamlit):** Building user interfaces by describing *what* should be displayed, and letting the framework handle *how* it gets rendered and updated.
    *   **State Management (`st.session_state`):** The mechanism for preserving information across user interactions. Since Streamlit re-runs the entire script on every action, `session_state` is essential for creating multi-step workflows.
    *   **Event-Driven Programming:** The application's flow is determined by user events like button clicks. The `if process_button:` block is a simple example of an event handler.
    *   **Conditional Rendering:** The practice of showing or hiding UI elements based on the current state of the application (e.g., the verification form only appears after data has been processed).

*   **Things to Research Further:**
    *   "How Streamlit works" to understand its script-rerun model.
    *   "Streamlit session state tutorial" for more advanced examples.
    *   "Declarative vs. Imperative UI programming" to appreciate the paradigm Streamlit uses.
    *   "Using Forms in Streamlit" to see how to group inputs to be submitted together.
      
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

### **Task 3: Information Extraction Service — Step-by-Step Implementation**

**Objective:** To develop a modular service that takes raw text as input and returns a structured `JobListing` object, progressively populating its fields using a tiered extraction strategy (Regex, spaCy). This module will import its core data structure from a central `models.py` file to ensure architectural integrity and prevent circular dependencies.

*   **Concepts Introduced:**
    *   **Separation of Concerns:** Creating a dedicated `processing.py` module for business logic, keeping it separate from the UI (`app.py`) and database (`db_utils.py`) code. This makes the system easier to maintain and test.
    *   **Data Modeling (`dataclasses`):** Using Python's `dataclass` to create a structured, type-hinted "schema" for your data. This is more robust and self-documenting than using plain dictionaries.
    *   **Regular Expressions (Regex):** A powerful tool for finding patterns in text. Ideal for fast, precise extraction when the text format is predictable.
    *   **Natural Language Processing (NLP) & Named Entity Recognition (NER):** Using a library like spaCy to understand the *meaning* of text and identify real-world entities like organizations (`ORG`) and locations (`GPE`).
    *   **Architectural Patterns (Orchestrator):** The `extract_job_details` function acts as an "orchestrator" or "manager," coordinating the work of smaller, specialized helper functions (`_extract_with_regex`, `_extract_with_spacy`).

*   **Things to Research Further:**
    *   "Python dataclasses vs dictionaries" to understand the benefits.
    *   "Regex tutorial for Python" (sites like regex101.com are excellent for interactive learning).
    *   "What is Named Entity Recognition (NER)?"
    *   "spaCy 101" to learn about its capabilities beyond NER (like tokenization and part-of-speech tagging).
    *   "Circular dependencies in Python" to understand why creating `models.py` is a good architectural decision.


---

#### **Step 3.1: Create the Central Data Model (`models.py`)**

*   **Goal:** To establish a central, independent file for shared data structures. This prevents circular dependencies by allowing multiple modules (`processing.py`, `db_utils.py`, etc.) to import from a common, neutral location.

*   **Instructions:**
    1.  In the root of your `resume_optimizer` project, create a new file named `models.py`.
    2.  Add the `JobListing` dataclass definition to this new file. This will be the single source of truth for the structure of your job data.

    ```python
    # In models.py
    from dataclasses import dataclass
    from typing import Optional

    """
    This file contains the core data structures (models) used across the application.
    """

    @dataclass
    class JobListing:
        """A structured representation of a job listing."""
        job_title: Optional[str] = None
        company: Optional[str] = None
        location: Optional[str] = None
        apply_url: Optional[str] = None
        description: str = ""
    ```

*   **Outcome:** You now have a `models.py` file. The project structure should be updated to include it:
    ```
    resume_optimizer/
    ├── ...
    ├── app.py
    ├── models.py  <-- NEW FILE
    └── requirements.txt
    ```

---

#### **Step 3.2: Create the Information Processing Service (`processing.py`)**

*   **Goal:** To establish the dedicated file for all backend extraction logic and to import the shared `JobListing` data model from `models.py`.

*   **Instructions & Pseudocode:**
    1.  Create a new file in your project's root directory named `processing.py`.
    2.  At the top of this file, import the necessary libraries (`re`, `spacy`, `hashlib`) and, crucially, import the `JobListing` dataclass from your new `models` module.

    ```python
    # In processing.py
    import re
    import spacy
    import hashlib
    from typing import Optional

    # --- 1. IMPORT SHARED DATA STRUCTURE ---
    # Import the JobListing model from the central models.py file.
    from models import JobListing
    ```

---

#### **Step 3.3: Implement the Main Orchestrator and Hashing Functions**

*   **Goal:** To create the single public function `extract_job_details` that `app.py` will call, and the `hash_description` utility that will be used in Task 4.

*   **Instructions & Pseudocode:**
    1.  In `processing.py`, define the `extract_job_details` function. It initializes a `JobListing` object and coordinates calls to the private extraction helpers.
    2.  Also define the `hash_description` function. Keeping it in `processing.py` is logical as it's a form of processing the description text.

    ```python
    # In processing.py (continued)

    # --- 2. MAIN ORCHESTRATOR & UTILITIES ---

    def extract_job_details(raw_text: str) -> JobListing:
        """
        Orchestrates the extraction process using a tiered approach.
        """
        job = JobListing(description=raw_text)
        job = _extract_with_regex(raw_text, job)
        job = _extract_with_spacy(raw_text, job)
        return job

    def hash_description(text: str) -> str:
        """
        Normalizes and hashes a string using SHA-26 to create a unique identifier.
        """
        normalized_text = "".join(text.lower().split())
        encoded_text = normalized_text.encode('utf-8')
        return hashlib.sha256(encoded_text).hexdigest()
    ```

---

#### **Step 3.4: Implement the Regex and spaCy Extraction Layers**

*   **Goal:** To build the private helper functions that perform the actual extraction, each one filling in the gaps left by the previous layer.

*   **Instructions & Pseudocode:**
    1.  Define the `_extract_with_regex` function to find explicitly labeled information.
    2.  Define the `_extract_with_spacy` function to find entities in less structured text. Load the spaCy model once at the module level for efficiency.

    ```python
    # In processing.py (continued)

    # --- 3. EXTRACTION LAYERS (PRIVATE HELPERS) ---

    # Layer 1: Regex
    REGEX_PATTERNS = {
        'job_title': re.compile(r"Job Title:\s*(.*)", re.IGNORECASE),
        'company': re.compile(r"Company:\s*(.*)", re.IGNORECASE),
        'location': re.compile(r"Location:\s*(.*)", re.IGNORECASE),
        'apply_url': re.compile(r"(https?://\S+apply\S*|\S+careers\S+|\S+jobs\S+)", re.IGNORECASE)
    }

    def _extract_with_regex(raw_text: str, job: JobListing) -> JobListing:
        for field, pattern in REGEX_PATTERNS.items():
            if getattr(job, field) is None:
                match = pattern.search(raw_text)
                if match:
                    setattr(job, field, match.group(1).strip())
        return job

    # Layer 2: spaCy NER
    try:
        NLP_MODEL = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model not found. Run: 'python -m spacy download en_core_web_sm'")
        NLP_MODEL = None

    def _extract_with_spacy(raw_text: str, job: JobListing) -> JobListing:
        if NLP_MODEL and (job.company is None or job.location is None):
            doc = NLP_MODEL(raw_text)
            for ent in doc.ents:
                if job.company is None and ent.label_ == "ORG":
                    job.company = ent.text.strip()
                if job.location is None and ent.label_ == "GPE":
                    job.location = ent.text.strip()
        return job
    ```

---

### **Verification and Definition of Done for Task 3**

Task 3 is complete when:

1.  A `models.py` file exists and contains the `JobListing` dataclass.
2.  The `processing.py` file exists and successfully imports `JobListing` from `models.py`.
3.  Calling `extract_job_details` with well-formatted text returns a `JobListing` object populated by the regex layer.
4.  Calling `extract_job_details` with less-structured text returns an object populated by the spaCy layer.
5.  **Unit Tests are Written** for the extraction helpers in a separate `test_processing.py` file to validate their accuracy on various sample job descriptions.
6.  The `app.py` file is updated to import `extract_job_details` from `processing.py` (instead of the placeholder) and successfully populates the "Edit and Confirm" form with real extracted data.

---

### **Task 4: Deduplication and Data Persistence — Step-by-Step Implementation**

**Objective:** To create a robust, modular data persistence layer in a dedicated `db_utils.py` file. This service will connect to the PostgreSQL database, check for duplicate entries using the hash generated by `processing.py`, and safely insert new, unique records.

*   **Concepts Introduced:**
    *   **Database Abstraction Layer:** Creating `db_utils.py` acts as a simple abstraction layer. `app.py` doesn't need to know *how* to connect to or query the database; it just calls functions like `insert_job_listing()`.
    *   **Cryptographic Hashing (SHA-256):** A one-way function that turns an input (the job description) into a unique, fixed-size string (the hash). It's used here to create a reliable "fingerprint" for deduplication.
    *   **Data Normalization:** The process of cleaning and standardizing data *before* processing it (e.g., converting text to lowercase and removing whitespace before hashing) to ensure consistency.
    *   **SQL Injection Prevention:** Using parameterized queries (`cur.execute(sql, values)`) is the most critical security practice for database programming. It treats user data as data, never as executable code.
    *   **Database Transactions (Commit/Rollback):** An `INSERT` operation is not permanent until `conn.commit()` is called. If an error occurs, `conn.rollback()` can undo the changes, ensuring the database remains in a consistent state.

*   **Things to Research Further:**
    *   "What is SQL Injection and how to prevent it?"
    *   "SHA-256 Hashing explained."
    *   "Python `psycopg2` tutorial" for more advanced usage.
    *   "Database Transactions and ACID properties (Atomicity, Consistency, Isolation, Durability)."
    *   
---

#### **Step 4.1: Create the Database Utility Module (`db_utils.py`)**

*   **Goal:** To create a centralized module for all database interactions, abstracting connection and query logic away from the main application. This module will import the shared `JobListing` model from `models.py`.

*   **Instructions & Pseudocode:**
    1.  Create a new file in your project's root directory named `db_utils.py`.
    2.  Import necessary libraries (`os`, `psycopg2`, `dotenv`) and, most importantly, import the `JobListing` dataclass from your central `models.py` file.
    3.  Create a private helper function, `_get_db_connection`, to handle reading credentials from the `.env` file and establishing the database connection.

    ```python
    # In db_utils.py
    import os
    import psycopg2
    from dotenv import load_dotenv

    # --- 1. IMPORT SHARED DATA STRUCTURE & SETUP ---
    # Import the JobListing model from the central models.py file.
    from models import JobListing

    load_dotenv() # Load environment variables from .env file

    def _get_db_connection():
        """Establishes and returns a connection to the PostgreSQL database."""
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
            print(f"Error: Could not connect to the database. {e}")
            return None
    ```

---

#### **Step 4.2: Implement Database Query Functions**

*   **Goal:** To build the two core public functions of the database module: one to check for duplicates and one to insert new data. These functions will use best practices like context managers and parameterized queries to ensure safety and reliability.

*   **Instructions & Pseudocode:**
    1.  In `db_utils.py`, define the `check_for_duplicate` function, which takes a hash string and returns a boolean.
    2.  Implement the `insert_job_listing` function, which takes a `JobListing` object and its corresponding hash.
    3.  Use `with conn.cursor() as cur:` to ensure resources are managed correctly.
    4.  **Use parameterized queries (e.g., `cur.execute(sql, (value,))`)** to prevent any risk of SQL injection.
    5.  Ensure you call `conn.commit()` after an insert and `conn.close()` in a `finally` block or by using the `with` statement for the connection itself.

    ```python
    # In db_utils.py (continued)

    # --- 2. PUBLIC DATABASE FUNCTIONS ---

    def check_for_duplicate(description_hash: str) -> bool:
        """Checks if a job listing with the given hash already exists."""
        conn = _get_db_connection()
        if not conn:
            return False

        is_duplicate = False
        try:
            with conn.cursor() as cur:
                sql = "SELECT 1 FROM job_listings WHERE description_hash = %s;"
                cur.execute(sql, (description_hash,))
                is_duplicate = cur.fetchone() is not None
        finally:
            conn.close()
        return is_duplicate

    def insert_job_listing(job: JobListing, job_hash: str):
        """Inserts a new job listing record into the database."""
        conn = _get_db_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO job_listings (job_title, company, location, apply_url, description, description_hash)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
                values = (job.job_title, job.company, job.location, job.apply_url, job.description, job_hash)
                cur.execute(sql, values)
                conn.commit()
        except Exception as e:
            print(f"Database insert failed: {e}")
            conn.rollback()
        finally:
            conn.close()
    ```

---

#### **Step 4.3: Integrate Persistence Logic into the Streamlit App (`app.py`)**

*   **Goal:** To connect the frontend workflow to the backend database logic. This involves calling the hashing and database functions when the user confirms the extracted data.

*   **Instructions & Pseudocode:**
    1.  Open `app.py`.
    2.  Import the necessary functions: `hash_description` from `processing`, the `JobListing` model from `models`, and your new database utilities from `db_utils`.
    3.  In the `if confirm_button:` block:
        a.  Re-assemble the final, user-verified data into a `JobListing` object.
        b.  Call `hash_description` using the original, full description text.
        c.  Call `db_utils.check_for_duplicate()`.
        d.  Conditionally call `db_utils.insert_job_listing()` and display the appropriate user feedback (`st.success` or `st.warning`).
        e.  Reset the session state to prepare for the next submission.

    ```python
    # In app.py
    # ... (other imports)
    from models import JobListing
    from processing import hash_description
    import db_utils

    # ... (inside the st.form block)
    if confirm_button:
        # 1. Gather final data from the form into a JobListing object.
        final_job_data = JobListing(
            job_title=edited_title,
            company=edited_company,
            location=edited_location,
            apply_url=edited_url,
            description=st.session_state.processed_data.description # Use original description
        )

        # 2. Generate the unique hash for the description.
        description_hash = hash_description(final_job_data.description)

        # 3. Check for duplicates and persist the data if unique.
        if db_utils.check_for_duplicate(description_hash):
            st.warning("This job listing is already in the database.")
        else:
            try:
                db_utils.insert_job_listing(final_job_data, description_hash)
                st.success("Job listing saved successfully!")
            except Exception as e:
                st.error(f"An error occurred while saving: {e}")

        # 4. Reset the application state for the next entry.
        st.session_state.processed_data = None
        st.session_state.job_text_input = ""
        st.rerun()
    ```

---

### **Verification and Definition of Done for Task 4**

Task 4 is complete when:

1.  A `db_utils.py` file exists and correctly imports the `JobListing` model from `models.py`.
2.  The `check_for_duplicate` function accurately returns `True` for an existing hash and `False` for a new one when tested.
3.  The `insert_job_listing` function successfully writes a new, complete row to the `job_listings` table, which can be verified in a SQL client.
4.  From the Streamlit UI, submitting a new job listing results in a success message and a new database entry.
5.  Submitting the same job listing a second time results in a duplicate warning message and **does not** create a second database entry.
   
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

