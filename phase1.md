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

**Task 3: Information Extraction Service**

*   **3.1. Implementation Steps:**
    *   Create a new file, `processing.py`.
    *   Define a Pydantic model or a simple Python `dataclass` to represent a structured `JobListing`.
    *   Implement the hybrid "fill-in-the-gaps" extraction function `extract_job_details(raw_text)` which:
        1.  Initializes an empty `JobListing` object.
        2.  Calls an internal `_extract_with_regex()` function to populate fields like URL, title, etc.
        3.  For any remaining empty fields, calls an internal `_extract_with_spacy()` function.
        4.  (Optional fallback) If key fields are still missing, calls an `_extract_with_llm()` function.
    *   Each extraction function will return the values it finds, never overwriting an existing value.
*   **3.2. Definition of Done:**
    *   The `extract_job_details` function can be called with raw text and reliably returns a structured `JobListing` object.
    *   **Unit tests are written** for the regex and spaCy extraction helpers to validate their accuracy on sample job descriptions.
*   **3.3. Potential Risks:**
    *   **Risk:** Regex patterns are too brittle.
    *   **Mitigation:** Rely more heavily on the spaCy NER as the primary method if regex proves unreliable during testing.

---

**Task 4: Deduplication and Data Persistence**

*   **4.1. Implementation Steps:**
    *   In `processing.py`, create a `hash_description(text)` function using `hashlib`.
    *   Create a database utility module (e.g., `db_utils.py`) to handle all database connections and queries. This avoids putting SQL in your main logic.
    *   Implement a `check_for_duplicate(description_hash)` function that queries the database.
    *   Implement an `insert_job_listing(job_listing_object)` function that inserts the validated data into the `job_listings` table.
*   **4.2. Definition of Done:**
    *   The `check_for_duplicate` function correctly returns `True` if a hash exists and `False` otherwise.
    *   The `insert_job_listing` function successfully saves a new record to the database.
    *   The Streamlit app provides clear feedback (`st.success` for new records, `st.warning` for duplicates).

**4. Testing Strategy**

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

