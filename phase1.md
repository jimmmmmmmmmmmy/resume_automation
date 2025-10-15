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

This is the core of the plan. Each task should have a clear `Definition of Done`.

---

**Task 1: Foundational Setup**

*   **1.1. Implementation Steps:**
    *   Initialize a Git repository.
    *   Create the project directory structure (`/resume_optimizer`, `/scripts`, `app.py`, etc.).
    *   Set up and activate a Python virtual environment (`venv`).
    *   Create a `requirements.txt` file and add initial dependencies: `streamlit`, `psycopg2-binary`, `spacy`, `python-dotenv`, `sentence-transformers`.
    *   Create a `.gitignore` file to exclude `venv`, `.env`, `__pycache__`, etc.
    *   Write a SQL initialization script (`/scripts/init_db.sql`) that creates the `job_listings` table with the enhanced schema (including `source_platform`, `metadata` JSONB, etc.).
*   **1.2. Definition of Done:**
    *   The virtual environment is active and all dependencies can be installed with `pip install -r requirements.txt`.
    *   The `init_db.sql` script can be run successfully against a local PostgreSQL instance to create the required table.

---

**Task 2: Streamlit UI and User Workflow**

*   **2.1. Implementation Steps:**
    *   In `app.py`, create the main title and instructional text.
    *   Use `st.text_area` for the job description input.
    *   Create an initial "Process Job Listing" button.
    *   **Implement the "Edit and Confirm" workflow:**
        1.  After the initial button press, the extraction logic is called.
        2.  The results are displayed in editable `st.text_input` fields for `job_title`, `company`, `location`, etc.
        3.  A "Confirm and Save" button is displayed, which triggers the deduplication and database insertion logic.
*   **2.2. Definition of Done:**
    *   The UI renders correctly.
    *   A user can paste text, click a button, and see the extracted (but not yet saved) data in editable fields.
    *   The "Confirm and Save" button is present but not yet functional.
*   **2.3. Potential Risks:**
    *   **Risk:** Streamlit's state management can be tricky. A page refresh might clear the extracted data before the user can confirm it.
    *   **Mitigation:** Use Streamlit's `st.session_state` to hold the extracted data between button clicks.

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

