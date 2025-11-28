#### **Phase 2: Resume Analysis and AI-Powered Optimization**

**1. Overview & Objectives**

*   **1.1. Vision:** To extend the job ingestion pipeline with resume parsing capabilities and AI-powered analysis that compares a user's resume against stored job listings, providing actionable optimization suggestions.
*   **1.2. Key Objectives for Phase 2:**
    *   Enable users to upload and parse their resume (PDF/DOCX).
    *   Extract structured information from resumes (skills, experience, education).
    *   Implement AI-powered comparison between resume and job descriptions.
    *   Generate tailored optimization suggestions and keyword recommendations.
    *   Store resume data and analysis results for future reference.

**2. Prerequisites**

Before starting Phase 2, ensure the following from Phase 1 are complete:
*   PostgreSQL database is running with `job_listings` table (updated schema with `source_url`).
*   The Streamlit app successfully ingests and stores job listings.
*   All Phase 1 tests pass (`pytest test_processing.py`).

---

### **Task 1: Resume Upload and Parsing**

**Objective:** Enable users to upload resume files and extract raw text content.

---

#### **TODO 1.1: Add File Upload to Streamlit UI**

- [ ] Add a new page or tab in `app.py` for "Resume Analysis"
- [ ] Implement `st.file_uploader()` accepting PDF and DOCX formats
- [ ] Add file size validation (recommend max 5MB)
- [ ] Display upload success/error feedback to user
- [ ] Store uploaded file in `st.session_state` for processing

**Files to modify:** `app.py`

**New dependencies:** None (Streamlit has built-in file upload)

---

#### **TODO 1.2: Implement PDF Text Extraction**

- [ ] Create new file `resume_parser.py` for resume processing logic
- [ ] Install and import `PyPDF2` or `pdfplumber` library
- [ ] Implement `extract_text_from_pdf(file_bytes) -> str` function
- [ ] Handle multi-page PDFs
- [ ] Handle encrypted/protected PDFs gracefully (return error message)
- [ ] Add fallback for PDFs with embedded images (OCR placeholder for future)

**Files to create:** `resume_parser.py`

**New dependencies:** Add to `requirements.txt`:
```
PyPDF2>=3.0.0
# OR
pdfplumber>=0.9.0
```

---

#### **TODO 1.3: Implement DOCX Text Extraction**

- [ ] Install and import `python-docx` library
- [ ] Implement `extract_text_from_docx(file_bytes) -> str` function
- [ ] Preserve paragraph structure where possible
- [ ] Handle tables in DOCX files (common in resumes)
- [ ] Handle corrupted files gracefully

**Files to modify:** `resume_parser.py`

**New dependencies:** Add to `requirements.txt`:
```
python-docx>=0.8.11
```

---

#### **TODO 1.4: Create Unified Resume Extraction Interface**

- [ ] Implement `parse_resume(file_bytes, file_type: str) -> str` dispatcher function
- [ ] Auto-detect file type from extension/MIME type
- [ ] Return cleaned, normalized text ready for further processing
- [ ] Add logging for debugging extraction issues

**Files to modify:** `resume_parser.py`

---

### **Task 2: Resume Information Extraction**

**Objective:** Extract structured data from resume text using NLP and pattern matching.

---

#### **TODO 2.1: Define Resume Data Model**

- [ ] Add `Resume` dataclass to `models.py`:
  ```python
  @dataclass
  class Resume:
      full_name: Optional[str] = None
      email: Optional[str] = None
      phone: Optional[str] = None
      location: Optional[str] = None
      summary: Optional[str] = None
      skills: List[str] = field(default_factory=list)
      experience: List[Dict[str, Any]] = field(default_factory=list)
      education: List[Dict[str, Any]] = field(default_factory=list)
      raw_text: str = ""
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```
- [ ] Add `to_dict()` and `from_dict()` methods
- [ ] Document expected structure for `experience` and `education` dicts

**Files to modify:** `models.py`

---

#### **TODO 2.2: Implement Contact Information Extraction**

- [ ] Create regex patterns for email extraction
- [ ] Create regex patterns for phone number extraction (handle multiple formats)
- [ ] Create regex patterns for LinkedIn URL extraction
- [ ] Use spaCy NER for name extraction (PERSON entities)
- [ ] Use spaCy NER for location extraction (GPE entities)

**Files to modify:** `resume_parser.py`

---

#### **TODO 2.3: Implement Skills Extraction**

- [ ] Create/source a comprehensive skills taxonomy (tech skills, soft skills)
- [ ] Implement keyword matching against skills taxonomy
- [ ] Use spaCy for additional skill entity extraction
- [ ] Normalize skill names (e.g., "JS" -> "JavaScript", "ML" -> "Machine Learning")
- [ ] Deduplicate extracted skills
- [ ] Consider using a skills database/API (optional enhancement)

**Files to modify:** `resume_parser.py`

**Optional new file:** `skills_taxonomy.py` or `data/skills.json`

---

#### **TODO 2.4: Implement Experience Section Extraction**

- [ ] Identify experience section boundaries using header detection
- [ ] Extract company names using spaCy ORG entities
- [ ] Extract job titles using pattern matching
- [ ] Extract date ranges (start/end dates) using regex
- [ ] Extract bullet points/responsibilities as list items
- [ ] Structure each experience entry as a dictionary

**Files to modify:** `resume_parser.py`

---

#### **TODO 2.5: Implement Education Section Extraction**

- [ ] Identify education section boundaries
- [ ] Extract institution names using spaCy ORG entities
- [ ] Extract degree types (BS, MS, PhD, etc.) using pattern matching
- [ ] Extract field of study
- [ ] Extract graduation dates
- [ ] Extract GPA if present (optional)

**Files to modify:** `resume_parser.py`

---

#### **TODO 2.6: Create Main Resume Extraction Orchestrator**

- [ ] Implement `extract_resume_details(raw_text: str) -> Resume` function
- [ ] Call all extraction helpers in sequence
- [ ] Track extraction methods in metadata (similar to job extraction)
- [ ] Return populated `Resume` object

**Files to modify:** `resume_parser.py`

---

### **Task 3: Database Schema Updates**

**Objective:** Extend the database to store resume data and analysis results.

---

#### **TODO 3.1: Create Resume Table Schema**

- [ ] Create `scripts/migrations/001_add_resumes.sql`:
  ```sql
  CREATE TABLE resumes (
      id SERIAL PRIMARY KEY,
      full_name VARCHAR(255),
      email VARCHAR(255),
      phone VARCHAR(50),
      location VARCHAR(255),
      summary TEXT,
      skills JSONB,
      experience JSONB,
      education JSONB,
      raw_text TEXT NOT NULL,
      file_hash VARCHAR(64) UNIQUE NOT NULL,
      metadata JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  CREATE INDEX idx_resume_file_hash ON resumes(file_hash);
  CREATE INDEX idx_resume_skills ON resumes USING GIN(skills);
  ```

**Files to create:** `scripts/migrations/001_add_resumes.sql`

---

#### **TODO 3.2: Create Analysis Results Table Schema**

- [ ] Create `scripts/migrations/002_add_analysis_results.sql`:
  ```sql
  CREATE TABLE analysis_results (
      id SERIAL PRIMARY KEY,
      resume_id INTEGER REFERENCES resumes(id) ON DELETE CASCADE,
      job_listing_id INTEGER REFERENCES job_listings(id) ON DELETE CASCADE,
      match_score DECIMAL(5,2),
      matching_skills JSONB,
      missing_skills JSONB,
      suggestions JSONB,
      analysis_metadata JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE(resume_id, job_listing_id)
  );

  CREATE INDEX idx_analysis_resume ON analysis_results(resume_id);
  CREATE INDEX idx_analysis_job ON analysis_results(job_listing_id);
  ```

**Files to create:** `scripts/migrations/002_add_analysis_results.sql`

---

#### **TODO 3.3: Update db_utils.py for Resume Operations**

- [ ] Implement `insert_resume(resume: Resume, file_hash: str) -> int` (returns resume ID)
- [ ] Implement `get_resume_by_id(resume_id: int) -> Optional[Dict]`
- [ ] Implement `get_resume_by_hash(file_hash: str) -> Optional[Dict]`
- [ ] Implement `get_all_resumes(limit: int) -> List[Dict]`
- [ ] Implement `delete_resume(resume_id: int) -> bool`
- [ ] Implement `update_resume(resume_id: int, resume: Resume) -> bool`

**Files to modify:** `db_utils.py`

---

#### **TODO 3.4: Add Analysis Results Database Functions**

- [ ] Implement `insert_analysis_result(resume_id, job_id, results) -> int`
- [ ] Implement `get_analysis_for_resume(resume_id) -> List[Dict]`
- [ ] Implement `get_analysis_for_job(job_id) -> List[Dict]`
- [ ] Implement `get_analysis_by_ids(resume_id, job_id) -> Optional[Dict]`

**Files to modify:** `db_utils.py`

---

### **Task 4: AI-Powered Resume Analysis**

**Objective:** Compare resumes against job listings and generate optimization suggestions.

---

#### **TODO 4.1: Define Analysis Data Model**

- [ ] Add `AnalysisResult` dataclass to `models.py`:
  ```python
  @dataclass
  class AnalysisResult:
      resume_id: int
      job_listing_id: int
      match_score: float
      matching_skills: List[str]
      missing_skills: List[str]
      keyword_suggestions: List[str]
      improvement_suggestions: List[str]
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```

**Files to modify:** `models.py`

---

#### **TODO 4.2: Implement Skills Matching Engine**

- [ ] Create new file `analysis.py` for analysis logic
- [ ] Implement `extract_skills_from_job(job: JobListing) -> List[str]`
- [ ] Implement `compare_skills(resume_skills, job_skills) -> Tuple[List, List]`
  - Returns (matching_skills, missing_skills)
- [ ] Implement fuzzy matching for similar skills (e.g., "React" matches "React.js")
- [ ] Calculate basic match score based on skill overlap percentage

**Files to create:** `analysis.py`

---

#### **TODO 4.3: Implement Keyword Analysis**

- [ ] Extract important keywords from job description (TF-IDF or frequency-based)
- [ ] Identify keywords present in resume
- [ ] Identify high-value missing keywords
- [ ] Rank keywords by importance/frequency in job posting

**Files to modify:** `analysis.py`

**Optional new dependency:**
```
scikit-learn>=1.0.0  # For TF-IDF
```

---

#### **TODO 4.4: Integrate LLM for Smart Suggestions (Optional but Recommended)**

- [ ] Add OpenAI/Anthropic API integration
- [ ] Create prompt template for resume optimization suggestions
- [ ] Implement `generate_ai_suggestions(resume, job) -> List[str]`
- [ ] Handle API rate limits and errors gracefully
- [ ] Add caching for repeated analyses
- [ ] Make AI integration optional (fallback to rule-based suggestions)

**Files to modify:** `analysis.py`

**New dependencies (optional):**
```
openai>=1.0.0
# OR
anthropic>=0.18.0
```

**Environment variables to add to `.env`:**
```
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here
```

---

#### **TODO 4.5: Implement Rule-Based Suggestions (Fallback)**

- [ ] Generate suggestions for missing required skills
- [ ] Suggest adding quantifiable achievements if missing
- [ ] Suggest keyword additions based on job description
- [ ] Check for common resume issues (too long, missing sections, etc.)
- [ ] Generate section-specific improvement tips

**Files to modify:** `analysis.py`

---

#### **TODO 4.6: Create Main Analysis Orchestrator**

- [ ] Implement `analyze_resume_against_job(resume: Resume, job: JobListing) -> AnalysisResult`
- [ ] Coordinate skills matching, keyword analysis, and suggestion generation
- [ ] Calculate overall match score
- [ ] Return populated `AnalysisResult` object

**Files to modify:** `analysis.py`

---

### **Task 5: Streamlit UI for Resume Analysis**

**Objective:** Build the user interface for resume upload, analysis, and results display.

---

#### **TODO 5.1: Create Resume Upload Page**

- [ ] Add navigation/tabs to switch between "Job Ingestion" and "Resume Analysis"
- [ ] Create resume upload interface with drag-and-drop support
- [ ] Display extracted resume information for user verification
- [ ] Allow editing of extracted resume data before saving
- [ ] Add "Save Resume" button to persist to database

**Files to modify:** `app.py`

---

#### **TODO 5.2: Create Job Selection Interface**

- [ ] Display list of saved job listings from database
- [ ] Add search/filter functionality for job listings
- [ ] Allow single or multiple job selection for analysis
- [ ] Show job preview on selection

**Files to modify:** `app.py`

---

#### **TODO 5.3: Create Analysis Results Display**

- [ ] Display overall match score with visual indicator (progress bar, gauge)
- [ ] Show matching skills with checkmarks
- [ ] Highlight missing skills with importance ranking
- [ ] Display keyword suggestions in a copy-friendly format
- [ ] Show AI-generated improvement suggestions as actionable items
- [ ] Add "Export Results" functionality (PDF or text)

**Files to modify:** `app.py`

---

#### **TODO 5.4: Create Analysis History View**

- [ ] Display past analysis results for a resume
- [ ] Allow comparison between different job analyses
- [ ] Show improvement tracking over time (if resume is updated)

**Files to modify:** `app.py`

---

### **Task 6: Testing**

**Objective:** Ensure Phase 2 functionality is reliable and well-tested.

---

#### **TODO 6.1: Unit Tests for Resume Parser**

- [ ] Create `test_resume_parser.py`
- [ ] Test PDF extraction with sample PDFs
- [ ] Test DOCX extraction with sample DOCX files
- [ ] Test contact information extraction patterns
- [ ] Test skills extraction accuracy
- [ ] Test experience section parsing
- [ ] Test education section parsing

**Files to create:** `test_resume_parser.py`

---

#### **TODO 6.2: Unit Tests for Analysis Module**

- [ ] Create `test_analysis.py`
- [ ] Test skills matching logic
- [ ] Test keyword extraction
- [ ] Test match score calculation
- [ ] Test suggestion generation

**Files to create:** `test_analysis.py`

---

#### **TODO 6.3: Integration Tests**

- [ ] Test full resume upload -> parse -> save flow
- [ ] Test full analysis workflow (resume + job -> results)
- [ ] Test database operations for new tables
- [ ] Test UI workflows end-to-end

**Files to modify:** Existing test files or create `test_integration.py`

---

#### **TODO 6.4: Create Sample Test Data**

- [ ] Create 2-3 sample resume PDFs for testing
- [ ] Create 2-3 sample resume DOCX files for testing
- [ ] Document expected extraction results for each sample

**Files to create:** `tests/fixtures/` directory with sample files

---

### **Task 7: Documentation and Cleanup**

**Objective:** Ensure the codebase is well-documented and maintainable.

---

#### **TODO 7.1: Update README**

- [ ] Document new Phase 2 features
- [ ] Update installation instructions with new dependencies
- [ ] Add usage examples for resume analysis
- [ ] Document environment variables needed

**Files to modify:** `README.md`

---

#### **TODO 7.2: Add Inline Documentation**

- [ ] Add docstrings to all new functions
- [ ] Add type hints throughout new code
- [ ] Add comments for complex logic

**Files to modify:** All new files created in Phase 2

---

#### **TODO 7.3: Create Migration Guide**

- [ ] Document database migration steps
- [ ] Provide rollback instructions if needed
- [ ] Note any breaking changes from Phase 1

**Files to create:** `MIGRATION.md` or add section to `README.md`

---

### **Verification and Definition of Done for Phase 2**

Phase 2 is complete when:

1. Users can upload PDF and DOCX resumes through the Streamlit UI
2. Resume text is extracted and structured data is parsed correctly
3. Resume data is persisted to the PostgreSQL database
4. Users can select a saved job listing for analysis
5. The system generates a match score and identifies skill gaps
6. Users receive actionable optimization suggestions
7. Analysis results are saved and can be reviewed later
8. All unit tests pass (`pytest test_resume_parser.py test_analysis.py`)
9. Manual testing confirms the end-to-end workflow functions correctly

---

### **Suggested File Structure After Phase 2**

```
resume_optimizer/
├── .git/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── phase1.md
├── phase2.md
├── app.py                    # Main Streamlit application
├── models.py                 # Data models (JobListing, Resume, AnalysisResult)
├── processing.py             # Job listing extraction logic
├── resume_parser.py          # NEW: Resume parsing and extraction
├── analysis.py               # NEW: Resume-job comparison and suggestions
├── db_utils.py               # Database operations
├── scripts/
│   ├── init_db.sql
│   └── migrations/
│       ├── 001_add_resumes.sql
│       └── 002_add_analysis_results.sql
├── tests/
│   ├── test_processing.py
│   ├── test_resume_parser.py  # NEW
│   ├── test_analysis.py       # NEW
│   └── fixtures/
│       ├── sample_resume.pdf
│       └── sample_resume.docx
└── data/
    └── skills_taxonomy.json   # Optional: Skills database
```

---

### **Notes **

1. **Start with Tasks 1-2** (Resume parsing) as they have no external API dependencies
2. **Task 4.4 (LLM integration) is optional** - the rule-based fallback in 4.5 should work standalone
3. **Database migrations (Task 3)** should be run before testing Task 5 UI
4. **Test incrementally** - don't wait until the end to test
5. **The `source_url` field change from Phase 1** - make sure your database is updated before starting
