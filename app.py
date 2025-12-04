"""
AI-Powered Resume Optimizer - Streamlit Application

Phase 1: Job Ingestion and Processing Pipeline
Phase 2: Resume Analysis and AI-Powered Optimization
Phase 3: Automated Web Scraping with Playwright

This application allows users to:
1. Scrape job postings from URLs using Playwright
2. Upload resumes and parse structured data
3. Compare resumes against job listings
4. Get actionable optimization suggestions
"""
import hashlib
import subprocess
import os
import streamlit as st

from models import JobListing, Resume
from processing import hash_description
from resume_parser import parse_resume, extract_resume_details
import db_utils
from llm_verify import auto_correct_resume
from job_scraper import (
    scrape_job_url,
    ScrapingError,
    PageLoadError,
    BlockedError,
    ExtractionError,
)


# --- Helper Functions for Shell Commands ---
def run_command(command: list[str], timeout: int = 10) -> tuple[bool, str]:
    """
    Run a shell command and return (success, output).

    Args:
        command: Command as list of strings (e.g., ['pg_isready'])
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except FileNotFoundError:
        return False, f"Command not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_postgres_running(port: int = 5432) -> tuple[bool, str]:
    """Check if PostgreSQL is running using pg_isready."""
    return run_command(['pg_isready', '-p', str(port)])


def detect_postgres_port() -> int | None:
    """Try to detect which port PostgreSQL is running on."""
    common_ports = [5432, 5433, 5434, 5435, 5436]
    for port in common_ports:
        success, _ = run_command(['pg_isready', '-p', str(port)])
        if success:
            return port
    return None


def detect_postgres_user() -> str | None:
    """Try to detect a valid PostgreSQL username."""
    import getpass
    system_user = getpass.getuser()

    # Common usernames to try (in order of likelihood)
    candidates = [system_user, 'postgres']

    port = detect_postgres_port() or 5432

    for user in candidates:
        try:
            # Try to connect with this user
            result = subprocess.run(
                ['psql', '-U', user, '-h', 'localhost', '-p', str(port), '-l', '-t'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return user
        except FileNotFoundError:
            # psql not installed
            return None
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    return None


def get_env_path() -> str:
    """Get the path to the .env file."""
    return os.path.join(os.path.dirname(__file__), '.env')


def check_env_exists() -> bool:
    """Check if .env file exists."""
    return os.path.exists(get_env_path())


def get_current_env_values() -> dict:
    """Read current values from .env file."""
    defaults = {
        'db_name': 'resume_optimizer',
        'db_user': 'postgres',
        'db_password': '',
        'db_host': 'localhost',
        'db_port': 5432
    }

    if not check_env_exists():
        return defaults

    try:
        with open(get_env_path(), 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'DB_NAME':
                        defaults['db_name'] = value
                    elif key == 'DB_USER':
                        defaults['db_user'] = value
                    elif key == 'DB_PASSWORD':
                        defaults['db_password'] = value
                    elif key == 'DB_HOST':
                        defaults['db_host'] = value
                    elif key == 'DB_PORT':
                        defaults['db_port'] = int(value) if value.isdigit() else 5432
    except Exception:
        pass

    return defaults


def check_database_exists(db_name: str = 'resume_optimizer') -> tuple[bool, str]:
    """Check if the database exists using credentials from .env."""
    env_vals = get_current_env_values()
    db_user = env_vals['db_user']
    db_host = env_vals['db_host']
    db_port = env_vals['db_port']
    db_password = env_vals['db_password']

    cmd = ['psql', '-h', db_host, '-p', str(db_port), '-U', db_user, '-l', '-t']

    env = os.environ.copy()
    if db_password:
        env['PGPASSWORD'] = db_password

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
        output = result.stdout + result.stderr

        if result.returncode == 0:
            if db_name in output:
                return True, f"Database '{db_name}' exists"
            else:
                return False, f"Database '{db_name}' not found"
        return False, output.strip()
    except Exception as e:
        return False, f"Error: {str(e)}"


def create_database(db_name: str = 'resume_optimizer') -> tuple[bool, str]:
    """Create the database using credentials from .env."""
    env_vals = get_current_env_values()
    db_user = env_vals['db_user']
    db_host = env_vals['db_host']
    db_port = env_vals['db_port']
    db_password = env_vals['db_password']

    cmd = [
        'createdb',
        '-h', db_host,
        '-p', str(db_port),
        '-U', db_user,
        db_name
    ]

    # Set PGPASSWORD environment variable for password
    env = os.environ.copy()
    if db_password:
        env['PGPASSWORD'] = db_password

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except Exception as e:
        return False, f"Error: {str(e)}"


def run_setup_script() -> tuple[bool, str]:
    """Run the database setup script using credentials from .env."""
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'setup_db.sql')
    if not os.path.exists(script_path):
        return False, f"Setup script not found: {script_path}"

    # Get credentials from .env
    env_vals = get_current_env_values()
    db_name = env_vals['db_name']
    db_user = env_vals['db_user']
    db_host = env_vals['db_host']
    db_port = env_vals['db_port']
    db_password = env_vals['db_password']

    # Build psql command with credentials
    cmd = [
        'psql',
        '-h', db_host,
        '-p', str(db_port),
        '-U', db_user,
        '-d', db_name,
        '-f', script_path
    ]

    # Set PGPASSWORD environment variable for password
    env = os.environ.copy()
    if db_password:
        env['PGPASSWORD'] = db_password

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except Exception as e:
        return False, f"Error: {str(e)}"


def create_default_env(
    db_name: str = "resume_optimizer",
    db_user: str = "postgres",
    db_password: str = "",
    db_host: str = "localhost",
    db_port: int = 5432,
    overwrite: bool = False
) -> tuple[bool, str]:
    """Create a .env file with the specified PostgreSQL settings."""
    env_path = get_env_path()

    if os.path.exists(env_path) and not overwrite:
        return False, ".env file already exists"

    content = f"""# Database Configuration
DB_NAME={db_name}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_HOST={db_host}
DB_PORT={db_port}
"""
    try:
        with open(env_path, 'w') as f:
            f.write(content)
        # Reload environment variables
        from dotenv import load_dotenv
        load_dotenv(override=True)
        return True, f"{'Updated' if overwrite else 'Created'} .env successfully"
    except Exception as e:
        return False, f"Failed to create .env: {e}"


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Resume Optimizer",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_description' not in st.session_state:
    st.session_state.original_description = None
if 'show_success' not in st.session_state:
    st.session_state.show_success = False
if 'current_resume' not in st.session_state:
    st.session_state.current_resume = None
if 'resume_file_hash' not in st.session_state:
    st.session_state.resume_file_hash = None
# Setup wizard state
if 'setup_step' not in st.session_state:
    st.session_state.setup_step = 1
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
# Scraper state
if 'scrape_error' not in st.session_state:
    st.session_state.scrape_error = None


def hash_file(file_bytes: bytes) -> str:
    """Generate SHA-256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


# --- DIALOG FUNCTIONS FOR VIEWING FULL DETAILS ---
@st.dialog("Job Listing Details", width="large")
def view_job_dialog(job_id: int):
    """Display full job listing details in a modal dialog."""
    job = db_utils.get_job_listing_by_id(job_id)
    if not job:
        st.error("Job listing not found.")
        return

    st.markdown(f"## {job.get('job_title', 'Unknown Position')}")
    st.markdown(f"**Company:** {job.get('company', 'N/A')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Location:** {job.get('location', 'N/A')}")
    with col2:
        st.markdown(f"**Added:** {job.get('created_at', 'N/A')}")

    if job.get('apply_url'):
        st.markdown(f"[Apply Here]({job['apply_url']})")

    if job.get('source_url'):
        st.markdown(f"**Source:** [{job['source_url'][:50]}...]({job['source_url']})")

    st.divider()
    st.markdown("### Job Description")
    description = job.get('description', 'No description available.')
    # Use a scrollable text area that respects Streamlit theming
    st.text_area(
        label="Job Description",
        value=description,
        height=400,
        disabled=True,
        label_visibility="collapsed"
    )

    # Show metadata if available
    if job.get('metadata'):
        with st.expander("Extraction Metadata"):
            st.json(job['metadata'])

    if st.button("Close", type="primary"):
        st.rerun()


@st.dialog("Resume Details", width="large")
def view_resume_dialog(resume_id: int):
    """Display full resume details in a modal dialog."""
    resume = db_utils.get_resume_by_id(resume_id)
    if not resume:
        st.error("Resume not found.")
        return

    st.markdown(f"## {resume.get('full_name', 'Unknown')}")

    # Contact Information
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Email:** {resume.get('email', 'N/A')}")
        st.markdown(f"**Phone:** {resume.get('phone', 'N/A')}")
    with col2:
        st.markdown(f"**Location:** {resume.get('location', 'N/A')}")
        st.markdown(f"**Added:** {resume.get('created_at', 'N/A')}")

    # Summary
    if resume.get('summary'):
        st.divider()
        st.markdown("### Summary")
        st.write(resume['summary'])

    # Skills
    skills = resume.get('skills', [])
    if skills:
        st.divider()
        st.markdown(f"### Skills ({len(skills)})")
        skills_html = " ".join([
            f'<span style="background-color: #e1e1e1 !important; padding: 4px 10px; margin: 3px; '
            f'border-radius: 15px; display: inline-block; font-size: 13px; color: #1a1a1a !important;">{skill}</span>'
            for skill in skills
        ])
        st.markdown(skills_html, unsafe_allow_html=True)

    # Experience
    experience = resume.get('experience', [])
    if experience:
        st.divider()
        st.markdown(f"### Experience ({len(experience)})")
        for exp in experience:
            st.markdown(f"**{exp.get('title', 'Position')}** at {exp.get('company', 'Company')}")
            if exp.get('dates'):
                st.caption(exp['dates'])
            if exp.get('description'):
                for bullet in exp['description']:
                    st.markdown(f"- {bullet}")
            st.markdown("")

    # Education
    education = resume.get('education', [])
    if education:
        st.divider()
        st.markdown(f"### Education ({len(education)})")
        for edu in education:
            st.markdown(f"**{edu.get('degree', 'Degree')}** - {edu.get('institution', 'Institution')}")
            details = []
            if edu.get('field'):
                details.append(edu['field'])
            if edu.get('dates'):
                details.append(edu['dates'])
            if edu.get('gpa'):
                details.append(f"GPA: {edu['gpa']}")
            if details:
                st.caption(" | ".join(details))

    # Projects
    projects = resume.get('projects', [])
    if projects:
        st.divider()
        st.markdown(f"### Projects ({len(projects)})")
        for proj in projects:
            st.markdown(f"**{proj.get('name', 'Project')}**")
            if proj.get('url'):
                st.markdown(f"[View Project]({proj['url']})")
            if proj.get('technologies'):
                tech_html = " ".join([
                    f'<span style="background-color: #d4edda !important; padding: 2px 6px; '
                    f'border-radius: 8px; font-size: 12px; color: #155724 !important;">{t}</span>'
                    for t in proj['technologies']
                ])
                st.markdown(tech_html, unsafe_allow_html=True)
            if proj.get('description'):
                for bullet in proj['description']:
                    st.markdown(f"- {bullet}")
            st.markdown("")

    if st.button("Close", type="primary"):
        st.rerun()


@st.dialog("Confirm Delete")
def confirm_delete_job_dialog(job_id: int, job_title: str, company: str):
    """Confirmation dialog for deleting a job listing."""
    st.markdown(f"Are you sure you want to delete this job listing?")
    st.markdown(f"**{job_title}** at **{company}**")
    st.warning("This action cannot be undone.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", type="primary", key="confirm_del_job"):
            if db_utils.delete_job_listing(job_id):
                st.session_state.delete_success = "Job listing deleted successfully!"
                st.rerun()
            else:
                st.error("Failed to delete job listing.")
    with col2:
        if st.button("Cancel", key="cancel_del_job"):
            st.rerun()


@st.dialog("Confirm Delete")
def confirm_delete_resume_dialog(resume_id: int, name: str):
    """Confirmation dialog for deleting a resume."""
    st.markdown(f"Are you sure you want to delete this resume?")
    st.markdown(f"**{name}**")
    st.warning("This action cannot be undone.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", type="primary", key="confirm_del_res"):
            if db_utils.delete_resume(resume_id):
                st.session_state.delete_success = "Resume deleted successfully!"
                st.rerun()
            else:
                st.error("Failed to delete resume.")
    with col2:
        if st.button("Cancel", key="cancel_del_res"):
            st.rerun()


# --- UI - HEADER ---
st.title("AI-Powered Resume Optimizer")

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["Job Ingestion", "Resume Analysis", "History"])


# =============================================================================
# TAB 1: JOB INGESTION (Phase 1)
# =============================================================================
with tab1:
    st.markdown("### Phase 1: Job Ingestion")

    # Show success message if set
    if st.session_state.show_success:
        st.success("Job listing saved successfully!")
        st.session_state.show_success = False

    # Show the verification form if we have processed data
    if st.session_state.processed_data:
        st.header("Step 2: Verify Extracted Information")
        st.warning("Please review and correct any extracted details below before saving.")

        data = st.session_state.processed_data

        # Display extraction metadata
        if data.metadata:
            with st.expander("Extraction Details"):
                # Scrape source info
                if data.metadata.get('scrape_source'):
                    source_map = {
                        'json_ld': 'JSON-LD Schema (most reliable)',
                        'css_selectors': 'CSS Selectors',
                        'llm': 'LLM Extraction'
                    }
                    source = data.metadata.get('scrape_source')
                    st.write(f"**Primary Source:** {source_map.get(source, source)}")

                if data.metadata.get('ats_platform'):
                    st.write(f"**ATS Platform:** {data.metadata['ats_platform']}")

                if data.metadata.get('llm_extraction'):
                    st.write("**LLM Enhancement:** Yes (two-pass extraction)")

                # Legacy regex/spacy info
                methods = data.metadata.get('extraction_methods', {})
                if methods.get('json_ld'):
                    st.write(f"**JSON-LD fields:** {', '.join(methods['json_ld'])}")
                if methods.get('css'):
                    st.write(f"**CSS extracted:** {', '.join(methods['css'])}")
                if methods.get('regex'):
                    st.write(f"**Regex extracted:** {', '.join(methods['regex'])}")
                if methods.get('spacy'):
                    st.write(f"**NLP (spaCy) extracted:** {', '.join(methods['spacy'])}")

        # Show description sections OUTSIDE the form (before editable fields)
        st.markdown("---")
        st.markdown("#### Job Description")

        llm_sections = data.metadata.get('llm_raw', {}).get('description_sections', []) if data.metadata else []

        if llm_sections:
            # Display each section with its header and bullet points
            for section in llm_sections:
                header = section.get('header', 'Description')
                content = section.get('content', [])

                with st.expander(f"**{header}**", expanded=True):
                    if isinstance(content, list):
                        for item in content:
                            if item and item.strip():
                                st.markdown(f"- {item}")
                    elif content:
                        st.markdown(content)
        else:
            # Fallback: show full description text in expander
            with st.expander("Full Description", expanded=True):
                # Show in a scrollable container
                st.markdown(
                    f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">'
                    f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{data.description}</pre>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Form for editing extracted data
        with st.form(key="confirmation_form"):
            st.markdown("#### Edit Extracted Fields")

            # Source URL (pre-filled from scrape)
            edited_source_url = st.text_input(
                "Source URL *",
                value=data.source_url or "",
                help="The URL where this job was found"
            )

            col1, col2 = st.columns(2)

            with col1:
                edited_title = st.text_input(
                    "Job Title",
                    value=data.job_title or "",
                    help="The position title"
                )
                edited_company = st.text_input(
                    "Company",
                    value=data.company or "",
                    help="The hiring company"
                )

            with col2:
                edited_location = st.text_input(
                    "Location",
                    value=data.location or "",
                    help="Job location (city, state, or Remote)"
                )
                edited_url = st.text_input(
                    "Apply URL",
                    value=data.apply_url or "",
                    help="Link to the application"
                )

            # Form buttons
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                confirm_button = st.form_submit_button("Confirm and Save", type="primary")
            with col2:
                cancel_button = st.form_submit_button("Cancel")

        # Handle form submission
        if confirm_button:
            if not edited_source_url or not edited_source_url.strip():
                st.error("Source URL is required.")
                st.stop()

            final_job_data = JobListing(
                job_title=edited_title or None,
                company=edited_company or None,
                location=edited_location or None,
                apply_url=edited_url or None,
                description=st.session_state.original_description,
                source_url=edited_source_url.strip(),
                metadata=data.metadata
            )

            description_hash = hash_description(final_job_data.description)

            if db_utils.check_for_duplicate(description_hash):
                st.warning("This job listing is already in the database.")
            else:
                success = db_utils.insert_job_listing(final_job_data, description_hash)
                if success:
                    st.session_state.show_success = True
                else:
                    st.error("Failed to save the job listing.")

            st.session_state.processed_data = None
            st.session_state.original_description = None
            st.rerun()

        if cancel_button:
            st.session_state.processed_data = None
            st.session_state.original_description = None
            st.rerun()

    else:
        # URL input form
        st.info("Enter a job posting URL below and click 'Scrape' to extract job details automatically.")

        # Show any previous scrape error
        if st.session_state.scrape_error:
            st.error(st.session_state.scrape_error)
            st.session_state.scrape_error = None

        job_url = st.text_input(
            "Job Posting URL",
            placeholder="https://jobs.example.com/position/12345",
            help="Paste the full URL of the job posting you want to analyze",
            key="job_url_input"
        )

        if st.button("Scrape Job Posting", type="primary"):
            if not job_url or not job_url.strip():
                st.error("Please enter a job posting URL.")
            elif not job_url.startswith(('http://', 'https://')):
                st.error("URL must start with http:// or https://")
            else:
                with st.spinner("Scraping job posting... This may take a few seconds."):
                    try:
                        job_listing = scrape_job_url(job_url.strip())
                        st.session_state.processed_data = job_listing
                        st.session_state.original_description = job_listing.description
                        st.rerun()
                    except PageLoadError as e:
                        st.session_state.scrape_error = f"Could not load page: {e}"
                        st.rerun()
                    except BlockedError as e:
                        st.session_state.scrape_error = f"Site blocked access: {e}. Try again later."
                        st.rerun()
                    except ExtractionError as e:
                        st.session_state.scrape_error = f"Could not extract job details: {e}"
                        st.rerun()
                    except ValueError as e:
                        st.session_state.scrape_error = str(e)
                        st.rerun()
                    except Exception as e:
                        st.session_state.scrape_error = f"Unexpected error: {e}"
                        st.rerun()



# =============================================================================
# TAB 2: RESUME ANALYSIS (Phase 2)
# =============================================================================
with tab2:
    st.markdown("### Resume Analysis")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)",
        type=['pdf', 'docx'],
        help="Maximum file size: 5MB"
    )

    if uploaded_file:
        # Check file size (5MB limit)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File too large. Please upload a file smaller than 5MB.")
        else:
            file_bytes = uploaded_file.read()
            file_hash = hash_file(file_bytes)

            # Auto-process new files
            if st.session_state.resume_file_hash != file_hash:
                with st.spinner("Parsing resume with AI..."):
                    try:
                        file_type = uploaded_file.name.split('.')[-1]
                        raw_text = parse_resume(file_bytes, file_type)
                        resume = extract_resume_details(raw_text)

                        # Auto-correct with LLM (runs silently)
                        resume = auto_correct_resume(resume)

                        st.session_state.current_resume = resume
                        st.session_state.resume_file_hash = file_hash
                    except Exception as e:
                        st.error(f"Error parsing resume: {e}")
                        st.session_state.current_resume = None

    # Display parsed resume info
    if st.session_state.current_resume:
        resume = st.session_state.current_resume
        st.success("Resume parsed successfully!")

        # Contact Information
        with st.expander("Contact Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {resume.full_name or 'Not detected'}")
                st.write(f"**Email:** {resume.email or 'Not detected'}")
            with col2:
                st.write(f"**Phone:** {resume.phone or 'Not detected'}")
                st.write(f"**Location:** {resume.location or 'Not detected'}")

            if resume.metadata.get('linkedin_url'):
                st.write(f"**LinkedIn:** {resume.metadata['linkedin_url']}")

        # Skills - Full List
        with st.expander(f"Skills ({len(resume.skills)} found)", expanded=True):
            if resume.skills:
                # Display as tags/chips style
                skills_html = " ".join([
                    f'<span style="background-color: #e1e1e1; padding: 2px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 14px;">{skill}</span>'
                    for skill in resume.skills
                ])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No skills detected. Try uploading a resume with a Skills section.")

        # Experience - Full Details
        with st.expander(f"Experience ({len(resume.experience)} entries)", expanded=False):
            if resume.experience:
                for i, exp in enumerate(resume.experience, 1):
                    st.markdown(f"**{i}. {exp.get('title', 'Position')}**")
                    if exp.get('company'):
                        st.write(f"   Company: {exp['company']}")
                    if exp.get('dates'):
                        st.write(f"   Dates: {exp['dates']}")
                    if exp.get('description'):
                        for bullet in exp['description']:
                            st.write(f"   - {bullet}")
                    st.divider()
            else:
                st.info("No experience entries detected.")

        # Education - Full Details
        with st.expander(f"Education ({len(resume.education)} entries)", expanded=False):
            if resume.education:
                for i, edu in enumerate(resume.education, 1):
                    degree = edu.get('degree', 'Degree')
                    institution = edu.get('institution', 'Institution')
                    st.markdown(f"**{i}. {degree}** - {institution}")
                    if edu.get('field'):
                        st.write(f"   Field: {edu['field']}")
                    if edu.get('dates'):
                        st.write(f"   Dates: {edu['dates']}")
                    if edu.get('gpa'):
                        st.write(f"   GPA: {edu['gpa']}")
            else:
                st.info("No education entries detected.")

        # Projects - Full Details
        with st.expander(f"Projects ({len(resume.projects)} entries)", expanded=False):
            if resume.projects:
                for i, proj in enumerate(resume.projects, 1):
                    st.markdown(f"**{i}. {proj.get('name', 'Project')}**")
                    if proj.get('url'):
                        st.markdown(f"   [Link]({proj['url']})")
                    if proj.get('technologies'):
                        tech_tags = " ".join([
                            f'<span style="background-color: #d4edda; padding: 2px 6px; margin: 1px; border-radius: 8px; font-size: 12px;">{t}</span>'
                            for t in proj['technologies'][:8]
                        ])
                        st.markdown(f"   Technologies: {tech_tags}", unsafe_allow_html=True)
                    if proj.get('description'):
                        for bullet in proj['description'][:3]:
                            st.write(f"   - {bullet[:150]}...")
                    st.divider()
            else:
                st.info("No project entries detected.")

        # Raw Text Preview
        with st.expander("Raw Text Preview", expanded=False):
            st.text_area(
                "Extracted text from resume",
                value=resume.raw_text[:3000] + ("..." if len(resume.raw_text) > 3000 else ""),
                height=200,
                disabled=True
            )

        # Extraction Metadata
        if resume.metadata.get('extraction_methods'):
            with st.expander("Extraction Methods", expanded=False):
                methods = resume.metadata['extraction_methods']
                if methods.get('regex'):
                    st.write(f"**Regex:** {', '.join(methods['regex'])}")
                if methods.get('spacy'):
                    st.write(f"**NLP (spaCy):** {', '.join(methods['spacy'])}")

                # LLM processing info
                if resume.metadata.get('llm_processed'):
                    st.write("**LLM:** Two-pass extraction complete")
                else:
                    st.write("**LLM:** Not processed (API unavailable?)")

                # Debug: show what LLM returned
                if resume.metadata.get('llm_pass1'):
                    with st.expander("LLM Pass 1 Response"):
                        st.json(resume.metadata['llm_pass1'])
                if resume.metadata.get('llm_pass2'):
                    with st.expander("LLM Pass 2 Response"):
                        st.json(resume.metadata['llm_pass2'])

        # Action buttons at bottom
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Resume to Database"):
                try:
                    if not db_utils.tables_ready():
                        st.error("Database tables not set up. Use the sidebar to run setup.")
                    else:
                        existing = db_utils.get_resume_by_hash(file_hash)
                        if existing:
                            st.warning("This resume is already saved in the database.")
                        else:
                            resume_id = db_utils.insert_resume(resume, file_hash)
                            if resume_id:
                                st.success(f"Resume saved! (ID: {resume_id})")
                            else:
                                st.error("Failed to save resume.")
                except Exception as e:
                    st.error(f"Database error: {e}")
        with col2:
            if st.button("Reprocess Resume", help="Force re-parsing with LLM"):
                st.session_state.resume_file_hash = None  # Clear hash to force reprocess
                st.rerun()


# =============================================================================
# TAB 3: HISTORY (Phase 2)
# =============================================================================
with tab3:
    st.markdown("### Analysis History")

    # Check if database is ready
    try:
        history_tables_ready = db_utils.tables_ready()
    except Exception:
        history_tables_ready = False

    if not history_tables_ready:
        st.warning("Database tables not set up yet. Use the Database Setup in the sidebar to create tables.")
    else:
        col_jobs, col_resumes = st.columns(2)

        with col_jobs:
            st.subheader("Saved Job Listings")
            # Show delete success message if set
            if st.session_state.get('delete_success'):
                st.success(st.session_state.delete_success)
                st.session_state.delete_success = None
            try:
                jobs = db_utils.get_all_job_listings(limit=20)
                if jobs:
                    for job in jobs:
                        job_title = job.get('job_title', 'Unknown')
                        company = job.get('company', 'Unknown')
                        with st.expander(f"{job_title} - {company}"):
                            st.write(f"**Location:** {job.get('location', 'N/A')}")
                            st.write(f"**Added:** {job.get('created_at', 'N/A')}")
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("View", key=f"view_job_{job['id']}", type="primary"):
                                    view_job_dialog(job['id'])
                            with btn_col2:
                                if st.button("Delete", key=f"del_job_{job['id']}"):
                                    confirm_delete_job_dialog(job['id'], job_title, company)
                else:
                    st.info("No saved job listings.")
            except Exception as e:
                st.error(f"Error loading jobs: {e}")

        with col_resumes:
            st.subheader("Saved Resumes")
            try:
                resumes = db_utils.get_all_resumes(limit=20)
                if resumes:
                    for res in resumes:
                        name = res.get('full_name') or res.get('email') or f"Resume #{res['id']}"
                        with st.expander(name):
                            st.write(f"**Email:** {res.get('email', 'N/A')}")
                            skills = res.get('skills', [])
                            if skills:
                                st.write(f"**Skills:** {', '.join(skills[:5])}...")
                            st.write(f"**Added:** {res.get('created_at', 'N/A')}")
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("View", key=f"view_res_{res['id']}", type="primary"):
                                    view_resume_dialog(res['id'])
                            with btn_col2:
                                if st.button("Delete", key=f"del_res_{res['id']}"):
                                    confirm_delete_resume_dialog(res['id'], name)
                else:
                    st.info("No saved resumes.")
            except Exception as e:
                st.error(f"Error loading resumes: {e}")


# --- SIDEBAR: Database Status ---
with st.sidebar:
    st.header("Settings")

    st.subheader("Database Status")

    # Check current state
    db_connected = False
    tables_exist = False
    try:
        db_connected = db_utils.test_connection()
        if db_connected:
            tables_exist = db_utils.tables_ready()
    except Exception:
        pass

    # Determine setup status
    env_exists = check_env_exists()
    pg_running, _ = check_postgres_running(detect_postgres_port() or 5432)

    # Check database exists - but also detect credential errors
    credentials_valid = True
    db_exists = False
    db_check_error = ""
    if pg_running:
        db_exists, db_check_error = check_database_exists()
        # Detect credential/authentication errors
        if "role" in db_check_error.lower() or "authentication" in db_check_error.lower() or "password" in db_check_error.lower():
            credentials_valid = False

    # Show current status
    if db_connected and tables_exist:
        st.success("Connected & Ready")
        st.session_state.setup_step = 5  # All done
    else:
        # Show what's missing
        status_items = [
            ("Environment (.env)", env_exists and credentials_valid),
            ("PostgreSQL Running", pg_running),
            ("Database Exists", db_exists),
            ("Tables Created", tables_exist),
        ]
        for item, ok in status_items:
            st.write(f"{'[x]' if ok else '[ ]'} {item}")

        # Show credential error if detected
        if not credentials_valid:
            st.error(f"Credential error: {db_check_error}")

    # Show setup wizard if not fully configured
    if not (db_connected and tables_exist):
        with st.expander("Database Setup Wizard", expanded=True):

            # Determine current step based on what's complete
            # Force step 1 if credentials are invalid
            if not env_exists or not credentials_valid:
                current_step = 1
            elif not pg_running:
                current_step = 2
            elif not db_exists:
                current_step = 3
            elif not tables_exist:
                current_step = 4
            else:
                current_step = 5

            st.progress(current_step / 4 if current_step <= 4 else 1.0)
            st.caption(f"Step {min(current_step, 4)} of 4")

            # ===================== STEP 1: Environment Config =====================
            if current_step == 1:
                st.markdown("### Step 1: Configure Database Connection")

                if env_exists and not credentials_valid:
                    st.error("Your current credentials are invalid. Please update them below.")
                else:
                    st.info("First, let's set up your database connection settings.")

                # Auto-detect port
                detected_port = detect_postgres_port()
                if detected_port:
                    st.success(f"Detected PostgreSQL on port {detected_port}")
                    default_port = detected_port
                else:
                    default_port = 5432

                # Auto-detect username
                detected_user = detect_postgres_user()
                if detected_user:
                    st.success(f"Detected valid user: **{detected_user}**")
                    default_user = detected_user
                else:
                    default_user = None

                # Get current values if .env exists but has bad values
                current_vals = get_current_env_values()

                # Determine best username to show
                # Priority: detected user > current env value (if valid)
                if default_user:
                    show_user = default_user
                elif current_vals['db_user'] not in ['your_username', 'postgres', '']:
                    show_user = current_vals['db_user']
                else:
                    show_user = default_user or current_vals['db_user']

                env_db_name = st.text_input("Database Name", value=current_vals['db_name'], key="env_db_name")
                env_db_user = st.text_input("Username", value=show_user, key="env_db_user",
                                           help="Usually 'postgres' or your system username")
                env_db_password = st.text_input("Password", value="", type="password", key="env_db_password",
                                               help="Leave empty if no password set")
                env_db_host = st.text_input("Host", value=current_vals['db_host'], key="env_db_host")
                env_db_port = st.number_input("Port", value=detected_port or current_vals['db_port'],
                                             min_value=1, max_value=65535, key="env_db_port")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Configuration", type="primary", key="save_env"):
                        success, output = create_default_env(
                            db_name=env_db_name,
                            db_user=env_db_user,
                            db_password=env_db_password,
                            db_host=env_db_host,
                            db_port=int(env_db_port),
                            overwrite=True
                        )
                        if success:
                            st.success(output)
                            st.rerun()
                        else:
                            st.error(output)
                with col2:
                    if env_exists:
                        st.caption("Will overwrite existing .env")

            # ===================== STEP 2: Check PostgreSQL =====================
            elif current_step == 2:
                st.markdown("### Step 2: Start PostgreSQL")
                st.warning("PostgreSQL doesn't appear to be running.")

                st.markdown("**Start PostgreSQL with:**")
                st.code("# macOS (Homebrew)\nbrew services start postgresql\n\n# Linux\nsudo systemctl start postgresql", language="bash")

                if st.button("Check Again", type="primary", key="check_pg_again"):
                    success, output = check_postgres_running(detect_postgres_port() or 5432)
                    if success:
                        st.success(f"PostgreSQL is running!")
                        st.rerun()
                    else:
                        st.error(f"Still not running: {output}")

            # ===================== STEP 3: Create Database =====================
            elif current_step == 3:
                st.markdown("### Step 3: Create Database")
                st.info("PostgreSQL is running. Now let's create the database.")

                env_vals = get_current_env_values()
                db_name = env_vals['db_name']

                st.write(f"Database to create: **{db_name}**")

                if st.button(f"Create '{db_name}' Database", type="primary", key="create_db"):
                    success, output = create_database(db_name)
                    if success:
                        st.success(f"Database '{db_name}' created!")
                        st.rerun()
                    elif "already exists" in output:
                        st.success(f"Database '{db_name}' already exists!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {output}")

            # ===================== STEP 4: Create Tables =====================
            elif current_step == 4:
                st.markdown("### Step 4: Create Tables")
                st.info("Database exists. Now let's create the tables.")

                # Show which tables are missing
                tables_status = db_utils.check_tables_exist()
                for table, exists in tables_status.items():
                    st.write(f"{'[x]' if exists else '[ ]'} {table}")

                if st.button("Create Tables", type="primary", key="run_setup"):
                    success, output = run_setup_script()
                    if success:
                        st.success("Tables created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Setup failed: {output}")
                        st.code(output, language="text")

            # ===================== STEP 5: Complete =====================
            else:
                st.markdown("### Setup Complete!")
                st.success("Your database is fully configured.")
                if st.button("Refresh", key="final_refresh"):
                    st.rerun()

            st.divider()

            # Reset option
            if current_step > 1:
                if st.button("Start Over", key="reset_setup"):
                    st.session_state.setup_step = 1
                    st.rerun()

    st.divider()

    st.subheader("Quick Stats")
    if db_connected and tables_exist:
        try:
            job_count = len(db_utils.get_all_job_listings(limit=1000))
            resume_count = len(db_utils.get_all_resumes(limit=1000))
            st.write(f"**Jobs saved:** {job_count}")
            st.write(f"**Resumes saved:** {resume_count}")
        except Exception:
            st.write("Stats unavailable")
    else:
        st.write("Complete setup to view stats")

# --- FOOTER ---
st.divider()
st.caption("Resume Optimizer v0.3 | Phase 3: Web Scraping")
