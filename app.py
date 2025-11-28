"""
AI-Powered Resume Optimizer - Streamlit Application

Phase 1: Job Ingestion and Processing Pipeline

This application allows users to:
1. Paste job listing text
2. Extract key information using regex and NLP
3. Review and edit extracted data
4. Save to a PostgreSQL database with deduplication
"""
import streamlit as st

from models import JobListing
from processing import extract_job_details, hash_description, clean_job_text
import db_utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Resume Optimizer",
    page_icon="📄",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
# This must be at the top of the script. It creates a dictionary-like object
# that persists across script reruns for a single user session.

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if 'original_description' not in st.session_state:
    st.session_state.original_description = None

if 'show_success' not in st.session_state:
    st.session_state.show_success = False


# --- UI - HEADER ---
st.title("AI-Powered Resume Optimizer")
st.markdown("### Phase 1: Job Ingestion")

# Show success message if set
if st.session_state.show_success:
    st.success("Job listing saved successfully!")
    st.session_state.show_success = False


# --- UI - SIDEBAR ---
with st.sidebar:
    st.header("Settings")

    # Source link input (mandatory)
    source_url = st.text_input(
        "Source Link *",
        placeholder="https://linkedin.com/jobs/view/...",
        help="Paste the URL where you found this job listing"
    )

    st.divider()

    # Database connection status
    st.subheader("Database Status")
    try:
        if db_utils.test_connection():
            st.success("Connected")
        else:
            st.error("Not connected")
    except Exception:
        st.warning("DB not configured")

    st.divider()

    # View saved listings
    st.subheader("Saved Listings")
    if st.button("View All Listings"):
        st.session_state.show_listings = True

    if st.session_state.get('show_listings', False):
        listings = db_utils.get_all_job_listings(limit=10)
        if listings:
            for listing in listings:
                with st.expander(f"{listing.get('job_title', 'Unknown')} - {listing.get('company', 'Unknown')}"):
                    st.write(f"**Location:** {listing.get('location', 'N/A')}")
                    st.write(f"**Date:** {listing.get('created_at', 'N/A')}")
        else:
            st.info("No saved listings yet.")


# --- UI - MAIN CONTENT ---

# Show the verification form if we have processed data
if st.session_state.processed_data:
    st.header("Step 2: Verify Extracted Information")
    st.warning("Please review and correct any extracted details below before saving.")

    data = st.session_state.processed_data

    # Display extraction metadata
    if data.metadata and 'extraction_methods' in data.metadata:
        methods = data.metadata['extraction_methods']
        regex_fields = methods.get('regex', [])
        spacy_fields = methods.get('spacy', [])

        with st.expander("Extraction Details"):
            if regex_fields:
                st.write(f"**Regex extracted:** {', '.join(regex_fields)}")
            if spacy_fields:
                st.write(f"**NLP (spaCy) extracted:** {', '.join(spacy_fields)}")

    # Form for editing extracted data
    with st.form(key="confirmation_form"):
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

        # Show truncated description
        st.text_area(
            "Description Preview",
            value=data.description[:500] + "..." if len(data.description) > 500 else data.description,
            height=150,
            disabled=True
        )

        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            confirm_button = st.form_submit_button(
                "Confirm and Save",
                type="primary"
            )

        with col2:
            cancel_button = st.form_submit_button("Cancel")

    # Handle form submission
    if confirm_button:
        # Validate mandatory source URL
        if not source_url or not source_url.strip():
            st.error("Source Link is required. Please enter the URL where you found this job listing.")
            st.stop()

        # Create the final JobListing object with edited data
        final_job_data = JobListing(
            job_title=edited_title or None,
            company=edited_company or None,
            location=edited_location or None,
            apply_url=edited_url or None,
            description=st.session_state.original_description,
            source_url=source_url,
            metadata=data.metadata
        )

        # Generate hash for deduplication
        description_hash = hash_description(final_job_data.description)

        # Check for duplicates
        if db_utils.check_for_duplicate(description_hash):
            st.warning("This job listing is already in the database. No duplicate was created.")
        else:
            # Insert into database
            success = db_utils.insert_job_listing(final_job_data, description_hash)
            if success:
                st.session_state.show_success = True
            else:
                st.error("Failed to save the job listing. Check the database connection.")

        # Reset state
        st.session_state.processed_data = None
        st.session_state.original_description = None
        st.rerun()

    if cancel_button:
        st.session_state.processed_data = None
        st.session_state.original_description = None
        st.rerun()

else:
    # --- UI - STEP 1: DATA INPUT ---
    st.info("Paste the full, unformatted text of a job listing below and click 'Process' to begin.")

    # Text input area
    job_listing_text = st.text_area(
        "Paste Job Listing Text Here",
        height=300,
        placeholder="Copy and paste the complete job listing text here...\n\nExample:\nJob Title: Data Scientist\nCompany: TechCorp\nLocation: San Francisco, CA\n\nAbout the role:\nWe are looking for a passionate data scientist...",
        key="job_text_input"
    )

    # Process button
    col1, col2 = st.columns([1, 4])
    with col1:
        process_button = st.button("Process Job Listing", type="primary")

    # Handle processing
    if process_button:
        if not job_listing_text or len(job_listing_text.strip()) < 50:
            st.error("Please paste a job listing with at least 50 characters.")
        else:
            with st.spinner("Analyzing job description..."):
                # Clean and extract
                cleaned_text = clean_job_text(job_listing_text)
                extracted_info = extract_job_details(cleaned_text)

                # Store in session state
                st.session_state.processed_data = extracted_info
                st.session_state.original_description = job_listing_text

            st.rerun()


# --- FOOTER ---
st.divider()
st.caption("Resume Optimizer v0.1 | Phase 1: Job Ingestion Pipeline")
