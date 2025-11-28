# AI-Powered Resume Optimizer

A local Streamlit application that helps optimize job applications by analyzing job listings, matching them to pre-tailored resumes, and generating personalized application materials.

## Overview

This tool allows you to:
1. Paste job listing text from sites like LinkedIn, Indeed, or Handshake
2. Automatically extract key details (title, company, location, etc.)
3. Score listings against multiple resume personas using semantic similarity
4. Generate optimized resumes and cover letters via LLM

## Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- spaCy English model

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd resume_automation
```

### 2. Create a virtual environment

```bash
python -m venv venv # If this doesnt work try: 'python3 -m venv venv'
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set up PostgreSQL

#### Installing PostgreSQL (First-time setup)

**macOS (using Homebrew):**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service (runs in background)
brew services start postgresql@15

# Add PostgreSQL to your PATH (add this to your ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Windows:**
1. Download the installer from https://www.postgresql.org/download/windows/
2. Run the installer and follow the prompts
3. Remember the password you set for the `postgres` superuser
4. The installer will add PostgreSQL to your PATH automatically

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Start on boot
```

#### Creating a Database User and Password

**macOS/Linux:**
```bash
# Connect to PostgreSQL as the default superuser
psql postgres

# Inside the PostgreSQL prompt, create a new user with a password
CREATE USER your_username WITH PASSWORD 'your_password';

# Grant the user permission to create databases
ALTER USER your_username CREATEDB;

# Exit PostgreSQL
\q
```

**Windows:**
```bash
# Open Command Prompt or PowerShell and connect as postgres user
psql -U postgres

# Enter the password you set during installation, then:
CREATE USER your_username WITH PASSWORD 'your_password';
ALTER USER your_username CREATEDB;
\q
```

#### Installing pgvector Extension

**macOS:**
```bash
brew install pgvector
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install postgresql-15-pgvector
```

**Windows:**
Follow the instructions at https://github.com/pgvector/pgvector#windows

#### Creating the Database

```bash
# Connect to PostgreSQL
psql -U your_username -d postgres

# Create the database
CREATE DATABASE resume_optimizer;

# Connect to the new database
\c resume_optimizer

# Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Exit
\q
```

#### Initialize the Schema

```bash
psql -U your_username -d resume_optimizer -f scripts/init_db.sql
```

**Troubleshooting:**
- If you get "connection refused", make sure PostgreSQL is running: `brew services list` (macOS) or `sudo systemctl status postgresql` (Linux)
- If you get "role does not exist", you may need to create your user first (see above)
- If you get "database does not exist", make sure you created the `resume_optimizer` database

### 6. Configure environment variables

Create a `.env` file and edit `.env` with your database credentials:

```
DB_NAME=resume_optimizer
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Select the source platform in the sidebar (LinkedIn, Indeed, etc.)
2. Paste the full job listing text into the text area
3. Click "Process Job Listing"
4. Review and edit the extracted information
5. Click "Confirm and Save" to store the listing

## Project Structure

```
resume_automation/
├── app.py              # Main Streamlit application
├── models.py           # Data models (JobListing, etc.)
├── processing.py       # Text extraction and cleaning logic
├── db_utils.py         # Database operations
├── scripts/
│   └── init_db.sql     # Database schema
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variables template
```

## Tech Stack

- **Frontend:** Streamlit
- **Database:** PostgreSQL + pgvector
- **NLP:** spaCy, Sentence Transformers
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

## Development

Run tests:

```bash
pytest test_processing.py
```
