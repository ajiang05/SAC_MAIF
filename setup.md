# Project Setup

Follow these instructions to set up your local development environment for the SAC_MAIF trading bot project.

## Prerequisites
- Python 3.9+ 
- macOS/Linux/Windows terminal

## 1. Create a Virtual Environment
It's highly recommended to run your code within a Python virtual environment to avoid package conflicts with your system or other projects. 

Run the following command in the root directory of the project:
```bash
python3 -m venv venv
```

## 2. Activate the Virtual Environment
Activate the virtual environment so that your terminal uses the local project dependencies instead of your global Python installation.

**On macOS and Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
.\venv\Scripts\activate
```
*(Note: Your terminal prompt should typically change to show `(venv)` indicating it is activated.)*

## 3. Install Dependencies
Once the virtual environment is activated, install the required packages using pip:
```bash
pip install -r requirements.txt
```

## 4. Deactivating (Optional)
When you're completely done working on the project, you can exit the virtual environment by simply running:
```bash
deactivate
```

You are now fully set up and ready to start coding!
