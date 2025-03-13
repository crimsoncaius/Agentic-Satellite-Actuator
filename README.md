# Demo Setup Guide

This guide helps you set up a **FastAPI backend** (running on port **8000** for REST API and **5015** for WebSockets) and a **Vite React frontend** (running on port **5173**).

---

## 1. Prerequisites

### Ensure You Are in the Root Directory

Before proceeding, confirm you are in the root directory of the project:

```
# Navigate to the root directory if not already there
cd /path/to/project
```

### Install Conda (if not installed)

If you don't have Conda installed, follow these steps:

#### **Windows:**

Download and install Miniconda from: 👉 [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

⚠ **Important:** During installation, select **"Add Conda to PATH"**.

If you skip this step, Conda commands won't work in your terminal.

After installation, restart your terminal and verify the installation:

```
conda --version
```

#### **Mac/Linux:**

Install Miniconda with:

```
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh
```

Restart the terminal and verify installation:

```
conda --version
```

### Install Python (if not installed)

Check if Python is installed:

```
python --version
```

If not installed, download and install it from: 👉 [Python Downloads](https://www.python.org/downloads/)

### Install Node.js (if not installed)

Check if Node.js is installed:

```
node -v
```

If not installed, download and install it from: 👉 [Node.js Downloads](https://nodejs.org/)

---

## 2. Backend Setup (FastAPI + Uvicorn with Conda)

### **1. Create and Activate Conda Environment**

```
# Navigate to backend directory
cd backend

# Create a Conda environment named "backend-env"
conda create --name backend-env python=latest -y

# Activate the environment
conda activate backend-env
```

### **2. Install Dependencies**

```
pip install -r requirements.txt
```

### **3. Run the Backend Server**

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **REST API runs on:** [http://localhost:8000](http://localhost:8000)  
✅ **WebSocket Server runs on port:** `5015`

---

## 3. Frontend Setup (Vite + React)

### **1. Install Dependencies**

```
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install
```

### **2. Run the Frontend**

```
npm run dev
```

✅ **Frontend runs on:** [http://localhost:5173](http://localhost:5173)

📌 Ensure WebSocket connections are directed to `ws://localhost:5015`.

---

## 4. WebSocket Port Warning ⚠

Your backend WebSocket server runs on port `5015`. If the port is already in use, the server may fail to start.

### **Check if Port 5015 is Open**

Run this command before starting the backend:

#### **Windows:**

```
netstat -ano | findstr :5015
```

#### **Mac/Linux:**

```
lsof -i :5015
```

### **If the port is in use, stop the process:**

#### **Windows:**

```
taskkill /PID <PID> /F
```

#### **Mac/Linux:**

```
kill -9 <PID>
```

---

## 5. Run the Unity Game

The full-stack application interfaces with the Unity game through a socket connection on port **5015**.

Email me at caiusaaronchew@gmail.com if you want access to the Unity Game, it's not stored here

### **Windows:**

1. Navigate to the Unity game directory:
    
    ```
    cd GSTCE_2025_demo
    ```
    
2. Run the Unity game executable:
    
    ```
    ./Unity\ Physics.exe
    ```
    

### **Mac:**

1. Navigate to the Unity game directory:
    
    ```
    cd GSTCE_2025_demo_mac
    ```
    
2. Run the Unity game:
    
    ```
    open GSTCE_2025_demo_mac.app
    ```
    

---

### ✅ **You're all set!** Now your demo should be running smoothly! 🚀

# Agentic Satellite Actuator

This project provides a system for controlling a satellite using natural language commands processed through LLMs.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   - Copy the `backend/.env.example` file to `backend/.env`
   - Fill in your API keys and configuration values

   ```
   cd backend
   cp .env.example .env
   # Then edit the .env file with your actual API keys
   ```

3. Run the application:
   ```
   python backend/llm_control.py
   ```

## Environment Variables

The following environment variables are used in this project (stored in `backend/.env`):

- `OPENAI_API_KEY`: The active OpenAI API key
- `PLAYDIALOG_API_KEY`: API key for PlayDialog text-to-speech service
- `PLAYDIALOG_USER_ID`: User ID for PlayDialog service
- `SOCKET_PORT`: Port number for socket communication (default: 5015)

## Usage

The system accepts text commands to control a satellite. You can:
- Point the satellite to specific orientations
- Add, modify, or delete orientation points
- Get status reports about the satellite's position, velocity, etc.
- Reset the simulation

Example commands:
- "Point to e1"
- "Add new orientation point mars with angles 120, 45, 90"
- "Get current orientation"
- "Reset simulation"
