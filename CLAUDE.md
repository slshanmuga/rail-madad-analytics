# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a full-stack web application for analyzing railway complaints data. The application consists of:

- **Backend**: FastAPI Python server with data processing capabilities
- **Frontend**: React application with Ant Design components for data visualization

## Architecture

### Backend (FastAPI)
- **Entry point**: `backend/main.py` - Main FastAPI application with CORS configuration
- **Core functionality**: 
  - CSV file upload and processing with pandas
  - In-memory caching system using file hashes
  - Data filtering, deduplication, and analytics
  - RESTful API endpoints for data operations
- **Key features**:
  - Multi-file upload support (up to 3 CSV files)
  - Year-over-year comparison capabilities
  - Optimized data processing with categorical dtypes
  - Timeline data aggregation with configurable granularity

### Frontend (React + Ant Design)
- **Entry point**: `frontend/src/App.js` - Main React component
- **Key components**:
  - File upload interface
  - Interactive filter sidebar with date ranges, categories, trains, and years
  - Statistics dashboard with key metrics
  - Data table with clickable rows for drill-down
  - Timeline charts (single-year area charts, multi-year line charts)
- **Visualization**: Uses Recharts for charting with responsive design

## Development Commands

### Backend
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
# Server runs on http://localhost:8000

# Run tests
pytest

# Production server (optional)
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
# Runs on http://localhost:3000 with proxy to backend

# Build for production
npm run build

# Run tests
npm test
```

## Data Requirements

The application expects CSV files with these columns:
- `createdOn` - Complaint timestamp (required)
- `subTypeName` - Complaint category (required)
- `trainStation` - Station name (required)
- `trainNameForReport` - Train identifier (required)
- `complaintRefNo` - Reference number (required)
- `contactId` - User identifier for deduplication (optional)
- `pnrUtsNo` - Train number for filtering (optional)

## Key API Endpoints

- `POST /upload` - Upload and process CSV files
- `GET /filter-options/{cache_key}` - Get available filter options
- `POST /analytics/{cache_key}` - Get statistics and analytics
- `POST /table/{cache_key}` - Get paginated table data
- `POST /timeline/{cache_key}` - Get timeline/chart data
- `GET /health` - Health check

## Development Notes

- Backend uses file content hashing for efficient caching
- Frontend state management uses React hooks (no external state library)
- The application supports multi-year data comparison when multiple CSV files are uploaded
- Date parsing is optimized using pandas with dayfirst=True for Indian date formats
- CORS is configured to allow frontend development server access