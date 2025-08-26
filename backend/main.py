from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import hashlib
import pickle
import io
import re
import calendar
from datetime import date, timedelta, datetime
from functools import lru_cache
import asyncio
import uvicorn

app = FastAPI(title="Rail Complaints Analytics API", version="1.0.0")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for processed data
data_cache: Dict[str, Any] = {}
file_cache: Dict[str, pd.DataFrame] = {}

# Pydantic models for request/response
class FilterParams(BaseModel):
    start_date: str
    end_date: str
    subtype: str = "All Categories"
    train: str = "All Trains"
    year: str = "All Years"
    remove_duplicates: bool = False

class StatsResponse(BaseModel):
    total_complaints: int
    unique_stations: int
    unique_trains: int
    top_category: str
    avg_daily: float
    duplicate_info: str = ""

class TableRow(BaseModel):
    rank: int
    station: str
    train: str
    complaints: int
    year_data: Optional[Dict[str, int]] = None

class TimelinePoint(BaseModel):
    date: str
    incidents: int
    period: Optional[str] = None
    year: Optional[str] = None

# ========== UTILITY FUNCTIONS ==========

def get_file_hash(content: bytes) -> str:
    """Generate hash for file content."""
    return hashlib.md5(content).hexdigest()

@lru_cache(maxsize=100)
def parse_dates_optimized(date_str: str) -> pd.Timestamp:
    """Optimized date parsing."""
    return pd.to_datetime(date_str, dayfirst=True, errors='coerce')

def load_and_process_data(file_content: bytes, file_name: str) -> tuple[pd.DataFrame, int]:
    """Load and process CSV data."""
    file_buffer = io.BytesIO(file_content)
    
    # Optimized dtypes
    dtype_dict = {
        "userMobile": "category",
        "pnrUtsNo": "category", 
        "contactId": "category",
        "subTypeName": "category",
        "trainStation": "category",
        "trainNameForReport": "category",
        "complaintRefNo": "string"
    }
    
    try:
        df = pd.read_csv(file_buffer, dtype=dtype_dict, low_memory=False)
    except (ValueError, KeyError):
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer, low_memory=False)
        
        # Convert existing columns to category
        for col, dtype in dtype_dict.items():
            if col in df.columns:
                if dtype == "category":
                    df[col] = df[col].astype("category")
                elif dtype == "string":
                    df[col] = df[col].astype("string")
    
    # Extract year from filename
    year_match = re.search(r'20\d{2}', file_name)
    file_year = int(year_match.group()) if year_match else 2024
    df['file_year'] = file_year
    
    # Process dates
    if "createdOn" in df.columns:
        df["createdOn"] = pd.to_datetime(df["createdOn"], dayfirst=True, errors='coerce')
        df["incident_date"] = df["createdOn"].dt.date
        df["month"] = df["createdOn"].dt.month
        df["year"] = df["createdOn"].dt.year
        df["month_year"] = df["createdOn"].dt.to_period('M')
    
    # Handle missing values
    cat_columns = ["subTypeName", "trainStation", "trainNameForReport", "pnrUtsNo", "contactId"]
    for col in cat_columns:
        if col in df.columns:
            if df[col].dtype == "category":
                if "Unknown" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(["Unknown"])
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna("Unknown")
    
    return df, file_year

def deduplicate_complaints(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate complaints."""
    if not all(col in df.columns for col in ["contactId", "incident_date", "subTypeName"]):
        return df, 0
    
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=['contactId', 'incident_date', 'subTypeName'], keep='first')
    removed_count = original_count - len(df_deduped)
    
    return df_deduped, removed_count

def filter_data(df: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    """Apply filters to dataframe."""
    filtered = df.copy()
    
    # Date filtering
    start_date = datetime.strptime(params.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(params.end_date, "%Y-%m-%d").date()
    
    if "incident_date" in filtered.columns:
        filtered = filtered[
            (filtered["incident_date"] >= start_date) & 
            (filtered["incident_date"] <= end_date)
        ]
    
    # Category filtering
    if params.subtype != "All Categories":
        filtered = filtered[filtered["subTypeName"] == params.subtype]
    
    # Train filtering
    if params.train != "All Trains" and "trainNameForReport" in filtered.columns:
        filtered = filtered[filtered["trainNameForReport"] == params.train]
    
    # Year filtering
    if params.year != "All Years" and "file_year" in filtered.columns:
        filtered = filtered[filtered["file_year"] == int(params.year)]
    
    return filtered

# ========== API ENDPOINTS ==========

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process CSV files."""
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 files allowed")
    
    all_dfs = []
    file_years = []
    file_hashes = []
    
    for file in files:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")
        
        content = await file.read()
        file_hash = get_file_hash(content)
        
        # Check cache first
        if file_hash in file_cache:
            df = file_cache[file_hash]
            file_year = df['file_year'].iloc[0]
        else:
            df, file_year = load_and_process_data(content, file.filename)
            file_cache[file_hash] = df
        
        all_dfs.append(df)
        file_years.append(file_year)
        file_hashes.append(file_hash)
    
    # Combine dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True, copy=False)
    
    # Store in cache
    cache_key = "_".join(file_hashes)
    data_cache[cache_key] = {
        'dataframe': combined_df,
        'file_years': file_years,
        'timestamp': datetime.now()
    }
    
    # Validate required columns
    required_cols = {"createdOn", "subTypeName", "trainStation", "trainNameForReport", "complaintRefNo"}
    missing = required_cols - set(combined_df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(sorted(missing))}")
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    total_records = int(combined_df.shape[0])
    
    # Handle date range conversion
    date_range = {"min": None, "max": None}
    if "incident_date" in combined_df.columns:
        min_date = combined_df["incident_date"].min()
        max_date = combined_df["incident_date"].max()
        if pd.notna(min_date):
            date_range["min"] = min_date.strftime('%Y-%m-%d')
        if pd.notna(max_date):
            date_range["max"] = max_date.strftime('%Y-%m-%d')
    
    return {
        "cache_key": cache_key,
        "total_records": total_records,
        "file_years": [int(year) for year in file_years],  # Ensure integers
        "date_range": date_range
    }

@app.get("/filter-options/{cache_key}")
async def get_filter_options(cache_key: str):
    """Get available filter options."""
    if cache_key not in data_cache:
        raise HTTPException(status_code=404, detail="Data not found")
    
    df = data_cache[cache_key]['dataframe']
    file_years = data_cache[cache_key]['file_years']
    
    # Sort subtypes by frequency (most complaints first)
    subtype_counts = df["subTypeName"].value_counts()
    sorted_subtypes = ["All Categories"] + [str(item) for item in subtype_counts.index.tolist()]
    
    # Use trainNameForReport instead of pnrUtsNo for better data availability
    train_options = ["All Trains"]
    if "trainNameForReport" in df.columns:
        train_counts = df["trainNameForReport"].value_counts()
        train_options += [str(item) for item in train_counts.index.tolist()]
    
    # Convert file_years to native Python integers
    years_list = ["All Years"] + [str(year) for year in sorted([int(year) for year in file_years], reverse=True)]
    
    # Handle months conversion
    months_list = []
    if "month_year" in df.columns:
        unique_months = df["month_year"].dropna().unique()
        months_list = [f"{calendar.month_name[m.month]} {m.year}" for m in sorted(unique_months)]
    
    return {
        "subtypes": sorted_subtypes,
        "trains": train_options,
        "years": years_list,
        "months": months_list
    }

@app.post("/analytics/{cache_key}")
async def get_analytics(cache_key: str, params: FilterParams):
    """Get analytics data based on filters."""
    if cache_key not in data_cache:
        raise HTTPException(status_code=404, detail="Data not found")
    
    df = data_cache[cache_key]['dataframe']
    file_years = data_cache[cache_key]['file_years']
    
    # Apply deduplication if requested
    duplicate_info = ""
    if params.remove_duplicates:
        df, removed_count = deduplicate_complaints(df)
        if removed_count > 0:
            duplicate_info = f"Removed {removed_count:,} duplicate complaints ({removed_count/(len(df)+removed_count)*100:.1f}%)"
    
    # Apply filters
    filtered = filter_data(df, params)
    
    # Calculate statistics
    stats = StatsResponse(
        total_complaints=len(filtered),
        unique_stations=filtered["trainStation"].nunique() if not filtered.empty else 0,
        unique_trains=filtered["trainNameForReport"].nunique() if not filtered.empty else 0,
        top_category=filtered["subTypeName"].value_counts().index[0] if not filtered.empty else "N/A",
        avg_daily=0,
        duplicate_info=duplicate_info
    )
    
    # Calculate average daily
    start_date = datetime.strptime(params.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(params.end_date, "%Y-%m-%d").date()
    days_diff = (end_date - start_date).days + 1
    if days_diff > 0:
        stats.avg_daily = stats.total_complaints / days_diff
    
    return {
        "stats": stats,
        "has_data": not filtered.empty
    }

@app.post("/table/{cache_key}")
async def get_table_data(
    cache_key: str, 
    params: FilterParams,
    max_rows: int = Query(20, ge=10, le=100)
):
    """Get table data for complaints by station and train."""
    if cache_key not in data_cache:
        raise HTTPException(status_code=404, detail="Data not found")
    
    df = data_cache[cache_key]['dataframe']
    file_years = data_cache[cache_key]['file_years']
    
    # Apply deduplication if requested
    if params.remove_duplicates:
        df, _ = deduplicate_complaints(df)
    
    # Apply filters
    filtered = filter_data(df, params)
    
    if filtered.empty:
        return {"rows": [], "total_combinations": 0}
    
    # Multi-year comparison logic
    if len(file_years) > 1 and params.year == "All Years":
        # Group by station, train, and year
        grouped = filtered.groupby(['trainStation', 'trainNameForReport', 'file_year']).size()
        pivot_data = grouped.unstack(fill_value=0)
        pivot_data['Total'] = pivot_data.sum(axis=1)
        table_df = pivot_data.reset_index()
        
        # Sort by latest year or total
        latest_year = max(file_years)
        if latest_year in table_df.columns:
            table_df = table_df.sort_values([latest_year, 'trainStation'], ascending=[False, True])
        else:
            table_df = table_df.sort_values(['Total', 'trainStation'], ascending=[False, True])
        
        table_df = table_df.head(max_rows)
        
        # Convert to response format
        rows = []
        for idx, row in table_df.iterrows():
            year_data = {}
            for year in file_years:
                if year in table_df.columns:
                    year_data[str(year)] = int(row[year])
            
            rows.append(TableRow(
                rank=len(rows) + 1,
                station=row['trainStation'],
                train=row['trainNameForReport'],
                complaints=int(row['Total']),
                year_data=year_data
            ))
    else:
        # Single year or filtered year
        station_train_agg = (
            filtered.groupby(["trainStation", "trainNameForReport"])
            .size()
            .reset_index(name="complaints")
            .sort_values(["complaints", "trainStation"], ascending=[False, True])
            .head(max_rows)
        )
        
        rows = []
        for idx, row in station_train_agg.iterrows():
            rows.append(TableRow(
                rank=len(rows) + 1,
                station=row['trainStation'],
                train=row['trainNameForReport'],
                complaints=int(row['complaints'])
            ))
    
    total_combinations = filtered.groupby(['trainStation', 'trainNameForReport']).ngroups
    
    return {
        "rows": [row.dict() for row in rows],
        "total_combinations": total_combinations
    }

@app.post("/timeline/{cache_key}")
async def get_timeline_data(
    cache_key: str,
    params: FilterParams,
    selected_station: Optional[str] = None,
    selected_train: Optional[str] = None
):
    """Get timeline data for charts."""
    if cache_key not in data_cache:
        raise HTTPException(status_code=404, detail="Data not found")
    
    df = data_cache[cache_key]['dataframe']
    file_years = data_cache[cache_key]['file_years']
    
    # Apply deduplication if requested
    if params.remove_duplicates:
        df, _ = deduplicate_complaints(df)
    
    # Apply filters
    filtered = filter_data(df, params)
    
    # Apply selection filter if provided
    if selected_station and selected_train:
        filtered = filtered[
            (filtered["trainStation"] == selected_station) &
            (filtered["trainNameForReport"] == selected_train)
        ]
    
    if filtered.empty or "createdOn" not in filtered.columns:
        return {"timeline": [], "multi_year": False}
    
    # Determine time granularity
    start_date = datetime.strptime(params.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(params.end_date, "%Y-%m-%d").date()
    date_range_days = (end_date - start_date).days
    
    if date_range_days <= 31:
        resample_rule = "D"
        time_unit = "Daily"
    elif date_range_days <= 90:
        resample_rule = "W"
        time_unit = "Weekly"
    else:
        resample_rule = "MS"
        time_unit = "Monthly"
    
    # Multi-year timeline - overlapping approach for easier comparison
    if len(file_years) > 1 and params.year == "All Years":
        # Create a comprehensive period range covering all years
        all_periods = set()
        year_data_dict = {}
        
        # First pass: collect all periods and data for each year
        for year in sorted(file_years):
            year_data = filtered[filtered["file_year"] == year]
            if not year_data.empty:
                year_clean = year_data[year_data["createdOn"].notna()].copy()
                if not year_clean.empty:
                    year_clean = year_clean.set_index("createdOn")
                    timeline = year_clean.resample(resample_rule).size().reset_index(name="incidents")
                    
                    # Create period mapping for this year
                    year_periods = {}
                    for _, point in timeline.iterrows():
                        if date_range_days <= 90:
                            period_key = point["createdOn"].strftime("%b")  # Just month name
                        else:
                            period_key = point["createdOn"].strftime("%b")  # Month name for overlapping
                        
                        period_sort_key = point["createdOn"].strftime("%m")  # For sorting
                        all_periods.add((period_key, period_sort_key))
                        year_periods[period_key] = int(point["incidents"])
                    
                    year_data_dict[year] = year_periods
        
        # Sort periods chronologically
        sorted_periods = sorted(all_periods, key=lambda x: x[1])  # Sort by month number
        
        # Create timeline data with overlapping structure
        timeline_data = []
        for period_key, _ in sorted_periods:
            period_entry = {"period": period_key}
            
            # Add data for each year
            for year in sorted(file_years):
                year_key = f"year_{year}"
                incidents = year_data_dict.get(year, {}).get(period_key, 0)
                period_entry[year_key] = incidents
            
            # Only include periods where at least one year has data
            if any(period_entry[f"year_{year}"] > 0 for year in file_years):
                timeline_data.append(period_entry)
        
        return {
            "timeline": timeline_data,
            "multi_year": True,
            "time_unit": time_unit,
            "years": [str(year) for year in sorted(file_years)]  # Available years for the frontend
        }
    
    else:
        # Single year timeline
        filtered_clean = filtered[filtered["createdOn"].notna()].copy()
        filtered_clean = filtered_clean.set_index("createdOn")
        timeline = filtered_clean.resample(resample_rule).size().reset_index(name="incidents")
        
        timeline_points = []
        for _, point in timeline.iterrows():
            timeline_points.append(TimelinePoint(
                date=point["createdOn"].isoformat(),
                incidents=int(point["incidents"])
            ))
        
        return {
            "timeline": [point.dict() for point in timeline_points],
            "multi_year": False,
            "time_unit": time_unit
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "cache_size": len(data_cache)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)