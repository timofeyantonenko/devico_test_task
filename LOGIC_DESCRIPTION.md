# Test Case Generator - Logic Description

## Overview/Summary
1. Read html/json files
2. Ask LLM to generate test cases
3. Ask LLM one more time to check the generated test cases
4. Save summary result to a json file and also to excel and json for each analyzed page.

## Main Process Flow

### 1. Input Processing
- Reads web page data from the `INPUT_PATH` directory
- Each page should have:
  - HTML content
  - Interactive elements description (JSON)
  - Basic page description
- Routes are read from `app_url_routes.json`

### 2. Test Case Generation
For each page, the system:
1. **Generates Page-Level Tests**
   - Analyzes the entire page structure
   - Creates test cases for main user flows
   - Considers functional testing aspects

2. **Generates Element-Level Tests**
   - Creates specific tests for each interactive element
   - Considers element context within the page

3. **Expert Validation**
   - Uses an "expert" LLM to review all test cases
   - Removes duplicates and improves quality
   - Validates priorities and categorization

### 3. Output Generation
The system produces:
1. **JSON Files**
   - Complete test cases with metadata
   - One file per page plus a summary file

2. **Excel Spreadsheets**
   - Formatted test documentation
   - One spreadsheet per page

### 4. Cost Tracking
- Monitors API usage
- Tracks token consumption
- Provides cost summary in the output

## Test Case Structure
Each test case includes:
- Title
- Steps
- Expected Result
- Priority (High/Medium/Low)
- Category
- Estimated Time
