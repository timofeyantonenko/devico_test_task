import json
from unittest.mock import Mock, patch

import pytest

from src.main import (
    MODEL_NAME,
    TestCaseGenerator,
    TestGenerationResultWithMetadata,
    TestPerspective,
)


class MockUsageInfo:
    """Mock class for OpenAI usage information"""

    @property
    def prompt_tokens(self):
        return 100

    @property
    def completion_tokens(self):
        return 50

    @property
    def total_tokens(self):
        return 150

    @property
    def prompt_tokens_details(self):
        class Details:
            cached_tokens = 20

        return Details()


@pytest.fixture
def sample_test_cases():
    return [
        {
            "title": "Verify login form submission",
            "steps": ["Navigate to login page", "Enter credentials", "Click submit"],
            "result": ["User should be logged in", "Dashboard should be displayed"],
            "priority": "High",
            "category": "functional",
            "estimated_time": "quick",
        }
    ]


@pytest.fixture
def mock_page_data():
    return {
        "description": "Login page",
        "html": "<form id='login-form'>...</form>",
        "elements": {
            "username": {"type": "input", "required": True},
            "password": {"type": "input", "required": True},
        },
    }


@pytest.fixture
def mock_routes():
    return {
        "privateRoutes": [
            {
                "path": "/dashboard",
                "children": [{"path": "/analytics"}],
            }
        ],
        "publicRoutes": [{"path": "/login"}],
    }


@pytest.fixture
def test_generator(tmp_path):
    """Create a TestCaseGenerator instance with temporary directory"""
    # Create necessary files in temporary directory
    input_path = tmp_path / "initial_data"
    input_path.mkdir()

    # Create app_url_routes.json
    routes_file = input_path / "app_url_routes.json"
    routes_file.write_text('{"privateRoutes": [], "publicRoutes": []}')

    # Create app_description.txt
    desc_file = input_path / "app_description.txt"
    desc_file.write_text("Test application description")

    return TestCaseGenerator(str(input_path))


def test_load_json(test_generator, tmp_path):
    """Test loading JSON file"""
    test_data = {"test": "data"}
    json_file = tmp_path / "initial_data" / "test.json"
    json_file.write_text(json.dumps(test_data))

    result = test_generator._load_json("test.json")
    assert result == test_data


def test_load_text(test_generator, tmp_path):
    """Test loading text file"""
    test_content = "Test content"
    text_file = tmp_path / "initial_data" / "test.txt"
    text_file.write_text(test_content)

    result = test_generator._load_text("test.txt")
    assert result == test_content


def test_extract_page_paths(test_generator, mock_routes):
    """Test extraction of page paths from routes"""
    test_generator.routes = mock_routes
    paths = test_generator._extract_page_paths(mock_routes)

    assert "dashboard" in paths
    assert "dashboard_analytics" in paths
    assert "login" in paths


def test_load_page_data(test_generator, tmp_path, mock_page_data):
    """Test loading page data from directory"""
    page_dir = tmp_path / "initial_data" / "login"
    page_dir.mkdir(parents=True)

    # Create test files
    html_file = page_dir / "page.html"
    html_file.write_text(mock_page_data["html"])

    json_file = page_dir / "elements.json"
    json_file.write_text(json.dumps(mock_page_data["elements"]))

    desc_file = page_dir / "page_description.txt"
    desc_file.write_text(mock_page_data["description"])

    result = test_generator._load_page_data("login")
    assert "html" in result
    assert "elements" in result
    assert "description" in result
    assert result["elements"] == mock_page_data["elements"]


@patch("src.main.OpenAI")
def test_generate_test_cases_with_retry(mock_openai, test_generator):
    """Test test case generation with retry logic"""
    # Create the mock response
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content=json.dumps(
                    {
                        "test_cases": [
                            {
                                "title": "Test Case",
                                "steps": ["Step 1", "Step 2"],
                                "result": ["Expected result 1"],
                                "priority": "High",
                                "category": "functional",
                                "estimated_time": "quick",
                            }
                        ]
                    }
                )
            )
        )
    ]
    mock_response.usage = MockUsageInfo()

    # Set up the OpenAI client mock
    mock_client = Mock()
    mock_client.beta.chat.completions.parse.return_value = mock_response
    mock_openai.return_value = mock_client
    test_generator._client = mock_client

    # Run the test
    result = test_generator._generate_test_cases_with_retry("Test prompt", 0.7)

    # Verify the result
    assert isinstance(result, TestGenerationResultWithMetadata)
    assert len(result.result.test_cases) > 0
    assert result.metadata.model_version == MODEL_NAME


def test_create_page_prompt(test_generator, mock_page_data):
    """Test creation of page-level prompt"""
    perspective = Mock()
    perspective.name = TestPerspective.FUNCTIONAL
    perspective.temperature = 0.7

    prompt = test_generator._create_page_prompt(mock_page_data, perspective)

    assert "Page Description" in prompt
    assert "Form Elements" in prompt
    assert "HTML Content" in prompt
    assert "FUNCTIONAL" in prompt


def test_extract_element_context(test_generator):
    """Test extraction of element context"""
    element_id = "test-input"
    element_data = {"type": "input", "required": True}
    html_content = "<div><input id='test-input' required /></div>"

    context = test_generator._extract_element_context(
        element_id, element_data, html_content
    )

    assert context.element_id == element_id
    assert context.element_type == "input"
    assert context.element_data == element_data
    assert context.surrounding_html == html_content


@pytest.mark.parametrize(
    "usage,model,expected_total",
    [
        (
            Mock(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                prompt_tokens_details=Mock(cached_tokens=200),
            ),
            "gpt-4o",
            0.00725,
        ),
    ],
)
def test_openai_api_calculate_cost(usage, model, expected_total):
    """Test OpenAI API cost calculation"""
    from src.utils import openai_api_calculate_cost

    result = openai_api_calculate_cost(usage, model)
    assert isinstance(result, dict)
    assert "total_cost" in result
    assert (
        abs(result["total_cost"] - expected_total) < 0.0001
    )  # Allow for small float differences
