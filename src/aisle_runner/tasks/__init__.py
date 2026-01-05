"""Task definitions for browser automation benchmarks."""

from ..models import Task

# Sample tasks for benchmarking
TASKS: dict[str, Task] = {
    "google-search": Task(
        id="google-search",
        name="Google Search",
        description="Navigate to Google, search for 'OpenAI GPT-5', and click on the first result.",
        start_url="https://www.google.com",
        success_criteria=[
            {"type": "url_contains", "value": "google.com/search"},
            {"type": "action_completed", "value": "click_result"},
        ],
        max_steps=10,
        timeout_seconds=120,
        tags=["search", "simple"],
    ),
    "wikipedia-navigate": Task(
        id="wikipedia-navigate",
        name="Wikipedia Navigation",
        description="Go to Wikipedia, search for 'Artificial Intelligence', and navigate to the 'Machine Learning' link within the article.",
        start_url="https://www.wikipedia.org",
        success_criteria=[
            {"type": "url_contains", "value": "Machine_learning"},
        ],
        max_steps=15,
        timeout_seconds=180,
        tags=["navigation", "simple"],
    ),
    "hackernews-top": Task(
        id="hackernews-top",
        name="Hacker News Top Story",
        description="Go to Hacker News, find the top story, and click on its comments link.",
        start_url="https://news.ycombinator.com",
        success_criteria=[
            {"type": "url_contains", "value": "item?id="},
        ],
        max_steps=10,
        timeout_seconds=120,
        tags=["navigation", "simple"],
    ),
    "github-repo": Task(
        id="github-repo",
        name="GitHub Repository Search",
        description="Go to GitHub, search for 'browser-use', and click on the first repository result.",
        start_url="https://github.com",
        success_criteria=[
            {"type": "url_pattern", "value": r"github\.com/[\w-]+/[\w-]+"},
        ],
        max_steps=15,
        timeout_seconds=180,
        tags=["search", "navigation"],
    ),
    "form-fill": Task(
        id="form-fill",
        name="Form Filling",
        description="Go to the test form page and fill out the contact form with: Name='Test User', Email='test@example.com', Message='This is a test message'. Then submit the form.",
        start_url="https://www.w3schools.com/html/html_forms.asp",
        success_criteria=[
            {"type": "form_submitted", "value": True},
        ],
        max_steps=20,
        timeout_seconds=180,
        tags=["forms", "input"],
    ),
    "multi-step-checkout": Task(
        id="multi-step-checkout",
        name="Multi-step Checkout Flow",
        description="Navigate to an e-commerce demo site, add a product to cart, proceed to checkout, and fill in shipping information.",
        start_url="https://demo.opencart.com",
        success_criteria=[
            {"type": "url_contains", "value": "checkout"},
            {"type": "step_reached", "value": "shipping"},
        ],
        max_steps=30,
        timeout_seconds=300,
        tags=["e-commerce", "complex", "multi-step"],
    ),
}


def get_task(task_id: str) -> Task | None:
    """Get a task by ID."""
    return TASKS.get(task_id)


def get_tasks_by_tag(tag: str) -> list[Task]:
    """Get all tasks with a specific tag."""
    return [t for t in TASKS.values() if tag in t.tags]


def list_tasks() -> list[Task]:
    """List all available tasks."""
    return list(TASKS.values())


__all__ = ["TASKS", "get_task", "get_tasks_by_tag", "list_tasks"]
