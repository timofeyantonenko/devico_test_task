# Test cases generator
This repo represents Tymofii Antonenko's implementation of test task by Devico.

## Logic of implementation
For logic details please take a look at `LOGIC_DESCRIPTION.md`.

## Prerequesites/tech details
1. Python 3.9+. In `Makefile` I'm using Python 3.11. If you don't have 3.11,
please select any other version which you have locally and which is at least 3.9 or newer. It's due to typing compitability.
2. OpenAI API key [https://platform.openai.com/api-keys]
3. `make` command (for Linux/Mac users)


## How to run

0. Clone the repository:
```bash
git clone git@github.com:timofeyantonenko/devico_test_task.git
cd devico_test_task
```
1. Run the installation:
```bash
make install
```
This will:
- Create a virtual environment
- Install dependencies
- Create a `.env` file from `.env.example`
2. Add `OPENAI_API_KEY` env variable into generated `.env` file.
```
OPENAI_API_KEY=your-api-key-here
```
3. (Optionally) Check and modify if necessary `INPUT_PATH` and `OUTPUT_PATH` into `.env` file
4. Place your inputs into `INPUT_PATH` (`initial_data` by default). Basically you should add direcotries, which contain:
```
initial_data/
├── [page_route]/
│   ├── page.html          # Simplified HTML of the page
│   ├── elements.json      # JSON file with interactive elements
│   └── page_description.txt   # Basic page information
```
The inputs will be analyses by using `app_url_routes.json`.  
So if the file doesn't have the routes - they will not be analysed.
5. `make run`
6. Check `OUTPUT_PATH` (`output` by default) for the results:
   - JSON files with test cases
   - Excel spreadsheets for each page
   - Summary reports with costs and metadata


## Assumptions
1. I assumed that for one route, its' HTML and JSON would fit into models' context window.
2. I didn't use Langchain etc., to simplify the dependencies of the project.
3. I assumed you have OpenAI API key


## How I selected LLM for this task:
1) It was recommended to use OpanAI LLMs
2) I wanted to use structured output feature
3) To choose a particular model, I took a look at a few leaderboards:
  - [Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/)
  - [Vellum](https://www.vellum.ai/llm-leaderboard)
  - [LM Arena](https://lmarena.ai/?leaderboard)
  - [Aider](https://aider.chat/docs/leaderboards/)
Based on all of this, I think `gpt-4o` should be fine as the default.


## TODO / Potential improvements
1) Handle case when html is too big and doesn't fit into one prompt
2) Add usage of multithreading/multiprocessing.  
I didn't do it because I already was getting `429 - Too Many Requests`.
