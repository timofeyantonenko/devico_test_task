# Test cases generator
This repo represents Tymofii Antonenko's implementation of test task by Devico.

## Prerequesites/tech details
1. Python 3.9+. In `Makefile` I'm using Python 3.11. If you don't have 3.11,
please select any other version which you have locally and which is at least 3.9 or newer. It's due to typing compitability.
2. OpenAI API key [https://platform.openai.com/api-keys]


## How to run (assuming you are on some Linux distro or on Mac and you have `make` command)
1. `make install`
2. Add `OPENAI_API_KEY` env variable into generated `.env` file
3. Check and modify if necessary `INPUT_PATH` and `OUTPUT_PATH` into `.env` file
4. Place your inputs into `INPUT_PATH` (`initial_data` by default). Basically you should add direcotries, which contain:
 - `.json` file with extracted interactive elements
 - `.html` code of simplified web pages
 - `page_description.txt` which gives a basic information about the page
5. `make run`
4. Check `OUTPUT_PATH` (`output` by default) for the results.



## Assumptions
1. I assumed that for one route, its' HTML and JSON would fit into models' context window.
2. I didn't use Langchain etc., to simplify the dependencies of the project.
3. I assumed you have OpenAI API key


## How I selected LLM for this task:
1) It was recommended to use OpanAI LLMs
2) I wanted to use structured output feature
3) I took a look at a few leaderboards: (alpaca_eval)[https://tatsu-lab.github.io/alpaca_eval/], (vellum)[https://www.vellum.ai/llm-leaderboard] and (lmarena)[https://lmarena.ai/?leaderboard] and (aider)[https://aider.chat/docs/leaderboards/]


## TODO / Potential improvements
1) Handle case when html is too big and doesn't fit into one prompt
2) Add usage of multithreading/multiprocessing.  
I didn't do it because I already was getting `429 - Too Many Requests`.