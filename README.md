# devico_test_task
devico_test_task

## Prerequesites/tech details
1. Python 3.11+
2. OpenAI API key [https://platform.openai.com/api-keys]
2. If installing dependencies fails, try to upgrade pip


## Assumptions
1. I assumed that for one route, its' HTML and JSON would fit into models' context window.
2. I didn't use Langchain etc., to simplify the dependencies of the project.
3. I assumed you have OpenAI API key



# How I selected LLM for this task:
1) It was recommended to use OpanAI LLMs
2) I wanted to use structured output feature
3) I took a look at a few leaderboards: (alpaca_eval)[https://tatsu-lab.github.io/alpaca_eval/], (vellum)[https://www.vellum.ai/llm-leaderboard] and (lmarena)[https://lmarena.ai/?leaderboard] and (aider)[https://aider.chat/docs/leaderboards/]

# TODO
1) Handle case when html is too big and doesn't fit into one prompt
2) 