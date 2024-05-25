## Install miniconda (if not already done)
```
brew install --cask miniconda
conda create --name vj python=3.9 --yes
conda init zsh
source activate vj
conda activate vj
```

## You may need to restart your terminal and try checking the python version
```
python --version
```

## Go to where the git clone is done and install all the required packages. There may be some ununsed packages (since I am testing with different document loaders...)
```
cd /Users/donthireddy/code/ollama/read-from-document
pip install -r ./requirements.txt
```

## Make sure the docker app is running on your Mac and pull the open source framework to run Large Language Models locally...
### pull the docker image ollama/ollama
### run the docker container and expose port 11434 so that application can connect to this framework via APIs
### pull all required LLMs and Embedding Models (here we are pulling mistral, llama3 and nomic-embed-text). But you can also use other models like gemma, etc.. Fora list of supported models, you can visit: https://ollama.com/library?sort=popular
```
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull mistral
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text
```

## If you want to run ollama in terminal, run the following command and you should see a prompt for you to ask questions
```
docker exec -it ollama ollama run llama3
```

## Run the code in terminal to see the response with no RAG
```
python norag.py
```

## Run the code in terminal to see the response with RAG
```
python withrag.py
```

## Please note that you may need to change the code to edit the question. Good luck and Thank you!