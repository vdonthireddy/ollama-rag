brew install --cask miniconda
conda create --name vj python=3.9 --yes
conda init zsh
# source activate vj
conda activate vj

# You may need to restart your terminal
python --version

cd /Users/donthireddy/code/ollama/read-from-document
pip install -r requirements.txt

# Make sure the docker is running
# docker pull ollama/ollama
# docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# docker exec -it ollama ollama pull llama3
# docker exec -it ollama ollama pull nomic-embed-text

# If you want to run ollama in terminal, run the following command and you should see a prompt for you to ask questions
# docker exec -it ollama ollama run llama3