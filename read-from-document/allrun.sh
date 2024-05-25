brew install --cask miniconda
conda create --name vj python=3.9 --yes
source activate vj
conda activate vj
python --version
pip install -r requirements.txt
