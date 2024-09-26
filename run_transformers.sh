
mkdir -p /var/tmp/strix
cd /var/tmp/strix

mkdir -p data

tmp_dir=$(uuidgen)

mkdir $tmp_dir
 
git clone https://github.com/spraakbanken/strix-sentence-transformers.git $tmp_dir &> /dev/null
cd $tmp_dir
echo "transformers_postprocess_dir: ../data" > config.yml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt &> /dev/null

python main.py $1

