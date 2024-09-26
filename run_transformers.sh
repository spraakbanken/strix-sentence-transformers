logfile=~/logs/$(date '+%Y-%m-%d_%H%M%S').log

echo "Running strix-sentence-transformers on $1" >> $logfile


mkdir -p /var/tmp/strix/envs
cd /var/tmp/strix/envs

tmp_dir=$(uuidgen)

mkdir $tmp_dir
 
git clone https://github.com/spraakbanken/strix-sentence-transformers.git $tmp_dir &>> $logfile
cd $tmp_dir
echo "transformers_postprocess_dir: ../../data" > config.yml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt &>> $logfile

python main.py $1 &>> $logfile

cd ..
rm -rf $tmp_dir
