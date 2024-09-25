import glob
import json
import logging
from pathlib import Path
import os
import sys
from sentence_transformers import SentenceTransformer
import yaml

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(FORMAT))
logging.root.setLevel(logging.INFO)
logging.root.addHandler(ch)

model = SentenceTransformer(
    "KBLab/sentence-bert-swedish-cased",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True},
)


def main(out, input):
    for line in input:
        [doc_id, text] = json.loads(line)
        vector = model.encode(text).tolist()
        out.write(f"{json.dumps([doc_id, vector], ensure_ascii=False)}\n")


if __name__ == "__main__":
    corpus = sys.argv[1]
    config = yaml.safe_load(open("config.yml"))
    for file in glob.glob(
        os.path.join(config["transformers_postprocess_dir"], corpus, "texts/*")
    ):
        with open(file) as fp:
            # check that user has set a directory for the transformers data and create directory structure
            if "transformers_postprocess_dir" not in config:
                raise RuntimeError("transformers_postprocess_dir not set in config")
            out_dir = os.path.join(
                config["transformers_postprocess_dir"], corpus, "vectors"
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(out_dir, os.path.basename(file)), "w") as fp_out:
                main(fp_out, fp)
