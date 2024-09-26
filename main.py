import glob
import json
import logging
from pathlib import Path
import os
import shutil
import sys
import yaml

import torch
from sentence_transformers import SentenceTransformer


FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(FORMAT))
logging.root.setLevel(logging.INFO)
logging.root.addHandler(ch)


def chunkify(lst, n):
    if n == 0:
        return [lst]
    return [(i, lst[i::n]) for i in range(n)]


def encode_lines(out, input, model):
    for line in input:
        [doc_id, text] = json.loads(line)
        vector = model.encode(text, show_progress_bar=False).tolist()
        out.write(f"{json.dumps([doc_id, vector], ensure_ascii=False)}\n")


def run_file(file, corpus, config, model):
    with open(file) as fp:
        # check that user has set a directory for the transformers data and create directory structure
        if "transformers_postprocess_dir" not in config:
            raise RuntimeError("transformers_postprocess_dir not set in config")
        out_dir = os.path.join(
            config["transformers_postprocess_dir"], corpus, "vectors"
        )
        try:
            shutil.rmtree(out_dir)
        except FileNotFoundError:
            pass
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_dir, os.path.basename(file)), "w") as fp_out:
            encode_lines(fp_out, fp, model)


def create_model(device):
    return SentenceTransformer(
        "KBLab/sentence-bert-swedish-cased",
        tokenizer_kwargs={"clean_up_tokenization_spaces": True},
        device=device,
    )


def run(n, chunk, corpus, config):
    model = create_model(n)
    for file in chunk:
        run_file(file, corpus, config, model)


def main(corpus):
    config = yaml.safe_load(open("config.yml"))
    files = glob.glob(
        os.path.join(config["transformers_postprocess_dir"], corpus, "texts/*")
    )
    if torch.cuda.is_available():
        """
        If there are GPUs available, split the input files into chunks and run each set on a distinct GPU 
        """
        processes = []
        for n, chunk in chunkify(files, torch.cuda.device_count()):
            p = torch.multiprocessing.Process(
                target=run, args=(n, chunk, corpus, config)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for file in files:
            run_file(file, corpus, config, create_model("cpu"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1])
