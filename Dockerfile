FROM guyzsarun/ml-toolkit-gpu:prebuild

COPY requirements.txt ./requirements.txt
COPY intentclassfier1.keras ./intentclassfier1.keras

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install  --user --ignore-installed -r requirements.txt

RUN conda remove wrapt

RUN conda install -y -c pytorch cuda100

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs=1.0.0 &&

RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -i -l -t -y

RUN git lfs install && \
    git clone https://huggingface.co/Chiizu/wav2vec2-base-vi-vlsp2020-demo && \
    git clone https://huggingface.co/anhtu77/videberta-base-finetuned-ner-2 && \
    git clone https://github.com/Trashists/spoken-norm-taggen.git

WORKDIR /spoken-norm-taggen

RUN curl -O http://27.71.27.81:3344/SLU-20230922T151745Z-001.zip
RUN unzip SLU-20230922T151745Z-001.zip
COPY tokenizer/base_sep_sfx.pkl ./base_sep_sfx.pkl
COPY tokenizer/UITws_vi.py ./UITws_vi.py
COPY test.py ./test.py
RUN python test.py