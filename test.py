from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from importlib.machinery import SourceFileLoader
from huggingface_hub import hf_hub_download
from huggingface_hub import notebook_login

processor = Wav2Vec2ProcessorWithLM.from_pretrained("nguyenvulebinh/wav2vec2-base-vi-vlsp2020")

model = SourceFileLoader("model", hf_hub_download(repo_id="nguyenvulebinh/wav2vec2-base-vi-vlsp2020", filename="model_handling.py")).load_module().Wav2Vec2ForCTC.from_pretrained("Chiizu/wav2vec2-base-vi-vlsp2020-demo")

import torch
use_gpu = torch.cuda.is_available()
if use_gpu:
  model.to('cuda')

import os

directory = 'public_test'
files = []
for file_path in os.listdir(directory):
  if os.path.isfile(os.path.join(directory, file_path)):
        files.append(file_path)

from datasets import Dataset, Audio
data = Dataset.from_dict({"audio": files}).cast_column("audio", Audio())

def map_to_result(batch):
    input_data = processor.feature_extractor(batch['audio']["array"], sampling_rate=batch['audio']["sampling_rate"], return_tensors='pt')
    if use_gpu:
      for key, val in input_data.items():
          input_data[key] = val.cuda()

    output = model(**input_data)

    batch["pred_str"] = processor.decode(output.logits.cpu().detach().numpy()[0], beam_width=100).text

    return batch  

text_data = data.map(map_to_result)

from IPython.display import display, HTML
import random
import pandas as pd
import numpy as np
def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

# %cd /content/drive/MyDrive/SLU_hackathon/SLU_dataset/spoken-norm-taggen

from infer import infer

import re
def get_spoken_norm(text_data):
  spoken_norm_fixed = [infer([item], ['phút | phút'])[0] for item in text_data]
  return spoken_norm_fixed

def reformat(text):
 # for text in text_data
  text = text.replace(' %', '%')
  for index, charc in enumerate(text):
    if index == 0:
      continue
    if charc == 'h' and (text[index-1]).isnumeric():
      text = text[:index] + ' giờ ' + text[index + 1:]
  re.sub(' +', ' ', text)
  return text

answer_text = map(reformat, get_spoken_norm(text_data))

answer_text = list(answer_text)

"""# Tokenize #"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/SLU_hackathon/tokenizer/

from UITws_v1 import UITws_v1
uitws_v1 = UITws_v1('base_sep_sfx.pkl')

data_prepared = uitws_v1.segment(texts = text_data['pred_str'], pre_tokenized = True, batch_size = 128)
data_prepared

"""# Intent classification #"""

from sentence_transformers import SentenceTransformer

sentence_encoder = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

encoded_sens = sentence_encoder.encode(answer_text)
encoded_sens.shape

import numpy as np

import keras, torch
intent_classifier = keras.models.load_model('intentclassifier1.keras')

y_pred = np.argmax(intent_classifier.predict(encoded_sens), axis = 1)

intent = ['mở thiết bị',
 'tăng âm lượng của thiết bị',
 'bật thiết bị',
 'kích hoạt cảnh',
 'hủy hoạt cảnh',
 'đóng thiết bị',
 'tăng mức độ của thiết bị',
 'giảm mức độ của thiết bị',
 'giảm nhiệt độ của thiết bị',
 'kiểm tra tình trạng thiết bị',
 'tắt thiết bị',
 'giảm độ sáng của thiết bị',
 'tăng độ sáng của thiết bị',
 'giảm âm lượng của thiết bị',
 'tăng nhiệt độ của thiết bị']
hash = [2,  7,  8, 13, 11,  4,  9,  3,  0,  6, 14,  1, 12, 10,  5]

def revert(y):
  return [intent[hash[ele]] for ele in y]

print(revert(y_pred))

"""# Named entity recognition #"""

ner_tags = {'device' : 1, 'location' : 2, 'command' : 3, 'time at' : 4, 'duration' : 5, 'target number' : 6, 'changing value' : 7, 'scene' : 8}
label_list = list(ner_tags.keys())
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "anhtu77/videberta-base-finetuned-ner-2",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
                                                       )
tokenizer = AutoTokenizer.from_pretrained("anhtu77/videberta-base-finetuned-ner-2")
nerpipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')

answers = nerpipeline(answer_text)

def reformat_answer(ans):
    result = dict()
    result["type"] = ans['entity_group']
    result["filler"] = ans['word'].replace('_', ' ').replace(' %', '%')
    return result
entities = [list(map(reformat_answer, answer)) for answer in answers]

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/SLU_hackathon/

import pandas as pd

# entities = json.load(open("/content/drive/MyDrive/SLU_hackathon/entities_newst_est.json"))

final_result = pd.DataFrame()
final_result['intent'] = revert(y_pred)
final_result['entities'] = entities
final_result['file'] = files
with open('predictions.jsonl', 'w', encoding='utf-8') as file:
    final_result.to_json(file, orient = 'records', lines = True, force_ascii=False)
# final_result.to_json('predictions.jsonl', orient = 'records', lines = True)

# import IPython.display as ipd
# ipd.Audio(data=np.asarray(data[4]['audio']["array"]), autoplay=True, rate=16000)