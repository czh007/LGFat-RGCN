import json
import os
import random
import traceback
from tqdm import tqdm


def is_topic_relevant(article, keywords: list = ['migra', 'flücht', 'asyl']):
    if not isinstance(article,dict):
        raise TypeError("article must be a dictionary")
    if not keywords or not isinstance(keywords, list) or not all(isinstance(kw,str) for kw in keywords) :
        raise TypeError("keywords must be non empty list of strings")
    
    if not all(key in article for key in ['date','title','text', 'url']):
        return False

    try:
        search_corpus = article['news_keywords'].lower()
    except KeyError:
        search_corpus = article['title'].lower() + article['text'].lower()
    except AttributeError:
        print(f"ERROR: News Keywords are {article['news_keywords']}")
        return False


    if any(keyword in search_corpus for keyword in keywords):
        return True 
    else:
        return False


def write_relevant_content_to_file(file_list, relevant_articles_base, search_keywords, 
                                   new=False, annotation=False,
                                   training_size: int = 1000, 
                                   seed=0):

    if new:
        try:
            os.remove(relevant_articles_base+"_evaluation.json")
            os.remove(relevant_articles_base+"_training.json")
        except FileNotFoundError:
            pass

    new_cont = {}
    for json_file in tqdm(file_list):
        try:
            with open(json_file, "r") as jf:
                content = json.load(jf)
                if(is_topic_relevant(content)):
                    new_cont[json_file] = content
        except TypeError:
            traceback.print_exc()
        

    if annotation:
        random.seed(seed)
        training_keys = random.sample(list(new_cont), training_size)

        train = {k: new_cont[k] for k in new_cont if k in training_keys}
        eval = {k: new_cont[k] for k in new_cont if k not in training_keys}

        list_train = list(train)
        ann_martin = {k: train[k] for k in list_train[: len(list_train)//3]}
        ann_josephine = {k: train[k] for k in list_train[len(list_train)//3: 2*len(list_train)//3]}
        ann_simon = {k: train[k] for k in list_train[2*len(list_train)//3:]}
    else:
        eval = new_cont

    try:
        with open(relevant_articles_base+"_evaluation.json", "r+") as ra:
            content_ra = json.load(ra)
            content_ra.update(eval)
            ra.seek(0)
            json.dump(content_ra, ra)
        if annotation:
            with open(relevant_articles_base+"_annotation_simon.json", "r+") as ra:
                content_ra = json.load(ra)
                content_ra.update(ann_simon)
                ra.seek(0)
                json.dump(content_ra, ra)
            with open(relevant_articles_base+"_annotation_josephine.json", "r+") as ra:
                content_ra = json.load(ra)
                content_ra.update(ann_josephine)
                ra.seek(0)
                json.dump(content_ra, ra)
            with open(relevant_articles_base+"_annotation_martin.json", "r+") as ra:
                content_ra = json.load(ra)
                content_ra.update(ann_martin)
                ra.seek(0)
                json.dump(content_ra, ra)

    except FileNotFoundError:
        with open(relevant_articles_base+"_evaluation.json", "w") as raf:
            json.dump(eval, raf)
        if annotation:
            with open(relevant_articles_base+"_annotation_simon.json", "w") as raf:
                json.dump(ann_simon, raf)
            with open(relevant_articles_base+"_annotation_josephine.json", "w") as raf:
                json.dump(ann_josephine, raf)
            with open(relevant_articles_base+"_annotation_martin.json", "w") as raf:
                json.dump(ann_martin, raf)
