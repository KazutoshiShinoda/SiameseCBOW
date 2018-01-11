import os
import re
import numpy as np
import pandas as pd
import random as rd
from .utils import padding

columns=["sentenceId","category","sectionType","sectionCategory","section4","5","6","7","8","9","10","content"]


class PathLineDocuments():
    """Load sentences through files"""
    def __init__(self, source, limit=None):
        self.source = source
        self.limit = limit
        if os.path.isfile(self.source):
            self.input_files = [self.source]
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + file for file in self.input_files]
            self.input_files.sort()
        else:
            raise ValueError('input is neither a file nor a path')
    
    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            yield self.read_tsv(file_name)
    
    def read_tsv(self, document):
        self._sentences=[]
        self._ids=[]
        self._section_titles=[]
        document = pd.read_csv(document, delimiter='\t', header=None, names=columns)
        sentence_ids = document["sentenceId"].values
        contents = document["content"].values
        section_types = document["sectionType"].values
        try:
            for sentence_id, content, section_type in zip(sentence_ids, contents, section_types):
                s_id = sentence_id.split('-')
                assert len(s_id) == 5
                sec_i = 0
                if s_id[1] == '0':
                    # main title
                    continue
                elif section_type in ['ReferenceHeader', 'AcknowledgementHeader']:
                    # Appendix
                    break
                else:
                    if section_type in ['Footnote', 'Caption']:
                        # Don't add contents other than the main sentences
                        continue
                    elif s_id[2]+s_id[3]+s_id[4]=='000':
                        # Header
                        title_match = re.match(r"[0-9]*[.{,1}[0-9]+]* .*", content)
                        if title_match:
                            # When the title match the type like '0.0.0 ***'
                            title = title_match.group()
                            pos = title.find(' ')
                            sec_title = title[pos+1:]
                        else:
                            # When the section title has no numbers at its head
                            sec_title = content
                        sec_i += 1
                        self._section_titles.append(sec_title)
                        par_i=0
                        sen_i=0
                    else:
                        if par_i != int(s_id[3]):
                            par_i = int(s_id[3])
                        self._sentences.append(content)
                        self._ids.append([sec_i, par_i, sen_i])
                        sen_i += 1
        except:
            raise ValueError('error while processing: {}'.format(document))
        else:
            return self._ids, self._sentences
        
class DataLoader():
    def __init__(self, batch_size, n_positive, n_negative, seq_length, token2id, random_seed=42):
        assert isinstance(documents, PathLineDocuments)
        self.batch_size = batch_size
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.seq_length = seq_length
        self.token2id = token2id
        self.unk = token2id['<UNK>']
        rd.seed(random_seed)
        
    def __iter__(self):
        batch_x=[]
        tar=[]
        pos=[[] for i in range(self.n_positive)]
        neg=[[] for i in range(self.n_negative)]
        batch_y=np.array(([1]*self.n_positive+[0]*self.n_negative)*self.batch_size).reshape(
            self.batch_size, self.n_positive+self.n_negative)
        for ids, document in documents:
            ids = np.array(ids)
            sections = np.unique(ids[:,0])
            current_section = ids[0,0]
            for t, s_id in enumerate(ids):
                if current_section != s_id[0]:
                    # new section
                    continue
                elif t==len(ids)-1:
                    # the end of a document
                    break
                elif current_section != ids[t+1,0]:
                    # the end of a section
                    continue
                else:
                    tar.append(self.get_id_sequence(document[t]))
                    pos[0].append(self.get_id_sequence(document[t-1]))
                    pos[1].append(self.get_id_sequence(document[t+1]))
                    for i, n in enumerate(rd.sample(self.other_than(document, t-1, t+1)), self.n_negative):
                        neg[i].append(self.get_id_sequence(n))
                    if len(tar)==self.batch_size:
                        yield ([np.array(tar)]+[np.array(p) for p in pos]+[np.array(n) for n in neg], batch_y)
                    
                    
    def get_id_sequence(self, line):
        line = line.strip().lower().split()
        return padding(line, self.seq_length, self.unk)
    
    def other_than(self, some_list, inf, sup):
        if inf==0:
            return some_list[sup+1:]
        elif sup==len(some_list)-1:
            return some_list[:inf]
        else:
            return some_list[:inf] + some_list[sup+1:]