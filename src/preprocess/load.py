import os
import pandas as pd
import re

columns=["sentenceId","category","sectionType","sectionCategory","section4","5","6","7","8","9","10","content"]


class PathLineSentences():
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
            yield self.read_csv(file_name)
    
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