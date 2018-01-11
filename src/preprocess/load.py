import os
import pandas as pd


def load(source):
    """Load documents
    
    Args
        source: should be a directory which includes documents
    """
    
    if os.path.isdir(source):
        source = os.path.join(source, '')  # ensures os-specific slash at end of path
        input_files = os.listdir(source)
        input_files = [source + file for file in input_files]  # make full paths
        input_files.sort()
    elif os.path.isfile(source):
        input_files = [source]
        
        
    return x, y

def read_tsv(self, document, document_id):
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
                    self._sections.append(sec_i)
                    self._ids.append((sec_i, par_i, sen_i))
                    sen_i += 1
    except:
        raise ValueError('error while processing: {}'.format(document_id))