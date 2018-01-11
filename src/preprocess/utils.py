def padding(line, seq_length, unk):
    if len(line) < seq_length:
        line.extend([unk] * (seq_length - len(line)))
    else:
        line = line[:seq_length]
    return line