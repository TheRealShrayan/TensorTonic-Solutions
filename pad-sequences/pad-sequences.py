import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    
    no_of_rows=len(seqs)
    

    if max_len is None:
        L = max(len(seq) for seq in seqs) if seqs else 0
       
       
    else:
       L=max_len

    result=np.full((no_of_rows,L),pad_value,dtype=int)
    for i ,seq in enumerate(seqs):
        min_length=min(len(seq),L)
        result[i,:min_length]=seq[:min_length]

    return result

