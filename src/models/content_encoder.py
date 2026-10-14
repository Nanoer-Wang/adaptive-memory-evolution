"""Pinned local MiniLM ONNX encoder, content-addressed persistent vector cache."""
import hashlib
import json
import sqlite3
import numpy as np
from src.config import resolve
from src.data.download import file_sha256


class ContentEncoder:
    def __init__(self,root='artifacts/encoders/minilm',cache='artifacts/content_vectors.sqlite3'):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root=resolve(root)
        source=json.loads((root/'source.json').read_text(encoding='utf-8'))
        for name in ('model_quint8_avx2.onnx','tokenizer.json'):
            if file_sha256(root/name)!=source['files'][name]['sha256']:
                raise ValueError('Encoder file hash differs from pinned manifest')
        self.signature=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()+':two256-mean-v1'
        self.tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(root/'model_quint8_avx2.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.inputs={i.name for i in self.session.get_inputs()}
        path=resolve(cache);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)')

    def key(self,text):
        return hashlib.sha256((self.signature+'\0'+text).encode()).hexdigest()

    def encode(self,texts):
        output=[None]*len(texts)
        missing={}
        for i,text in enumerate(texts):
            key=self.key(text)
            found=self.db.execute('SELECT value FROM vectors WHERE key=?',(key,)).fetchone()
            if found:
                output[i]=np.frombuffer(found[0],dtype=np.float32).copy()
            else:
                missing.setdefault(key,{'text':text,'indices':[]})['indices'].append(i)
        pairs=list(missing.items())
        for start in range(0,len(pairs),16):
            batch=pairs[start:start+16]
            windows=[];owners=[]
            for j,(_,item) in enumerate(batch):
                ids=self.tokenizer.encode(item['text'],add_special_tokens=False).ids
                parts=[ids] if len(ids)<=254 else [ids[:254],ids[-254:]]
                for part in parts:
                    windows.append([101]+part+[102]);owners.append(j)
            length=max(map(len,windows))
            tokens=np.zeros((len(windows),length),dtype=np.int64)
            mask=np.zeros_like(tokens)
            for j,w in enumerate(windows):tokens[j,:len(w)]=w;mask[j,:len(w)]=1
            feeds={'input_ids':tokens,'attention_mask':mask,'token_type_ids':np.zeros_like(tokens)}
            raw=self.session.run(None,{k:v for k,v in feeds.items() if k in self.inputs})[0]
            pooled=(raw*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1)[:,None],1)
            for j,(key,item) in enumerate(batch):
                vector=pooled[np.asarray(owners)==j].mean(axis=0)
                vector=(vector/max(float(np.linalg.norm(vector)),1e-8)).astype(np.float32)
                if vector.shape!=(384,) or not np.isfinite(vector).all():raise ValueError('Invalid content vector')
                self.db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)',(key,vector.tobytes()))
                for index in item['indices']:output[index]=vector.copy()
            self.db.commit()
        return np.stack(output) if output else np.empty((0,384),np.float32)

    def close(self):
        self.db.close()
