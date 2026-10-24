"""Fixed maps of public bytes only; raw full views are separately reconstructible."""
import hashlib
import numpy as np
import epoch_protocol as ep
import historical_features as hf

def sketch(data):
    a=np.frombuffer(data,dtype=np.uint8);assert len(a)
    hist=np.bincount(a,minlength=256).reshape(64,4).sum(axis=1)/len(a)
    x=a.astype(np.uint32);h=((x[:-1]*np.uint32(2654435761))^(x[1:]*np.uint32(2246822519)))&63
    pairs=np.bincount(h.astype(int),minlength=64)/max(1,len(h))
    return np.r_[hist,pairs]
def gamma_public(pair):
    from pathlib import Path
    g=hf.gamma_features(pair['pair_id'],pair['gamma1_path'],pair['gamma2_path'],32)
    numeric={k:float(v) for k,v in g.items() if k.startswith('gamma_')}
    data=[Path(pair[f'gamma{j}_path']).read_bytes() for j in [1,2]]
    return dict(numeric=numeric,sketch=np.r_[sketch(data[0]),sketch(data[1])],
                digest=hashlib.sha256(ep.enc([(b'g1',data[0]),(b'g2',data[1])])).digest()[:8])
def image_public(path):
    from pathlib import Path
    from PIL import Image
    v=hf.image_features(str(path),32);numeric={k:float(x) for k,x in v.items() if k.startswith('img_')}
    with Image.open(path) as im:prefix=np.asarray(im,dtype=np.uint8).ravel()[:8].copy()
    return dict(numeric=numeric,sketch=sketch(Path(path).read_bytes()),prefix=prefix)
def content_vector(g,img):
    cross=hf.cross_features(img['numeric'],g['numeric'],32)
    a=np.frombuffer(g['digest'],dtype=np.uint8);b=img['prefix']
    return np.r_[list(img['numeric'].values()),list(g['numeric'].values()),list(cross.values()),g['sketch'],img['sketch'],
                 np.mean(a==b),np.unpackbits(np.bitwise_xor(a,b)).mean()]
def metadata_vector(e):
    raw=e.sid+ep.uint(e.epoch)+e.hdr+e.nonce_a+e.nonce_b+e.nonce_c+ep.uint(e.payload_bits)+ep.VERSION+ep.CODEC
    # Every length is public. Fixed-size prefixes and PGM framing keep this Z matched.
    return np.r_[np.frombuffer(raw,dtype=np.uint8)/255.,np.array([len(e.gamma1),len(e.gamma2),len(e.stego)])/2**20]
def auth_vector(e):
    b=np.frombuffer(e.tag,dtype=np.uint8)
    hist=np.bincount((b>>4).astype(int),minlength=16)/32
    positive=hist[hist>0]
    public=hashlib.sha256(ep.transcript(e)).digest()
    return np.r_[b/255.,hist,np.unpackbits(b).mean(),-np.sum(positive*np.log2(positive)),
                 np.unpackbits(np.bitwise_xor(b,np.frombuffer(public,dtype=np.uint8))).mean()]
def challenge_vector(e,g,img):
    return np.r_[content_vector(g,img),metadata_vector(e),auth_vector(e)]
def full_vector(e,g,img,history):
    core=content_vector(g,img);current=np.r_[core,metadata_vector(e),auth_vector(e)]
    prior=[np.r_[h['vector'],sketch(h['query'])] for h in history]
    dif=core-np.mean([h['content'] for h in history],axis=0)
    tagbits=np.unpackbits(np.frombuffer(e.tag,dtype=np.uint8))
    comparisons=[np.mean(tagbits!=np.unpackbits(np.frombuffer(h['epoch'].tag,dtype=np.uint8))) for h in history]
    return np.r_[current,*prior,dif,comparisons].astype(np.float32)
def columns(e,g,img,history):
    nc=len(content_vector(g,img));nm=len(metadata_vector(e));na=len(auth_vector(e));nf=len(full_vector(e,g,img,history))
    return dict(content_only=list(range(nc)),metadata_only=list(range(nc,nc+nm)),
        authentication_only=list(range(nc,nc+nm+na)),full_challenge=list(range(nc+nm+na)),full_history=list(range(nf)))
