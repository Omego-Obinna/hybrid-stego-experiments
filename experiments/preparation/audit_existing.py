#!/usr/bin/env python3
"""Read-only inventory and explicitly scoped schema/data verification."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,os,re,sys,time,traceback,zipfile,zlib
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent
BOSS=Path('/home/omego/Downloads/BOSSbase')
OUT=ROOT/'audit'
START=time.time()
STATE={'state':'running','stage':'Inventorying experiment files','progress':0,'files':0,'covers_checked':0,'findings':[]}
def atomic(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str));temp.replace(path)
def update(**kw):
 STATE.update(kw);STATE['updated_at']=datetime.now(timezone.utc).isoformat();STATE['elapsed_sec']=round(time.time()-START,2);atomic(ROOT/'status/audit.json',STATE)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def excluded(p):return any(x in ('myenv','site-packages','__pycache__','.git','.cache','node_modules') for x in p.parts)
def scan_csv(p):
 width=Counter();payload=Counter();headers=[];rows=0;errors=[];sampled=p.stat().st_size>256*1024**2
 try:
  with p.open(newline='',errors='replace') as f:
   rd=csv.reader(f);head=next(rd,[]);headers.append(head)
   index=next((head.index(k) for k in ('payload_bpp','target_bpp','payload','bpp','payload_rate') if k in head),None)
   for row in rd:
    if not row:continue
    if row==head:continue
    rows+=1;width[len(row)]+=1
    if index is not None and len(row)==len(head):payload[str(row[index])]+=1
    if sampled and rows>=8192:break
 except Exception as e:errors.append(str(e))
 return {'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'coverage':'first 8192 rows only' if sampled else 'complete','header':headers[0] if headers else [],'rows_read':rows,'row_widths':dict(width),'payload_counts_for_header_width':dict(payload),'errors':errors}
def scan_jsonl(p):
 parts=Counter();schemas=Counter();ok=Counter();rows=0;errors=[];sampled=p.stat().st_size>512*1024**2
 try:
  with p.open(errors='replace') as f:
   for line in f:
    if not line.strip():continue
    rows+=1
    try:
     r=json.loads(line)
     if isinstance(r,dict):
      a=r.get('algo_params') or {}
      if isinstance(a,str):
       try:a=json.loads(a)
       except (ValueError,TypeError):a={}
      a=a if isinstance(a,dict) else {}
      pay=r.get('payload_bpp',r.get('target_bpp',r.get('payload',r.get('bpp'))))
      version=r.get('version',r.get('method_version',a.get('method_version',a.get('version'))))
      run=r.get('run_id',r.get('run'))
      prof=(str(run),str(pay),str(version),str(a.get('base_cost_map',a.get('base_cost'))),str(a.get('fused_suniward_weight')),str(a.get('cf_use_slic',a.get('slic_enabled'))))
      parts[prof]+=1;schemas[len(r)]+=1
      ok[(str(r.get('ok')),str(r.get('success_extract')),str(r.get('stc_backend_actual')),str(r.get('ber')))]+=1
    except (json.JSONDecodeError,TypeError,ValueError) as e:
     if len(errors)<5:errors.append({'row':rows,'error':str(e)})
    if sampled and rows>=8192:break
 except Exception as e:errors.append(str(e))
 return {'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'coverage':'first 8192 rows only' if sampled else 'complete','rows_read':rows,'top_level_key_counts':dict(schemas),'partitions':[dict(run_id=k[0],payload=k[1],version=k[2],base_cost=k[3],fused_weight=k[4],slic=k[5],rows=v) for k,v in parts.items()],'outcomes':[dict(ok=k[0],success_extract=k[1],backend=k[2],ber=k[3],rows=v) for k,v in ok.items()],'errors':errors}
def main():
 update()
 entries=[];top=defaultdict(lambda:{'files':0,'bytes':0,'suffixes':Counter()});csvfiles=[];jsonlfiles=[];jsonfiles=[];sources=[];warnings=[]
 with (OUT/'file_inventory.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['relative_path','bytes','mtime_ns','is_symlink','content_scan_excluded'])
  for dirpath,dirs,files in os.walk(DATA,followlinks=False):
   if Path(dirpath)==DATA:dirs[:]=[d for d in dirs if d!=ROOT.name]
   dirs[:]=[d for d in dirs if d!='.git']
   for name in files:
    p=Path(dirpath)/name
    try:st=p.lstat()
    except OSError as e:warnings.append(str(e));continue
    rel=p.relative_to(DATA);ex=excluded(rel);link=p.is_symlink();w.writerow([str(rel),st.st_size,st.st_mtime_ns,link,ex]);STATE['files']+=1
    t=top[rel.parts[0]];t['files']+=1;t['bytes']+=st.st_size;t['suffixes'][p.suffix.lower()]+=1
    if not ex and not link:
     if p.suffix.lower()=='.csv':csvfiles.append(p)
     elif p.suffix.lower()=='.jsonl':jsonlfiles.append(p)
     elif p.suffix.lower()=='.json':jsonfiles.append(p)
     elif p.suffix.lower() in ('.py','.sh','.toml') and st.st_size<8*1024**2:sources.append(p)
    if STATE['files']%20000==0:update(detail=f"Indexed {STATE['files']:,} files",progress=3)
 atomic(OUT/'inventory_summary.json',{'roots':[str(DATA),str(BOSS)],'total_files':STATE['files'],'top_level':{k:{**v,'suffixes':dict(v['suffixes'])} for k,v in top.items()},'csv_files':len(csvfiles),'jsonl_files':len(jsonlfiles),'json_files':len(jsonfiles),'source_files':len(sources),'warnings':warnings})
 atomic(OUT/'source_hashes.json',[{'path':str(p.relative_to(DATA)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sources])
 update(stage='Verifying the full BOSSBase image directory against the archive',progress=10,detail='Checking every cover image: header, byte count, SHA-256 and archive CRC')
 zpath=BOSS/'BOSSbase_1.01.zip';covers=sorted((BOSS/'BOSSbase_1.01').glob('*.pgm'),key=lambda p:int(p.stem) if p.stem.isdigit() else 100000)
 with zipfile.ZipFile(zpath) as z:zi={Path(i.filename).name:i for i in z.infolist() if i.filename.lower().endswith('.pgm')}
 bad=[];hashes=defaultdict(list)
 with (OUT/'bossbase_manifest.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['cover_id','path','bytes','width','height','max_value','sha256','crc32','zip_crc_match'])
  for i,p in enumerate(covers,1):
   b=p.read_bytes();tokens=[m.group() for m in re.finditer(rb'#[^\r\n]*|[^\s#]+',b[:1024]) if not m.group().startswith(b'#')][:4]
   dims=[int(v) for v in tokens[1:4]] if len(tokens)==4 and tokens[0]==b'P5' else [None]*3
   sh=hashlib.sha256(b).hexdigest();crc=zlib.crc32(b)&0xffffffff;info=zi.get(p.name);match=bool(info and crc==info.CRC and len(b)==info.file_size);hashes[sh].append(p.stem)
   w.writerow([p.stem,str(p),len(b),*dims,sh,f'{crc:08x}',match])
   if dims!=[512,512,255] or not match:bad.append({'cover':p.name,'header':dims,'archive_crc_match':match})
   STATE['covers_checked']=i
   if i%250==0:update(progress=10+int(25*i/max(1,len(covers))),detail=f'{i:,} / {len(covers):,} covers checked')
 dup=[ids for ids in hashes.values() if len(ids)>1]
 atomic(OUT/'bossbase_verification.json',{'directory':str(BOSS/'BOSSbase_1.01'),'archive':str(zpath),'directory_images':len(covers),'archive_images':len(zi),'all_directory_crc_match_archive':not bad,'nonconforming_or_mismatched':bad,'duplicate_byte_identical_cover_groups':dup,'ids_complete_1_to_10000':{p.stem for p in covers}=={str(i) for i in range(1,10001)},'archive_check':'CRC and uncompressed-size matches against ZIP central directory; no archive extraction or modification'})
 update(stage='Checking saved CSV and JSONL schemas and run partitions',progress=36,detail=f'{len(csvfiles)} CSV files and {len(jsonlfiles)} JSONL files located')
 summaries=[];allfiles=[('csv',p) for p in csvfiles]+[('jsonl',p) for p in jsonlfiles]
 for i,(kind,p) in enumerate(allfiles,1):
  info=scan_csv(p) if kind=='csv' else scan_jsonl(p);summaries.append(info)
  if i%10==0:update(progress=36+int(50*i/max(1,len(allfiles))),detail=f'{i} / {len(allfiles)} structured logs checked; currently {p.parent.name}')
 atomic(OUT/'structured_file_audit.json',summaries)
 meta=[]
 for p in jsonfiles:
  if p.stat().st_size>16*1024**2:meta.append({'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'coverage':'metadata only; large JSON'});continue
  if any(x in p.name.lower() for x in ('credential','secret','token','keyfile')):continue
  try:
   obj=json.loads(p.read_text());keys=list(obj) if isinstance(obj,dict) else None
   meta.append({'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'parse_valid':True,'type':type(obj).__name__,'keys':keys,'length':len(obj) if isinstance(obj,(list,dict)) else None})
  except Exception as e:meta.append({'path':str(p.relative_to(DATA)),'parse_valid':False,'error':str(e)})
 atomic(OUT/'json_metadata_index.json',meta)
 issues=[{'path':x['path'],'row_widths':x.get('row_widths'),'errors':x['errors']} for x in summaries if x['errors'] or (len(x.get('row_widths',{}))>1)]
 atomic(OUT/'schema_issues.json',issues)
 update(stage='Inventory, dataset verification and structured-log scan complete',progress=90,detail='Preparing source-specific reuse decisions; audit report follows',findings=[f'{len(covers):,} dataset images checked against archive CRCs',f'{len(csvfiles)} CSV and {len(jsonlfiles)} JSONL files indexed',f'{len(issues)} structured files need schema/error review'])
 atomic(OUT/'scan_complete.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_sec':time.time()-START,'files':STATE['files'],'covers':len(covers),'structured_files':len(summaries),'full_scan_files':sum(x['coverage']=='complete' for x in summaries),'sampled_large_files':[{'path':x['path'],'bytes':x['bytes'],'coverage':x['coverage']} for x in summaries if x['coverage']!='complete'],'status':'scan complete; source-specific reuse analysis pending'})
 print(json.dumps({'files':STATE['files'],'covers':len(covers),'structured_files':len(summaries),'issues':len(issues)}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:update(state='failed',stage='Audit needs attention',last_error=str(e));traceback.print_exc();raise
