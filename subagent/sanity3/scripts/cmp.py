import json,sys
R='/home/user/perseus-speedrun/tmodel/em/'
for t in range(30,50,2):
    ref={json.loads(l)['name']:json.loads(l) for l in open(R+f'n32a4_t{t}.jsonl')}
    ev={json.loads(l)['name']:json.loads(l) for l in open(R+f'even_n32a4_t{t}.jsonl')}
    mine={}
    for f in (f'ship_t{t}.jsonl',f'mine_t{t}.jsonl'):
        for l in open(f):
            d=json.loads(l); mine[d['name']]=d
    out=[]
    for n in ['Shipped_K16','Shipped_K24','D27_K16','D27_K24']:
        a=ref[n]; b=mine[n]
        same = a['oplist']==b['oplist'] and a['relin']==b['relin'] and a['levels']==b['levels']
        out.append(f"{n}:{'OK' if same else 'DIFF'}")
    for n in ['Even_z11_K16','Even_z13_K16','Even_z13_K24']:
        a=ev[n]; b=mine[n]
        same = a['oplist']==b['oplist'] and a['relin']==b['relin'] and a['levels']==b['levels']
        out.append(f"{n}:{'OK' if same else 'DIFF'}")
    z=mine['Even_z11_K16']; s=mine['Shipped_K16']
    print(t, ' '.join(out), '| z11 lv',z['levels'],'relin',z['relin'],z['relin_n'],'resc',z['rescale'],'| ship lv',s['levels'],s['relin'],s['relin_n'],s['rescale'])
