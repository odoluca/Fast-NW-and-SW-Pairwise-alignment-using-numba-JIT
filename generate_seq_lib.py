from random import randint
import pickle
from gotoh_jit_traceback import report_alignment
# import MiscTools.gotoh_jit_traceback as gj_t

# sm=gj.generate_scoring_matrix(4)
# sm[2,2]=2
# sm[1,2]=sm[2,1]=-2
#
# psm=delete(sm,[4,5],axis=1)
# psm= dict( zip(product("TAGC","TAGC"), psm.reshape(16,)))


init_cython_py="""
import gotoh_cython_py as g
sm=g.generate_scoring_matrix(4)

"""

cmd_cython_py_gms="""
g.globalms(a,b,
"""


def generate(length):
    alphabet=4
    seq=[]
    for i in range(length):
        seq.append(randint(0,alphabet-1))
    return seq

def mutate(seq):
    alphabet=4
    pos=randint(0,len(seq)-1)
    mutation_type=randint(0,100)
    indel=20
    if mutation_type>indel:
        while True:
            old=seq[pos]
            seq[pos]=randint(0,alphabet-1)
            if old!=seq[pos]:
                break
    elif mutation_type>indel/2:
        seq = seq[:pos]+[randint(0,alphabet-1)]+seq[pos:]
    else:
        seq = seq[:pos]+ seq[pos+1:]
    return seq

def num_to_str(seq,visualize=True):
    conversion_dict = dict(zip(range(-1, 5), "-TAGC"))
    result= "".join([conversion_dict[x] for x in seq])
    if visualize: print (result)
    return result

def pair(size,target_similarity):
        import gotoh_jit as gj
        b = a = generate(size)
        while True:
            b = mutate(b)
            similarity=gj.globalms(a,b,1,-1,-1,-1)/min(len(a),len(b))
            # num_to_str(b)
            if similarity<=target_similarity:
                #print(similarity)
                break
        return a,b

def pairs(count,size,target_similarity):
    a=[]
    for i in range(count):
        print(i)
        a.append(pair(size,target_similarity))
    return a




settings=(100,400,0.2)

for similarity in [0.2,0.4,0.6,0.8]:
    for seq_size in [100,200,400,600,800,1000]:
        settings=(100,seq_size,similarity)
        print(settings)
        a=pairs(*settings)
        with open("_".join([str(x) for x in settings])+".pickle","wb") as f:
            pickle.dump(a,f)
#
# a=pairs(100,100,0.2)
# with open(r"100_200_0.2.pickle","wb") as f:
#     pickle.dump(a,f)
#
# a=pairs(100,400,0.8)
# with open(r"100_1000_0.8.pickle","wb") as f:
#     pickle.dump(a,f)
#
# a=pairs(100,400,0.2)
# with open(r"100_1000_0.2.pickle","wb") as f:
#     pickle.dump(a,f)
#
# a=pairs(100,400,0.8)
# with open(r"100_1000_0.8.pickle","wb") as f:
#     pickle.dump(a,f)
#
# a=pairs(100,400,0.2)
# with open(r"100_1000_0.2.pickle","wb") as f:
#     pickle.dump(a,f)

# gj_t_r=gj_t.globalms(*a[0],1,-2,-3,-1)

