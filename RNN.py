import numpy as np

alphabets = "abcdefghijklmnopqrstuvwxyz#"
def one_hot_encode(name):
    name = name.lower()
    data = []
    name = name+'#'
    for l in name:
        temp = np.zeros((1,len(alphabets)))
        temp[0][alphabets.index(l)] = 1
        data.append(temp)
    return data

def get_letter(data):
    return alphabets[np.argmax(data)]

def normalize_data(data):
    data  = data - np.mean(data)
    data/=np.max(data)
    return data

def data_set():
    alpha = "abcdefghijklmnopqrstuvwxyz"
    with open("IndianNames.csv", encoding="utf8") as f:
        names = []
        for name in f.read().split():        
            good_name = True
            for c in name.lower():
                if c not in alpha:
                    good_name = False

            if good_name:
                names.append(name)
                
    from collections import Counter
    return list(dict(Counter(names)).keys())

def softmax(data):
    data = data - np.max(data)
    e = np.exp(data)
    out = e/np.sum(e)
    return out

def tanh_prime(data):
#   data is tanh output
    return 1-data**2

def cross_entropy(c_output,a_output):
    return -np.sum(a_output * np.log(c_output))

def dsoftmax_cross_entropy(c_output,a_output):
    return c_output - a_output

wx = np.random.normal(0,1,(27,30))
ws = np.random.normal(0,1,(30,30))
wy = np.random.normal(0,1,(30,27))

def train(wx,wy,ws,name):

    xt = one_hot_encode(name)[:-1]
    yt = one_hot_encode(name)[1:]
    error = 0
    
    #     forward pass
    st_p = [np.zeros((1,30))]
    yhatt = []
    
    for i in range(len(xt)):
        s = np.tanh(np.matmul(xt[i],wx)+np.matmul(st_p[i],ws))
        st_p.append(s)
        yhat = softmax(np.matmul(s,wy))
        yhatt.append(yhat)
           
        error += np.sum(cross_entropy(yhat,xt[i]))

    #       BPTT
    dEdyt = [];dsds = [];dEdst = []

    temp = ws
    for i in range(len(xt)-1):
        dsds.append(temp)
        temp = np.matmul(ws,temp)
        
    dsds = [normalize_data(data) for data in dsds]

    dwy = np.zeros_like(wy)
    dwx = np.zeros_like(wx)
    dws = np.zeros_like(ws)
    alpha = 0.0005

    for i in range(len(xt)):
        dEdy = dsoftmax_cross_entropy(yhatt[i],yt[i])
        dEdyt.append(dEdy)

        dEdwy = np.matmul(st_p[i+1].T,dEdy)

        dEds = np.matmul(dEdy,wy.T)
        dEdst.append(dEds)

        dEdws = np.matmul(st_p[i].T,dEds)
        for j in range(i):
            dEdws += np.matmul(st_p[i-1-j].T,np.matmul(dEds,dsds[j].T))

        dEdwx = np.matmul(xt[i].T,tanh_prime(st_p[i])) * dEds
        for j in range(i):
            dEdwx += np.matmul(xt[i-1-j].T,tanh_prime(st_p[i-j])) * np.matmul(dEds,dsds[j].T)

        dwy += dEdwy
        dwx += dEdwx
        dws += dEdws

    wx -= alpha * dwx
    ws -= alpha * dws
    wy -= alpha * dwy

    return wx,wy,ws

names = data_set()[:20]
# names = ["preetham","goutham","praveen","gali"]
for _ in range(1000):
    for name in names:
        wx,wy,ws = train(wx,wy,ws,name)
        
def predict(wx,wy,ws,name):
    
    xt = one_hot_encode(name)
    #     forward pass
    s = np.zeros((1,30))
    for i in range(20):
        
        x = y if i >= len(name) else xt[i]    
        s = np.tanh(np.matmul(x,wx)+np.matmul(s,ws))
        y = softmax(np.matmul(s,wy))
        y = one_hot_encode(get_letter(y))[0]

        if get_letter(x) == '#':
            return
        
        print(get_letter(x),end='')
        
predict(wx,wy,ws,'r')
