import numpy as np

alphabets = "abcdefghijklmnopqrstuvwxyz#"
def one_hot_endcode(name):
    name = name.lower()
    data = []
    name = name+'#'
    for l in name:
        temp = np.zeros((1,len(alphabets)))
        temp[0][alphabets.index(l)] = 1
        data.append(temp)
    return data

def normalize_data(data):
    data  = data - np.mean(data)
    data/=np.max(data)
    return data

def get_letter(data):
    return alphabets[np.argmax(data)]

def softmax(data):
    data = data - np.max(data)
    e = np.exp(data)
    out = e/np.sum(e)
    return out

def tanh_prime(data):
#     data is tanh ouput
    return 1-data**2

def cross_entropy(c_output,a_output):
    return -np.sum(a_output * np.log(c_output))

def dsoftmax_cross_entropy(c_output,a_output):
    return c_output - a_output

input_size = 26+1
hidden_size = input_size+3
output_size = input_size

# 27x30
U = np.random.normal(0,1,(input_size,hidden_size))

# 30x30
W = np.random.normal(0,1,(hidden_size,hidden_size))

# 30x27
V = np.random.normal(0,1,(hidden_size,output_size))

def train(U,V,W,name):
    import time

    start = time.time()
    h_t, h_activation_t, h_out_t, h_out_activation_t, yhat_t = [],[],[],[],[]
    dEdV_t, dEdW_t, dEdU_t = np.zeros_like(V),np.zeros_like(W),np.zeros_like(U)

    name = name
    data = one_hot_endcode(name)
    x = data[0]
    x_t = []
    x_t.append(x)
    y_t = data[1:]

    """

            yhat                     yhat
            ^                        ^
            |                        |           
          h_out_activation      h_out_activation
          h_out                    h_out 
            ^                        ^
            | V            W         | V            W
          h_activation ------>  h_activation  ------>  
            h                        h
            ^                        ^
            | U                      | U  
            x_t                      x_t


    """

    mess = ""
    # forward pass
    error = 0
    for i in range(len(y_t)):

        # 1x27 --> 1x30
        h = np.matmul(x,U) if i==0 else np.matmul(x,U)+np.matmul(h_activation_t[i-1],W)
        h_activation = np.tanh(h)
        # 1x30 --> 1x27
        h_out = np.matmul(h_activation,V)
        h_out_activation = np.tanh(h_out)
        yhat = softmax(h_out_activation)
        x = one_hot_endcode(alphabets[np.argmax(yhat)])[0]
        x_t.append(x)
        error+= cross_entropy(yhat,y_t[i])
            
        h_activation_t.append(h_activation)
        h_out_activation_t.append(h_out_activation)
        yhat_t.append(yhat)
        mess = mess+get_letter(yhat)

#     print("=======>",error)
    print(mess)

    dh_activation_dW_t = []
    dh_activation_dU_t = []
    dh_activation_dh_t = []
    dh_activation_dh_activation_pre = []

    temp = W
    for i in range(len(x_t)-1):
        dh_activation_dh_activation_pre.append(temp)
        temp = np.matmul(temp,W)

    # back prop
    for i in range(len(y_t)):

        # dE/dh out activation
        dE_dh_out_activation = dsoftmax_cross_entropy(yhat_t[i],y_t[i])
        # dh out activation/dh out
        dh_out_activation_dh_out = tanh_prime(h_out_activation_t[i])
        # dE/dh out
        dE_dh_out = dh_out_activation_dh_out*dE_dh_out_activation
        

        # dh out/dV
        dh_out_dV = h_activation_t[i]
        #dh out/dh activaion
        dh_out_dh_activation = V
        

        # dE/dh activation
        dE_dh_activation = np.matmul(dh_out_dh_activation,(dE_dh_out).T).T
        # dh_activation_dh_activation_pre = np.zeros_like(W) if i==0 else W

        # dh activation/dh
        dh_activation_dh = tanh_prime(h_activation_t[i])
        dh_activation_dh_t.append(dh_activation_dh)
        # dh/dU
        dh_dU = x_t[i]

        # dh activation/dcU
        dh_activation_current_dU = np.matmul(dh_dU.T,dh_activation_dh)
        # dh activation/dcW
        dh_activation_current_dW = np.zeros_like(h_activation_t[0]) if i==0 else h_activation_t[i-1]
        # dh activation/dW
        dh_activation_dW = dh_activation_current_dW
        # dh activation/dU
        dh_activation_dU = dh_activation_current_dU


        for j in range(i):
            dh_activation_dW = dh_activation_dW + np.matmul(dh_activation_dW_t[i-j-1],dh_activation_dh_activation_pre[j])
            dh_activation_dU = dh_activation_dU + np.matmul(dh_activation_dU_t[i-j-1],dh_activation_dh_activation_pre[j])
        dh_activation_dW_t.append(dh_activation_dW)
        dh_activation_dU_t.append(dh_activation_dU)

        dE_dV = np.matmul(dh_out_dV.T,dE_dh_out)
        dE_dW = np.matmul(dE_dh_activation.T,dh_activation_dW)
        dE_dU = dE_dh_activation*dh_activation_dU
        
        dEdV_t+=dE_dV
        dEdW_t+=dE_dW
        dEdU_t+=dE_dU
        


    alpha = 0.01
    U = normalize_data(U - alpha * dEdU_t)
    V = normalize_data(V - alpha * dEdV_t)
    W = normalize_data(W - alpha * dEdW_t)
    
    
    if name+'#' == get_letter(x_t[0]) + mess:
        print("yaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(get_letter(x_t[0]) + mess)
    return U,V,W
    
for i in range(100):
    for name in ["goutham","preetham"]:
        U,V,W = train(U,V,W,name)

def test(U,V,W,name):
    
    name = name
    data = one_hot_endcode(name)
    x = data[0]

    mess = get_letter(x)
    yhat = x
    # forward pass
    error = 0
    h_activation = np.zeros((1,30))
    while(get_letter(yhat)!='#'):
        h = np.matmul(x,U)+np.matmul(h_activation,W)
        h_activation = np.tanh(h)
        h_out = np.matmul(h_activation,V)
        h_out_activation = np.tanh(h_out)
        yhat = softmax(h_out_activation)
        x = one_hot_endcode(alphabets[np.argmax(yhat)])[0]
        print(get_letter(yhat),end = "")

#         mess = mess+get_letter(yhat)
#         print(mess)

test(U,V,W,"g")
