#------------------SHELL MODELS LIBRARY----------------------#
from libraries import *

##################----------------DEVICE/CPU/GPU---------###################
device = th.device('cpu')
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    device = th.device('cuda')
    print("device = ", device)
print("device:",device) 

######################--------LOAD DATA-------------------################

def load_data(path,filename,nshells=12,sampling=1
,difference=False,normalize01=True,normalize11=False,verbose=False):
    """ This function loads the dataset of a given path 
    and does the preprocessing. 
    You can select:
    -nshells
    -sampling
    -difference  : instead of x_n it uses x_n'=x_n-x_{n-1} -- it should shows better performances
    -normalize01 : to have data that are in range (0,1)
    -normalize11 : to have data that are in range (-1,1)
    -verbose

    """
    try:
        alldata = np.load(path+filename)[2:-2]
        print(path+filename)
    except FileNotFoundError:
        print("file not found !!")
        print(path+filename)
    
    if verbose:
        print("Original data shape : ", alldata.shape)

    data = np.expand_dims(alldata.T, axis=1)
    data = data[:,:,:nshells]
    data = data[::sampling]
    data = np.abs(data)
    L = len(data)
    min=data.min(axis=0)
    max=data.max(axis=0)
    
    print("After subsampling : ", data.shape )
    if difference:
        data = data[1:] - data[:-1]  # x_n-x_{n-1}

    if normalize01:
        
        data = (data - min) / (max - min)
       # if normalize11:
       #     data = data * 2 - 1  

    if normalize11:
        data = (data - min) / (max - min)
        data = data * 2 - 1    
    
   
    return data ,(max , min)

######################--------SPLIT DATASET-------------------################

def split_data(data,start=0):
    """
        To use after load_data
        split the dataset in three parts : 

        - train  
        - valid
        - test

        start :   begin of training set
        
        train :   from start            ---> start+train
        val   :   from start+train      ---> start+train+val
        test  :   from start+train+val  ---> start+train+val+test

        return train/val/test
    """
    train=len(data)*8//10
    val=len(data)*15//100
    test=len(data)*5//10

    train_data = th.FloatTensor((data[start:start+train,:,:]))#.to(device)
    valid_data = th.FloatTensor((data[start+train:start+train+val,:,:]))#.to(device)
    test_data = th.FloatTensor((data[start+train+val:start+train+val+test,:,:]))#.to(device)
    print("train data: ", train_data.shape)
    print("valid data: ", valid_data.shape)
    print("test data: ", test_data.shape)

    return train_data, valid_data, test_data

def struct_func(u_n,order,h,f,title,save=False,folder=''):
    """
        Compute the Structure function given |U_n| :
            S_q= mean ( |U_n(t)|^q ) 
        In which mean is the time average.
    """
    print("evaluate structure function S_",order)
    Sq=np.mean(np.power(u_n,order),0)
        
    if save:
        np.save('./'+str(folder)+'/S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title),Sq.T)

        print('saved S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title))
    return Sq.T

######################--------BATCH-------------------#################
def batchify(data, indices, history, future, model):
    
    """
    Function that given data of dimension (L,bs,features)  ex: (10_000,100,12)

    Creates a batch with dimension (h,100,12) and (f,100,12) for lstm
    
    Creates a batch with dimension (100,h*10_000) and (100,f*10_000)

    """

    
    
    
    bs = len(indices)
    S = data.shape[-1] # number of shells


    if isinstance(model, Sequence):
        
        outX = th.zeros(history, bs, S)
        outY = th.zeros(future, bs, S)
        for i in range(bs):
            start = indices[i]
            outX[:, i:i + 1, :] = data[start:start + history].to(device)
            outY[:, i:i + 1, :] = data[start + history:start + history + future].to(device)
        return outX, outY

    if isinstance(model, MLP):
        
        outX = th.zeros(bs, history * S)
        outY = th.zeros(bs, future * S)
        for i in range(bs):
            start = indices[i]  
            outX[i, :] = data[start:start + history].flatten().to(device)
            outY[i, :] = data[start + history:start + history + future].flatten().to(device)
        return outX, outY

#-----------------------------------------------------TRAIN FUNCTIONS-----------------------------------------------------------------------#

class TrainFunction:
    def __init__(self, 
                model,
                optimizer,
                train,
                valid,
                history=100,
                future=10,
                nepochs=10,
                bsize=100,
                nprints=10,
                save=False
                ):

        self.model=model
        self.optimizer=optimizer
        self.train=train
        self.valid=valid
        self.history=history
        self.future=future
        self.nepochs=nepochs
        self.nprints=nprints
        self.save=save


class STDtraining(TrainFunction):


    def __init__(self, model, optimizer, train, valid, history=1, future=15, nepochs=10, bsize=100, nprints=10, save=False):
        super().__init__(model, optimizer, train, valid, history, future, nepochs, bsize, nprints, save)


    def get_param(self,model,optimizer):

        #------------------------------------------chose model---------------------------------------#
        if isinstance(model, Sequence):
            model_name='lstm'
        elif isinstance(model,MLP):
            model_name='mlp'
        
        #------------------------------------------chose optimizer-----------------------------------#

        if isinstance(optimizer, th.optim.Adam):
            optimizer_name='adam'
        elif isinstance(optimizer, th.optim.LBFGS):
            optimizer_name='lbfgs'

        return model_name,optimizer_name

    def train_model(self, nepochs=10, nprints=10,folder=''):
        """
        Basic training

        The training sequence is processed as a long sequence. 
        For each time step       
        y_{t+1} = model(x_t,h_t)
        
        """
        model=self.model
        optimizer=self.optimizer
        train=self.train.to(device)
        valid=self.valid.to(device)
        nepochs=self.nepochs
        save=self.save
        train_loss = []
        valid_loss = []
        i = 0
        min_perf=900000          # param used to find (and save) min validation loss model
        criterion = nn.MSELoss()
        

        
        model_name,optimizer_name=self.get_param(model,optimizer)
        print("DEBUG")
        if optimizer_name=='adam':
            while i < nepochs:
                model.train()
                optimizer.zero_grad()
                if model_name=='lstm':
                    out, (h, c) = model(train[:-1])
                    loss = criterion(out.to(device), train[1:])
                elif model_name=='mlp':
                    out = model(train[:-1])
                    loss = criterion(out.to(device), train[1:])
                        
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                # validation
                with th.no_grad():
                    model.eval()
                    if model_name=='lstm':
                        pred, _ = model(valid[:-1])
                        validloss = criterion(pred.to(device), valid[1:])
                    elif model_name=='mlp':
                        pred = model(valid[:-1])
                        validloss = criterion(pred.to(device), valid[1:])
                    
                    
                    valid_loss.append(validloss.item())
                i += 1
                if i % (nepochs // nprints) == 0:
                    print(i, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                
                if validloss.item() < min_perf :
                    min_perf = validloss.item()
                    bestmodel=model
                    if save: 
                        th.save(model.state_dict(), './'+str(folder)+'/models/epochs_'+str(nepochs))
                        bestmodel.load_state_dict(th.load('./'+str(folder)+'/models/epochs_'+str(nepochs)))
            model=bestmodel

            return train_loss, valid_loss

        if optimizer_name=='lbfgs':
            optim = th.optim.LBFGS(model.parameters())                
            train_loss = []
            valid_loss = []
            e = 0
            min_perf=900000          # param used to find (and save) min validation loss model

            criterion = nn.MSELoss()
            while e < nepochs:
                def closure(): 
                    optim.zero_grad()
                    if model_name=='lstm':
                        out, (h, c) = model(train[:-1].to(device))
                        loss = criterion(out.to(device), train[1:].to(device))
                    if model_name=='mlp':
                        out = model(train[:-1])
                        loss = criterion(out, train[1:])
                    loss.backward()
                    return loss
                model.train()
                loss_epoch = optim.step(closure)
                train_loss.append(loss_epoch.item())    
                # validation
                with th.no_grad():
                    model.eval()
                    if model_name=='lstm':
                        pred, _ = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])
                    if model_name=='mlp':
                        pred = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])

                    valid_loss.append(validloss.item())
                e += 1
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                if validloss.item() < min_perf :
                    min_perf = validloss.item()
                    th.save(model.state_dict(), './results_LBFGS/models/epochs_'+str(nepochs))
                    bestmodel=model.load_state_dict(th.load('./results_LBFGS/models/epochs_'+str(nepochs)))
            model=bestmodel
            return train_loss, valid_loss


class FORtraining(TrainFunction):

    def __init__(self, model, optimizer, train, valid, history=1, future=15,
     nepochs=10, bsize=100, nprints=10, save=False):
        super().__init__(model, optimizer, train, valid, history, future,nepochs, bsize, nprints, save)


    def get_param(self,model,optimizer):

            #------------------------------------------chose model---------------------------------------#
            if isinstance(model, Sequence):
                model_name='lstm'
            elif isinstance(model,MLP):
                model_name='mlp'
            
            #------------------------------------------chose optimizer-----------------------------------#

            if isinstance(optimizer, th.optim.Adam):
                optimizer_name='adam'
            elif isinstance(optimizer, th.optim.LBFGS):
                optimizer_name='lbfgs'

            return model_name,optimizer_name

    def train_model(self,
                    nepochs=10,
                    bsize=100,
                    nprints=10,
                    patience=50,
                    folder=''):


        """## With forecast training as fine tuning 

        forecast training takes as parameters: 
        - history: $h$,  the time span read by the LSTM
        $$
        \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
        $$
        - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
        $$
        \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
        $$

        """
        model=self.model
        optimizer=self.optimizer
        train=self.train
        valid=self.valid
        nepochs=self.nepochs
        history=self.history
        future=self.future
        save=self.save
        #model=model.to(device)
        ## Data splitting
        # Train
        L = len(train)
        nex = (L - future - history)
        nbatch = nex // bsize
        # Valid
        LV = len(valid)
        nexV = (LV - future - history)
        nbatchV = nexV // bsize
        print("n batch valid loss ",nbatchV)
        # Random split of the train
        indices = np.arange(nex)
        indicesV = np.arange(nexV)
        np.random.shuffle(indices)
        # Init.
        #optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)



        
        train_loss = []
        valid_loss = []
        e = 0
        counter=0
        loss_lag=42
        min_perf=900000          # param used to find (and save) min validation loss model
        criterion = nn.MSELoss()

        model_name,optimizer_name=self.get_param(model,optimizer)


        if optimizer_name=='lbfgs':
            
            optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

            while e < nepochs:
                global eloss 
                eloss = 0
                evalloss=0
                model.train()
                for i in range(nbatch):
                    optimizer.zero_grad()
                    bidx = indices[i * bsize:min((i + 1) * bsize, L)]
                    inputs, refs = batchify(train, bidx, history, future,model)
                       
                    def closure():
                        optimizer.zero_grad()
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                        loss = criterion(outputs, refs.to(device))
                        loss.backward()
                        global eloss
                        eloss += loss
                        return eloss
        
                loss_b=optimizer.step(closure)
                train_loss.append(loss_b.item())
            
                
                if (np.abs(loss_lag-loss_b.item())<1e-5):
                    counter+=1
                    #print("counter=",counter)
                    #print("loss_lag=",loss_lag)
                    #print("loss_b =",loss_b)
                    #print("patience=",patience)
                    if (counter==patience):
                        print("loss is not decreasing anymore")
                        if e>100:
                            print("training stopped at epoch ",e," and model saved")
                            break
                        else:
                            print("training stopped at epoch ",e," and model not saved")
                            sys.exit()
                else:
                    counter=0
                    loss_lag=loss_b.item()
                    
                #validation
                with th.no_grad():
                    model.eval()
                    for i in range(nbatchV):
                        #print("!!! in batch !!")
                        bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
                        inputs, refs = batchify(valid, bidx, history, future)
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                        validloss = criterion(outputs, refs.to(device))
                        #print(" gave value to validloss")
                    valid_loss.append(validloss.item())
                e += 1
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                if save:
                    if validloss.item() < min_perf :
                        min_perf = validloss.item()
                        name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
                        th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
                        bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
            model=bestmodel
            return train_loss, valid_loss

        if optimizer_name=='adam':
            criterion = nn.MSELoss()
            while e < nepochs:
                model.train()
                eloss = 0
                for i in range(nbatch):
                    optimizer.zero_grad()
                    bidx = indices[i * bsize:min((i + 1) * bsize, L)]
                    inputs, refs = batchify(train, bidx, history, future,model)
                    if model_name=='lstm':
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                    elif model_name=='mlp':
                        outputs = model.forward(inputs.to(device))
                    loss = criterion(outputs, refs.to(device))
                    loss.backward()
                    optimizer.step()
                    eloss += loss.item()
                
                train_loss.append(eloss)

                if (np.abs(loss_lag-train_loss[e])<1e-5):
                    counter+=1
                    #print("counter=",counter)
                    #print("loss_lag=",loss_lag)
                    #print("loss_b =",loss_b)
                    #print("patience=",patience)
                    if (counter==patience):
                        print("loss is not decreasing anymore")
                        if e>100:
                            print("training stopped at epoch ",e," and model saved")
                            break
                        else:
                            print("training stopped at epoch ",e," and model not saved")
                            sys.exit()
                else:
                    counter=0
                    loss_lag=train_loss[e]



                # validation
                with th.no_grad():
                    model.eval()
                    for i in range(nbatchV):
                        bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
                        inputs, refs = batchify(valid, bidx, history, future,model)

                        if model_name=='lstm':
                            h0, c0 = model.get_init(bsize)
                            outputs, (h, c) = model.generate(inputs, future, h0, c0)
                        elif model_name=='mlp':    
                            outputs = model.forward(inputs.to(device))    

                        validloss = criterion(outputs, refs.to(device))
                    valid_loss.append(validloss.item())
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])

                if save:
                    if validloss.item() < min_perf :
                        min_perf = validloss.item()
                        name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
                        th.save(model.state_dict(), './results/models/'+name)
                e += 1

            return train_loss, valid_loss    
            
#-----------------------------------------------------TRAIN PLOT-----------------------------------------------------------------------#

def plot_train(tl, vl,save=True,show=False,logplot=False):
    plt.figure(figsize=(10,8))
    plt.plot(tl,label='train_loss')
    plt.plot(vl,label='val_loss')
    plt.xlabel(r'$epochs$', fontsize=15)
    plt.ylabel(r'$Loss$', fontsize=15)
    plt.legend()
    if logplot:
        plt.yscale('log')     
    if save:
        plt.savefig('./'+str(folder_name)+'/plots/loss/learningCurve_'+str(len(vl))+'_epoch'+
        '_sampling_'+str(sampling)+'h_'+str(history)+'_f_'+str(future)+'_run_'+str(n))
    if show:
        plt.show()

#######################--------MODEL INIT----#################

class Sequence(nn.Module):
    def __init__(self,
                 hidden=50,
                 layer=1,
                 nfeatures=3,
                 dropout=0,
                 device=device):                #changed here the device 
        print("inside seq device=",device)
        super(Sequence, self).__init__()
        self.hidden = hidden
        self.layer = layer
        self.nfeatures = nfeatures
        self.device = device
        self.lstm1 = nn.LSTM(self.nfeatures,
                             self.hidden,
                             self.layer,
                             dropout=dropout).to(device)
        self.linear = nn.Linear(self.hidden, self.nfeatures).to(device)
        self.h_0 = nn.Parameter(th.randn(
            layer,
            1,
            hidden,
        ).to(device) / 100)
        self.c_0 = nn.Parameter(th.randn(
            layer,
            1,
            hidden,
        ).to(device) / 100)

    def forward(self, input, h_t=None, c_t=None):
        L, B, F = input.shape
        if h_t is None:
            h_t = self.h_0
        if c_t is None:
            c_t = self.c_0
        self.lstm1.flatten_parameters()
        out, (h_t, c_t) = self.lstm1(input, (h_t, c_t))
        output = out.view(input.size(0), input.size(1), self.hidden)
        output = th.sigmoid(self.linear(out)) #changed with sigmoid

        output = output.view(input.size(0), input.size(1), self.nfeatures).to(device)   #added to(device)    
        return output, (h_t, c_t)

    def get_init(self, bsize=1):
        assert bsize > 0
        if bsize == 1:
            return self.h_0, self.c_0
        return self.h_0.repeat(1, bsize, 1), self.c_0.repeat(1, bsize, 1)

    def generate(self, init_points, N, h_t=None, c_t=None, full_out=False):
        """
        - the init_points are used to initialize the memory of the LSTM
        - N is the number of generated points by the model on its own
        
        Return a tensor of size (N+L, 1, 12)
        with L the length of init_points
        """
        L, B, _ = init_points.shape
        if h_t is None or c_t is None:
            h_t, c_t = self.get_init(B)
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl, B, self.nfeatures).to(device)
        # the init_points are used to initialize the memory of the LSTM
        init_pred, (h_t, c_t) = self.forward(init_points.to(device), h_t, c_t)
        if full_out:
            output[:offset] = init_pred
        #print("generate", h_t.shape,c_t.shape,init_pred[-1].shape)
        inp = init_pred[-1].unsqueeze(0).to(device)
        for i in range(offset, N + offset):
            inp, (h_t, c_t) = self.forward(inp.to(device), h_t, c_t)
            output[i] = inp
        return output, (h_t, c_t)


    def generate_new(self, init_points, N, h_t=None, c_t=None, full_out=False,train=True):
        """
        - the init_points are used to initialize the memory of the LSTM
        - N is the number of generated points by the model on its own
        
        Return a tensor of size (N+L, 1, 12)
        with L the length of init_points
        """
        L, B, _ = init_points.shape
        if h_t is None or c_t is None:
            h_t, c_t = self.get_init(B)
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl, B, self.nfeatures).to(device)
        # the init_points are used to initialize the memory of the LSTM
        init_pred, (h_t, c_t) = self.forward(init_points.to(device), h_t, c_t)
        if full_out:
            output[:offset] = init_pred
        #print("generate", h_t.shape,c_t.shape,init_pred[-1].shape)
        inp = init_pred[-1].unsqueeze(0).to(device)
        S=np.zeros([N+offset,L])
        for i in range(offset, N + offset):
            inp, (h_t, c_t) = self.forward(inp.to(device), h_t, c_t)
            output[i] = inp
            #-------------------TRY------------------#
            h_t.detach()
            c_t.detach()
            if (train==False):
                order=2
                if (i>5):
                    S[i]+=np.power(inp.cpu(),order)
        if (train==False):
            print(S/(N-5))
        return output, (h_t, c_t)    

class MLP(nn.Module):
    def __init__(self,
                 neurons=100,
                 nfeatures=12,
                 h=1,
                 f=25,
                 dropout=0,
                 device=device):                #changed here the device 
        print("inside seq device=",device)
        super(MLP, self).__init__()
        self.history=h
        self.future=f
        self.neurons = neurons
        self.nfeatures = nfeatures
        self.device = device

        self.MLP = nn.Sequential(
            nn.Linear(self.nfeatures*h, 2*self.neurons),
            nn.ReLU(),
            nn.Linear(2*self.neurons, self.neurons+100),
            nn.ReLU(),
            nn.Linear(self.neurons+100,self.nfeatures*f),
            nn.Sigmoid()
        ).to(device)    




    def forward(self, input):
        output = self.MLP(input.to(device))
        #print("forward output size =",output.view(-1, self.nfeatures *self.future).shape)
        return output#.view(-1, self.future, self.nfeatures)




    def generate_new(self, init_points, N, full_out=False):
        L, B, _ = init_points.shape
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl*self.future, B, self.nfeatures).to(device)
        if full_out:
            output[:offset*self.future] = init_points.repeat(self.future, 1, 1)
        # inp is of expected size 1, B, D (number of features)
        inp = init_points[-1].unsqueeze(0).to(device)
        for i in range(offset*self.future, (N + offset)*self.future):
            inp = self.forward(inp.to(device))
            output[i] = inp
            #output[i*self.future:i*self.future+self.future] = inp.view(self.future, -1)
            if self.future==1:
                inp = inp.unsqueeze(0)
            else:
                inp = inp.view(1, B, self.nfeatures*self.future)
                if i%self.future==self.future-1:
                    inp = inp[:, :, -self.nfeatures:]
        if self.future==1:
            return output.view(-1, B, self.nfeatures)
        else:
            return output.view(-1, B, self.nfeatures*self.future)



    def generate(self, init_points, N, full_out=False):

        #forecast case, the model receives data in shape (BatchSize,f*nshell) 
        
        if self.future==1:          #   std case, the model receives data in shape (L,bs,nshells)  tested ok
                
            L, B, _ = init_points.shape
            outl = N
            offset = 0
            if full_out:
                outl = N + L
                offset = L
            output = th.zeros(outl, B, self.nfeatures).to(device)
            if full_out:
                output[:offset] = init_points
            # inp is of expected size 1, B, D (number of features)
            inp = init_points[-1].unsqueeze(0).to(device)
            for i in range(offset, N + offset):
                inp = self.forward(inp.to(device))
                output[i] = inp
            return output 


            
            """ 
            
            #init_points=init_points.flatten()
            print("generating function")
            print(init_points.shape)
            L, B,_ = init_points.shape
            f=self.future
            outl = N
            offset = 0
            if full_out:
                outl = N + L
                offset = L
            output = th.zeros(B, L * self.nfeatures).to(device)
            #output = th.zeros(outl, B, self.nfeatures).to(device)
            print("ouput",output.shape)
            if full_out:
                output[:offset] = init_points
            # inp is of expected size 1, B, D (number of features)

            #inp = init_points[-1].unsqueeze(0).to(device)
            inp = init_points[-1].to(device)

            print("inp",inp.shape)
            
            for i in range(offset, (N + offset)//f):
                inp = self.forward(inp.to(device))           
                print("forward dimension=",inp.shape)
                output[i] = inp
            print("output",output.shape)
            return output.view(N,self.features)  """
        else:      #forecast case, training data are in dimension (bs,f*nshells)  

            L, B, _ = init_points.shape
            print("init point shape= ",init_points.shape)

            outl = N
            offset = 0
            if full_out:
                outl = N + L
                offset = L
            output = th.zeros( outl*self.future,B, self.nfeatures).to(device)
            
            if full_out:
                output[:offset] = init_points
            # inp is of expected size 1, B, D (number of features)
            inp = init_points[-1].unsqueeze(0).to(device)
            for i in range(offset, (N + offset)//self.future):
                print("begin input shape=",inp.shape)
                if i==0:
                    output=self.forward(inp.to(device)).view(self.future,1 , self.nfeatures)
                    inp=output[-1,:,:].unsqueeze(0)
                else:
                    future_points=self.forward(inp.to(device)).view( self.future,1, self.nfeatures)
                    output =th.cat((output,future_points),dim=0)
                    inp=future_points[-1,:,:].unsqueeze(0)
                print("forward shape",output.shape)
                print("inp shape =",inp.shape)
                
            #print(output[:100])
            print("max output",output.max(axis=1))
            return output#.view(N,1,self.nfeatures)



#######################--------STATISTICAL ANALYSIS----#################

def struct_func(u_n,order,h,f,title,save=False,folder=''):
    """
        Compute the Structure function given |U_n| :
            S_q= mean ( |U_n(t)|^q ) 
        In which mean is the time average.
    """
    print("evaluate structure function S_",order)
    Sq=np.mean(np.power(u_n,order),0)
        
    if save:
        np.save('./'+str(folder)+'/S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title),Sq.T)

        print('saved S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title))
    return Sq.T 



#path_data="./data/"
#path_model="./results_LBFGS/rawdata/spectrum/"
#name_data="Uf_N12.npy"     # Not so long dataset 


#path_data='./dataset/N12/'

#n=1
#sampling=30
#N=12
#knn=np.power(2,np.arange(N+4))
#kn=knn*np.power(2,-4.)
#U_true=load_data(path_data,name_data,sampling=sampling)



def load_struc_func(path):
    return np.load(path)[:,0]  # load_data add a dimension for training


def mean_struct_func(S_n):
    return np.mean(S_n,axis=0), np.std(S_n,axis=0)

#S_2_true=struct_func(U_true,2,h='nd',f='nd',title='S2')[:,0]
#S_4_true=struct_func(U_true,4,h='nd',f='nd',title='S4')[:,0]
#S_6_true=struct_func(U_true,6,h='nd',f='nd',title='S6')[:,0]

def load_S_n_vector(N,order,h,f,s=0.1,long=False,verbose=False):
    true=[]
    model=[]
    for i in range(1,N+1):
        try:

        
            if long:
                mod=load_struc_func(path_model+'S_'+str(order)+'_h_'+str(h)+'_f_'+str(f)+'empiricalrun_'+str(i)+'100K.npy')
                tru=load_struc_func(path_model+'S_'+str(order)+'_h_'+str(h)+'_f_'+str(f)+'truerun_'+str(i)+'100K.npy')
            
            else:
                mod=load_struc_func(path_model+'S_'+str(order)+'_h_'+str(h)+'_f_'+str(f)+'empiricalrun_'+str(i)+'.npy')
                tru=load_struc_func(path_model+'S_'+str(order)+'_h_'+str(h)+'_f_'+str(f)+'truerun_'+str(i)+'.npy')
            #print("mod shape = ",mod.shape,"true shape = ",tru.shape)
         
            model.append(mod)   
            true.append(tru)
            
            if verbose:
                print("discarded spectrum run ",i)
                print("diff =",np.linalg.norm(mod-tru))
            
        except FileNotFoundError:
            print("file",i," not found")
    model=np.asarray(model)
    true= np.asarray(true)
    #print("model shape ",model.shape,"true shape",true.shape)
    print("number of used spectrum=",len(model))
    #print("number of discarded spectrum=",N-len(model))
    return model, true



def plot_struct(S_true,S_model,S_model_std,h,f,title,save=True,show=False,long=False):
    N=12
    knn=np.power(2,np.arange(N+4))
    kn=knn*np.power(2,-4.)
    l=['model','true']

    Sdiff=S_model-S_model_std
    

    plt.figure(figsize=(14,10)) 

    plt.plot((kn[2:-2]),(S_true),'o-',label=rf'$S_2$'+str(l[1]))

    plt.errorbar((kn[2:-2]),(S_model),yerr=np.abs(S_model_std),label=rf'$S_2$'+str(l[0]))
    #plt.plot((kn[2:-2]),(S_model+S_model_std),'o-')
    #plt.plot((kn[2:-2]),np.abs(S_model-S_model_std),'o-',label='caz')
    #plt.fill_between((kn[2:-2]), np.abs(Sdiff), S_model+S_model_std,alpha=0.4,interpolate=True,color='blue')
    
    plt.xlabel(r'$k_n$', fontsize=40)
    plt.ylabel(r'$'+str(title)+'(k_n)$', fontsize=40)

    # plt.ylim([np.min(Sq[1])/10,np.max(Sq[1])*10]) # aesthetic options
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.loglog()
    if save:
        if long:
            plt.savefig('./results_LBFGS/plots/spectrum/strucfunc/statistic/'+title+'h_'+str(h)+'f_'+str(f)+'new2_100k')
        else:
            plt.savefig('./results_LBFGS/plots/spectrum/strucfunc/statistic/'+title+'h_'+str(h)+'f_'+str(f)+'new2')

def fit_exp(S_model,S_true,show=False,save=False):

    plt.figure(figsize = (14,10))
    ax = sns.regplot(x=np.log(kn[2:-2][2:-3]), y=np.log(S_model[2:-3]), color="g")
    plt.plot(np.log(kn[2:-2]),np.log(S_model),'o-',label=r'$S_1$')

    plt.legend()
    plt.savefig('./results_LBFGS/plots/spectrum/strucfunc/statistic')

def exponents(S,kn,nshells):


    if nshells==12:
            #print("shape inside exponents computing ",S.shape)
            return stats.linregress(np.log(kn[2:-2][2:-3]),np.log(S[2:-3]))
    if nshells==19:
            return stats.linregress(np.log(kn[2:-2][3:-4]),np.log(S[3:-4]))
    

def genSTDplots(Nsample,order,h,f,long=False,s=0.1,save=False):


    modelSn,trueSn=load_S_n_vector(Nsample,order,h,f,s,long)

    trueSn_mean,_=mean_struct_func(trueSn)
    modelSn_mean,_=mean_struct_func(modelSn)
    _,modelSn_std=mean_struct_func(modelSn)


    #print("diff = " ,modelSn_mean-modelSn_std)

    #print("S_",order," model mean =",modelSn_mean.shape,"\n model std =",modelSn_std,"\n","true",trueSn_mean.shape)
    expTrue=exponents(trueSn_mean)
    expModel=exponents(modelSn_mean)


    plot_struct(trueSn_mean,modelSn_mean,modelSn_std,h,f,title='S_'+str(order),long=True)


    print("S_",order,"true = ",expTrue.slope,"\n","S_",order," model = ",expModel.slope)
    if save:
        np.save(path_model+'/fit/fit_model_S_'+str(order)+'_h_'+str(h)+'_f_'+str(f),expModel)
