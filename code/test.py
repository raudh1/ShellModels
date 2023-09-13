import argparse
from shellmodels_lib import *
from libraries import *

parser = argparse.ArgumentParser(description='Train and generate trajectories using a shell model.')
parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'LSTM'], help='Type of model to use')
parser.add_argument('--h', type=int, default=1, help='Number of historical timesteps to consider')
parser.add_argument('--f', type=int, default=25, help='Number of future timesteps to predict')
parser.add_argument('--training', type=str, default='STD', choices=['STD', 'FOR'], help='Type of training to use')
parser.add_argument('--optim', type=str, default='adam',choices=['adam','lbfgs'], help='Choose the optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--patience', type=int, default=50, help='Number of epochs of loss not decreasing to stop the train')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
parser.add_argument('--print-every', type=int, default=10, help='Print progress every N epochs')
parser.add_argument('--noise', type=float, default=3.141592, help='Adversarial noise')
parser.add_argument('--nshells', type=int, default=12, help='Number of shells')
parser.add_argument('--dataset', type=str, default='10k',choices=['10k','50k'], help='choose dataset length')
parser.add_argument('--sampling', type=int, default=30, help='Sampling rate for the data')
parser.add_argument('--run', type=int, default=1, help='Run number')
parser.add_argument('--savemode', type=bool, default=False, help='Save all results')


args = parser.parse_args()
save_model=args.savemode
patience=args.patience
run=args.run
dataset=args.dataset
model_type = args.model
h = args.h
f = args.f
training_type = args.training
if training_type == 'STD':
    f=1
lr = args.lr
optimizer_type=args.optim
nepochs = args.epochs
bsize = args.batch_size
nprints = args.print_every
nshells=args.nshells
datapath ='../dataset/N'+str(nshells)+'/'
sampling = args.sampling
if nshells==12:
    sampling=30
    if dataset=='10k':
        datafile = 'Uf_N12.npy'
elif nshells==19:
    sampling=20
    if dataset=='10k':
        datafile = 'Uf_N19_200k_.npy'        #'Uf_N19_50k_2nd_test.npy' #'Uf_N19_1100k_.npy'
    elif dataset=='50k':
        datafile = 'Uf_N19_1100k_.npy'

#----------------------standard deviation of noise--------------#

noise=args.noise
print("noise inside test",noise)
#----------------------param physical data----------------------#
N=nshells
knn=np.power(2,np.arange(N+4))
kn=knn*np.power(2,-4.)


def folder_generator(save_data=False,save_plot=False,save_model=False,
                     training_type=training_type,model_type=model_type,nshells=nshells):

    folder='../results/'+str(training_type)+'/'+str(model_type)+'/'+'N'+str(nshells)+'/'
    
    if save_model:
        return folder+'model/'
        #print("folder",folder_generator(save_model=True))
    if save_data:
        return folder+'raw/'
        #print("folder",folder_generator(save_data=True))

    if save_plot:
        return folder+'plot/'
        #print("folder",folder_generator(save_plot=True))



if model_type == 'MLP':
    model = MLP(100, nshells, h, f, 0)
elif model_type == 'LSTM':
    model = Sequence(100, 1, nshells)

if optimizer_type=='adam':
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
elif optimizer_type=='lbfgs':
    optimizer = th.optim.LBFGS(model.parameters(),lr=lr)

data, (max, min) = load_data(datapath, datafile, sampling=sampling,nshells=nshells)
train_data, valid_data, test_data = split_data(data)

if training_type == 'STD':
        trainer = STDtraining(model, optimizer, train_data, valid_data,
                          history=h, future=f, nepochs=nepochs,patience=patience, bsize=bsize, nprints=nprints,noise=noise, save=save_model)
elif training_type == 'FOR':
    trainer = FORtraining(model, optimizer, train_data, valid_data,
                          history=h, future=f, nepochs=nepochs,patience=patience, bsize=bsize, nprints=nprints,noise=noise, save=save_model)

train_loss, valid_loss = trainer.train_model()

if model_type=='MLP':
    traj = model.generate(train_data, len(data)).cpu().detach().numpy()
if model_type=='LSTM':
    traj = model.generate(train_data, len(data))[0].cpu().detach().numpy()
print("TRAJ SHAPE ",traj.shape)
regen=traj*(max-min)+min
#print("max, min = ",max,min)
#print(traj[:,:,:])
model_exps=np.zeros([5,nshells])
true_exps=np.zeros([5,nshells])

data_true=data*(max-min)+min
for i in range(0,5): #evaluate struc funct from S_2 to S_10
    s=struct_func(regen,(i+1)*2,h=h,f=f,title='empirical'+'run_'+str(args.run),save=False)
    model_exps[i]=s[:,0]

    truth=struct_func(data_true,(i+1)*2,h=h,f=f,title='real'+'run_'+str(args.run),save=False)
    true_exps[i]=truth[:,0]

    #print(i)
    #print(s.shape)

model_exps_list=[]
true_exps_list=[]

for i in model_exps:
    #np.save(folder_generator(),-exponents(i,kn=kn,nshells=nshells))
    model_exps_list.append(-exponents(i,kn=kn,nshells=nshells).slope)
    #print(model_exps)
for j in true_exps:
    true_exps_list.append(-exponents(j,kn=kn,nshells=nshells).slope)

print("model exponents:",model_exps_list,"run number",str(run))
print("true exponents:",true_exps_list,"run number",str(run))
