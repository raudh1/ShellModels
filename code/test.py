from shellmodels_lib import *
from libraries import *



h = 1
f = 1
model=MLP(100,12,h,f,0)
#model=Sequence(100,1,12)
optimizer=th.optim.Adam(model.parameters(),lr=0.001)
data=load_data('../dataset/N12/','Uf_N12.npy',sampling=20)
train_data,valid_data,test_data=split_data(data)


#standard=STDtraining(mlp,optimizer,train_set,valid)





trainer = STDtraining(model, optimizer, train_data, valid_data, history=h, future=f, nepochs=100, bsize=32, nprints=2, save=False)
t, v = trainer.train_model()


#print(get_param(standard,optimizer))
#train,val=standard.train(mlp,optimizer,train_set,valid,10,10,save=False)
#print(standard)
#orecast=FORtraining()
#print(forecast)
