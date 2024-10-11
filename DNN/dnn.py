import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(15, 15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(15, 15)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(15, 15)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class dnn():
   
    def __init__(self, n_epochs = 250, batch_size=10):
        self.n_epochs = n_epochs
        self. batch_size = batch_size
        self.model = None
        
    def fit(self, X_train0, y_train0, weights0=0):
    
        if len(weights0)==1:
            weights0 = np.ones(X_train0.shape[0])
        
        weights0 = weights0/np.sum(weights0)
        
        X_train = torch.tensor(X_train0, dtype=torch.float32)
        y_train = torch.tensor(y_train0, dtype=torch.float32).reshape(-1, 1)
        weights = torch.tensor(weights0, dtype=torch.float32).reshape(-1, 1)

        
        self.model = DeepNN()
        
        # loss function and optimizer
        loss_fn = nn.BCELoss(reduction='none')  # binary cross entropy
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
   
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        batch_start = torch.arange(0, len(X_train), batch_size)


        for epoch in range(n_epochs):
            # print(epoch)
            self.model.train()
            with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    weights_batch = weights[start:start+batch_size]
                    # forward pass
                    y_pred = self.model(X_batch)
                    loss = weights_batch* loss_fn(y_pred, y_batch)
                    loss = loss.sum()
                
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()

            self.model.eval()
            y_pred_train = self.model(X_train)
            loss1 = weights* loss_fn(y_pred_train, y_train)
            loss1 = loss1.sum()
            #print(loss1.float())
 
        
        return self

        
    def predict_proba(self, X_val0):
        X_val = torch.tensor(X_val0, dtype=torch.float32)
        y_prob_tensor = self.model(X_val)
        y_prob = y_prob_tensor.detach().numpy()[:,0]
        q_xall = np.zeros((X_val0.shape[0],2))
        q_xall[:,0] = 1-y_prob
        q_xall[:,1] = y_prob
       
        return q_xall 






