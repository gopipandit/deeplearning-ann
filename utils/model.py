class Perceptron:
    
    def __init__ (self, eta, epochs):
        self.eta = eta #learning rate
        self.epochs = epochs #No of iterations
        self.weights = np.random.randn(3) * 1e-4 #weight initiliztion
        print(f"initial weights before training: \n{self.weights}")
        
        
    def activateFunction(self, inputs, weights):
        z  =np.dot(inputs,weights) # z = W*X
        return np.where(z>0, 1, 0)
        
    def fit(self, X,y):
        self.X = X
        self.y = y
        
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        print(f"X with bias: \n{X_with_bias}")
        for epoch in range(self.epochs):
            print('--'*10)
            print(f"for epoch: {epoch}")
            print('--'*10)
            
            y_hat = self.activateFunction(X_with_bias, self.weights) #Forward propogation
            print(f"predicted value after forward pass: \n{y_hat}")
            self.error = self.y  -y_hat
            print(f"error: \n{self.error}")
            self.weights = self.weights + self.eta*np.dot(X_with_bias.T,self.error) #Backword Propogation
            print(f"updated weights after epoch:\n{epoch}/{self.epochs} : \n{self.weights}")
            print("####"*10)
            
            
    def predict(self, x):
        x_with_bias = np.c_[x, -np.ones((len(x), 1))]
        return self.activateFunction(x_with_bias, self.weights)
        
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total loss : {total_loss}")
        return total_loss
        