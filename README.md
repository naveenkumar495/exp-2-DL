# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.


## DESIGN STEPS
### STEP 1: 
Load the existing market customer data with features and their corresponding segment labels (A, B, C, D).

### STEP 2: 
Clean the dataset by handling missing values, encoding categorical variables, and normalizing numerical features.


### STEP 3: 
Divide the data into training and testing sets to evaluate the model’s performance.



### STEP 4: 
Initialize the neural network by defining input layer, hidden layers, output layer, activation functions, and loss function.

### STEP 5: 
Train the neural network using the training data by adjusting weights through backpropagation to minimize classification error.


### STEP 6: 
Evaluate the model using the test dataset and measure accuracy, precision, or loss.

### STEP 7:
Use the trained model to classify customers in the new market into segments A, B, C, or D.





## PROGRAM

### Name:Naveenkumar M

### Register Number:212224230183

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
        
    def forward(self, x):
       x=F.relu(self.fc1(x))
       x=F.relu(self.fc2(x))
       x=F.relu(self.fc3(x))
       x=self.fc4(x)
       return x



        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
   model.train()
  for epoch in range(epochs):
    for inputs,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Initialize model
model =PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

```

### Dataset Information
<img width="1240" height="226" alt="image" src="https://github.com/user-attachments/assets/b5cf2974-b617-4e31-9f2e-cacfe851296d" />

### OUTPUT

## Confusion Matrix
<img width="615" height="516" alt="image" src="https://github.com/user-attachments/assets/3353277a-2b6f-4393-ae33-6674c27348b0" />




## Classification Report
<img width="659" height="398" alt="image" src="https://github.com/user-attachments/assets/b8f5c8f9-6c1f-43f0-bf46-aa2cf12b3260" />


### New Sample Data Prediction
<img width="389" height="93" alt="image" src="https://github.com/user-attachments/assets/d46d1e6a-ffa4-47a7-96cb-2c9d4075c7cc" />



## RESULT
The neural network classification model was executed successfully and customer segments (A, B, C, and D) were predicted accurately for the given dataset.
