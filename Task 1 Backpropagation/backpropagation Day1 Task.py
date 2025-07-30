import numpy as np 

#initializing all  input matrix their weights and biases 
#let say we have 2 input features 2 nuron in first hidden layer 2 in second and finally 1 in output layers

x= np.array([[1],[2]])     #input layer
y=np.array([[1]])           # actual_output
w1 = np.array([[0.5 , -0.7],
               [2.6, 0.9]])     # weights of the input layers that will pass to first hidde layer
biases1= np.array([[0.1],[1.1]])  # biases of the input layers that will pass to first hidde layer


w2 = np.array([[2.4 , 2.0],
               [0.4 , 0.7]])        # weights of the FHL layers that will pass to second hidde layer
biases2= np.array([[0.2],[0.2]])      # biases of the FHL layers that will pass to second hidde layer



w3 = np.array([[0.2 ,0.6 ]])        # weights of the SHL layers that will pass to output layer
biases3= np.array([[0.3]])      # biases of the SHL layers that will pass to output layer



learning_rate = 0.1        #learning rate that will be use in gradient 


#as we are using Sigmoid acivation function so we will define a function for it
def sigmoid(z):
                return 1 / (1 + np.exp(-z))
            
   
#To calculate sigmoid dericatives at each layers during backpropagation            
def sigmoid_derivative(a):
                return a * (1 - a)



#main training function that just need number of epochs(number of repetitions in training the neural network to reach at good accuracy (near actual output))
def training(epochs):
    global w1, w2, w3, biases1, biases2, biases3 
    epochs_counter=0
    while(epochs>epochs_counter):
        
            
            #FeedForwardNetwork :
            # we need to multiply weights and inputs and add biases to them (for each layers)and resultant sum will be send to sigmoid function 

            z1 = np.dot(w1,x )+ biases1

            

            a1= sigmoid(z1)

            #now moving toward the next layer(2)

            z2 = np.dot(w2,a1)+biases2
            a2= sigmoid(z2)

            #now applying the same with third and final(output layer where we will get predicted output)

            z3= np.dot(w3,a2)+biases3
            y_hat = sigmoid(z3)
            print(f"Predicted Output of Epoch {epochs_counter+1}:", y_hat)




            #BackPropagation
            #now we need to do backpropagate to minimize error 


            #finding Mean Square Error MSE
            L = 0.5*(y-y_hat)**2
            print(L)


            #This loss is dependent on the weights and biases of all the hidden and output layers so inorder to minimize it we need to change all the weights and biases for which actually this backpropagation theorem is used so we will use the chain rule and finding the derivatives we wil change the weights and biases.




            #Outputlayer
            delta3 = (y_hat- y)*sigmoid_derivative(y_hat)
            dw3= np.dot(delta3, a2.T)
            db3 = delta3

            #Secondlayer

            delta2 = np.dot(w3.T,delta3)* sigmoid_derivative(a2)
            dw2 = np.dot(delta2, a1.T)                               
            db2 = delta2


            #Firstlayer

            delta1 = np.dot(w2.T,delta2)* sigmoid_derivative(a1)
            dw1 = np.dot(delta1, x.T)                               
            db1 = delta1



            #updating weights for next epoches
            w3 -= learning_rate * dw3
            biases3 -= learning_rate * db3
            w2 -= learning_rate * dw2
            biases2 -= learning_rate * db2
            w1 -= learning_rate * dw1
            biases1 -= learning_rate * db1
            
            
            epochs_counter+=1
            
            
            
#taking the number of epochs which will send to training function as a perameter
epochs= int(input("Enter the Number of epochs you want to Train for ?    "))
training(epochs)