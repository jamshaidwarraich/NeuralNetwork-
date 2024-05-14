import numpy as np
import scipy.special
class neural_network:
    def __init__(self,inodes,hnodes,onodes,learningrates):
        self.inputnode=inodes
        self.hiddennode=hnodes
        self.outputnode=onodes
        self.rate=learningrates
        self.wih=np.random.normal(0.0,pow(self.hiddennode,-0.5),(self.hiddennode,self.inputnode))
        self.who=np.random.normal(0.0,pow(self.hiddennode,-0.5),(self.outputnode,self.hiddennode))
        self.activation= lambda x: scipy.special.expit(x)    
        pass
    def train (self,user_input,labeled_output):
        inputs=np.array(user_input,ndmin=2).T                   #inputs given by user
        labeledoutputs=np.array(labeled_output,ndmin=2).T       #desired outputs
        
        hinput=np.dot(self.wih, inputs)                         #input for hidden layer
        houtput=self.activation(hinput)                         #output from hidden layer
        
        finalinput=np.dot(self.who, houtput)                     #output layer input
        finaloutput=self.activation(finalinput)                  #finaloutput
        
        outputerror=finaloutput-finaloutput              #error from final layer
        hiddenerror=np.dot(self.who.T, outputerror)               #error from hidden layer
        
        #updating the weights
        self.who += self.rate * np.dot((outputerror * finaloutput * (1 - finaloutput)), np.transpose(houtput))
        self.wih += self.rate * np.dot((hiddenerror * houtput * (1 - houtput)), np.transpose(inputs))

        pass 
    def query (self,user_input):
        inputs=np.array(user_input,ndmin=2).T
        hinput=np.dot(self.wih, inputs)                         #input for hidden layer
        houtput=self.activation(hinput)                         #output from hidden layer
        
        finalinput=np.dot(self.who, houtput)                     #output layer input
        finaloutput=self.activation(finalinput)                  #finaloutput
        return finaloutput
inodes=784
hnodes=24
onodes=23
learningrates=0.3
NN=neural_network(inodes, hnodes, onodes, learningrates) 
data_file = open('B:/Spyder python/Data1/mnist_train.csv', 'r')

data_list=data_file.readlines()
data_file.close()

for record in data_list:
    all_values=record.split(',')
    inputs=(np.asfarray(all_values[1:])/255*0.99)+0.01
    targets=np.zeros(onodes)+0.01
    targets[int(all_values[0])]=0.99
    NN.train(inputs, targets)
    pass

#testing
testing_file = open('B:/Spyder python/Data1/mnist_test.csv', 'r')
testing_list = testing_file.readlines()
testing_file.close()

scorecard = []
# Iterate over a subset of the testing data, e.g., the first 10 records
for record in testing_list[:10]:
    each_entry = record.split(',')
    test_inputs = (np.asfarray(each_entry[1:]) * 255 * 0.99) + 0.01
    target_label = int(each_entry[0])
    print("target label=", target_label)
    outputs = NN.query(test_inputs)
    NN_label = np.argmax(outputs)
    print("network given label=", NN_label)
    if (NN_label == target_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
pass

print(scorecard)
scorecard_array=np.array(scorecard)
performance=scorecard_array.sum()/scorecard_array.size
print(performance)

      
        

