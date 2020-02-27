import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename
from tkinter import *

class Fraud:
    
    def __init__(self, master):
        self.master = master
        master.title("Fraud Detector")
        
        self.label = Label(master, text="Choose the CSV file to work on: ")
        
        self.uploadbutton = Button(master, text="Upload", command=self.uploadfunc)
        
        #layout
        self.label.grid(row=0, column=0, sticky=W)
        self.uploadbutton.grid(row=3, column=3)
        
    def uploadfunc(self):
        filename = askopenfilename()
        dataset = pd.read_csv(filename)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        self.success = Label( text="File successfully uploaded")
        self.success.grid(row=4, column=4)
        self.graphbutton = Button(text="Generate Graph", command=lambda: self.graph(dataset,X,y))
        self.graphbutton.grid(row=5, column=4)

    def graph(self,dataset,X,y):
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        X = sc.fit_transform(X)


        from minisom import MiniSom
        som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
        som.random_weights_init(X)
        som.train_random(data = X, num_iteration = 100)


        from pylab import bone, pcolor, colorbar, plot, show
        bone()
        pcolor(som.distance_map().T)
        colorbar()
        markers = ['o', 's']
        colors = ['r', 'g']
        for i, x in enumerate(X):
            w = som.winner(x)
            plot(w[0] + 0.5,
                 w[1] + 0.5,
                 markers[y[i]],
                 markeredgecolor = colors[y[i]],
                 markerfacecolor = 'None',
                 markersize = 10,
                 markeredgewidth = 2)
        show()
        
        self.entry1_value = IntVar()
        self.entry2_value = IntVar()
        self.entry3_value = IntVar()
        self.entry4_value = IntVar()
        self.entry1 = Entry(root,textvariable=self.entry1_value,width=25)
        self.entry2 = Entry(root,textvariable=self.entry2_value,width=25)
        self.entry3 = Entry(root,textvariable=self.entry3_value,width=25)
        self.entry4 = Entry(root,textvariable=self.entry4_value,width=25)
        self.entry1.grid(row=6, column=2)
        self.entry2.grid(row=6, column=4)
        self.entry1.grid(row=7, column=2)
        self.entry1.grid(row=7, column=4)
        
        mappings = som.win_map(X)
        frauds = np.concatenate((mappings[(self.entry1,self.entry2)], mappings[(self.entry3,self.entry4)]), axis = 0)
        frauds = sc.inverse_transform(frauds)
        
        self.trainbutton= Button(text="Get probabilities", command=lambda: self.neural(dataset,frauds))
        self.trainbutton.grid(row=8, column=4)


    def neural(self,dataset,frauds):
        customers = dataset.iloc[:, 1:].values

        
        is_fraud = np.zeros(len(dataset))
        
        for i in range(len(dataset)):
            if dataset.iloc[i,0] in frauds:
                is_fraud[i] = 1
        
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        customers = sc.fit_transform(customers)
        
       
        from keras.models import Sequential
        from keras.layers import Dense
        
        
        classifier = Sequential()
        
        classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
        
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)
        
        y_pred = classifier.predict(customers)
        y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
        y_pred = y_pred[y_pred[:, 1].argsort()] 
    
    

root = Tk()
my_gui = Fraud(root)
root.mainloop()