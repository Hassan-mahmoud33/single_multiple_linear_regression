
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def delete():
    os.system('cls' if os.name == 'nt' else 'clear')
delete()

path = r"C:\Users\B-UNIT\Desktop\ML\codes\datasets\profit\\data_single_var.txt"

data = pd.read_csv(path , header=None , names=["population", "profit"], skipinitialspace=True)



# to convert them into numeric
data['population'] = pd.to_numeric(data['population'] , errors='coerce') 
data['profit'] = pd.to_numeric(data['profit'] , errors='coerce') 


# show the data
def display():

    print(f"{'-' * 50}\n")
    print(data.info())
    print(f"{'-' * 50}\n")

    print(data.describe())
    print(f"{'-' * 50}\n")
    print(data.nunique())
    print(f"{'-' * 50}\n")

    data.plot(kind = 'scatter' , x = 'population', y = 'profit', figsize = (7 , 5))
    plt.show()

# display()


# column of ones
data.insert( 0 , 'ones' , 1)

# print(data.head(2))


# select input and output

cols = data.shape[1]  
X = data.iloc[ : , 0 : cols - 1]
y = data.iloc[ : , cols - 1 : cols]

# print(X.head(2))
# print(f"{'-' * 50}")
# print(y.head(2))

# seperate features for sklearn ( before converting them into matrix)
X_df = data[['population']]
y_df = data[['profit']]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix( np.array( [0 , 0]) )

# print(X[ : 3 ])
# print(f"{'-' * 50}")
# print(y[ : 3])
# print(f"{'-' * 50}")
# print(theta)


#***************************************************************************************
# manually calculate cost fun and gredient descent 

def compute_cost( X , y , theta) : 

    z = np.power ( ( X * theta.T ) - y  , 2 )
    # print(z)

    return np.sum(z) / ( 2 * len(X))

# print(f"compute_cost( X , y , theta) = {compute_cost( X , y , theta)} ")



def gredient( X , y , theta , alpha , iters):

    temp = np.matrix ( np.zeros( theta.shape))
    parameters = int ( theta.ravel().shape[1])
    cost = np.zeros(iters)

    exiting = 0

    for i in range(iters):

        error = ( X * theta.T ) - y

        for j in range(parameters) :
            
            term = np.multiply(error , X[0 : , j])
            temp[0 , j] = theta[0 , j] - ( ( alpha / len(X)) * np.sum(term) )

        theta = temp
        cost[i] = compute_cost( X , y , theta)

    # setup iteration when the cost fun decreases a too little it stops iterations
    # at alpha = 0.01

        if i > 0 and abs ( cost[i - 1] - cost[i] ) < 0.00003 :
            exiting += 1
            
            if exiting > 20 :
                cost = cost[ : i]
                break
        else :
            exiting = 0


    return theta , cost

alpha = 0.01
iters = 5000

theta , cost = gredient( X , y , theta , alpha , iters)

# print(f"cost = {cost}")
# print(f"cost = {len(cost)}")
# print(f"theta = {theta}")


# drawing

x = np.linspace(data['population'].min() , data['population'].max(), 100)
f = theta[ 0 , 0] + ( theta[ 0 , 1] * x )

def draw():

    fig , ax1  = plt.subplots(  figsize = (8 , 5))

    ax1.plot(x , f , 'red', label = 'prediction line' )
    ax1.scatter(data.population , data.profit , label = 'training data')
    
    ax1.set_xlabel('population')
    ax1.set_ylabel('profit')
    ax1.set_title('predicted profit vs population size')

    plt.show()

# draw()

# ***************************************************************************************


# prediction by sklearn

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit( X_df , y_df )

# # y_expected = lin_reg.predict([[70000]])
y_expected = lin_reg.predict(pd.DataFrame([[7] , [1] , [4]] , columns = ['population']))
print(f"predicted profit for 70k people = {y_expected[0][0]*10000:.2f}\n")
print(f"predicted profit for 10k people = {y_expected[1][0]*10000:.2f}")
print(f"predicted profit for 40k people = {y_expected[2][0]*10000:.2f}")


#***************************************************

# drawing

def draw_by_sklearn():

    x = np.linspace(data['population'].min() , data['population'].max(), 100)
    f = lin_reg.predict(pd.DataFrame( x , columns=['population']))

    fig , ( ax1 , ax2 )= plt.subplots( 1 , 2 , figsize = ( 8 , 5))

    ax1.plot( x , f , 'red', label = 'prediction')
    ax1.scatter(data.population , data.profit , label = 'training data')
    ax1.set_xlabel("population")
    ax1.set_ylabel("profit")
    ax1.set_title('predicted profit vs population size')

    ax2.plot( np.arange( len(cost)) , cost  )
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('cost')
    ax2.set_title('Erros vs Training epoch')

    plt.show()

draw_by_sklearn()