import pandas as pd
import sys
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.optimize import minimize, Bounds

# enivorment tf2.4

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

df = pd.read_pickle('/home/halin/Master/Transformer/Test/ModelsResults/collections/dataframe.pkl')
#col_1 = 'model_num'
col_2 = 'encoder_type'
col_3 = 'embed_type'#
col_4 = 'final_type' 
#col_6 = 'num_param'
col_5 = 'normalization'
col_7 = 'pos_enc_type'
col_9 = 'seq_len'
col_10 = 'd_model'
col_11 = 'd_ff'
col_12 = 'N'
col_13 = 'h' 
col_14 = 'data_type'
col_15 = 'NSE_AT_10KNRF'
#col_16 = 'TRESH_AT_10KNRF'
col_17 = 'data_type'
#col_18 = 'MACs'
print(df[[col_4, col_5,col_7,  col_9, col_10, col_11, col_12, col_13,  col_15]])
df_2 = df[[col_2, col_4, col_5, col_7, col_9, col_10, col_11, col_12, col_13,  col_15]].dropna()

df_2[col_4] = pd.factorize(df_2[col_4])[0]+ 1
df_2[col_5] = pd.factorize(df_2[col_5])[0]+ 1
df_2[col_7] = pd.factorize(df_2[col_7])[0]+ 1

print(df_2)
correlation_matrix = df_2.corr()
print(correlation_matrix)

X = df_2[[col_4, col_5, col_7, col_9, col_10, col_11, col_12, col_13 ]]
y = df_2[col_15]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# To retrieve the intercept:
print('Intercept:', regressor.intercept_)

# For retrieving the slope (coefficient of x):
print('Coefficients:', regressor.coef_)

def func_to_minimize(variables, coefficients, intercept):
    final, norm, pos, seq, d_model, d_ff, N, h =  variables 
    
    return coefficients[0]*final + coefficients[1]*norm + coefficients[2]*pos + coefficients[3]*seq + coefficients[4]*d_model + coefficients[5]*d_ff + coefficients[6]*N + coefficients[7]*h + intercept - 1
# 'final_type' 'normalization' 'pos_enc_type' 'seq_len' 'd_model' 'd_ff' 'N' 'h'
initial_guess = [1,1,1,128,64,32,2,2]
bounds = Bounds([1,1,1,64,16,16,1,1], [3,3,3,2048,2048,2048,16,2048])
result = minimize(func_to_minimize, initial_guess, args=(regressor.coef_, regressor.intercept_), bounds=bounds)
print(result.x)