import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


# Constants : 
NAN_VALUE_INT = 8888
NAN_STRING_TO_REPLACE = 'ZZZZ'


# DESCRIPTION : Ballance data for a pd.DataFrame and binary classification (labels (0,1) )
# INPUT ARGUMENTS :
# 	df : pd.DataFrame without the Id column, containing the label column
# OUTPUT : 
# X_res: pd.Dataframe for training ballanced containing the label column
def ballance_data_with_y(df, target_column_name):
    df_1 =  df[df[target_column_name]==1]
    df_0 =  df[df[target_column_name]==0]
    len1 = df_1.shape[0]
    len0 = df_0.shape[0]
    
    vmax = 0
    vmin = 1
    if len1 > len0:
        vmax = 1
        vmin = 0
        df_max = df_1
        df_min = df_0
    elif len1 < len0:
        vmax = 0
        vmin = 1
        df_max = df_0
        df_min = df_1
    else:
        return (df, Y)
    
    len_max = df_max.shape[0]
    len_min = df_min.shape[0]
    
    to_multiply = int(round(len_max/len_min))
    df_to_append = pd.concat([df_min] * to_multiply, ignore_index=True)
    
    len_append = df_to_append.shape[0]
    
    X_res = pd.concat([df_max, df_to_append], ignore_index=True)
    
    to_add = len_max - len_append
    if to_add > 0:
        df_to_add = df_min.sample(n=to_add, random_state=1)
        X_res = pd.concat([X_res, df_to_add], ignore_index=True)
    
    X_res = X_res.reset_index(drop=True)
    return X_res

# DESCRIPTION : Convert categories to numeric
# INPUT ARGUMENTS :
# 	df : train set
# 	df_all : train set + test set -- use it in order to cover all values in features
# 	columns_for_ordinal_encoder : list of columns for ordinal encoding
# 	ar_train_transformed : base array transformed with ordinal encoding
# 	enc : fit encoder
# OUTPUT : 
# df: encoded dataframe, ar_train_transformed_new: fit basis array, enc_new : encoder 
def convert_data_to_numeric(df, df_all, columns_for_ordinal_encoder, ar_train_transformed=None, enc=None):
    enc_new = enc
    
    # Fillna with ZZZZ which will be the last value in alphabetical order. 
    df_ordinal = df[columns_for_ordinal_encoder].fillna(NAN_STRING_TO_REPLACE).applymap(lambda x: str(x))
    df_ordinal_all = df_all[columns_for_ordinal_encoder].fillna(NAN_STRING_TO_REPLACE).applymap(lambda x: str(x))
    ar_train_transformed_new = ar_train_transformed
    if enc == None and ar_train_transformed == None:
        enc_new = OrdinalEncoder(dtype=np.int16)
        enc_new.fit(df_ordinal_all)
        
        ar_train_transformed_new = enc_new.transform(df_ordinal_all)
    
    ar_ordinal_transformed = enc_new.transform(df_ordinal)
    count_columns = 0
    for cn in columns_for_ordinal_encoder:
        this_col_all = ar_train_transformed_new[:,count_columns]
        # Find the index of ZZZZ, last one always, in our case, as OrdinalEncoder encodes by alphabetic order
        mx = this_col_all.max()
        
        this_col_train = ar_ordinal_transformed[:,count_columns]
        this_col_nan_train = np.where(this_col_train==mx, NAN_VALUE_INT, this_col_train)
        
        ar_ordinal_transformed[:,count_columns] = this_col_nan_train
        count_columns = count_columns+1

    df_ordinal_transformed = pd.DataFrame(ar_ordinal_transformed, columns=columns_for_ordinal_encoder)
    
    df.update(df_ordinal_transformed)      
    
    # Fill NaN with -1 first : 
    df = df.fillna(-1)
    
    # Optimize : 
    columns_only_int8 = [cn for cn in df.columns if df[cn].nunique() < 128]   
    columns_only_int16 = [cn for cn in df.columns if df[cn].nunique()>=128 & df[cn].nunique()<32768]  
    columns_only_int32 = [cn for cn in df.columns if df[cn].nunique()>=32768 & df[cn].nunique()<2147483648]    
    
    df[columns_only_int8] = df[columns_only_int8].astype('int8')
    df[columns_only_int16] = df[columns_only_int16].astype('int16')
    df[columns_only_int32] = df[columns_only_int32].astype('int32')
    
    df = df.applymap(lambda x: NAN_VALUE_INT if x == -1 else x)
    
    return (df, ar_train_transformed_new, enc_new)

# DESCRIPTION : Parse OneHotEncoding for train and test, and for the features in cols
# INPUT ARGUMENTS :
# 	df_train : train set
#	df_test : test set
# 	cols : list of columns to encode
# OUTPUT : 
# df_train: encoded train set;  df_test : encoded test set
def get_ohe(df_train, df_test, cols):
    new_cols = ["ohe_"+col for col in cols]
    
    size_train = df_train.shape[0]
    size_test = df_test.shape[0]    
        
    df = df_train.append(df_test, ignore_index=True, sort=False)
    df_ohe = df.loc[:, cols].astype('category')
    # replace NAN_VALUE_INT with nan, required by the function :
    df_ohe = df_ohe.applymap(lambda x: np.nan if x==NAN_VALUE_INT else x)
    
    df_ohe = pd.get_dummies(df_ohe, prefix=new_cols, sparse=True, columns=cols)
     
    index_test_start = size_train
    index_test_end = size_train+size_test-1
    
    df_train = df_ohe.loc[0:size_train-1,:].astype('int16')
    df_test = df_ohe.loc[index_test_start:index_test_end,:].astype('int16')
    return (df_train, df_test)


# DESCRIPTION : Encode pandas DataDrame with SimpleImputer
# INPUT ARGUMENTS :
# 	df : input dataframe
# 	cols : columns to encode
#	strategy : strategy to encode 
# 	si : encoder already fit, to apply
# OUTPUT : 
# df : encoded dataframe, si : encoder already fit 

def encode_missing_simple_imputer(df, cols, strategy, si=None):
    df_to_transform = df[cols]
    
    if si==None:
        si = SimpleImputer(strategy=strategy, missing_values=NAN_VALUE_INT)
        ar = si.fit_transform(df_to_transform)
    else:
        ar = si.transform(df_to_transform)
    
    df_transformed = pd.DataFrame(ar, columns=cols)
    if strategy == 'most_frequent':
        df_transformed = df_transformed.applymap(lambda x: str(x)).astype('category')
       
    df.update(df_transformed)
    
    return (df, si)
