import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 



vars_float = [
              
]

vars_cat = [
  'Ticket',
  'Name',
  'Cabin',
  'Embarked'    

]

vars_int = [
            

]




def return_df_dummies(df, vars_cat):

  '''
  Função criada para realizar one-hot encoding das variáveis
  '''

  all_dummies = [] 
  for var in vars_cat:
    num_cat_unicos = df[var].value_counts().index
    if num_cat_unicos > 2:
      dummies_var = pd.get_dummies(df[var], drop_first =  True)
      all_dummies.append(dummies_var)

    elif num_cat_unicos == 2:
      dummie_double = np.where(df[var] == df[var].unique()[0], 0 )
      all_dummies.append(dummie_double)
      
  df_dummies = pd.concat(all_dummies)


def sweetviz_univariada(df):
    import sweetviz as sv 

    report = sv.analyze(df)
    return report.show_notebook()


def transformacoes(df, tipo_arquivo):
    if tipo_arquivo = 'csv':
      try:
        df = pd.read_csv(df)
      except:
        df = pd.read_csv(df, engine = 'python')
    
    return df 




class Separa_df:

  def __init__(self ,df, var_safra ,max_dev, coluna_target):
      self.df = df 
      self.var_safra = var_safra
      self.max_dev = max_dev 
      self.coluna_target = coluna_target


  def one_hot(self):
    from sklearn.preprocessing import OneHotEncoder 

    
    drop_enc = OneHotEncoder(drop = 'first').fit()

  def _divide_dev_oot(self):
      df_dev = self.df[self.df[self.var_safra] <= self.max_dev]
      df_oot = self.df[self.df[self.var_safra] >  self.max_dev]

      return df_dev, df_oot 

  
  def split_(self, cols_to_drop = []):
      from sklearn.model_selection import train_test_split 

      tmpX = self.df.drop(cols_to_drop)
      tmpXcol = tmpX.columns 

      X_train, X_test, y_train, y_test = train_test_split(self.df[tmpXcol], self.df['Target'])

      return X_train, X_test, y_train, y_test 




def corrige_tipo_variavel(df, var_explicat_cont = [], var_explicat_disc = [], var_explicat_cat = []):

  for var in var_explicat_cont:
    df[var] = df[var].astype(float)

  for var in var_explicat_disc:
    df[var] = df[var].astype(int)

  for var in var_explicat_cat:
    df[var] = df[var].astype(str)

  return df 

def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)