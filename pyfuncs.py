import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import OneHotEncoder
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




def sweetviz_univariada(df):
    import sweetviz as sv 

    report = sv.analyze(df)
    return report.show_notebook()


def transformacoes(df, tipo_arquivo):
    if tipo_arquivo == 'csv':
      try:
        df = pd.read_csv(df)
      except:
        df = pd.read_csv(df, engine = 'python')
    
    return df 



class Preprocessamento:

  def __init__(self, df, vars_continuas, vars_discretas):
    self.df = df
    self.vars_cat = vars_cat
    self.vars_continuas = vars_continuas
    self.vars_discretas = vars_discretas 

  def return_one_hot(self, N = 100):
  
    var_cat_to_one_hot = [var for var in self.vars_cat if self.df[var].nunique() < N]

    enc = OneHotEncoder(handle_unknown = 'ignore')

    colunas_one_hot = enc.fit_transform(self.df[var_cat_to_one_hot]).toarray()

    #tmpDf = pd.DataFrame(colunas_one_hot, columns = enc.get_feature_names()) 

    self.df[enc.get_feature_names()] = colunas_one_hot

    self.df = self.df.drop(var_cat_to_one_hot, axis = 1)


  def padroniza(self, cols_escalonar):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    colunas_escalonadas = scaler.fit_transform(self.df[cols_escalonar])

    self.df[cols_escalonar] = self.df[colunas_escalonadas]

    self.df = self.df.drop(cols_escalonar)
    #tmpDf = pd.DataFrame(colunas_escalonadas, columns = cols_escalonar)

  @staticmethod
  def _retorna_tipos_validos(self, tipo, tipos_validos):
    if tipo not in tipos_validos:
      raise ValueError(f"Preencha corretamente, conforme opções em {tipos_validos}")

  

class split:

  def __init__(self,df):
    self.df = df 

  def split_df(self, divisao, var_explicat, target):
    from sklearn.model_selection import train_test_split 
    divisoes = ('OOS','OOT', 'Ambos')

    if divisao == 'OOS':
      X, y = train_test_split(self.df[var_explicat], self.df[target])    


  def amostragem(self, tipo, frac):
    
    tipos_validos = ('desbalanceado', 'under', 'over')

    resultado = self._retorna_tipos_validos(tipo = tipo, tipos_validos = tipos_validos)


    if tipo == 'desbalanceado':
      self.df = self.df.sample(frac = frac)

    elif tipo == 'balanceado':

    return 

  
  


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