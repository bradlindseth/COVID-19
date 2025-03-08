import os
import pandas as pd
import numpy as np
import argparse

from dateutil.parser import parse, ParserError
from datetime import date
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

import random

class DateParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, parse(values).date())

parser = argparse.ArgumentParser(
    prog = 'Fiddler v7',
    description = 'Work out Coefficients',
    epilog = 'By Brad Lindseth',
    )
parser.add_argument('-A','--dateA', action=DateParser, help='String Date', default=date.today())
parser.add_argument('-B','--dateB', action=DateParser, help='String Date', default=date.today())
args = parser.parse_args()

class SIRDV_model():

    def get_N0(self, Target):
        Target_pd = pd.to_datetime(Target.strftime("%Y-%m-%d"))
        
        df_InputA = pd.read_csv('Values.csv')
        df_Input = df_InputA.loc[(df_InputA['Name'] == 'Population')]
        
        # df_main = pd.DataFrame({'Date':pd.date_range('2020-01-01',Today.strftime("%Y-%m-%d"))})
        
        df_Input['DateB'] = df_Input['DateB'].apply(pd.to_datetime)
        df_Input['DateA'] = df_Input['DateA'].apply(pd.to_datetime)
        df_Input['Days_Diff'] = (df_Input['DateB'] - df_Input['DateA']).dt.days
        df_Input['Slope'] = (df_Input['ValueB'] - df_Input['ValueA']) / df_Input['Days_Diff']

        df1 = df_Input.loc[(df_Input['DateB']>= Target_pd) & 
            (df_Input['DateA']<= Target_pd)]
        if df1.shape[0] == 1:
            for _, row1 in df1.iterrows():
                Value = row1['ValueA'] + row1['Slope']*(Target_pd - row1['DateA']).days
                # print(int(Value))
        elif df1.shape[0] == 0:
            Slope = df_Input['Slope'].mean()
            df2 = df_Input.sort_values(by=['DateA'], ascending=False).head(1)
            for _, row1 in df2.iterrows():
                Value = row1['ValueA'] + Slope*(Target_pd - row1['DateA']).days
                # print(int(Value))
    
        return int(Value)
    
    def __init__(self):
        self.Coefficient = {}
        self.count = 0
        
        self.df_Coefficients = pd.read_csv('Coefficients.csv')
        
        self.df_Data = pd.read_csv('output.csv')
        
        self.DateA = args.dateA 
        self.DateB = args.dateB

        ndf = self.df_Data.loc[
            (self.df_Data['Date'] >= self.DateA.strftime("%Y-%m-%d")) &
            (self.df_Data['Date'] <= self.DateB.strftime("%Y-%m-%d"))
            ]
        self.H_actual = ndf['Hbar'].to_numpy() # ndf['H: Hospitalizations'].to_numpy()    
        self.V_actual = ndf['V'].to_numpy()
        
        self.H0 = ndf['H: Hospitalizations'].to_numpy()[0]
        self.D0 = 0
        self.C0 = 0
        if np.isnan(self.V_actual[0]):
            self.V0 = 0
        else:
            self.V0 = 1*self.V_actual[0]
        self.Vdot = ndf['Vdot'].to_numpy()
        
        self.t = range(ndf.shape[0])

        self.Coefficients = {}
        df1 = self.df_Coefficients.loc[self.df_Coefficients['Fixed?'] == False]
        self.CoefficientList = df1['Coefficient'].tolist()

        df1 = self.df_Coefficients.loc[(self.df_Coefficients['Type'] == 'Int') & 
            (self.df_Coefficients['Fixed?'] == False) ]
        self.CoefficientListInt = df1['Coefficient'].tolist()

        df1 = self.df_Coefficients.loc[(self.df_Coefficients['Pegged?'] == True) ]
        self.CoefficientPegged = df1['Coefficient'].tolist()

        self.ColumnList = ['ssd']
        for a in self.CoefficientList:
            self.ColumnList.append(a)

        self.df_output = pd.DataFrame(columns=self.ColumnList)

        for _, row1 in self.df_Coefficients.iterrows():
            self.Coefficients[row1['Coefficient']] = {}
            self.Coefficients[row1['Coefficient']]['Tolerance'] = row1['Tolerance']
            if row1['Coefficient'] in self.CoefficientList:
                self.Coefficients[row1['Coefficient']]['Bounds'] = (row1['Low'], row1['High'])
                self.Coefficients[row1['Coefficient']]['Space'] =  row1['Space']
            else:
                self.Coefficients[row1['Coefficient']]['Value'] = row1['Low']
    
        self.N0 = self.get_N0(self.DateA)
    
    def random_guesses(self):
        
        self.count += 1
        
        for var in self.CoefficientList:
            if var in self.CoefficientListInt:
                self.Coefficients[var]['Value'] = round(random.choice(self.Coefficients[var]['Range']))
            else:
                self.Coefficients[var]['Value'] = random.choice(self.Coefficients[var]['Range'])
        
        if 'E' in self.CoefficientPegged:
            self.Coefficients['E']['Value'] = round(self.Coefficients['beta']['Value']*self.Coefficients['I']['Value']*(self.N0-self.Coefficients['I']['Value']-self.Coefficients['R']['Value']-self.V0)/self.N0/(self.Coefficients['eta']['Value'] + self.Coefficients['epsilon']['Value'] ))
            
        S = self.N0 - self.Coefficients['E']['Value'] - self.Coefficients['I']['Value'] - self.Coefficients['R']['Value'] - self.V0
        X0 = np.array([self.N0,S,self.Coefficients['E']['Value'], self.Coefficients['I']['Value'], self.Coefficients['R']['Value'],self.D0,self.C0,self.H0,self.V0])
        y = self.rk4( self.SIRDV, X0, self.t )
        
        H_predicted = y[:,7]
        
        Dict = {}
        Dict['ssd'] = self.ssd(H_predicted, self.H_actual)
        for var in self.CoefficientList:
            Dict[var] = self.Coefficients[var]['Value']   
        
        self.df_output = self.df_output.append(Dict, ignore_index = True)
        
        if self.count % 1000 == 0:
            os.system('clear')
            print(self.df_output[self.ColumnList].sort_values(by=['ssd'], ascending=True).head(50))
            print('Random Guesses!')
            
    def redo_bounds(self):
        print('Redoing Bounds!')
        df_slice = self.df_output.sort_values(by=['ssd'], ascending=True).head(25)
        
        for var in self.CoefficientList:
            if var in self.CoefficientListInt:
                self.Coefficients[var]['Bounds'] = (round(np.maximum(df_slice[var].mean()-df_slice[var].std(), self.Coefficients[var]['Bounds'][0])), round(df_slice[var].mean()+df_slice[var].std()))
            else:
                self.Coefficients[var]['Bounds'] = (np.maximum(df_slice[var].mean()-df_slice[var].std(), self.Coefficients[var]['Bounds'][0]),
                    (df_slice[var].mean()+df_slice[var].std()))       
        
        self.df_output = df_slice

    def test_bounds(self, n=3):
        print('testing Bounds!!!')
        N_Checks = n
    
        df_slice = self.df_output.sort_values(by=['ssd'], ascending=True).head(25)
        df_top = self.df_output.sort_values(by=['ssd'], ascending=True).head(1)
        for var in self.CoefficientList:
            self.Coefficients[var]['Value'] = df_top[var].mean() 
            #print(var, df_top[var].mean())

        # Reset self.df_output
        self.df_output = pd.DataFrame(columns=self.ColumnList)

        aaa = [x for x in self.CoefficientList if x not in self.CoefficientPegged] # self.CoefficientList #
        random.shuffle(aaa)
        
        for var in aaa:
            var_range = np.linspace(self.Coefficients[var]['Bounds'][0], self.Coefficients[var]['Bounds'][1], n)
            for value in var_range:
                self.count += 1
                if var in self.CoefficientListInt:
                    self.Coefficients[var]['Value'] = round(value)
                else:
                    self.Coefficients[var]['Value'] = value
                
                if 'E' in self.CoefficientPegged:
                    self.Coefficients['E']['Value'] = round(self.Coefficients['beta']['Value']*self.Coefficients['I']['Value']*(self.N0-self.Coefficients['I']['Value']-self.Coefficients['R']['Value']-self.V0)/self.N0/(self.Coefficients['eta']['Value'] + self.Coefficients['epsilon']['Value'] ))
                S = self.N0 - self.Coefficients['E']['Value'] - self.Coefficients['I']['Value'] - self.Coefficients['R']['Value'] - self.V0
                X0 = np.array([self.N0,S,self.Coefficients['E']['Value'], self.Coefficients['I']['Value'], self.Coefficients['R']['Value'],self.D0,self.C0,self.H0,self.V0])
                y = self.rk4( self.SIRDV, X0, self.t )
                
                H_predicted = y[:,7]
                
                Dict = {}
                Dict['ssd'] = self.ssd(H_predicted, self.H_actual)
                for var in self.CoefficientList:
                    Dict[var] = self.Coefficients[var]['Value']  
                
                self.df_output = self.df_output.append(Dict, ignore_index = True)
                
                if self.count % 1000 == 0:
                    os.system('clear')
                    print(self.df_output[self.ColumnList].sort_values(by=['ssd'], ascending=True).head(50))
                    
            best = self.df_output.sort_values(by=['ssd'], ascending=True).head(1)
            self.Coefficients[var]['Value'] = best[var].mean()
        

    def squeeze_bounds(self, n=3):
        print('Squeezing Bounds!')
        N_Checks = n
    
        df_slice = self.df_output.sort_values(by=['ssd'], ascending=True).head(25)
        df_top = self.df_output.sort_values(by=['ssd'], ascending=True).head(1)
        for var in self.CoefficientList:
            self.Coefficients[var]['Value'] = df_top[var].mean()
            if var in ['E','I','R']:
                self.Coefficients[var]['Range'] = np.linspace(round(self.Coefficients[var]['Value']-df_slice[var].std()),
                    round(self.Coefficients[var]['Value']+df_slice[var].std())
                    ,N_Checks)
            else:
                self.Coefficients[var]['Range'] = np.linspace(np.maximum(self.Coefficients[var]['Value']-df_slice[var].std(), 0),
                    self.Coefficients[var]['Value']+df_slice[var].std(),
                    N_Checks)

        # Reset self.df_output
        self.df_output = pd.DataFrame(columns=self.ColumnList)

        aaa = [x for x in self.CoefficientList if x not in ['E']]
        random.shuffle(aaa)
        
        print(aaa)
        
        count = 1
        for var in aaa:
            count = count * len(self.Coefficients[var]['Range'])
            random.shuffle(self.Coefficients[var]['Range'])
            print(var, self.Coefficients[var]['Range'])
        
        i = 0
        for var0 in self.Coefficients[aaa[0]]['Range']:
            for var1 in self.Coefficients[aaa[1]]['Range']:
                for var2 in self.Coefficients[aaa[2]]['Range']:
                    for var3 in self.Coefficients[aaa[3]]['Range']:
                        for var4 in self.Coefficients[aaa[4]]['Range']:
                            for var5 in self.Coefficients[aaa[5]]['Range']:
                                for var6 in self.Coefficients[aaa[6]]['Range']:
                                    for var7 in self.Coefficients[aaa[7]]['Range']:
                                        i += 1
                                        self.count += 1
                                        
                                        var = [var0, var1, var2, var3, var4, var5, var6, var7]
                                        # print(var)
                                        I = round(var[aaa.index('I')])
                                        R = round(var[aaa.index('R')])
                                        
                                        self.Coefficients['gamma']['Value'] = var[aaa.index('gamma')]
                                        self.Coefficients['epsilon']['Value'] = var[aaa.index('epsilon')]
                                        self.Coefficients['delta']['Value'] = var[aaa.index('delta')]
                                        self.Coefficients['beta']['Value'] = var[aaa.index('beta')]
                                        self.Coefficients['alpha']['Value'] = var[aaa.index('alpha')]
                                        self.Coefficients['phi']['Value'] = var[aaa.index('phi')]
                                    
                                        self.Coefficients['E']['Value'] = round(self.Coefficients['beta']['Value']*self.Coefficients['I']['Value']*(self.N0-self.Coefficients['I']['Value']-self.Coefficients['R']['Value']-self.V0)/self.N0/(self.Coefficients['eta']['Value'] + self.Coefficients['epsilon']['Value'] ))
                                        S = self.N0 - self.Coefficients['E']['Value'] - self.Coefficients['I']['Value'] - self.Coefficients['R']['Value'] - self.V0
                                        X0 = np.array([self.N0,S,self.Coefficients['E']['Value'], self.Coefficients['I']['Value'], self.Coefficients['R']['Value'],self.D0,self.C0,self.H0,self.V0])
                                        y = self.rk4( self.SIRDV, X0, self.t )
                                        
                                        H_predicted = y[:,7]
                                        
                                        Dict = {}
                                        Dict['ssd'] = self.ssd(H_predicted, self.H_actual)
                                        Dict['E'] = self.Coefficients['E']['Value']
                                        Dict['I'] = self.Coefficients['I']['Value']
                                        Dict['R'] = self.Coefficients['R']['Value']
                                        Dict['gamma'] = self.Coefficients['gamma']['Value']
                                        Dict['epsilon'] = self.Coefficients['epsilon']['Value']
                                        Dict['delta'] = self.Coefficients['delta']['Value']
                                        Dict['beta'] = self.Coefficients['beta']['Value']
                                        Dict['alpha'] = self.Coefficients['alpha']['Value']
                                        Dict['phi'] = self.Coefficients['phi']['Value']    
                                        
                                        self.df_output = self.df_output.append(Dict, ignore_index = True)
                                        
                                        if self.count % 1000 == 0:
                                            os.system('clear')
                                            print(self.df_output[['ssd', 'E', 'I', 'R', 'gamma', 'epsilon', 'delta', 'beta', 'alpha', 'phi' ]].sort_values(by=['ssd'], ascending=True).head(50))
                                            print('Squeezing Bounds is at %d of %d complete!' % (i, count) )

        return 

    def ssd(self, A,B):
      dif = A.ravel() - B.ravel()
      return np.dot( dif, dif ) / np.size(A)
    
    def display1(self):
        os.system('clear')
        print('Inital Conditions and Coefficients')
        print("Hospitalizations: %d " % (self.H0))
        
        for var in self.CoefficientList:
            if var in ['E','I','R']:
                print("%s: %d, %d" % (var, self.Coefficients[var]['Bounds'][0], self.Coefficients[var]['Bounds'][1]))
            else:
                print("%s: %.5f, %.5f" % (var, self.Coefficients[var]['Bounds'][0], self.Coefficients[var]['Bounds'][1]))
                

    def rk4(self, f, x0, t ):
        """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

        USAGE:
            x = rk4(f, x0, t)

        INPUT:
            f     - function of x and t equal to dx/dt.  x may be multivalued,
                    in which case it should a list or a NumPy array.  In this
                    case f must return a NumPy array with the same dimension
                    as x.
            x0    - the initial condition(s).  Specifies the value of x when
                    t = t[0].  Can be either a scalar or a list or NumPy array
                    if a system of equations is being solved.
            t     - list or NumPy array of t values to compute solution at.
                    t[0] is the the initial condition point, and the difference
                    h=t[i+1]-t[i] determines the step size h.

        OUTPUT:
            x     - NumPy array containing solution values corresponding to each
                    entry in t array.  If a system is being solved, x will be
                    an array of arrays.
        """

        n = len( t )
        x = np.array( [ x0 ] * n )
        for i in range( n - 1 ):
            h = t[i+1] - t[i]
            k1 = h * f( x[i], t[i] )
            k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h )
            k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h )
            k4 = h * f( x[i] + k3, t[i+1] )
            # print(k1, k2, k3, k4)
            x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

        return x

    def SIRDV(self, XX,t):
        dNN = np.array(len(XX))
        NN = XX[0]
        SS = XX[1]
        EE = XX[2]
        II = XX[3]
        RR = XX[4]
        DD = XX[5]
        CC = XX[6]
        HH = XX[7]
        VV = XX[8]
        
        dD = (self.Coefficients['zeta']['Value']*HH)
        
        dV = 0
        try:
            dV = self.Vdot[int(np.floor(t))]
            if np.isnan(dV):
                dV = 0
        except:
            pass
            
        dN = (self.Coefficients['Lambda']['Value'] - self.Coefficients['eta']['Value']*NN - dD)
        dS = (self.Coefficients['Lambda']['Value'] - self.Coefficients['eta']['Value']*SS - self.Coefficients['beta']['Value']*II*SS/NN+self.Coefficients['delta']['Value']*RR - dV)
        dE = (self.Coefficients['beta']['Value']*II*SS/NN - (self.Coefficients['epsilon']['Value']+self.Coefficients['eta']['Value'])*EE)
        dI = ((self.Coefficients['epsilon']['Value'])*EE-(self.Coefficients['gamma']['Value']+self.Coefficients['eta']['Value'])*II)
        dR = (self.Coefficients['gamma']['Value']*II-(self.Coefficients['delta']['Value']+self.Coefficients['eta']['Value'])*RR)
        dC = (self.Coefficients['epsilon']['Value']*EE)
        dH = (self.Coefficients['alpha']['Value']*(II-HH)-self.Coefficients['phi']['Value']*HH)
        
        dNN = np.array([dN,dS,dE,dI,dR,dD,dC,dH,dV])
        
        return dNN

    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

#-----------------------------------------------------------------------------

def main():
    Model = SIRDV_model()
    
    Model.display1()
    
    N = 11
    
    for var in Model.CoefficientList:
        if Model.Coefficients[var]['Space'] == 'geomspace':
            Model.Coefficients[var]['Range'] = np.geomspace(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], N)
        elif Model.Coefficients[var]['Space'] == 'linspace':
            Step = (Model.Coefficients[var]['Bounds'][1]-Model.Coefficients[var]['Bounds'][0])/N
            if Step <= Model.Coefficients[var]['Tolerance']:
                Model.Coefficients[var]['Range'] = np.arange(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], Model.Coefficients[var]['Tolerance'])
            else:
                Model.Coefficients[var]['Range'] = np.linspace(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], N)
        print(var)
        print(Model.Coefficients[var]['Range'])

    blah = 1
    for var in Model.CoefficientList:
        blah *= len(Model.Coefficients[var]['Range'])
    
    print(blah)

    # exit()
    
    print('Doing Random Guesses!')
    for i in tqdm(range(10000)):
        Model.random_guesses()
    Model.redo_bounds()
    Model.display1()

    N = 11

    for var in Model.CoefficientList:
        print(var)
        if Model.Coefficients[var]['Space'] == 'geomspace':
            Model.Coefficients[var]['Range'] = np.geomspace(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], N)
        elif Model.Coefficients[var]['Space'] == 'linspace':
            Step = (Model.Coefficients[var]['Bounds'][1]-Model.Coefficients[var]['Bounds'][0])/N
            if Step <= Model.Coefficients[var]['Tolerance'] and Step > 0.0:
                Model.Coefficients[var]['Range'] = np.arange(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], Model.Coefficients[var]['Tolerance'])
            else:
                Model.Coefficients[var]['Range'] = np.linspace(Model.Coefficients[var]['Bounds'][0], Model.Coefficients[var]['Bounds'][1], N)
        print(Model.Coefficients[var]['Range'])

    blah = 1
    for var in Model.CoefficientList:
        blah *= len(Model.Coefficients[var]['Range'])
    
    print(blah)
    # exit()

    print('Doing Random Guesses!')
    for _ in range(10000):
        Model.random_guesses()
    Model.redo_bounds()
    print(Model.V0)
    Model.display1()    
    
    # Model.test_bounds(n=101)
    # print(Model.df_output.sort_values(by=['ssd'], ascending=True).head(50))
    # Model.redo_bounds()
    # Model.display1()
    
    # df_output.at[i,'ssd'] = ssd(H_predicted,H_actual)
    
if __name__ == "__main__":
    main()
