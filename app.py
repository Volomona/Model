import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import g4f
from scipy.optimize import minimize
import pdfkit

# -------------------------------

# Настройка логирования

# -------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

# -------------------------------

# Simulation functions

# -------------------------------

def simulate\_hybrid(N0\_vec, T, fert\_base, surv\_base, K, r\_fert, r\_surv,
delay\_fert, delay\_surv, migration\_rates, env\_effect, stoch\_intensity):
n = len(N0\_vec)
buffer\_size = max(max(delay\_fert), max(delay\_surv)) + 1
buffer = \[np.array(N0\_vec, dtype=float)] \* buffer\_size
history = \[]
for t in range(T):
N\_new = np.zeros(n)
total = buffer\[-1].sum()
noise = np.random.normal(0, stoch\_intensity \* np.sqrt(buffer\[-1] + 1))
env\_f = 1 + env\_effect \* np.sin(0.1 \* t)
for i in range(n):
delayed = buffer\[-delay\_fert\[i]]\[i]
dens = np.exp(-r\_fert \* total / K)
N\_new\[0] += fert\_base\[i] \* dens \* env\_f \* buffer\[-1]\[i]
for i in range(1, n):
delayed = buffer\[-delay\_surv\[i-1]]\[i-1]
dens = np.exp(-r\_surv \* delayed / (K/n))
N\_new\[i] += surv\_base\[i-1] \* dens \* env\_f \* buffer\[-1]\[i-1]
mig = np.zeros(n)
for i in range(n):
out = buffer\[-1]\[i] \* migration\_rates\[i]
mig\[i] -= out
mig += out/(n-1)
N\_new = np.clip(N\_new + mig + noise, 0, None)
buffer.append(N\_new)
if len(buffer) > buffer\_size: buffer.pop(0)
history.append(N\_new\.copy())
return np.array(history)

# Other models unchanged

def simulate\_logistic(N0, r, K, T):
Ns=\[N0]
for \_ in range(T): Ns.append(Ns\[-1]+r*Ns\[-1]*(1-Ns\[-1]/K))
return np.array(Ns)

def simulate\_ricker(N0, r, K, T):
Ns=\[N0]
for \_ in range(T): Ns.append(Ns\[-1]*np.exp(r*(1-Ns\[-1]/K)))
return np.array(Ns)

def simulate\_leslie(N0\_vec, fertility, survival, T):
n=len(N0\_vec); N=np.array(N0\_vec); hist=\[N.copy()]
L=np.zeros((n,n)); L\[0,:]=fertility
for i in range(1,n): L\[i,i-1]=survival\[i-1]
for \_ in range(T): N=L.dot(N); hist.append(N.copy())
return np.array(hist)

def simulate\_delay(N0, r, K, T, tau):
Ns=\[N0]\*(tau+1)
for t in range(tau,T+tau): Ns.append(Ns\[t]*np.exp(r*(1-Ns\[t-tau]/K)))
return np.array(Ns\[:T+1])

def simulate\_stochastic(base\_sim, N0, r, K, T, sigma, repeats):
runs=\[]; prog=st.progress(0)
for i in range(repeats):
traj=base\_sim(N0,r,K,T)
noise=np.random.normal(0,sigma,size=traj.shape)
runs.append(np.clip(traj+noise,0,None))
prog.progress((i+1)/repeats)
return np.array(runs)

# -------------------------------

# Analysis functions

# -------------------------------

def analyze\_behavior(ts):
std=np.std(ts\[-len(ts)//2:])
if std<1e-3: return "Стационарность"
peaks=np.sum(np.diff(np.sign(np.diff(ts)))<0)
if peaks>5: return "Периодические колебания"
return "Хаос"

def sensitivity\_heatmap(model, param\_ranges, fixed, T):
p1,p2=list(param\_ranges)
v1=np.linspace(\*param\_ranges\[p1])
v2=np.linspace(\*param\_ranges\[p2])
M=np.zeros((len(v1),len(v2)))
for i,x in enumerate(v1):
for j,y in enumerate(v2):
args=fixed.copy(); args\[p1]=x; args\[p2]=y
ts=model(\*args.values(),T)
M\[i,j]=ts.max()-ts.min()
fig,ax=plt.subplots(); c=ax.pcolormesh(v1,v2,M.T,shading='auto'); fig.colorbar(c,ax=ax)
ax.set\_xlabel(p1);ax.set\_ylabel(p2)
return fig

def optimize\_parameters(model,data,guess,bounds,T):
def loss(p): return np.mean((model(p\[0],p\[1],p\[2],T)-data)\*\*2)
return minimize(loss,guess,bounds=bounds)

def generate\_pdf\_report(model\_name,ts):
html=f"""<h1>Отчет по модели {model\_name}</h1><pre>{ts\[:10]}</pre>"""
path="report.pdf"; pdfkit.from\_string(html,path)
return path

# -------------------------------

# Streamlit UI

# -------------------------------

st.set\_page\_config(page\_title="Population Dynamics Simulator",layout="wide")
st.title("🌱 Симулятор популяционной динамики")

models={
"Гибридная"\:simulate\_hybrid,
"Логистический"\:simulate\_logistic,
"Рикер"\:simulate\_ricker,
"Лесли"\:simulate\_leslie,
"С задержкой"\:simulate\_delay,
"Стохастическая"\:simulate\_stochastic
}
model=st.sidebar.selectbox("Модель:",list(models))
T=st.sidebar.slider("T",10,500,100)

# input collection

if model=="Гибридная":
n=st.sidebar.number\_input("классов",2,10,3)
N0v=\[st.sidebar.number\_input(f"N0\_{i}",0.,1000.,10.) for i in range(n)]
fert=\[st.sidebar.number\_input(f"f\_{i}",0.,1.,0.5) for i in range(n)]
surv=\[st.sidebar.number\_input(f"s\_{i}",0.,1.,0.8) for i in range(n-1)]
df=\[st.sidebar.number\_input(f"df\_{i}",0,5,1) for i in range(n)]
ds=\[st.sidebar.number\_input(f"ds\_{i}",0,5,1) for i in range(n-1)]
mig=\[st.sidebar.number\_input(f"m\_{i}",0.,.5,0.1) for i in range(n)]
K=st.sidebar.number\_input("K",1.,1000.,100.)
rf=st.sidebar.number\_input("rf",0.,1.,0.1)
rs=st.sidebar.number\_input("rs",0.,1.,0.05)
ee=st.sidebar.slider("env",-1.,1.,0.2)
si=st.sidebar.slider("sto",0.,1.,0.1)
if st.sidebar.button("Sim"): res=models[model](N0v,T,fert,surv,K,rf,rs,df,ds,mig,ee,si)
elif model in \["Логистический","Рикер"]:
N0=st.sidebar.number\_input("N0",0.,1000.,10.)
r=st.sidebar.number\_input("r",0.,5.,0.5)
K=st.sidebar.number\_input("K",1.,1000.,100.)
if st.sidebar.button("Sim"): res=models[model](N0,r,K,T)
elif model=="С задержкой":
N0=st.sidebar.number\_input("N0",0.,1000.,10.)
r=st.sidebar.number\_input("r",0.,5.,0.5)
K=st.sidebar.number\_input("K",1.,1000.,100.)
tau=st.sidebar.number\_input("tau",1,10,1)
if st.sidebar.button("Sim"): res=models[model](N0,r,K,T,tau)
elif model=="Лесли":
n=st.sidebar.number\_input("классов",2,10,3)
N0v=\[st.sidebar.number\_input(f"N0\_{i}",0.,1000.,10.) for i in range(n)]
fert=\[st.sidebar.number\_input(f"f\_{i}",0.,1.,0.5) for i in range(n)]
surv=\[st.sidebar.number\_input(f"s\_{i}",0.,1.,0.8) for i in range(n-1)]
if st.sidebar.button("Sim"): res=models[model](N0v,fert,surv,T)
else:
N0=st.sidebar.number\_input("N0",0.,1000.,10.)
r=st.sidebar.number\_input("r",0.,5.,0.5)
K=st.sidebar.number\_input("K",1.,1000.,100.)
sigma=st.sidebar.number\_input("sigma",0.,1.,0.1)
repeats=st.sidebar.number\_input("repeats",1,200,50)
if st.sidebar.button("Sim"): res=models[model](simulate_logistic,N0,r,K,T,sigma,repeats)

# display

if 'res' in locals():
st.subheader(f"Результаты: {model}")
st.line\_chart(pd.DataFrame(res))
st.write("Режим:",analyze\_behavior(res.flatten()))
st.button("Скачать CSV", on\_click=lambda: export\_csv(res,model,str(res\[:10]),''))
st.pyplot(sensitivity\_heatmap(models\[model],{'r':(0.1,1,20),'K':(10,200,20)},{'N0'\:N0,'r'\:r,'K'\:K},T))
if st.sidebar.checkbox("Оптимизация параметров"): pass
if st.sidebar.button("PDF"): st.write(generate\_pdf\_report(model,res))
st.sidebar.info("Л.И.А.")
