import FinalLearnM
import pandas as pd
import matplotlib as plt
import numpy as np
from tkinter import messagebox
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt; 
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import plotly.plotly as py
import plotly.figure_factory as ff
plt.rcdefaults()

class ABC(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()   
        
root = Tk()
app = ABC(master=root)
app.master.title("Energon")

frame1 = Frame(root,height=300,width=600)
frame1.pack()
frame2 = Frame(root,height=10,width=600)
frame2.pack()
frame3 = Frame(root,height=10,width=600)
frame3.pack()
frame4 = Frame(root,height=30,width=600)
frame4.pack()
frame5 = Frame(root,height=30,width=600)
frame5.pack()
frame6 = Frame(root,height=30,width=600)
frame6.pack()
frame7 = Frame(root,height=30,width=600)
frame7.pack()
frame8 = Frame(root,height=30,width=600)
frame8.pack()
frame9 = Frame(root,height=30,width=600)
frame9.pack()
frame10 = Frame(root,height=30,width=600)
frame10.pack()

image = Image.open('s.gif')
photo_image = ImageTk.PhotoImage(image)
label6 = Label(frame1, image = photo_image)
label6.pack()  

def submit():
    state = entry1.get()
    df = pd.read_csv("book1.csv")
    
    df_state = df.iloc[:,0]
    
    if state not in list(df_state):
        messagebox.showerror("ERROR!!!","Invalid State")
   
def Pie():
    state = entry1.get()
    year = int(entry2.get())
    df = pd.read_csv("book1.csv")
         
    df_total = df.iloc[:,-2]
    total = list(df_total)
    df_states = df.iloc[:,0]
    df_states = list(df_states)
    total_array = np.array(total,dtype=float)

    coal = list(df.iloc[:,1])
    gas = list(df.iloc[:,2])
    disel = list(df.iloc[:,3])
    nuclear= list(df.iloc[:,4])
    hydro = list(df.iloc[:,5])
    res = list(df.iloc[:,6])

    ix = df_states.index(state)

    if year==2013:
        ix=ix
    elif year==2014:
        ix=ix+1
    elif year==2015:
        ix=ix+2
    elif year==2016:
        ix=ix+3
    else:
        ix=ix+4
    
    c=coal[ix]
    g=gas[ix]
    d=disel[ix]
    n=nuclear[ix]
    h=hydro[ix]
    r=res[ix]
    width=0.1
    
    print (c,g,d,n,h,r)
    labels2 = [' ',' ',' ',' ', ' ',' ']
    labels = ['Coal','Gas','Disel','Nuclear','Hydro','Renewabel']
    sizes = [c,g,d,n,h,r]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red','blue']
    explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1)  # explode 1st slice
    
    texts = plt.pie(sizes, explode=explode, labels=labels2, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.legend(texts[0],labels, loc="right")
    plt.axis('equal')
    plt.show() 

    
def Bar():
    state = entry1.get()    
    df = pd.read_csv("book1.csv") 
    
    df_total = df.iloc[:,-2]
    total = list(df_total)
    df_states = df.iloc[:,0]
    df_states = list(df_states)
    total_array = np.array(total,dtype=float)
    
    coal = df.iloc[:,1]
    gas = df.iloc[:,2]
    disel = df.iloc[:,3]
    nuclear= df.iloc[:,4]
    hydro = df.iloc[:,5]
    res = df.iloc[:,6]
    
    ix = df_states.index(state)
    
    c1=coal[ix:ix+5]
    g1=gas[ix:ix+5]
    d1=disel[ix:ix+5]
    n1=nuclear[ix:ix+5]
    h1=hydro[ix:ix+5]
    r1=res[ix:ix+5]
    
    listl = [0,1,2,3,4]
    ind = np.array(listl)
    
    labels1 = [2013,2014,2015,2016,2017]
    
    width = 0.1
    fig=plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind,c1,width,color='gold')
    rects2 = ax.bar(ind+width,g1,width,color='yellowgreen')
    rects3 = ax.bar(ind+width*2,d1,width,color='lightcoral')
    rects4 = ax.bar(ind+width*3,n1,width,color='lightskyblue')
    rects5 = ax.bar(ind+width*4,h1,width,color='red')
    rects6 = ax.bar(ind+width*5,r1,width,color='blue')
    
    ax.set_ylabel('Energy')
    ax.set_xlabel('Year')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(labels1)
    
    plt.show()
    
label1 = Label(frame2,text = "Enter State ",fg = 'green')
label1.grid(row = 0)
label5 = Label(frame2,text = "Year ",fg = 'green')
label5.grid(row = 0,column=4)
entry1 = Entry(frame2,bd=4)
entry1.grid(row=0,column=1)
entry2 = Entry(frame2,bd=4,width=8)
entry2.grid(row=0,column=6)

label2 = Label(frame6,text = "                   ")
label2.grid(row = 0,column=3)
label3 = Label(frame8,text = "2017 Actual India Population Map ")
label3.grid(row = 0)
label4 = Label(frame10,text = "2017 Prediction Of Eneregy In India   ")
label4.grid(row = 0)

button1 = Button(frame4,text="Submit",activeforeground='red',bd=4,command=submit)
button1.grid(row=2,column=1)

button2 = Button(frame6,text="Pie Chart",command=Pie,fg='blue',activeforeground='red',bd=4)
button2.grid(row=0,column=0)

button3 = Button(frame6,text="Bar Graph",command=Bar,fg='blue',activeforeground='red',bd=4)
button3.grid(row=0,column=4,rowspan=5)

button4 = Button(frame8,text="Click Here",activeforeground='red',bd=4,command=FinalLearnM.actual)
button4.grid(row=0,column=1)

button4 = Button(frame10,text="Predict",activeforeground='red',bd=4,command=FinalLearnM.predict,fg='blue')
button4.grid(row=0,column=1)


root.mainloop()