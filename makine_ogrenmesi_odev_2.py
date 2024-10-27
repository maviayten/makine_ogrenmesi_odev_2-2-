#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Makine Öğrenmesi 1. Ödev
#1.) Önceki ödev de oluşturduğumuz veri setinin işlenmemiş hali(veri ön işleme öncesi)
#1.1 Veri Setini Oluştuma

import pandas as pd
import numpy as np

data = {
    'Boy': [165,170,175,np.nan,180,220,160,155,np.nan,210],
    'Kilo': [60,70,80,90,120,150,55,45,65,100],
    'Medeni_Durum': ['Bekar','Evli',np.nan, 'Bekar','Boşanmış','Evli','Bekar','Bekar','Boşanmış','Evli'],
    'Eğitim_Seviyesi': ['Lise', 'Lisans', 'Yüksek Lisans', 'Lise', 'Doktora', 'Yüksek Lisans', 'Doktora', 'Lisans', 'Doktora', 'Yüksek Lisans'] 
}

df = pd.DataFrame(data)


# In[2]:


#1.2 Verinin Özeti

df.info()
print("\nHer sütündaki eksik değerlerin sayısı:")


# In[3]:


print(df.isnull().sum())


# In[4]:


print(df.head())


# In[5]:


#1.3 Sayısal değişkenlerin özeti

print("\nVeri Özeti:")
print(df.describe())


# In[6]:


#1.4 Eksik Değerlerin Görselleştirmesi

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Ham veri:Eksik değerler")
plt.show()


# In[7]:


#2.)Önceki ödev de oluşturduğumuz veri setinin işlenmiş hali(veri ön işleme sonrası)
# 2.1 veri seti oluşturma

import pandas as pd
import numpy as np
np.random.seed(0)
data = {
    'Boy': [165,170,175,np.nan,180,220,160,155,np.nan,210],
    'Kilo': [60,70,80,90,120,150,55,45,65,100],
    'Medeni_Durum': ['Bekar','Evli',np.nan, 'Bekar','Boşanmış','Evli','Bekar','Bekar','Boşanmış','Evli'],
    'Eğitim_Seviyesi': ['Lise', 'Lisans', 'Yüksek Lisans', 'Lise', 'Doktora', 'Yüksek Lisans', 'Doktora', 'Lisans', 'Doktora', 'Yüksek Lisans'] 
}

df = pd.DataFrame(data)
df


# In[8]:


#2.2 Pandas ile histogram 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df.hist(figsize=(15,10),bins=50)
plt.tight_layout()
plt.show()


# In[9]:


df['Boy'].hist(
               figsize=(5,3),
               bins=10,color='#008080',
               edgecolor='black',
               alpha=0.75
              )

plt.title("Boy Dağılımı",fontsize=20)
plt.xlabel("Boy Değerleri",fontsize=15)
plt.ylabel("sayı ",fontsize=15)
plt.grid(True)
plt.show()


# In[10]:


df["Kilo"].hist(
    figsize=(6,4),
    bins=10,
    color='#CAE1FF',
    edgecolor='black',
    alpha=0.50
)

plt.title("Kilo Dağılımı",fontsize=20)
plt.xlabel("Kilo Değerleri",fontsize=15)
plt.ylabel("sayı",fontsize=15)
plt.grid(True)
plt.show()


# In[11]:


#2.3 Seaborn ile histogram

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df['Boy'])
plt.title('Boy Dağılımı')
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.histplot( 
    df['Boy'],
    bins=20,
    kde=True,
    color="blue",
    label='Boy Dağılımı'
)
plt.title('Boy Dağılımı')
plt.xlabel('Boy')
plt.ylabel('Sayı')
plt.legend()

plt.grid(True,which="both",ls="--",c='0.65')
plt.show()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()


# In[14]:


#2.4 pandas ile kutu grafiği

import pandas as pd
import matplotlib.pyplot as plt

df.boxplot(figsize=(5,3))
plt.tight_layout()
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
df.boxplot(column=['Kilo','Boy'],figsize=(8,4),grid=True,vert=False)
plt.title('Boy,Kilo')
plt.show()


# In[16]:


#2.5 Seaborn ile kutu grafiği

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df)
plt.show()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
palette=sns.color_palette("viridis",2)

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.boxplot(y=df['Boy'], color=palette[0])
plt.title('Boy Kutu Grafiği')
plt.xlabel('Sayı')
plt.ylabel('Boy')

plt.subplot(2,2,2)
sns.boxplot(y=df['Kilo'],color=palette[1])
plt.title('Kilo kutu grafiği')
plt.xlabel('Sayı')
plt.ylabel('Kilo')

plt.tight_layout()
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
palette=sns.color_palette("viridis",2)

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.boxplot(y=df['Boy'], color=palette[0])
sns.swarmplot(y=df['Boy'],color='black',size=15)
plt.title('Boy Kutu Grafiği')

plt.subplot(2,2,2)
sns.boxplot(y=df['Kilo'],color=palette[1])
sns.swarmplot(y=df['Kilo'],color='black',size=15)
plt.title('Kilo kutu grafiği')

plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


boy_counts = df['Boy'].value_counts()

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.pie(
    boy_counts, 
    labels=boy_counts.index, 
    autopct='%1.1f%%', 
    colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'], 
    explode=[0.05] * len(boy_counts),  
    shadow=True, 
    startangle=90,  
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Boy Dağılımı')


kilo_kategorileri = pd.cut(df['Kilo'], bins=[20, 30, 40, 50, 60, 65], labels=['20-30', '30-40', '40-50', '50-60', '60-65'])
kilo_counts = kilo_kategorileri.value_counts()


plt.subplot(1, 2, 2)
plt.pie(
    kilo_counts, 
    labels=kilo_counts.index, 
    autopct='%1.1f%%', 
    colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'],  
    explode=[0.05] * len(kilo_counts),  
    shadow=True,  
    startangle=90, 
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Kilo Dağılımı')

plt.tight_layout()
plt.show()


# In[20]:


#2.6 seaborn ile pasta grafiği

import matplotlib.pyplot as plt

boy_sayimlari = df['Boy'].value_counts()

plt.figure(figsize=(10,7))

plt.pie(
    boy_sayimlari,
    labels=boy_sayimlari.index,
    autopct='%1.1f%%',
    startangle=90,
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'],
    explode=[0.1 if i ==0 else 0 for i in range(len(boy_sayimlari))],
    shadow=True,
    wedgeprops={'linewidth':1.5,'edgecolor':'black'}
)

plt.title('Boylara göre dağılım seaborn ile')
plt.show()


# In[21]:


#2.7 pandas ile çubuk grafiği

import pandas as pd
import matplotlib.pyplot as plt

kilo_sayimlari = df['Kilo'].value_counts()

kilo_sayimlari.plot(kind='bar',
                    figsize=(8,6),
                    color='skyblue',
                    edgecolor='black',
                    rot=60,
                    grid=True,
                    legend=True,
                    fontsize=8,
                    width=0.5
                    
)

plt.title('Boylara göre sayımlar pandas ile',fontsize=15)
plt.ylabel('Sayı',fontsize=10)
plt.xlabel('Boy',fontsize=10)
plt.show()


# In[22]:


# 2.8 seaborn ile çubuk grafiği

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df['Boy_Aralığı'] = pd.cut(df['Boy'], bins=[20, 30, 40, 50, 60, 70], labels=['20-30', '30-40', '40-50', '50-60', '60+'])
plt.figure(figsize=(6,4))
sns.countplot(x='Kilo',
              data=df,
              palette='coolwarm',
              order=df['Kilo'],
              dodge=True,
              linewidth=2,
              edgecolor='black',
              saturation=0.8,
              width=0.7
             
)
plt.title('Kilo ve Boy aralıklarına göre dağılım seaborn ile',fontsize=10)
plt.xlabel('Boy',fontsize=10)
plt.ylabel('Sayı',fontsize=10)
plt.xticks(rotation=45,fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Boy Aralığı',loc='upper right',fontsize=7,title_fontsize=7)
plt.grid(True,which='both',axis='y',linestyle='--',linewidth=0.7)
plt.show()


# In[23]:


#2.9 pandas ile saçılım garfiği

import pandas as pd
import matplotlib.pyplot as plt

df.plot(
    kind='scatter',           
    x='Boy',                 
    y='Kilo',                
    color='blue',            
    figsize=(5, 3),          
        
    alpha=0.70,                
    marker='o',              
    edgecolor='black',       
    linewidth=2,              
    grid=True,                
    title='Boy ve Kilo Arasındaki İlişki' 
)

plt.xlabel('Boy')
plt.ylabel('Kilo')

plt.show()


# In[24]:


#2.10 seaborn ile şaçılım grafiği

plt.figure(figsize=(15,8))

sns.scatterplot(
    x='Boy',
    y='Kilo',
    data=df,
    hue='Kilo',
    palette='coolwarm',
    alpha=0.7,
    edgecolor='black',
    linewidth=1.6
    
)

plt.title('Kilo-Boy saçılım grafiği',fontsize=16)
plt.xlabel('Boy',fontsize=15)
plt.ylabel('Kilo',fontsize=15)

plt.show()


# In[25]:


#2.11 pandas ile çizgi grafiği

df.plot(
    kind='line',
    x='Boy',
    y='Kilo',
    color='skyblue',
    figsize=(10,6),
    linewidth=2,
    marker='o',
    markersize=5,
    alpha=0.7
)
plt.title('Kilo-Boy çizgi grafiği pandas ile', fontsize=15)

plt.xlabel('Kilo',fontsize=10)
plt.ylabel('Boy', fontsize=10)

plt.show()


# In[26]:


#2.12 seaborn ile çizgi grafiği

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df['Boy'] = pd.to_datetime(df['Boy'],errors='coerce')
print(df.head())

plt.figure(figsize=(10,8))

sns.lineplot(
    x='Boy',
    y='Kilo',
    data=df,
    markers=True,
    dashes=False,
    palette='coolwarm',
    ci=95,
    linewidth=2,
    legend='full'
    
)

plt.title('Boy-Kilo çizgi grafiği seaborn ile',fontsize=16)
plt.xlabel('Boy',fontsize=15)
plt.ylabel('Kilo',fontsize=15)

plt.show()


# In[27]:


#2.13 seaborn ile yoğunluk grafiği

df['Kilo'].plot(
    kind='kde',              
    bw_method=0.3,             
    ind=1000,                  
    color='red',             
    linestyle='--',           
    linewidth=2.5,             
    alpha=0.8,                 
    figsize=(10, 6),           
    title='Kilo Yoğunluk Grafiği (Gelişmiş)', 
    grid=True                  
)


plt.xlabel('Kilo')
plt.ylabel('Yoğunluk')


plt.show()


# In[28]:


plt.figure(figsize=(10, 6))


sns.kdeplot(
    x=df['Kilo'],             
    shade=True,               
    color='blue',             
    bw_adjust=0.3,            
    alpha=0.8,                
    fill=True,               
    cumulative=False,        
    linewidth=2.5             
)


plt.title('Boy Yoğunluk Grafiği (Seaborn - Tüm Parametreler)', fontsize=16) 
plt.xlabel('Kilo', fontsize=14) 
plt.ylabel('Yoğunluk', fontsize=14)  


plt.xlim(0, 100)  
plt.ylim(0, 0.05) 


plt.show()


# In[29]:


#2.14 pandas ile ile ısı haritası

plt.figure(figsize=(10, 7)) 

numeric_df = df.select_dtypes(include=['number'])


corr_matrix = numeric_df.corr()

sns.heatmap(
    corr_matrix, 
    annot=True,              
    cmap='coolwarm',       
    linewidths=0.5,          
    linecolor='white',       
    vmin=0, vmax=1,          
    cbar=True,               
    square=True,             
    fmt=".3f",               
    annot_kws={"size": 10, "color": "black"}  
)


plt.show()


# In[ ]:


#Dipnot
yanlış veri seçiminden kaynaklı olup grafiklerin absürt çıktığının farkındayım.


# In[ ]:





# In[ ]:




