import netCDF4 as nc
import numpy as np
from math import *
import pandas as pd
import xlrd
import os
from matplotlib import pyplot as plt
import random as rd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import folium

os.chdir("C:\\Users\\mathi\\OneDrive - Université Paris Sciences et Lettres\\Desktop\\Cours\\CPES3\\S6\\Mémoire\\means")

fn = 'C:\\Users\\mathi\\OneDrive - Université Paris Sciences et Lettres\\Desktop\\Cours\\CPES3\\S6\\Mémoire\\Data\\2021.nc'
ds = nc.Dataset(fn)
lo=np.zeros(56)
la=np.zeros(35)
t=np.zeros(8760)

def nettonpliste(l,tr,ds):
  i = 0
  for var in ds.variables[tr]:
        l[i] = var
        i += 1

nettonpliste(lo,'longitude',ds)
nettonpliste(la,'latitude',ds)
nettonpliste(t,'time',ds)

df=pd.read_excel(r'C:\\Users\\mathi\\OneDrive - Université Paris Sciences et Lettres\\Desktop\\Cours\\CPES3\\S6\\Mémoire\\Data\\power wind data (1).xlsx')
def H(Wind,Power,x):
    n = len(Wind)
    if n != len(Power):
        print("Error : lists lengh isn't the same")
        return -1
    if x > 25 :
        return 0
    a = 0
    b = 1
    for i in range(len(Wind)-1):
        if (x>=a) and (x<b):
            # print((Power[i+1]-Power[i])*(x-a) + Power[i])
            return ((Power[i+1]-Power[i])/(Wind[i+1]-Wind[i]))*(x-Wind[i]) + Power[i]
        a += 1
        b += 1


def Hsiemens(x):
  return H(df["speed"],df["power"],x)


datafm=np.zeros((len(t),len(la),len(lo)))
datafa=np.zeros((len(t),len(la),len(lo)))
data = np.zeros((len(t),len(la),len(lo)))

def wind():
    y=0
    u100 = ds.variables["u100"]
    v100 = ds.variables["v100"]
    for i in range(len(t)):
        if y%100==0:
            print(y)
        y=y+1
    #print(i)
        for j in range(len(la)):
            #print("j",j)
            for r in range(len(lo)):

            #print("r",r)
                u = u100[i,j,r]
                v = v100[i,j,r]
                value = np.sqrt(u**2 + v**2)
                data[i,j,r] = value
                #datafa[i,j,r]=load_factor(data[i,j,r],df)/2300
                datafm[i,j,r]=Hsiemens(value)/90


# df=pd.read_csv(r'C:/Users/PC/43.05.cvs')
# df1=pd.read_csv(r'C:/Users/PC/43.3.cvs')
# l=[]
# m=[]
# for i in range(25):
#     l.append(df.columns[-26:-1][i])
#     m.append(df1.columns[-26:-1][i])
# for i in range(len(l)):
#     df[l[i]]=df[l[i]]*100
#     df1[m[i]]=df1[m[i]]*100
#
#
# datafm=np.zeros((len(df),2,len(l)))
#
# for i in range(len(df)):
#     for r in range(len(l)):
#         datafm[i,0,r]=df[l[r]][i]
#         datafm[i,1,r]=df1[m[r]][i]


# Functions
def bihisto(title,data,labelx,labely,sg,grid,colors):
    sns.set()
    # sns.set_style("whitegrid")
    # sns.set_style("ticks")
    fig,axes = plt.subplots(1,1,figsize=(10,10))
    plt.hist(data, bins = sg, histtype='step',label = labelx, fill=False,color = colors, alpha = 0.8)
    plt.title(f"{title}")
    plt.ylabel(f"{labely}")
    plt.legend(prop={'size': 10})
    axes.xaxis.set_major_locator(MaxNLocator(grid))
    axes.yaxis.set_major_locator(MaxNLocator(grid))
    plt.show()

def twinhisto(title,data,labelx,labely,sg,grid,colors):
    sns.set()
    # sns.set_style("whitegrid")
    sns.set_style("ticks")
    fig,axes = plt.subplots(1,1,figsize=(10,10))
    plt.hist(data[0], bins = sg, histtype='step',label = labelx[0], fill=False,color = colors[0], alpha = 0.8)
    plt.legend(loc = "upper left",prop={'size': 10})
    ax2 = plt.gca().twinx()
    ax2.hist(data[1], bins = sg, histtype='step',label = labelx[1], fill=False,color = colors[1], alpha = 0.8)
    plt.title(f"{title}")
    plt.xlabel("G magnitude")
    plt.ylabel(f"{labely}")
    plt.legend(loc = "upper right",prop={'size': 10})
    axes.xaxis.set_major_locator(MaxNLocator(grid))
    axes.yaxis.set_major_locator(MaxNLocator(grid))
    plt.show()

def scat(title,datax,datay,labelx,labely,grid):
    sns.set()
    fig,axes = plt.subplots(1,1,figsize=(10,10)) #de base 10 10
    axes.xaxis.set_major_locator(MaxNLocator(grid)) #grid de base pour le même intervalle option int(grid/2)
    axes.yaxis.set_major_locator(MaxNLocator(grid))
    plt.scatter(datax,datay,marker = ".",color = "green",alpha =0.8,s=50) # Alpha usuel = 0.2, le max = 0.005
    # plt.title(f"{title}")
    plt.xlabel(f"{labelx}")
    plt.ylabel(f"{labely}")
#     y = x //////////////////
    #lims = [
    #np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
    #np.max([axes.get_xlim(), axes.get_ylim()]),  # max of both axes


    # now plot both limits against eachother
    #axes.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #axes.set_aspect('equal')
    #axes.set_xlim(lims)
    # altenative pour fixer soi même:
    # axes.set_xlim(minx, maxx)
    #axes.set_ylim(lims)

#   ///////////////////
    plt.show()


#indla : listes des indices choisis pour latitude
#indlo: listes d'indice de longitude pour chaque latitude
#fenetre: fenetre de temps choisi en heure
def fmean(indla,indlo,fenetre):
    datafma=np.zeros(len(datafm))

    for i in range(len(datafma)):   #moyennes sur les longitude et latitude
        mean=0
        co=0
        for la in range(len(indla)):
            for lo in indlo[la]:
                mean=mean+datafm[i][indla[la]][lo]
                co=co+1
        mean=mean/co
        datafma[i]=mean

    if fenetre!=1:
        datafmt=np.zeros(len(datafma)//fenetre)  #moyennes pour les différents temps
        for i in range(len(datafmt)):
            t=0+i*fenetre
            mean=0
            while t!=fenetre+i*fenetre:
                mean=mean+datafma[t]
                t=t+1
            datafmt[i]=mean/fenetre
        return datafmt
    else:
        return datafma

def singmean(loc):
    m = 0
    for time in range(len(t)):
        m += datafm[time][loc[0]][loc[1]]
        # print("m",m)
    return m/len(t)


def inproba(ar,p,x):

    for i in range(len(ar)):
        if (x>=p*i) and (x<p*(i+1)):
            ar[i]=ar[i]+1
        elif x == 100:
            ar[len(ar)-1] += 1

def fdproba(p,indla,indlo,fenetre):
    arx=np.zeros(int(100/p)+1)
    leng = 100//p
    ary = np.zeros(leng+1)
    for i in range(leng+1):
      ary[i] = i*p
    datafma=fmean(indla,indlo,fenetre)
    for m in datafma:
        inproba(arx,p,m)
    arx=arx/np.sum(arx)
    return arx,ary

def fdr(p,indla,indlo,fenetre):
    a = fdproba(p,indla,indlo,fenetre)
    ar = a[0]
    # print(a)
    for i in range(len(ar)-2,-1,-1):
        ar[i]=ar[i+1]+ar[i]
        # print(ar[i])
    return a[1],ar*100

def fvar(list_monotone):
    m=np.mean(list_monotone)
    s=0
    for i in list_monotone:
        x=(i-m)**2
        #print(x)
        s=s+x
    return sqrt(s/len(liste_monotone))

# def tcritique(list_monotone,list_temps,seuil):
#     i=len(list_monotone)-1
#     while list_monotone[i]<=seuil:
#         i=i-1
#     return list_temps[i]

def tcritique(list_monotone,list_temps,seuil):
    return list_temps[int(seuil)]

def t_critique_compare (list_localisation,nombre_de_loc_voulu,p,fenetre,seuil):
    list_t_critique=[]
    l=list(combinations(list_localisation,nombre_de_loc_voulu))
    maxi = len(l)
    print(maxi)
    cnt = 0
    peri = 0.1
    for i in l:
        cnt += 1
        la=[]
        lo=[]
        for loc in i:
            la.append(loc[0])
            lo.append([loc[1]])
        a=fdr(p,la,lo,fenetre)
        list_t_critique.append(tcritique(a[0],a[1],seuil))
        per = cnt/maxi * 100
        if per >= peri:
            print(per)
            peri += 0.1
    index=list_t_critique.index(np.max(list_t_critique))
    # print(list_t_critique)
    return l[index],np.max(list_t_critique),list_t_critique





def ind_to_loc(indla,indlo):
    list_localisation=[]
    for la in indla:
        for lo in indlo:
            list_localisation.append([la,lo])
    return list_localisation




def pre_selection(indla,indlo,fenetre,seuil):
    for la in indla:
        for lo in indlo[indla[indla.index(la)]]:
            mean=fmean(la,lo)
            if mean<seuil:
                indlo[indlo.index(la)].remove(lo)
    for j in indlo:
        if j==0:
            indla.remove(indla[j])

    indlo=list(filter(None, indlo))
    return


wind()


indla= [i for i in range(len(la))]
indlo= [[i] for i in range(len(lo))]
localisations = ind_to_loc(indla,indlo)
print(localisations)
print(len(localisations))
Touteslesloc = localisations[:]

def pre_selection_loc(locatisations,seuil):
    saveallloc = localisations[:]
    for j in range(len(saveallloc)):
        loc =saveallloc[j]
        mean=singmean(loc)
        # print(mean)
        if mean<seuil:
            # print(loc)
            localisations.remove(loc)
    return


# pre_selection_loc(localisations,70)
pre_selection_loc(localisations,50)
# print(localisations)

def removeengland():
    saveallloc = localisations[:]
    for i in range(len(saveallloc)):
        if saveallloc[i][1][0] <= 29:
            if saveallloc[i][0] <= -8/29*saveallloc[i][1][0]+8:
                localisations.remove(saveallloc[i])
                # print(saveallloc[i])
    return

removeengland()

def removegolf():
    saveallloc = localisations[:]
    for i in range(len(saveallloc)):
        if saveallloc[i][0] >= 7/13*saveallloc[i][1][0]+15:
            localisations.remove(saveallloc[i])
            # print(saveallloc[i])
    return

removegolf()

#Etude localle du foisonnemenent à  parcs éoliens

def randomsample(N):
    localea = []
    for i in range(N):
        a = rd.randint(0,len(localisations)-1)
        localea.append(localisations[a])
    return localea

#Trouver l'indice de foisonnement avec t critique compare, seuil à 15%???

def randomopti(N,n,p,fenetre, seuil):
    save =[]
    for i in range(n):
        sample = randomsample(N)
        a = fdr(p,[x[0] for x in sample],[y[1] for y in sample],fenetre)
        save.append(tcritique(a[1],a[0],seuil))
    return save

# opti = randomopti(2,1000,1,12, 15)
# bihisto("indice de criticalité",opti,"pourcentage du temps au dessus de 15%","nombre", 100,20, "red")
# bihisto("indice de criticalité",opti,"pourcentage du temps au dessus de 15%","nombre", 100,20, "red")

def covar(list_monotone1,list_monotone2):
    m1=np.mean(list_monotone1)
    m2=np.mean(list_monotone2)
    s=0
    sig1 = np.std(list_monotone1)
    sig2 = np.std(list_monotone2)
    for i in range(len(list_monotone1)):
        x=(list_monotone1[i]-m1)*(list_monotone2[i]-m2)
        #print(x)
        s=s+x
    return s/((len(list_monotone1)-1)*sig1*sig2)

def varianceloc():
    covarhorizon = []
    covarverti = []
    for loc in localisations:
        # print(loc)
        if [loc[0]+1,loc[1]] in localisations:
            covarhorizon.append(covar(datafm[:,loc[0],loc[1][0]],datafm[:,loc[0]+1,loc[1][0]]))
        if [loc[0],[loc[1][0]+1]] in localisations:
            covarverti.append(covar(datafm[:,loc[0],loc[1][0]],datafm[:,loc[0],loc[1][0]+1]))
    return covarhorizon,covarverti

# a,b = varianceloc()
# # scat("",a, b, "variance horizontale", "variance verticale", 10 )
# bihisto("covariance entre voisin vertical",b,"covariance","nombre", 50,10, "red")
# bihisto("covariance entre voisin horizontal",a,"covariance","nombre", 50,10, "red")

def distance(loc1,loc2):
    l1,l2 = lo[loc1[1][0]]*2*np.pi/360,lo[loc2[1][0]]*2*np.pi/360
    phi1,phi2 = la[loc1[0]]*2*np.pi/360,la[loc2[0]]*2*np.pi/360
    d = 2*6371*np.arcsin(np.sqrt((np.sin((phi2-phi1)/2))**2+np.cos(phi1)*np.cos(phi2)*(np.sin((l2-l1)/2))**2))
    return d

def codist(n):
    savecovar =[]
    saved = []
    for i in range(n):
        loc1,loc2 = randomsample(2)
        savecovar.append(covar(datafm[:,loc1[0],loc1[1][0]],datafm[:,loc2[0],loc2[1][0]]))
        saved.append(distance(loc1,loc2))
    return savecovar,saved

# aa,bb = codist(1000)
# scat("Corrélation en fonction de la distance",bb,aa,"distance","corrélation",10)

# def cadrillage(pas):
#     zones = np.zeros(((len(indla)//pas)+1,(len(indlo)//pas)+1,pas*pas))
#     zones.fill(-1)
#     # print(zones)
#     cnt = np.zeros((len(indla)//pas+1,len(indlo)//pas+1))
#     for i in range(len(localisations)):
#         loc = localisations[i]
#         # print(loc)
#         k,l = loc[0]//pas,loc[1][0]//pas
#         # print(k,l)
#         # print(cnt[k,l])
#         # print(zones[k,l,int(cnt[k,l])])
#         zones[k,l,int(cnt[k,l])] = i
#         cnt[k,l] += 1
#     return zones
#
#
# temp = cadrillage(2)
#
# zones = []
# for x in temp:
#     for y in x:
#         zones.append(y)
#
# def reduction(zones):
#     saveactualloc = localisations[:]
#     for z in zones:
#         savemax = 0
#         savej = 0
#         for j in range(len(z)):
#             # print(z)
#             if z[j] != -1:
#                 # print(z[j])
#                 loc = saveactualloc[int(z[j])]
#                 a=fdr(1,[loc[0]],[loc[1]],12)
#                 criticality = tcritique(a[1],a[0],12)
#                 if criticality >= savemax:
#                     savemax = criticality
#                     # print(saveactualloc[int(z[savej])])
#                     if savej != 0:
#                         print("yes")
#                         localisations.remove(saveactualloc[int(z[savej])])
#                     savej = j
#                 else:
#                     localisations.remove(loc)
#                     print("no")
#
# reduction(zones)

def easyprered():
    north = []
    middle = []
    south = []
    for i in range(len(localisations)):
        if localisations[i][0] < 10:
            north.append(i)
        elif localisations[i][0] > 28:
            south.append(i)
        else:
            middle.append(i)
    return np.array([north,middle,south])

n,m,s = easyprered()

reduction(easyprered())



def loctoloc(localisations):
    laloc = []
    loloc = []
    for x in localisations:
        laloc.append(la[x[0]])
        loloc.append(lo[x[1]])
    return laloc[:],loloc[:]

# x,y = loctoloc(localisations)
# scat("",x, y, "la", "lo", 10 )

def createmap(locations):
    carte = folium.Map(location = [46.89, 1.25], zoom_start = 6)
    for x in locations:
        (folium.Circle([la[x[0]],lo[x[1]]])).add_to(carte)
    carte.save("result3_5.html")

result3_5 = resultat[:]

createmap(s)
createmap(localisations)
createmap(resultat[0])
resultat = t_critique_compare(localisations,3,1,12,15)

for i in range(6):
    localisations.append(localisations[i])

scat("",a[1],a[0],"","",1)
a = fdr(1,[10],[[0]],240)
