import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self):
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.transo = [0, 0, 0, 0, 0, 0]
        self.transw = [0, 0, 0, 0, 0, 0]
        self.markbc = -2
        self.press = 0
        self.Sw = 0
        self.markbc_Sw=0
        self.markwell=-1
        self.mobiw=0
        self.mobio=0
        self.mobit=0

print("build Grid")

grid_size=7

ddx=100/grid_size
ddy=100/grid_size
ddz=5.0

dxvec=[0]
for i in range(0, grid_size):
    dxvec.append(ddx)

dyvec=[0]
for i in range(0, grid_size):
    dyvec.append(ddy)

dzvec=[0,ddz]


nx=len(dxvec)-1
ny=len(dyvec)-1
nz=len(dzvec)-1


nodelist=[]

llz = 0
for k in range(0, nz+1):
    llz = llz + dzvec[k]
    lly=0
    for j in range(0, ny+1):
        lly = lly + dyvec[j]
        llx = 0
        for i in range(0, nx+1):
            llx = llx + dxvec[i]
            node=NODE()
            node.x=llx
            node.y=lly
            node.z=llz
            nodelist.append(node)

# build connectivity and neighbors
celllist=[]

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            if i>0:
                cell.neighbors[0] = nc - 1
            if i<nx-1:
                cell.neighbors[1] = nc + 1
            if j>0:
                cell.neighbors[2] = nc - nx
            if j<ny-1:
                cell.neighbors[3] = nc + nx
            if k>0:
                cell.neighbors[4] = nc - nx*ny
            if k<nz-1:
                cell.neighbors[5] = nc + nx * ny
            
            i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            cell.dx = nodelist[i1].x - nodelist[i0].x
            cell.dy = nodelist[i2].y - nodelist[i0].y
            cell.dz = nodelist[i4].z - nodelist[i0].z
            
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7

            cell.xc = 0.125 * (nodelist[i0].x + nodelist[i1].x + nodelist[i2].x + nodelist[i3].x + nodelist[i4].x + nodelist[i5].x + nodelist[i6].x + nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
           
            cell.volume=cell.dx*cell.dy*cell.dz
            
            celllist.append(cell)

cellvolume=celllist[0].volume
ncell=len(celllist)

print("define properties")
mu_o = 1.8e-3
mu_w = 1e-3
chuk = 15e-15
poro = 0.2
Siw=0.2
Bo = 1.02
Bw = 1.0
Cr = 10 * 1e-6 / 6894
Cw = 4 * 1e-6 / 6894
Co = 100 * 1e-6 / 6894
p_init = 20e6
p_e = 20e6

print("set properties to grid and initial conditions")
for i in range(0, ncell):
    celllist[i].porosity=poro
    celllist[i].kx = chuk
    celllist[i].ky = chuk
    celllist[i].kz = chuk
    celllist[i].Sw = Siw
    celllist[i].press=p_init


print("set well conditions")
celllist[0].markwell = 0
celllist[0].markbc = -1
celllist[0].markbc_Sw = 1
celllist[0].Sw = 1
celllist[ncell - 1].markwell = 1
celllist[ncell - 1].markbc = -1

print("mobility function")
def computemobi(P):
    for ie in range(0, ncell):
        sw=celllist[ie].Sw
        a=(1-sw)/(1-Siw)
        b=(sw-Siw)/(1-Siw)
        kro=a*a*(1-b*b)
        krw=b*b*b*b
        vro=kro*(1+Co*(P[ie]-p_init))/(mu_o*Bo)
        vrw=krw*(1+Cw*(P[ie]-p_init))/(mu_w*Bw)
        celllist[ie].mobio=vro
        celllist[ie].mobiw=vrw
        celllist[ie].mobit=vro+vrw

print("transmissibility function")
def computetrans():
    for ie in range(0, ncell):
        for j in range(0, 4):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                mt1=celllist[ie].mobit
                mt2=celllist[je].mobit
                mt3=celllist[ie].mobiw
                mt4=celllist[je].mobiw
                mt5=celllist[ie].mobio
                mt6=celllist[je].mobio
                k1 = celllist[ie].kx
                k2 = celllist[je].kx
                t1=mt1*k1*ddy*ddz/(ddx/2)
                t2=mt2*k2*ddy*ddz/(ddx/2)
                t3=mt3*k1*ddy*ddz/(ddx/2)
                t4=mt4*k2*ddy*ddz/(ddx/2)
                t5=mt5*k1*ddy*ddz/(ddx/2)
                t6=mt6*k2*ddy*ddz/(ddx/2)
                tt=1 / (1 / t1 + 1 / t2)
                t=1 / (1 / t3 + 1 / t4)
                ttt=1 / (1 / t5 + 1 / t6)
                celllist[ie].trans[j] =tt
                celllist[ie].transw[j] =t
                celllist[ie].transo[j] =ttt

print("neighbour tensor")

neiborvec_w = torch.zeros(ncell).type(torch.long).to(device)
neiborvec_e = torch.zeros(ncell).type(torch.long).to(device)
neiborvec_s = torch.zeros(ncell).type(torch.long).to(device)
neiborvec_n = torch.zeros(ncell).type(torch.long).to(device)

for ie in range(ncell):
    neibor_w = celllist[ie].neighbors[0]
    neibor_e = celllist[ie].neighbors[1]
    neibor_s = celllist[ie].neighbors[2]
    neibor_n = celllist[ie].neighbors[3]
    if neibor_w < 0:
        neibor_w = 0
    if neibor_e < 0:
        neibor_e = 0
    if neibor_s < 0:
        neibor_s = 0
    if neibor_n < 0:
        neibor_n = 0
    neiborvec_w[ie] = neibor_w
    neiborvec_e[ie] = neibor_e
    neiborvec_n[ie] = neibor_n
    neiborvec_s[ie] = neibor_s

print("trans tensor")
transvec_w = torch.zeros(ncell).to(device)
transvec_e = torch.zeros(ncell).to(device)
transvec_s = torch.zeros(ncell).to(device)
transvec_n = torch.zeros(ncell).to(device)

transvec_w_o = torch.zeros(ncell).to(device)
transvec_e_o = torch.zeros(ncell).to(device)
transvec_s_o = torch.zeros(ncell).to(device)
transvec_n_o = torch.zeros(ncell).to(device)

transvec_w_w = torch.zeros(ncell).to(device)
transvec_e_w = torch.zeros(ncell).to(device)
transvec_s_w = torch.zeros(ncell).to(device)
transvec_n_w = torch.zeros(ncell).to(device)

def transcell2tensor():
    for ie in range(ncell):
        transvec_w[ie] = celllist[ie].trans[0]
        transvec_e[ie] = celllist[ie].trans[1]
        transvec_s[ie] = celllist[ie].trans[2]
        transvec_n[ie] = celllist[ie].trans[3]
        transvec_w_o[ie] = celllist[ie].transo[0]
        transvec_e_o[ie] = celllist[ie].transo[1]
        transvec_s_o[ie] = celllist[ie].transo[2]
        transvec_n_o[ie] = celllist[ie].transo[3]
        transvec_w_w[ie] = celllist[ie].transw[0]
        transvec_e_w[ie] = celllist[ie].transw[1]
        transvec_s_w[ie] = celllist[ie].transw[2]
        transvec_n_w[ie] = celllist[ie].transw[3]

print("define NN model, criterion, optimizer and scheduler")

# RBF layer
class RBF(nn.Module):

    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        # nn.init.constant_(self.centres, 0)
        nn.init.constant_(self.sigmas, 0.5)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / self.sigmas.unsqueeze(0)
        gaussian = torch.exp(-1*distances.pow(2))
        return gaussian

#RBFNN
class RBF_Net(nn.Module):
    
    def __init__(self, layer_widths, layer_centres):
        super(RBF_Net, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i]))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))
    
    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

layer_widths=[ncell,ncell]
layer_centres=[500]

model1 = RBF_Net(layer_widths, layer_centres)
model1 = model1.to(device)
criterion1 = nn.MSELoss(reduction='mean')
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)

model2 = RBF_Net(layer_widths, layer_centres)
model2 = model2.to(device)
criterion2 = nn.MSELoss(reduction='mean')
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)


print("train model")

def pdeimplicit(p, presslast, alphavec, qtvec):
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    pde1=torch.zeros_like(p).to(device)
    pde1[:] = pde1[:] - transvec_w[:] * (pp[neiborvec_w[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_e[:] * (pp[neiborvec_e[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_s[:] * (pp[neiborvec_s[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_n[:] * (pp[neiborvec_n[:]] - pp[:])
    pde1[:] = pde1[:] - qtvec[:] + (pp[:]-presslast[:])*alphavec[:]
    return pde1

def pde_oil(p, presslast, Sw, Swlast, alphavec_o, betavec_o, qvec_o):
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    SwSw = torch.zeros_like(Sw).to(device)
    SwSw[:]=abs(Sw[:])
    pde1=torch.zeros_like(p).to(device)
    pde1[:] = pde1[:] - transvec_w_o[:] * (pp[neiborvec_w[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_e_o[:] * (pp[neiborvec_e[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_s_o[:] * (pp[neiborvec_s[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_n_o[:] * (pp[neiborvec_n[:]] - pp[:])
    pde1[:] = pde1[:]-qvec_o[:]+(pp[:]-presslast[:])*alphavec_o[:]+(Swlast[:]-SwSw[:])*betavec_o
    return pde1

def pde_water(p, presslast, Sw, Swlast, alphavec_w, betavec_w, qvec_w):
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    SwSw = torch.zeros_like(Sw).to(device)
    SwSw[:]=abs(Sw[:])
    pde2=torch.zeros_like(p).to(device)
    pde2[:] = pde2[:] - transvec_w_w[:] * (pp[neiborvec_w[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_e_w[:] * (pp[neiborvec_e[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_s_w[:] * (pp[neiborvec_s[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_n_w[:] * (pp[neiborvec_n[:]] - pp[:])   
    pde2[:] = pde2[:]-qvec_w[:]+(pp[:]-presslast[:])*alphavec_w[:]+(SwSw[:]-Swlast[:])*betavec_w
    return pde2

print("construct input tensor")

inputtensor = torch.ones(1, ncell).to(device)
inputSw = torch.ones(1, ncell).to(device)
resultout=torch.ones(ncell).to(device)
resultSw=torch.ones(ncell).to(device)
presslast=torch.ones(ncell).to(device)
Swlast=torch.ones(ncell).to(device)
qtvec=torch.zeros(ncell).to(device)
qvec_o=torch.zeros(ncell).to(device)
qvec_w=torch.zeros(ncell).to(device)

for i in range(ncell):
    resultout[i]=(celllist[i].press)/p_init
    presslast[i]=celllist[i].press
    if celllist[i].markbc_Sw==0:
        resultSw[i]=celllist[i].Sw
        Swlast[i]=celllist[i].Sw

qin=5.0/86400
qtvec[0]= qin
qtvec[nx*nx-1]=-qin
qvec_w[0]= qin
qvec_o[nx*nx-1]=-qin

print("Time Iteration")
nt = 500
dt = 7200

alphavec = torch.zeros(ncell).to(device)
alphavec_o = torch.zeros(ncell).to(device)
betavec_o = torch.zeros(ncell).to(device)
alphavec_w = torch.zeros(ncell).to(device)
betavec_w = torch.zeros(ncell).to(device)
re = 0.14*(ddx*ddx + ddy*ddy)**0.5
SS=3
rw=0.05

num_epochs=2000
num_epochs1=1000

totaltime=0

pwf_all = []
pwf_all_water = []

for t in range(nt):
    print('Time is ', t)
    if t>0:
        num_epochs=500
        num_epochs1=300
    lowestloss = 10000000000
    lowestloss2 = 10000000000
    
    computemobi(resultout*p_init)
    computetrans()
    transcell2tensor()
    
    for ie in range(ncell):
        alphavec[ie]=celllist[ie].porosity*(1-celllist[ie].Sw)*(Cr+Co)/Bo+celllist[ie].porosity*celllist[ie].Sw*(Cr+Cw)/Bw
        alphavec[ie] = alphavec[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        alphavec_o[ie]=celllist[ie].porosity*(1-celllist[ie].Sw)*(Cr+Co)/Bo
        alphavec_o[ie] = alphavec_o[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        betavec_o[ie]=celllist[ie].porosity*(1+(Cr+Co)*(celllist[ie].press-p_e))/Bo
        betavec_o[ie] = betavec_o[ie]*celllist[ie].volume/dt

    for ie in range(ncell):
        alphavec_w[ie]=celllist[ie].porosity*celllist[ie].Sw*(Cr+Cw)/Bw
        alphavec_w[ie] = alphavec_w[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        betavec_w[ie]=celllist[ie].porosity*(1+(Cr+Cw)*(celllist[ie].press-p_e))/Bw
        betavec_w[ie] = betavec_w[ie]*celllist[ie].volume/dt
    
    inputtensor[0] = resultout
    inputSw[0] = resultSw
    
    print("NN Implicit Solver")
    
    starttime = time.time()
        
    for epoch2 in range(num_epochs):
        outputtensor = model2(inputtensor)  # on gpu
        resultnext = outputtensor[0].clone().detach()
        diff2=pdeimplicit(outputtensor[0], presslast, alphavec, qtvec)
        loss2 = criterion2(diff2, diff2 * 0)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        if loss2 < lowestloss2:
            lowestloss2 = loss2
            resultout = resultnext
        if epoch2 % 49 == 0:
            print('epoch2 is ', epoch2, 'lowestloss2 is: ', lowestloss2, '\n')
            # floss.write("%e\n" % lowestloss2)
        if (loss2 < 8e-20):#3e-19
            break

    endtime = time.time()
    simtime = endtime - starttime
    totaltime=totaltime+simtime
    
    for ie in range(ncell):
        resultout[ie]=abs(resultout[ie])
    for ie in range(ncell):
        celllist[ie].press = resultout[ie]*p_init
        # celllist[nx-1].press = p_e
        # celllist[nx*nx-nx].press = p_e

    PI = 2 * 3.14 * ddz * celllist[nx * nx - 1].kx * celllist[nx * nx - 1].mobio / (math.log(re / rw) + SS)
    pwf=celllist[nx*nx-1].press-qin/PI
    # print("pwf:",pwf)
    PI_water = 2 * 3.14 * ddz * celllist[0].kx * celllist[0].mobiw / (math.log(re / rw) + SS)
    pwf_water = celllist[0].press+qin/PI_water
    
    pwf = pwf.unsqueeze_(0)
    pwf_all.append(pwf) 
    
    pwf_water = pwf_water.unsqueeze_(0)
    pwf_all_water.append(pwf_water) 

    for epoch in range(num_epochs1):
        outputSw = model1(inputSw)
        resultnextSw = outputSw[0].clone().detach()
        diff=pde_oil(resultout, presslast, outputSw[0], Swlast, alphavec_o, betavec_o, qvec_o)
        # diff=pde_water(resultout, presslast, outputSw[0], Swlast, alphavec_w, betavec_w, qvec_w)
        loss = criterion1(diff, diff * 0)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        if loss < lowestloss:
            lowestloss = loss
            resultSw = resultnextSw
        if epoch % 49 == 0:
            print('epoch is ', epoch, 'lowestloss is: ', lowestloss, '\n')


    for ie in range(ncell):
        resultSw[ie]=abs(resultSw[ie])
    for ie in range(ncell):
        if celllist[ie].markbc_Sw==0:
            celllist[ie].Sw = resultSw[ie]

    for ie in range(ncell):
        presslast[ie] = celllist[ie].press
        if celllist[ie].markbc_Sw==0:
            Swlast[ie] = celllist[ie].Sw




pwf_all = torch.cat(pwf_all,dim=0)
pwf_all = pwf_all.detach().cpu().numpy()
pwf_all = np.insert(pwf_all,0,[20000000.])
pwf_all = pwf_all/1e6
plt.plot(pwf_all,label='Bottom-Hole Pressure by PICNN')
plt.xlabel('Time (h)')
plt.ylabel('Pressure (MPa)')
plt.legend()
plt.savefig("./Bottom-Hole Pressure.pdf") 
plt.savefig("./Bottom-Hole Pressure.png", dpi=600)
plt.show() 




p_all=torch.zeros(ncell)

for ie in range(ncell):
    p_all[ie]=celllist[ie].press

p_all = p_all.detach().cpu().numpy()
p_all = p_all.reshape(grid_size,grid_size)/1e6
n_rows = p_all.shape[0]

for i in range(n_rows // 2):
    p_all[[i,n_rows-i-1],:] = p_all[[n_rows-i-1,i],:]

ax = sns.heatmap(p_all, cmap='coolwarm',cbar_kws={'label': 'Pressure (MPa)'})
plt.axis('off')
plt.savefig("./Pressure.pdf")
plt.savefig("./Pressure.png", dpi=600)
plt.show()


Sw_all=torch.zeros(ncell)

for ie in range(ncell):
    Sw_all[ie]=celllist[ie].Sw

Sw_all = Sw_all.detach().cpu().numpy()
Sw_all = Sw_all.reshape(grid_size,grid_size)
n_row = Sw_all.shape[0]

for j in range(n_row // 2):
    Sw_all[[j,n_row-j-1],:] = Sw_all[[n_row-j-1,j],:] 

ax = sns.heatmap(Sw_all, cmap='coolwarm',cbar_kws={'label': 'Water Saturation'}) 
plt.axis('off')
plt.savefig("./Sw.pdf")
plt.savefig("./Sw.png", dpi=600)
plt.show()
