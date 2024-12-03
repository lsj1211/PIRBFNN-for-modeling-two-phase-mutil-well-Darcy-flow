# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:25:53 2023

@author: lsj
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# aa=torch.tensor([1, 2, 3, 4])
# bb=aa.reshape(2,2)

# f=open('samplesperm.txt','r')
# data = f.readlines()  # 将txt中所有字符串读入data
# permvec = list(map(float, data))
# f.close()
# for i in range(400):
#     permvec[i]=permvec[i]*1e-15

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self): # 不加self就变成了对所有类对象同时更改
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1] # 存储相邻信息
        self.dx = 0 # x方向上的步长
        self.dy = 0 # y方向上的步长
        self.dz = 0 # z方向上的步长
        self.volume = 0 # 体积
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0 # 孔隙度
        self.kx = 0 # 三个方向的渗透率，均质时三个值相同
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0] # 传导率
        self.transo = [0, 0, 0, 0, 0, 0]
        self.transw = [0, 0, 0, 0, 0, 0]
        self.markbc = -2
        self.press = 0 # 压力
        self.Sw = 0
        self.markbc_Sw=0
        self.markwell=-1
        self.mobiw=0 # water mobility
        self.mobio=0 # oil mobility
        self.mobit=0 # total mobility

print("build Grid")

grid_size=7

ddx=100/grid_size
ddy=100/grid_size
ddz=5.0

dxvec=[0]
for i in range(0, grid_size):
    dxvec.append(ddx)  # dxvec: [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

dyvec=[0]
for i in range(0, grid_size):
    dyvec.append(ddy)  # dyvec: [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

dzvec=[0,ddz]


nx=len(dxvec)-1  # nx=20
ny=len(dyvec)-1  # ny=20
nz=len(dzvec)-1  # nz=1


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
            nodelist.append(node)  # nodelist: 类型为list, 尺寸为882, 每个元素node为类NODE的对象。
            # 第0个元素对象的属性值分别为nodelist[0].x,nodelist[0].y,nodelist[0].z=0,0,0; 第1个为5,0,0; 第三个为10,0,0;...; 第881个为100,100,5

# build connectivity and neighbors
celllist=[]

for k in range(0, nz): #nz=1
    for j in range(0, ny): #ny=20
        for i in range(0, nx): #nx=20
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            # celllist中每个cell对象保存的是与该网格(对象)相邻的网格的索引
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
            cell.dx = nodelist[i1].x - nodelist[i0].x  # x方向上的步长
            cell.dy = nodelist[i2].y - nodelist[i0].y  # y方向上的步长
            cell.dz = nodelist[i4].z - nodelist[i0].z  # z方向上的步长
            
            # vertices没有用到，只是将其保存在文件中了
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7
            
            # xc, yc, zc没有用到
            cell.xc = 0.125 * (nodelist[i0].x + nodelist[i1].x + nodelist[i2].x + nodelist[i3].x + nodelist[i4].x + nodelist[i5].x + nodelist[i6].x + nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
           
            cell.volume=cell.dx*cell.dy*cell.dz # 网格体积，定值125
            
            celllist.append(cell) # celllist: 类型为list, 尺寸为400, 每个元素cell为类CELL的对象。

cellvolume=celllist[0].volume # cellvolume=125
ncell=len(celllist) # ncell=400

print("define properties")
mu_o = 1.8e-3 # 油的粘度
mu_w = 1e-3 # 水的粘度
chuk = 15e-15 #渗透率
poro = 0.2 # 孔隙度
Siw=0.2  # 水饱和度
Bo = 1.02 # 油体积系数
Bw = 1.0 # 水体积系数
Cr = 10 * 1e-6 / 6894 # 岩石压缩系数，1psi=6894pa
Cw = 4 * 1e-6 / 6894 # 水压缩系数
Co = 100 * 1e-6 / 6894 # 油压缩系数
p_init = 20e6 # 初始压力
p_e = 20e6 # 参考压力



print("set properties to grid and initial conditions")
for i in range(0, ncell): # 为400个网格设置属性
    celllist[i].porosity=poro # 孔隙度
    # celllist[i].kx = permvec[i]
    # celllist[i].ky = permvec[i]
    # celllist[i].kz = permvec[i]
    celllist[i].kx = chuk #x,y,z三个方向的渗透率
    celllist[i].ky = chuk
    celllist[i].kz = chuk
    celllist[i].Sw = Siw # 水饱和度
    celllist[i].press=p_init # 初始压力


print("set well conditions") # 对应论文中图3(a)
## celllist[0]左下角网格
celllist[0].markwell = 0  # 注水井
celllist[0].markbc = -1
celllist[0].markbc_Sw = 1
celllist[0].Sw = 1
## celllist[ncell - 1]右上角网格
celllist[ncell - 1].markwell = 1  # 生产井
celllist[ncell - 1].markbc = -1
## celllist[nx - 1]右下角网格
# celllist[nx - 1].markbc = 1
# celllist[nx - 1].press = p_e
## celllist[nx * nx - nx]左上角网格
# celllist[nx * nx - nx].markbc = 1
# celllist[nx * nx - nx].press = p_e



print("mobility function")
def computemobi(P):
    for ie in range(0, ncell): # 为400个网格计算mobility
        sw=celllist[ie].Sw # celllist[0].Sw=1, celllist[1].Sw, ..., celllist[399].Sw=0.2
        a=(1-sw)/(1-Siw)
        b=(sw-Siw)/(1-Siw)
        kro=a*a*(1-b*b) # 对应论文中公式(16)
        krw=b*b*b*b # 对应论文中公式(15)
        vro=kro*(1+Co*(P[ie]-p_init))/(mu_o*Bo) # 对应论文中公式(14)下面
        vrw=krw*(1+Cw*(P[ie]-p_init))/(mu_w*Bw)
        celllist[ie].mobio=vro
        celllist[ie].mobiw=vrw
        celllist[ie].mobit=vro+vrw

print("transmissibility function")
def computetrans():
    for ie in range(0, ncell): # 为400个网格计算transmissibility
        for j in range(0, 4): # j取range(0, 4),是因为celllist[ie].neighbors的后两个元素都是-1，不用判断,即neighbors保存的相邻信息只体现在前四个元素。
            je = celllist[ie].neighbors[j]
            if je >= 0:
                mt1=celllist[ie].mobit # 注意索引为ie
                mt2=celllist[je].mobit # 注意索引为je,je是与ie相邻的网格的索引
                mt3=celllist[ie].mobiw # 注意索引为ie
                mt4=celllist[je].mobiw
                mt5=celllist[ie].mobio # 注意索引为ie
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

# neighbors保存的相邻信息只体现在前四个元素.
# 分别记为west:neiborvec_w, east:neiborvec_e, south:neiborvec_s, north:neiborvec_n。
# 条件语句是将neighbors保存的相邻信息(前四个元素)中的-1元素变为0
# 最终的neiborvec_w保存的是每个网格的左边(west西边)相邻网格的索引。
# 部分输出：neiborvec_w: [  0,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  18,   0,  20,  21,  22,  23, ..., 394, 395, 396, 397, 398]

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

# 与上面neighbour tensor一样的存储方式
# trans保存的transmissibility信息只体现在前四个元素

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

#定义BPNN
class BP_Net(nn.Module):
    def __init__(self, input_unit, hidden_unit, out_unit): #实例化类的对象时须传入的参数ResBlock,input_unit,hidden_unit
        super(BP_Net, self).__init__()
        self.layer1 = nn.Linear(input_unit, hidden_unit, bias=True)
        self.layer2 = nn.Linear(hidden_unit, out_unit, bias=True) #残差网络最后一层为线性层
    
    def forward(self, x):
        x = self.layer1(x)
        outputs = self.layer2(x)
        return outputs

# 定义RBF层
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

#定义RBFNN
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

#定义残差块
class Res_Block(nn.Module):
    def __init__(self, input_unit, hidden_unit): #实例化类的对象时须传入的参数input_unit,hidden_unit
        super(Res_Block, self).__init__()
        self.input_unit = input_unit
        self.hidden_unit = hidden_unit
        self.layer1 = nn.Linear(input_unit, hidden_unit,bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(input_unit, hidden_unit,bias=True)
        
    def forward(self, inputs):
        x = inputs
        if self.input_unit == self.hidden_unit: #输入与输出维度一致
            x = self.layer1(x)
            outputs = self.relu(x + inputs)
        else: #输入与输出维度不一致
            x = self.layer1(x)
            outputs = self.relu(x + self.layer2(inputs))
        return outputs

#定义残差网络
class Res_Net(nn.Module):
    def __init__(self, ResBlock, input_unit, hidden_unit, out_unit): #实例化类的对象时须传入的参数ResBlock,input_unit,hidden_unit
        super(Res_Net, self).__init__()
        self.layer1 = ResBlock(input_unit, hidden_unit)
        self.layer2 = nn.Linear(hidden_unit, out_unit, bias=True) #残差网络最后一层为线性层
    
    def forward(self, b):
        x = b
        b = self.layer1(b)
        outputs = self.layer2(b) + x
        return outputs

#设置各层神经元个数
n_input = ncell #BPNN输入神经元个数
n_hidden = 600 #BPNN隐藏神经元个数
n_output = ncell #BPNN输出神经元个数

layer_widths=[ncell,ncell] # RBFNN输入和输出神经元个数
layer_centres=[500] # RBFNN隐藏神经元个数

# 实例化模型
# model1 = BP_Net(n_input, n_hidden, n_output)
# model1 = model1.to(device)
# criterion1 = nn.MSELoss(reduction='mean')
# optimizer1 = optim.Adam(model1.parameters(), lr=0.01)

model1 = RBF_Net(layer_widths, layer_centres)
model1 = model1.to(device)
criterion1 = nn.MSELoss(reduction='mean')
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)

model2 = RBF_Net(layer_widths, layer_centres)
model2 = model2.to(device)
criterion2 = nn.MSELoss(reduction='mean')
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

model3 = Res_Net(Res_Block, n_input, n_hidden, n_output)
model3 = model3.to(device)
criterion3 = nn.MSELoss(reduction='mean')
optimizer3 = optim.Adam(model3.parameters(), lr=0.01)


print("训练模型")

# 输入的四个参数p, presslast, alphavec, qtvec尺寸都是(400, )，p表示当前压力，presslast表示上一时步压力。
def pdeimplicit(p, presslast, alphavec, qtvec):  #bound指的是确保effectively输出是在0到1之间起作用, 因此用abs。
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    # pp[nx-1]=p_e # 右下角Dirichelt
    # pp[nx*nx-nx]=p_e # 左上角Dirichelt
    pde1=torch.zeros_like(p).to(device)
    pde1[:] = pde1[:] - transvec_w[:] * (pp[neiborvec_w[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_e[:] * (pp[neiborvec_e[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_s[:] * (pp[neiborvec_s[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_n[:] * (pp[neiborvec_n[:]] - pp[:])
    # 经过上面四行代码，pde1保存的是论文中公式(10)的求和符号那一项。
    pde1[:] = pde1[:] - qtvec[:] + (pp[:]-presslast[:])*alphavec[:] # 对应论文中公式(10)
    return pde1

def pde_oil(p, presslast, Sw, Swlast, alphavec_o, betavec_o, qvec_o):  #bound指的是确保effectively输出是在0到1之间起作用, 因此用abs。
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    SwSw = torch.zeros_like(Sw).to(device)
    SwSw[:]=abs(Sw[:])
    pde1=torch.zeros_like(p).to(device)
    pde1[:] = pde1[:] - transvec_w_o[:] * (pp[neiborvec_w[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_e_o[:] * (pp[neiborvec_e[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_s_o[:] * (pp[neiborvec_s[:]] - pp[:])
    pde1[:] = pde1[:] - transvec_n_o[:] * (pp[neiborvec_n[:]] - pp[:])
    # 经过上面四行代码，pde1保存的是论文中公式(10)的求和符号那一项。
    pde1[:] = pde1[:]-qvec_o[:]+(pp[:]-presslast[:])*alphavec_o[:]+(Swlast[:]-SwSw[:])*betavec_o # 对应论文中公式(10)
    return pde1

def pde_water(p, presslast, Sw, Swlast, alphavec_w, betavec_w, qvec_w):  #bound指的是确保effectively输出是在0到1之间起作用, 因此用abs。
    pp = torch.zeros_like(p).to(device)
    pp[:]=abs(p[:])*p_init
    SwSw = torch.zeros_like(Sw).to(device)
    SwSw[:]=abs(Sw[:])
    pde2=torch.zeros_like(p).to(device)
    pde2[:] = pde2[:] - transvec_w_w[:] * (pp[neiborvec_w[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_e_w[:] * (pp[neiborvec_e[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_s_w[:] * (pp[neiborvec_s[:]] - pp[:])
    pde2[:] = pde2[:] - transvec_n_w[:] * (pp[neiborvec_n[:]] - pp[:])   
    # 经过上面四行代码，pde1保存的是论文中公式(10)的求和符号那一项。
    pde2[:] = pde2[:]-qvec_w[:]+(pp[:]-presslast[:])*alphavec_w[:]+(SwSw[:]-Swlast[:])*betavec_w # 对应论文中公式(10)
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

# qin_w=6.0/86400
qin=5.0/86400
qtvec[0]= qin # 注水井流量
qtvec[nx*nx-1]=-qin # 生产井流量
qvec_w[0]= qin # 注水井流量
qvec_o[nx*nx-1]=-qin # 生产井流量

print("Time Iteration")
nt = 500 # 预测nt个时间步
dt = 7200 # 时间步长dt

alphavec = torch.zeros(ncell).to(device)
alphavec_o = torch.zeros(ncell).to(device)
betavec_o = torch.zeros(ncell).to(device)
alphavec_w = torch.zeros(ncell).to(device)
betavec_w = torch.zeros(ncell).to(device)
re = 0.14*(ddx*ddx + ddy*ddy)**0.5 # 井的等效半径
SS=3 # 表皮系数
rw=0.05 # 井半径

# floss = open('All_losslowest_PIResNet.txt','w')
# ftime = open('Time_cost_PIResNet.txt','w')

num_epochs=2000
num_epochs1=1000

totaltime=0

pwf_all = [] # 存储所有时刻的井底流压数据
pwf_all_water = [] # 存储所有时刻的水井井底流压数据

def water(Sw, Swlast, pnew, presslast):
    pp = torch.zeros_like(pnew).to(device)
    pp[:]=abs(pnew[:])*p_init
    SwSw = torch.zeros_like(Sw).to(device)
    SwSw[:]=abs(Sw[:])
    pde3=torch.ones_like(Sw).to(device)
    for ie in range(ncell):
        if celllist[ie].markbc_Sw==0: # celllist[0].markbc_Sw=1, celllist[j].markbc_Sw=0, j=1...399
            tfluxsw=0
            tfluxin=0
            pi=pp[ie]
            for i in range(4):
                je=celllist[ie].neighbors[i]
                if je>=0:
                    pj=pp[je]
                    if pj>pi:
                        fluxin=(pj-pi)*celllist[ie].trans[i]
                        tfluxin += fluxin
                        tfluxsw += fluxin*celllist[je].mobiw/celllist[je].mobit
            tfluxout=-tfluxin
            tfluxsw += tfluxout*celllist[ie].mobiw/celllist[ie].mobit
            if ie==0:
                tfluxsw += qin
            sw=Swlast[ie]
            tfluxsw += -(pi-presslast[ie])/dt*poro*sw*celllist[ie].volume/Bw*(Cr+Cw)
            pde3[ie] = SwSw[ie] - Swlast[ie] - tfluxsw*dt/celllist[ie].volume*(Bw/(poro*(1+(Cr+Cw)*(pp[ie]-p_e))))
            return pde3

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
        # 循环后alphavec公式为论文中等式(10)左边的分式
        alphavec[ie]=celllist[ie].porosity*(1-celllist[ie].Sw)*(Cr+Co)/Bo+celllist[ie].porosity*celllist[ie].Sw*(Cr+Cw)/Bw
        alphavec[ie] = alphavec[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        alphavec_o[ie]=celllist[ie].porosity*(1-celllist[ie].Sw)*(Cr+Co)/Bo
        alphavec_o[ie] = alphavec_o[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        betavec_o[ie]=celllist[ie].porosity*(1+(Cr+Co)*(celllist[ie].press-p_e))/Bo
        betavec_o[ie] = betavec_o[ie]*celllist[ie].volume/dt

    for ie in range(ncell):
        # 循环后alphavec公式为论文中等式(10)左边的分式
        alphavec_w[ie]=celllist[ie].porosity*celllist[ie].Sw*(Cr+Cw)/Bw
        alphavec_w[ie] = alphavec_w[ie]*celllist[ie].volume/dt
        
    for ie in range(ncell):
        betavec_w[ie]=celllist[ie].porosity*(1+(Cr+Cw)*(celllist[ie].press-p_e))/Bw
        betavec_w[ie] = betavec_w[ie]*celllist[ie].volume/dt
    
    # inputtensor = inputtensor.numpy()
    
    inputtensor[0] = resultout
    inputSw[0] = resultSw
    
    print("NN Implicit Solver")
    
    starttime = time.time()
    
    # for epoch1 in range(300):
    #     output = model1(inputtensor)  # on gpu
    #     # resultnext = outputtensor[0].clone().detach()
    #     diff1=pdeimplicit(output[0], presslast, alphavec, qtvec)
    #     # 计算损失并利用反向传播计算损失对各参数梯度
    #     loss1 = criterion1(diff1, diff1 * 0)
    #     optimizer1.zero_grad()
    #     loss1.backward()
    #     optimizer1.step()
    #     if loss1 < lowestloss1:
    #         lowestloss1 = loss1
    #         # resultout = resultnext
    #     if epoch1 % 99 == 0:
    #         print('epoch1 is ', epoch1, 'lowestloss1 is: ', lowestloss1, '\n')
    #         # floss.write("%e\n" % lowestloss1)
    #     output = output.cpu().detach().numpy()
    #     output = torch.tensor(output)
    #     output = output.to(device)
    #     if (loss1 < 1e-14):
    #         break
        
    for epoch2 in range(num_epochs):
        outputtensor = model2(inputtensor)  # on gpu
        resultnext = outputtensor[0].clone().detach()
        diff2=pdeimplicit(outputtensor[0], presslast, alphavec, qtvec)
        # 计算损失并利用反向传播计算损失对各参数梯度
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
    pwf_all.append(pwf) #将每次的pwf保存在pwf_all中
    
    pwf_water = pwf_water.unsqueeze_(0)
    pwf_all_water.append(pwf_water) #将每次的pwf保存在pwf_all_water中

    # #update saturation(饱和度).对应论文中公式(14)，显式计算下一时步的Sw
    # for ie in range(ncell):
    #     if celllist[ie].markbc_Sw==0: # celllist[0].markbc_Sw=1, celllist[j].markbc_Sw=0, j=1...399
    #         tfluxsw=0
    #         tfluxin=0
    #         pi=celllist[ie].press
    #         for i in range(4):
    #             je=celllist[ie].neighbors[i]
    #             if je>=0:
    #                 pj=celllist[je].press
    #                 if pj>pi:
    #                     fluxin=(pj-pi)*celllist[ie].trans[i]
    #                     tfluxin += fluxin
    #                     tfluxsw += fluxin*celllist[je].mobiw/celllist[je].mobit
    #         tfluxout=-tfluxin
    #         tfluxsw += tfluxout*celllist[ie].mobiw/celllist[ie].mobit
    #         if ie==0:
    #             tfluxsw += qin
    #         sw=celllist[ie].Sw
    #         tfluxsw += -(pi-presslast[ie])/dt*poro*sw*celllist[ie].volume/Bw*(Cr+Cw)
    #         celllist[ie].Sw = celllist[ie].Sw + tfluxsw*dt/celllist[ie].volume*(Bw/(poro*(1+(Cr+Cw)*(celllist[ie].press-p_e))))

    for epoch in range(num_epochs1):
        outputSw = model1(inputSw)  # on gpu
        resultnextSw = outputSw[0].clone().detach()
        # diff = water(outputSw[0],Swlast,resultout,presslast)
        # diff=pde_water(resultout, presslast, outputSw[0], Swlast, alphavec_w, betavec_w, qvec_w)
        diff=pde_oil(resultout, presslast, outputSw[0], Swlast, alphavec_o, betavec_o, qvec_o)
        # diff=pde_water(resultout, presslast, outputSw[0], Swlast, alphavec_w, betavec_w, qvec_w)+pde_oil(resultout, presslast, outputSw[0], Swlast, alphavec_o, betavec_o, qvec_o)
        # 计算损失并利用反向传播计算损失对各参数梯度
        loss = criterion1(diff, diff * 0)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        if loss < lowestloss:
            lowestloss = loss
            resultSw = resultnextSw
        if epoch % 49 == 0:
            print('epoch is ', epoch, 'lowestloss is: ', lowestloss, '\n')
            # floss.write("%e\n" % lowestloss2)
        # if (loss < 8e-20):#3e-19
        #     break

    for ie in range(ncell):
        resultSw[ie]=abs(resultSw[ie])
    for ie in range(ncell):
        if celllist[ie].markbc_Sw==0:
            celllist[ie].Sw = resultSw[ie]

    for ie in range(ncell):
        presslast[ie] = celllist[ie].press
        if celllist[ie].markbc_Sw==0:
            Swlast[ie] = celllist[ie].Sw

# ftime.write("%0.3f\n" % totaltime)
# floss.close()
# ftime.close()
print(totaltime)



# 所有时刻井底流压pwf画图
pwf_all = torch.cat(pwf_all,dim=0)
pwf_all = pwf_all.detach().cpu().numpy()
pwf_all = np.insert(pwf_all,0,[20000000.]) # 在pwf_all中为0的位置插入[20000000]
pwf_all = pwf_all/1e6
plt.plot(pwf_all,label='Bottom-Hole Pressure by PICNN')
plt.xlabel('Time (h)')
plt.ylabel('Pressure (MPa)')
plt.legend()
plt.savefig("./Bottom-Hole Pressure.pdf") # 图像保存为PDF格式
plt.savefig("./Bottom-Hole Pressure.png", dpi=600)
plt.show() # 显示图像




# 最后时刻的压力p分布图
p_all=torch.zeros(ncell)

for ie in range(ncell):
    p_all[ie]=celllist[ie].press

p_all = p_all.detach().cpu().numpy()
p_all = p_all.reshape(grid_size,grid_size)/1e6
n_rows = p_all.shape[0] # 获取行数

for i in range(n_rows // 2): # //为向下取整
    p_all[[i,n_rows-i-1],:] = p_all[[n_rows-i-1,i],:] # 交换行

ax = sns.heatmap(p_all, cmap='coolwarm',cbar_kws={'label': 'Pressure (MPa)'}) # 绘制热力图
plt.axis('off') # 隐藏坐标轴
plt.savefig("./Pressure.pdf") # 图像保存为PDF格式
plt.savefig("./Pressure.png", dpi=600)
plt.show() # 显示图像



# 最后时刻的水饱和度Sw分布图
Sw_all=torch.zeros(ncell)

for ie in range(ncell):
    Sw_all[ie]=celllist[ie].Sw

Sw_all = Sw_all.detach().cpu().numpy()
Sw_all = Sw_all.reshape(grid_size,grid_size)
n_row = Sw_all.shape[0] # 获取行数

for j in range(n_row // 2): # //为向下取整
    Sw_all[[j,n_row-j-1],:] = Sw_all[[n_row-j-1,j],:] # 交换行

ax = sns.heatmap(Sw_all, cmap='coolwarm',cbar_kws={'label': 'Water Saturation'}) # 绘制热力图
plt.axis('off') # 隐藏坐标轴
plt.savefig("./Sw.pdf") # 图像保存为PDF格式
plt.savefig("./Sw.png", dpi=600)
plt.show() # 显示图像