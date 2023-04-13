from __future__ import print_function
import torch
import numpy as np
import unittest


def removeRBM():

    # for ALA-2
    res = 0
    nAtomPerRes = 10
    nOffset = 6

    idN = nOffset + nAtomPerRes * res + 0
    idCA = nOffset + nAtomPerRes * res + 2
    idC = nOffset + nAtomPerRes * res + 8
    dataRBM = removeRBMsingle(idN, idCA, idC)
    return dataRBM

def Rx(angle):
  rx = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
  return rx

def Ry(angle):
  ry = np.array([[np.cos(angle), 0, np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]])
  return ry

def Rz(angle):
  rz = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
  return rz


def AngBetVec(uu, vv):
    dotp = np.dot(uu, vv)
    normU = np.linalg.norm(uu)
    normV = np.linalg.norm(vv)

    cosAng = dotp / (normU * normV)
    ang = np.arccos(cosAng)

    return ang

def quadrant(_aa, xx, yy):
    ab = 0

    if (xx < 0 and yy > 0):
        ab = np.pi - _aa
    elif (xx < 0 and yy < 0):
        ab = np.pi + _aa
    elif (xx > 0 and yy < 0):
        ab = 2 * np.pi - _aa
    elif (xx > 0 and yy > 0):
        ab = _aa

    return ab

def removeRBMsingle(i, j, k):
    # i j k for the three particles which should define
    a1 = dat[i, :]
    a2 = dat[j, :]
    a3 = dat[k, :]

    v = a2 - a1

    vxy = np.copy(v)
    vxy[2] = 0

    vx = np.copy(vxy)
    vx[1] = 0
    a = AngBetVec(vxy, vx)
    aa = quadrant(a, v[0], v[1])
    phiZ = aa

    datR = np.copy(dat)
    datM = np.copy(dat)
    datR = dat - a1

    for l in range(0, len(dat)):
        datM[l, :] = np.dot(datR[l, :], Rz(phiZ))

    v = datM[j, :] - datM[i, :]
    vx = np.copy(v)
    vx[2] = 0
    vx[1] = 0

    a = AngBetVec(v, vx)
    a = quadrant(a, v[0], v[2])
    thetaY = np.pi / 2 - a

    for l in range(0, len(dat)):
        datM[l, :] = np.dot(datM[l, :], Ry(thetaY))

    u = datM[k, :] - datM[j, :]

    uxy = np.copy(u)
    uxy[2] = 0
    ux = np.copy(uxy)
    ux[1] = 0

    a = AngBetVec(uxy, ux)
    a = quadrant(a, u[0], u[1])
    psiZ = a

    for l in range(0, len(dat)):
        datM[l, :] = np.dot(datM[l, :], Rz(psiZ))

    return datM


### Convert ??? 
def convertToReducedCartersianCoordinates(data, breduce, dofsnp=np.array([18, 19, 20, 24, 25, 43], dtype=int), x_dim_red=60, x_dim_original=66):
    if breduce:
        dofs = torch.from_numpy(dofsnp).long()
        mask = torch.ones(data.shape[0], data.shape[1]).byte()
        for i in dofs:
            mask[:, i] = 0.
        data_red = data[mask].unsqueeze_(1).view(-1, x_dim_red)
    else:
        data_red = data
    return data_red



def convertToFullCartersianCoordinates(data, dofsnp=np.array([18, 19, 20, 24, 25, 43], dtype=int), x_dim_red=60, x_dim_original=66):

    dofs = torch.from_numpy(dofsnp).long()

    batch_size = data.shape[0]
    data_ext = torch.zeros(batch_size, x_dim_original)

    data_ext[:, 0:dofs[0]] = data[:, 0:dofs[0]]
    icountskip = 0
    removeddofs = dofs.shape[0]
    for i in range(dofs[0], x_dim_original):
        if icountskip >= removeddofs:
            data_ext[:, i:] = data[:, (i-icountskip):]
            break
        elif i == dofs[icountskip]:
            icountskip += 1
            continue
        else:
            data_ext[:, i] = data[:, i-icountskip]
    #data_ext[:, 0:dofs[0]] = data[:, 0:dofs[0]]
    #data_ext[:, dofs[2]+1:dofs[3]] = data[:, dofs[0]:dofs[0]+3]
    #data_ext[:, dofs[4]+1:dofs[5]] = data[:, dofs[0]+3:dofs[0]+3+(dofs[5]-dofs[4]-1)]
    #data_ext[:, dofs[5]+1:] = data[:, dofs[0]+3+(dofs[5]-dofs[4]-1):]

    return data_ext


### convert data that is not given in the units used by this framework to the correct units
def convertReferenceDataToUnit(data, lengthfactor, useangulardat):
    # cartesian representation
    if useangulardat is 'no':
        data.mul_(lengthfactor)
    elif 'ang_auggrouped' in useangulardat:
        dim_x = data.shape[1]
        dim_r = int(dim_x / 5)
        data[:, 0:dim_r].mul_(lengthfactor)

def getGroupedRsinphicosphisinthetacostheta(rcoord):

    n = rcoord.shape[0]
    dim = rcoord.shape[1]

    # specify the size of one coordinate point: here (r, sin \theta, cos \theta, sin \psi, cos \psi)
    sizeofcoord = 5
    nparticles = int(dim / sizeofcoord + 1)
    ncoordtuples = int(nparticles - 1)

    r = rcoord[:, 0 * ncoordtuples:1 * ncoordtuples]
    sinphi = rcoord[:, 1 * ncoordtuples:2 * ncoordtuples]
    cosphi = rcoord[:, 2 * ncoordtuples:3 * ncoordtuples]
    sintheta = rcoord[:, 3 * ncoordtuples:4 * ncoordtuples]
    costheta = rcoord[:, 4 * ncoordtuples:5 * ncoordtuples]

    return r, sinphi, cosphi, sintheta, costheta


def jacobiMatrixdxbardr(rcoord, bGPU=False):

    mint, maxt = -100., 100.

    n = rcoord.shape[0]
    dim = rcoord.shape[1]

    # specify the size of one coordinate point: here (r, sin \theta, cos \theta, sin \psi, cos \psi)
    sizeofcoord = 5
    nparticles = int(dim / sizeofcoord + 1)
    ncoordtuples = nparticles - 1

    r, sinphi, cosphi, sintheta, costheta = getGroupedRsinphicosphisinthetacostheta(rcoord)

    # TODO check division by 0
    tanphi = sinphi / cosphi
    cotphi = 1. / tanphi

    tantheta = sintheta / costheta
    cottheta = 1. / tantheta

    # bound the extreme values for tan x and cot x
    tanphi.clamp_(min=mint, max=maxt)
    cotphi.clamp_(min=mint, max=maxt)
    tantheta.clamp_(min=mint, max=maxt)
    cottheta.clamp_(min=mint, max=maxt)

    # rcood.view(N, ncoordtuples, sizeofcoord)
    if bGPU:
        jacobi = torch.zeros(n, (nparticles - 1) * 3, (nparticles - 1) * sizeofcoord, device=torch.device("cuda"))
    else:
        jacobi = torch.zeros(n, (nparticles - 1) * 3, (nparticles-1) * sizeofcoord)

    for i in range(ncoordtuples):
        # dx dr
        jacobi[:, i * 3 + 0, ncoordtuples * 0 + i] = sinphi[:, i] * costheta[:, i]
        jacobi[:, i * 3 + 1, ncoordtuples * 0 + i] = sinphi[:, i] * sintheta[:, i]
        jacobi[:, i * 3 + 2, ncoordtuples * 0 + i] = cosphi[:, i]
        # dx dsinphi
        jacobi[:, i * 3 + 0, ncoordtuples * 1 + i] = r[:, i] * costheta[:, i]
        jacobi[:, i * 3 + 1, ncoordtuples * 1 + i] = r[:, i] * sintheta[:, i]
        jacobi[:, i * 3 + 2, ncoordtuples * 1 + i] = - r[:, i] * tanphi[:, i]
        # dx dcosphi
        jacobi[:, i * 3 + 0, ncoordtuples * 2 + i] = - r[:, i] * cotphi[:, i] * costheta[:, i]
        jacobi[:, i * 3 + 1, ncoordtuples * 2 + i] = - r[:, i] * cotphi[:, i] * sintheta[:, i]
        jacobi[:, i * 3 + 2, ncoordtuples * 2 + i] = r[:, i]
        # dx dsintheta
        jacobi[:, i * 3 + 0, ncoordtuples * 3 + i] = - r[:, i] * sinphi[:, i] * tantheta[:, i]
        jacobi[:, i * 3 + 1, ncoordtuples * 3 + i] = r[:, i] * sinphi[:, i]
        jacobi[:, i * 3 + 2, ncoordtuples * 3 + i] = 0.
        # dx dcostheta
        jacobi[:, i * 3 + 0, ncoordtuples * 4 + i] = r[:, i] * sinphi[:, i]
        jacobi[:, i * 3 + 1, ncoordtuples * 4 + i] = - r[:, i] * sinphi[:, i] * cottheta[:, i]
        jacobi[:, i * 3 + 2, ncoordtuples * 4 + i] = 0.

    return jacobi

def jacobiMatrixdxdxbar(bGPU=False):
    # number of residues
    nACE = 1
    nALA = 1
    nNME = 1

    ACEleng = 6
    ALAleng = 10
    NMEleng = 6

    iFirstALA = ACEleng
    iFirstNME = ACEleng + nALA * ALAleng

    natoms = ACEleng + ALAleng + NMEleng

    if bGPU:
        jacobimatrix = torch.zeros(natoms*3, (natoms-1)*3, device=torch.device("cuda"))
    else:
        jacobimatrix = torch.zeros(natoms*3, (natoms-1)*3)

    idx = 0
    for i in range(idx, idx+3):
        jacobimatrix[i, i] = 1.
    idx += 3
    for k in range(2, natoms):
        idx += 3
        for i in range(idx, idx+3):
            jacobimatrix[i, i-3] = 1.

    return jacobimatrix

def buildConnectionMatrix():
    # number of residues
    nACE = 1
    nALA = 1
    nNME = 1

    ACEleng = 6
    ALAleng = 10
    NMEleng = 6

    iFirstALA = ACEleng
    iFirstNME = ACEleng + nALA * ALAleng

    natoms = ACEleng + ALAleng + NMEleng

    natomconnection = torch.zeros(natoms, natoms)

    natomconnection[0, 1] = 1
    natomconnection[1, 0] = 1

    natomconnection[1, 2] = 1
    natomconnection[2, 1] = 1
    natomconnection[1, 3] = 1
    natomconnection[3, 1] = 1
    natomconnection[1, 4] = 1
    natomconnection[4, 1] = 1

    natomconnection[4, 5] = 1
    natomconnection[5, 4] = 1

    natomconnection[4, iFirstALA] = 1
    natomconnection[iFirstALA, 4] = 1

    for iALA in range(0, nALA):
        natomconnection[iFirstALA + iALA * ALAleng + 0, iFirstALA + iALA * ALAleng + 1] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 1, iFirstALA + iALA * ALAleng + 0] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 0, iFirstALA + iALA * ALAleng + 2] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 2, iFirstALA + iALA * ALAleng + 0] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 2, iFirstALA + iALA * ALAleng + 3] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 3, iFirstALA + iALA * ALAleng + 2] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 2, iFirstALA + iALA * ALAleng + 4] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 4, iFirstALA + iALA * ALAleng + 2] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 2, iFirstALA + iALA * ALAleng + 8] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 8, iFirstALA + iALA * ALAleng + 2] = 1

        natomconnection[iFirstALA + iALA * ALAleng + 4, iFirstALA + iALA * ALAleng + 5] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 5, iFirstALA + iALA * ALAleng + 4] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 4, iFirstALA + iALA * ALAleng + 6] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 6, iFirstALA + iALA * ALAleng + 4] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 4, iFirstALA + iALA * ALAleng + 7] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 7, iFirstALA + iALA * ALAleng + 4] = 1

        natomconnection[iFirstALA + iALA * ALAleng + 8, iFirstALA + iALA * ALAleng + 9] = 1
        natomconnection[iFirstALA + iALA * ALAleng + 9, iFirstALA + iALA * ALAleng + 8] = 1
        if iALA+1 < nALA:
            natomconnection[iFirstALA + (iALA+1) * ALAleng + 0, iFirstALA + iALA * ALAleng + 8] = 1
            natomconnection[iFirstALA + iALA * ALAleng + 8, iFirstALA + (iALA+1) * ALAleng + 0] = 1
    natomconnection[iFirstNME - 2, iFirstNME + 0] = 1
    natomconnection[iFirstNME + 0, iFirstNME - 2] = 1

    natomconnection[iFirstNME + 0, iFirstNME + 1] = 1
    natomconnection[iFirstNME + 1, iFirstNME + 0] = 1
    natomconnection[iFirstNME + 0, iFirstNME + 2] = 1
    natomconnection[iFirstNME + 2, iFirstNME + 0] = 1
    natomconnection[iFirstNME + 2, iFirstNME + 3] = 1
    natomconnection[iFirstNME + 3, iFirstNME + 2] = 1
    natomconnection[iFirstNME + 2, iFirstNME + 4] = 1
    natomconnection[iFirstNME + 4, iFirstNME + 2] = 1
    natomconnection[iFirstNME + 2, iFirstNME + 3] = 1
    natomconnection[iFirstNME + 3, iFirstNME + 2] = 1

    return natomconnection


def getAbsCoordinates(xyz, bGPU=False):
    if bGPU:
        _xyzAbs = torch.zeros([xyz.shape[0] + 1, xyz.shape[1], xyz.shape[2]], device=torch.device('cuda'))
    else:
        _xyzAbs = torch.zeros([xyz.shape[0] + 1, xyz.shape[1], xyz.shape[2]])

    nsamples =  xyz.shape[2]

    # number of residues
    nACE = 1
    nALA = 1
    nNME = 1

    ACEleng = 6
    ALAleng = 10
    NMEleng = 6

    # go through every residue
    if bGPU:
        aACE = torch.zeros([nACE * ACEleng, 3, nsamples], device=torch.device('cuda'))
        aALA = torch.zeros([nALA * ALAleng, 3, nsamples], device=torch.device('cuda'))
        aNME = torch.zeros([nNME * NMEleng, 3, nsamples], device=torch.device('cuda'))
    else:
        aACE = torch.zeros([nACE * ACEleng, 3, nsamples])
        aALA = torch.zeros([nALA * ALAleng, 3, nsamples])
        aNME = torch.zeros([nNME * NMEleng, 3, nsamples])

    # 1HH3 = CH3 + (1HH3 - CH3)
    aACE[0, :, :] = xyz[0, :, :]
    # CH3 = 0
    # aACE[1,:] = 0
    # 2HH3 = CH3 + (2HH3 - CH3)
    aACE[2, :, :] = xyz[1, :, :]
    # 3HH3 = CH3 + (3HH3 - CH3)
    aACE[3, :, :] = xyz[2, :, :]
    # C = CH3 + (C - CH3)
    aACE[4, :, :] = xyz[3, :, :]
    # O = C + (O - C)
    aACE[5, :, :] = aACE[4, :, :] + xyz[4, :, :]

    # first N coordinate
    aALA[0, :, :] = aACE[4, :, :] + xyz[5, :, :]

    for iALA in range(0, nALA):
        # N = C + (N - C)
        if iALA > 0:
            aALA[iALA * ALAleng + 0, :, :] = aALA[iALA * ALAleng - 2, :, :] + xyz[ACEleng + iALA * ALAleng - 1, :, :]
        # H = N + (H - N)
        aALA[iALA * ALAleng + 1, :, :] = aALA[iALA * ALAleng + 0, :, :] + xyz[ACEleng + iALA * ALAleng + 0, :, :]
        # CA = N + (CA - N)
        aALA[iALA * ALAleng + 2, :, :] = aALA[iALA * ALAleng + 0, :, :] + xyz[ACEleng + iALA * ALAleng + 1, :, :]
        # HA = CA + (HA - CA)
        aALA[iALA * ALAleng + 3, :, :] = aALA[iALA * ALAleng + 2, :, :] + xyz[ACEleng + iALA * ALAleng + 2, :, :]
        # CB = CA + (CB - CA)
        aALA[iALA * ALAleng + 4, :, :] = aALA[iALA * ALAleng + 2, :, :] + xyz[ACEleng + iALA * ALAleng + 3, :, :]
        # HB1 = CB + (HB1 - CB)
        aALA[iALA * ALAleng + 5, :, :] = aALA[iALA * ALAleng + 4, :, :] + xyz[ACEleng + iALA * ALAleng + 4, :, :]
        # HB2 = CB + (HB2 - CB)
        aALA[iALA * ALAleng + 6, :, :] = aALA[iALA * ALAleng + 4, :, :] + xyz[ACEleng + iALA * ALAleng + 5, :, :]
        # HB3 = CB + (HB3 - CB)
        aALA[iALA * ALAleng + 7, :, :] = aALA[iALA * ALAleng + 4, :, :] + xyz[ACEleng + iALA * ALAleng + 6, :, :]
        # C = CA + (C - CA)
        aALA[iALA * ALAleng + 8, :, :] = aALA[iALA * ALAleng + 2, :, :] + xyz[ACEleng + iALA * ALAleng + 7, :, :]
        # O = C + (O - C)
        aALA[iALA * ALAleng + 9, :, :] = aALA[iALA * ALAleng + 8, :, :] + xyz[ACEleng + iALA * ALAleng + 8, :, :]

    # Last part
    # N = C + (N - C)
    aNME[0, :, :] = aALA[nALA * ALAleng - 2, :, :] + xyz[ACEleng + nALA * ALAleng - 1, :, :]
    # H = N + (H - N)
    aNME[1, :, :] = aNME[0, :, :] + xyz[ACEleng + nALA * ALAleng + 0, :, :]
    # CH3 = N + (CH3 - N)
    aNME[2, :, :] = aNME[0, :, :] + xyz[ACEleng + nALA * ALAleng + 1, :, :]
    # 1HH3 = CH3 + (1HH3 - CH3)
    aNME[3, :, :] = aNME[2, :, :] + xyz[ACEleng + nALA * ALAleng + 2, :, :]
    # 2HH3 = CH3 + (2HH3 - CH3)
    aNME[4, :, :] = aNME[2, :, :] + xyz[ACEleng + nALA * ALAleng + 3, :, :]
    # 3HH3 = CH3 + (2HH3 - CH3)
    aNME[5, :, :] = aNME[2, :, :] + xyz[ACEleng + nALA * ALAleng + 4, :, :]

    _xyzAbs[0:ACEleng, :, :] = aACE
    _xyzAbs[ACEleng:(ACEleng + nALA * ALAleng), :, :] = aALA
    _xyzAbs[(ACEleng + nALA * ALAleng):, :, :] = aNME

    return _xyzAbs

def getCartesianT(rphitheta, dataaugmented=False, bGPU=False):
    rphithetaShape = rphitheta.shape

    if dataaugmented:
        if bGPU:
            _xyz = torch.zeros([rphithetaShape[0], 3, rphithetaShape[2]], device=torch.device('cuda'))
        else:
            _xyz = torch.zeros([rphithetaShape[0], 3, rphithetaShape[2]])
        r = rphitheta[:, 0, :]
        sinphi = rphitheta[:, 1, :]
        cosphi = rphitheta[:, 2, :]
        sintheta = rphitheta[:, 3, :]
        costheta = rphitheta[:, 4]
        _xyz[:, 0, :] = r * costheta * sinphi
        _xyz[:, 1, :] = r * sintheta * sinphi
        _xyz[:, 2, :] = r * cosphi
    else:
        _xyz = np.zeros(rphithetaShape)
        _xyz[:, 0] = rphitheta[:, 0] * torch.cos(rphitheta[:, 2]) * torch.sin(rphitheta[:, 1])
        _xyz[:, 1] = rphitheta[:, 0] * torch.sin(rphitheta[:, 2]) * torch.sin(rphitheta[:, 1])
        _xyz[:, 2] = rphitheta[:, 0] * torch.cos(rphitheta[:, 1])

    xyzAbs = getAbsCoordinates(xyz=_xyz, bGPU=bGPU)

    return xyzAbs


def convertangularaugmenteddataset(data, bgrouped=False):

    #data = torch.from_numpy(datanp)

    #outname = 'samples.txt'
    #data = np.loadtxt('dataset_mixed_1527_ang.txt')

    dim = data.shape[0]
    n = data.shape[1]

    # specify the size of one coordinate point: here (r, sin \theta, cos \theta, sin \psi, cos \psi)
    sizeofcoord = int(5)
    nparticles = int(dim / sizeofcoord + 1)
    ncoordtuples = int(nparticles - 1)

    dataUse = torch.zeros_like(data)
    if bgrouped:
        # sorted dataset r1 r2 r3 r4 , sin sin sin
        # temporary dataset for
        r = data[0 * ncoordtuples:1 * ncoordtuples, :]
        sinphi = data[1 * ncoordtuples:2 * ncoordtuples, :]
        cosphi = data[2 * ncoordtuples:3 * ncoordtuples, :]
        sintheta = data[3 * ncoordtuples:4 * ncoordtuples, :]
        costheta = data[4 * ncoordtuples:5 * ncoordtuples, :]
        for i in range(0, ncoordtuples):
            dataUse[i * sizeofcoord + 0, :] = r[i, :]
            dataUse[i * sizeofcoord + 1, :] = sinphi[i, :]
            dataUse[i * sizeofcoord + 2, :] = cosphi[i, :]
            dataUse[i * sizeofcoord + 3, :] = sintheta[i, :]
            dataUse[i * sizeofcoord + 4, :] = costheta[i, :]
    else:
        dataUse.data = data.clone()

    rphithetaaugmented = dataUse.view([int(dim/sizeofcoord), sizeofcoord, n])
    datacatout = getCartesianT(rphitheta=rphithetaaugmented, dataaugmented=True)

    # rphithetaaugmented
    # for sample in dataUse.transpose(0,1):
    #     rphithetaaugmented = torch.zeros([dim/sizeofcoord, sizeofcoord])
    #
    #
    #
    # for j in range(0, n):
    #     sample = dataUse[:, j]
    #     rphithetaaugmented = np.zeros([dim/sizeofcoord, sizeofcoord])
    #     for i in range(0, rphithetaaugmented.shape[0]):
    #         rphithetaaugmented[i, 0] = sample[i * sizeofcoord + 0]
    #         rphithetaaugmented[i, 1] = sample[i * sizeofcoord + 1]
    #         rphithetaaugmented[i, 2] = sample[i * sizeofcoord + 2]
    #         rphithetaaugmented[i, 3] = sample[i * sizeofcoord + 3]
    #         rphithetaaugmented[i, 4] = sample[i * sizeofcoord + 4]
    #
    #     datacoord = getCartesian(rphitheta=rphithetaaugmented, dataaugmented=True)
    #     datacoordvec = np.reshape(datacoord, nparticles * 3)
    #     datacatout[:, j] = np.copy(datacoordvec)

    return datacatout


class TestTools(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(138323)
        #self.data_ang_grouped = np.loadtxt('/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide/dataset_mixed_1527_ang_auggrouped.pdb')
        path = '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide/dataset_10.txt'
        self.dat = torch.from_numpy(np.loadtxt(path)).float().transpose(0, 1)

    def test_ConvertToFullCartersianCoordinates(self):
        indzero = self.dat.abs() < 0.00001
        indactive = (indzero.long() - 1).abs()
        dat_red = self.dat[indactive.byte()].unsqueeze_(1).view(-1, 60)

        dat_reconst = convertToFullCartersianCoordinates(data=dat_red)
        #print(dat_reconst.abs() == self.dat.abs())
        #print(dat_reconst)
        #print(self.dat)
        ind_rec_zeros = dat_reconst.abs() < 0.00001
        ind_dat_zeros = self.dat.abs() < 0.00001
        res = (ind_rec_zeros == ind_dat_zeros).all()
        print(res)
        self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()

