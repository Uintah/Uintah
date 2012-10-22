/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/Node27Interpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace SCIRun;
using namespace Uintah;

    
Node27Interpolator::Node27Interpolator()
{
  d_size = 27;
  d_patch = 0;
}

Node27Interpolator::Node27Interpolator(const Patch* patch)
{
  d_size = 27;
  d_patch = patch;
}
    
Node27Interpolator::~Node27Interpolator()
{
}

Node27Interpolator* Node27Interpolator::clone(const Patch* patch)
{
  return scinew Node27Interpolator(patch);
}
    
void Node27Interpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni, 
                                            vector<double>& S,
                                            const Matrix3& size,
                                            const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  double lz = size(2,2)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }
  
  ni[0]  = IntVector(ix,    iy,      iz);
  ni[1]  = IntVector(ix+1,  iy,      iz);
  ni[2]  = IntVector(ix+nnx,iy,      iz);
  ni[3]  = IntVector(ix,    iy+1,    iz);
  ni[4]  = IntVector(ix+1,  iy+1,    iz);
  ni[5]  = IntVector(ix+nnx,iy+1,    iz);
  ni[6]  = IntVector(ix,    iy+nny,  iz);
  ni[7]  = IntVector(ix+1,  iy+nny,  iz);
  ni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  ni[9]  = IntVector(ix,    iy,      iz+1);
  ni[10] = IntVector(ix+1,  iy,      iz+1);
  ni[11] = IntVector(ix+nnx,iy,      iz+1);
  ni[12] = IntVector(ix,    iy+1,    iz+1);
  ni[13] = IntVector(ix+1,  iy+1,    iz+1);
  ni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  ni[15] = IntVector(ix,    iy+nny,  iz+1);
  ni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  ni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  ni[18] = IntVector(ix,    iy,      iz+nnz);
  ni[19] = IntVector(ix+1,  iy,      iz+nnz);
  ni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  ni[21] = IntVector(ix,    iy+1,    iz+nnz);
  ni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  ni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  ni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  ni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  ni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
  
  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double pz0 = cellpos.z() - (iz);
  double pz1 = cellpos.z() - (iz+1);
  double pz2 = cellpos.z() - (iz + nnz);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX},
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX},
         fz[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
  if(px0 <= lx){
    fx[0] = 1. - (px0*px0 + (lx)*(lx))/(2*lx);
    fx[1] = (1. + lx + px1)*(1. + lx + px1)/(4*lx);
    fx[2] = (1. + lx - px2)*(1. + lx - px2)/(4*lx);
  }
  else if(px0 > lx && px0 <= (1.-lx)){
    fx[0] = 1. - px0;
    fx[1] = 1. + px1;
    fx[2] = 0.;
  }
  else if(px0 > (1.-lx)){
    fx[0] = (1. + lx - px0)*(1. + lx - px0)/(4*lx);
    fx[1] = 1. - (px1*px1 + (lx)*(lx))/(2*lx);
    fx[2] = (1. + lx + px2)*(1. + lx + px2)/(4*lx);
  }
  
  if(py0 <= ly){
    fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2*ly);
    fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4*ly);
    fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4*ly);
  }
  else if(py0 > ly && py0 <= (1.-ly)){
    fy[0] = 1. - py0;
    fy[1] = 1. + py1;
    fy[2] = 0.;
  }
  else if(py0 > (1.-ly)){
    fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4*ly);
    fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2*ly);
    fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4*ly);
  }
  
  if(pz0 <= lz){
    fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
    fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4*lz);
    fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4*lz);
  }
  else if(pz0 > lz && pz0 <= (1.-lz)){
    fz[0] = 1. - pz0;
    fz[1] = 1. + pz1;
    fz[2] = 0.;
  }
  else if(pz0 > (1.-lz)){
    fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4*lz);
    fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2*lz);
    fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4*lz);
  }
  
  S[0]  = fx[0]*fy[0]*fz[0];
  S[1]  = fx[1]*fy[0]*fz[0];
  S[2]  = fx[2]*fy[0]*fz[0];
  S[3]  = fx[0]*fy[1]*fz[0];
  S[4]  = fx[1]*fy[1]*fz[0];
  S[5]  = fx[2]*fy[1]*fz[0];
  S[6]  = fx[0]*fy[2]*fz[0];
  S[7]  = fx[1]*fy[2]*fz[0];
  S[8]  = fx[2]*fy[2]*fz[0];
  S[9]  = fx[0]*fy[0]*fz[1];
  S[10] = fx[1]*fy[0]*fz[1];
  S[11] = fx[2]*fy[0]*fz[1];
  S[12] = fx[0]*fy[1]*fz[1];
  S[13] = fx[1]*fy[1]*fz[1];
  S[14] = fx[2]*fy[1]*fz[1];
  S[15] = fx[0]*fy[2]*fz[1];
  S[16] = fx[1]*fy[2]*fz[1];
  S[17] = fx[2]*fy[2]*fz[1];
  S[18] = fx[0]*fy[0]*fz[2];
  S[19] = fx[1]*fy[0]*fz[2];
  S[20] = fx[2]*fy[0]*fz[2];
  S[21] = fx[0]*fy[1]*fz[2];
  S[22] = fx[1]*fy[1]*fz[2];
  S[23] = fx[2]*fy[1]*fz[2];
  S[24] = fx[0]*fy[2]*fz[2];
  S[25] = fx[1]*fy[2]*fz[2];
  S[26] = fx[2]*fy[2]*fz[2];

}
 
void Node27Interpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  double lz = size(2,2)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }
  
  ni[0]  = IntVector(ix,    iy,      iz);
  ni[1]  = IntVector(ix+1,  iy,      iz);
  ni[2]  = IntVector(ix+nnx,iy,      iz);
  ni[3]  = IntVector(ix,    iy+1,    iz);
  ni[4]  = IntVector(ix+1,  iy+1,    iz);
  ni[5]  = IntVector(ix+nnx,iy+1,    iz);
  ni[6]  = IntVector(ix,    iy+nny,  iz);
  ni[7]  = IntVector(ix+1,  iy+nny,  iz);
  ni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  ni[9]  = IntVector(ix,    iy,      iz+1);
  ni[10] = IntVector(ix+1,  iy,      iz+1);
  ni[11] = IntVector(ix+nnx,iy,      iz+1);
  ni[12] = IntVector(ix,    iy+1,    iz+1);
  ni[13] = IntVector(ix+1,  iy+1,    iz+1);
  ni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  ni[15] = IntVector(ix,    iy+nny,  iz+1);
  ni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  ni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  ni[18] = IntVector(ix,    iy,      iz+nnz);
  ni[19] = IntVector(ix+1,  iy,      iz+nnz);
  ni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  ni[21] = IntVector(ix,    iy+1,    iz+nnz);
  ni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  ni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  ni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  ni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  ni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
  
  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double pz0 = cellpos.z() - (iz);
  double pz1 = cellpos.z() - (iz+1);
  double pz2 = cellpos.z() - (iz + nnz);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         fz[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfy[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfz[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
  if(px0 <= lx){
    fx[0]  = 1. - (px0*px0 + (lx)*(lx))/(2.*lx);
    fx[1]  = (1. + lx + px1)*(1. + lx + px1)/(4.*lx);
    fx[2]  = (1. + lx - px2)*(1. + lx - px2)/(4.*lx);
    dfx[0] = -px0/lx;
    dfx[1] =  (1. + lx + px1)/(2.*lx);
    dfx[2] = -(1. + lx - px2)/(2.*lx);
  }
  else if(px0 > lx && px0 <= (1-lx)){
    fx[0]  = 1. - px0;
    fx[1]  = 1. + px1;
    fx[2]  = 0.;
    dfx[0] = -1.;
    dfx[1] =  1.;
    dfx[2] =  0.;
  }
  else if(px0 > (1-lx)){
    fx[0]  = (1. + lx - px0)*(1. + lx - px0)/(4.*lx);
    fx[1]  = 1. - (px1*px1 + (lx)*(lx))/(2.*lx);
    fx[2]  = (1. + lx + px2)*(1. + lx + px2)/(4.*lx);
    dfx[0] = -(1. + lx - px0)/(2.*lx);
    dfx[1] = -px1/lx;
    dfx[2] = (1. + lx + px2)/(2.*lx);
  }
  
  if(py0 <= ly){
    fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2.*ly);
    fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4.*ly);
    fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4.*ly);
    dfy[0] = -py0/ly;
    dfy[1] =  (1. + ly + py1)/(2.*ly);
    dfy[2] = -(1. + ly - py2)/(2.*ly);
  }
  else if(py0 > ly && py0 <= (1-ly)){
    fy[0] = 1. - py0;
    fy[1] = 1. + py1;
    fy[2] = 0.;
    dfy[0] = -1.;
    dfy[1] =  1.;
    dfy[2] =  0.;
  }
  else if(py0 > (1-ly)){
    fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4.*ly);
    fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2.*ly);
    fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4.*ly);
    dfy[0] = -(1. + ly - py0)/(2.*ly);
    dfy[1] = -py1/ly;
    dfy[2] = (1. + ly + py2)/(2.*ly);
  }
  
  if(pz0 <= lz){
    fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
    fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4.*lz);
    fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4.*lz);
    dfz[0] = -pz0/lz;
    dfz[1] =  (1. + lz + pz1)/(2.*lz);
    dfz[2] = -(1. + lz - pz2)/(2.*lz);
  }
  else if(pz0 > lz && pz0 <= (1-lz)){
    fz[0] = 1. - pz0;
    fz[1] = 1. + pz1;
    fz[2] = 0.;
    dfz[0] = -1.;
    dfz[1] =  1.;
    dfz[2] =  0.;
  }
  else if(pz0 > (1-lz)){
    fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4.*lz);
    fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2.*lz);
    fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4.*lz);
    dfz[0] = -(1. + lz - pz0)/(2.*lz);
    dfz[1] = -pz1/lz;
    dfz[2] = (1. + lz + pz2)/(2.*lz);
  }
  
  d_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
  d_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
  d_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
  d_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
  d_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
  d_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
  d_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
  d_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
  d_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);
  
  d_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
  d_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
  d_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
  d_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
  d_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
  d_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
  d_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
  d_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
  d_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);
  
  d_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
  d_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
  d_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
  d_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
  d_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
  d_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
  d_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
  d_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
  d_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

}

void 
Node27Interpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  double lz = size(2,2)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }
  
  ni[0]  = IntVector(ix,    iy,      iz);
  ni[1]  = IntVector(ix+1,  iy,      iz);
  ni[2]  = IntVector(ix+nnx,iy,      iz);
  ni[3]  = IntVector(ix,    iy+1,    iz);
  ni[4]  = IntVector(ix+1,  iy+1,    iz);
  ni[5]  = IntVector(ix+nnx,iy+1,    iz);
  ni[6]  = IntVector(ix,    iy+nny,  iz);
  ni[7]  = IntVector(ix+1,  iy+nny,  iz);
  ni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  ni[9]  = IntVector(ix,    iy,      iz+1);
  ni[10] = IntVector(ix+1,  iy,      iz+1);
  ni[11] = IntVector(ix+nnx,iy,      iz+1);
  ni[12] = IntVector(ix,    iy+1,    iz+1);
  ni[13] = IntVector(ix+1,  iy+1,    iz+1);
  ni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  ni[15] = IntVector(ix,    iy+nny,  iz+1);
  ni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  ni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  ni[18] = IntVector(ix,    iy,      iz+nnz);
  ni[19] = IntVector(ix+1,  iy,      iz+nnz);
  ni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  ni[21] = IntVector(ix,    iy+1,    iz+nnz);
  ni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  ni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  ni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  ni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  ni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
  
  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double pz0 = cellpos.z() - (iz);
  double pz1 = cellpos.z() - (iz+1);
  double pz2 = cellpos.z() - (iz + nnz);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX},
         fz[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfy[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfz[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
  if(px0 <= lx){
    fx[0]  = 1. - (px0*px0 + (lx)*(lx))/(2.*lx);
    fx[1]  = (1. + lx + px1)*(1. + lx + px1)/(4.*lx);
    fx[2]  = (1. + lx - px2)*(1. + lx - px2)/(4.*lx);
    dfx[0] = -px0/lx;
    dfx[1] =  (1. + lx + px1)/(2.*lx);
    dfx[2] = -(1. + lx - px2)/(2.*lx);
  }
  else if(px0 > lx && px0 <= (1-lx)){
    fx[0]  = 1. - px0;
    fx[1]  = 1. + px1;
    fx[2]  = 0.;
    dfx[0] = -1.;
    dfx[1] =  1.;
    dfx[2] =  0.;
  }
  else if(px0 > (1-lx)){
    fx[0]  = (1. + lx - px0)*(1. + lx - px0)/(4.*lx);
    fx[1]  = 1. - (px1*px1 + (lx)*(lx))/(2.*lx);
    fx[2]  = (1. + lx + px2)*(1. + lx + px2)/(4.*lx);
    dfx[0] = -(1. + lx - px0)/(2.*lx);
    dfx[1] = -px1/lx;
    dfx[2] = (1. + lx + px2)/(2.*lx);
  }
  
  if(py0 <= ly){
    fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2.*ly);
    fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4.*ly);
    fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4.*ly);
    dfy[0] = -py0/ly;
    dfy[1] =  (1. + ly + py1)/(2.*ly);
    dfy[2] = -(1. + ly - py2)/(2.*ly);
  }
  else if(py0 > ly && py0 <= (1-ly)){
    fy[0] = 1. - py0;
    fy[1] = 1. + py1;
    fy[2] = 0.;
    dfy[0] = -1.;
    dfy[1] =  1.;
    dfy[2] =  0.;
  }
  else if(py0 > (1-ly)){
    fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4.*ly);
    fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2.*ly);
    fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4.*ly);
    dfy[0] = -(1. + ly - py0)/(2.*ly);
    dfy[1] = -py1/ly;
    dfy[2] = (1. + ly + py2)/(2.*ly);
  }
  
  if(pz0 <= lz){
    fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
    fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4.*lz);
    fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4.*lz);
    dfz[0] = -pz0/lz;
    dfz[1] =  (1. + lz + pz1)/(2.*lz);
    dfz[2] = -(1. + lz - pz2)/(2.*lz);
  }
  else if(pz0 > lz && pz0 <= (1-lz)){
    fz[0] = 1. - pz0;
    fz[1] = 1. + pz1;
    fz[2] = 0.;
    dfz[0] = -1.;
    dfz[1] =  1.;
    dfz[2] =  0.;
  }
  else if(pz0 > (1-lz)){
    fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4.*lz);
    fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2.*lz);
    fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4.*lz);
    dfz[0] = -(1. + lz - pz0)/(2.*lz);
    dfz[1] = -pz1/lz;
    dfz[2] = (1. + lz + pz2)/(2.*lz);
  }
  
  S[0]  = fx[0]*fy[0]*fz[0];
  S[1]  = fx[1]*fy[0]*fz[0];
  S[2]  = fx[2]*fy[0]*fz[0];
  S[3]  = fx[0]*fy[1]*fz[0];
  S[4]  = fx[1]*fy[1]*fz[0];
  S[5]  = fx[2]*fy[1]*fz[0];
  S[6]  = fx[0]*fy[2]*fz[0];
  S[7]  = fx[1]*fy[2]*fz[0];
  S[8]  = fx[2]*fy[2]*fz[0];
  S[9]  = fx[0]*fy[0]*fz[1];
  S[10] = fx[1]*fy[0]*fz[1];
  S[11] = fx[2]*fy[0]*fz[1];
  S[12] = fx[0]*fy[1]*fz[1];
  S[13] = fx[1]*fy[1]*fz[1];
  S[14] = fx[2]*fy[1]*fz[1];
  S[15] = fx[0]*fy[2]*fz[1];
  S[16] = fx[1]*fy[2]*fz[1];
  S[17] = fx[2]*fy[2]*fz[1];
  S[18] = fx[0]*fy[0]*fz[2];
  S[19] = fx[1]*fy[0]*fz[2];
  S[20] = fx[2]*fy[0]*fz[2];
  S[21] = fx[0]*fy[1]*fz[2];
  S[22] = fx[1]*fy[1]*fz[2];
  S[23] = fx[2]*fy[1]*fz[2];
  S[24] = fx[0]*fy[2]*fz[2];
  S[25] = fx[1]*fy[2]*fz[2];
  S[26] = fx[2]*fy[2]*fz[2];
  
  d_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
  d_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
  d_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
  d_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
  d_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
  d_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
  d_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
  d_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
  d_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);
  
  d_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
  d_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
  d_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
  d_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
  d_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
  d_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
  d_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
  d_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
  d_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);
  
  d_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
  d_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
  d_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
  d_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
  d_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
  d_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
  d_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
  d_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
  d_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

}

int Node27Interpolator::size()
{
  return d_size;
}
