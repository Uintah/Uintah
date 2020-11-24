/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <Core/Grid/GIMPInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace Uintah;
using namespace std;
    
GIMPInterpolator::GIMPInterpolator()
{
  d_size = 27;
  d_patch = 0;
}

GIMPInterpolator::GIMPInterpolator(const Patch* patch)
{
  d_size = 27;
  d_patch = patch;
}
    
GIMPInterpolator::~GIMPInterpolator()
{
}

GIMPInterpolator* GIMPInterpolator::clone(const Patch* patch)
{
  return scinew GIMPInterpolator(patch);
}
    
int GIMPInterpolator::findCellAndWeights(const Point& pos,
                                         vector<IntVector>& ni, 
                                         vector<double>& S,
                                         const Matrix3& size)
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

  IntVector tni[27];
  double tS[27];
  
  tni[0]  = IntVector(ix,    iy,      iz);
  tni[1]  = IntVector(ix+1,  iy,      iz);
  tni[2]  = IntVector(ix+nnx,iy,      iz);
  tni[3]  = IntVector(ix,    iy+1,    iz);
  tni[4]  = IntVector(ix+1,  iy+1,    iz);
  tni[5]  = IntVector(ix+nnx,iy+1,    iz);
  tni[6]  = IntVector(ix,    iy+nny,  iz);
  tni[7]  = IntVector(ix+1,  iy+nny,  iz);
  tni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  tni[9]  = IntVector(ix,    iy,      iz+1);
  tni[10] = IntVector(ix+1,  iy,      iz+1);
  tni[11] = IntVector(ix+nnx,iy,      iz+1);
  tni[12] = IntVector(ix,    iy+1,    iz+1);
  tni[13] = IntVector(ix+1,  iy+1,    iz+1);
  tni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  tni[15] = IntVector(ix,    iy+nny,  iz+1);
  tni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  tni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  tni[18] = IntVector(ix,    iy,      iz+nnz);
  tni[19] = IntVector(ix+1,  iy,      iz+nnz);
  tni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  tni[21] = IntVector(ix,    iy+1,    iz+nnz);
  tni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  tni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  tni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  tni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  tni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
  
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
  double fx[3] = {NAN,NAN,NAN};
  double fy[3] = {NAN,NAN,NAN};
  double fz[3] = {NAN,NAN,NAN};;

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
  
  tS[0]  = fx[0]*fy[0]*fz[0];
  tS[1]  = fx[1]*fy[0]*fz[0];
  tS[2]  = fx[2]*fy[0]*fz[0];
  tS[3]  = fx[0]*fy[1]*fz[0];
  tS[4]  = fx[1]*fy[1]*fz[0];
  tS[5]  = fx[2]*fy[1]*fz[0];
  tS[6]  = fx[0]*fy[2]*fz[0];
  tS[7]  = fx[1]*fy[2]*fz[0];
  tS[8]  = fx[2]*fy[2]*fz[0];
  tS[9]  = fx[0]*fy[0]*fz[1];
  tS[10] = fx[1]*fy[0]*fz[1];
  tS[11] = fx[2]*fy[0]*fz[1];
  tS[12] = fx[0]*fy[1]*fz[1];
  tS[13] = fx[1]*fy[1]*fz[1];
  tS[14] = fx[2]*fy[1]*fz[1];
  tS[15] = fx[0]*fy[2]*fz[1];
  tS[16] = fx[1]*fy[2]*fz[1];
  tS[17] = fx[2]*fy[2]*fz[1];

  tS[18] = fx[0]*fy[0]*fz[2];
  tS[19] = fx[1]*fy[0]*fz[2];
  tS[20] = fx[2]*fy[0]*fz[2];
  tS[21] = fx[0]*fy[1]*fz[2];
  tS[22] = fx[1]*fy[1]*fz[2];
  tS[23] = fx[2]*fy[1]*fz[2];
  tS[24] = fx[0]*fy[2]*fz[2];
  tS[25] = fx[1]*fy[2]*fz[2];
  tS[26] = fx[2]*fy[2]*fz[2];

  int count = 0;
  for(int i=0;i<27;i++){
   if(tS[i]>0.0){
    S[count] =tS[i];
    ni[count]=tni[i];
    count++;
   }
  }

  return count;
}
 
int GIMPInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                  vector<IntVector>& ni,
                                                  vector<Vector>& d_S,
                                                  const Matrix3& size)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  double lz = size(2,2)/2.;
  
  IntVector tni[27];
  double tS[27];
  Vector td_S[27];

  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }
  
  tni[0]  = IntVector(ix,    iy,      iz);
  tni[1]  = IntVector(ix+1,  iy,      iz);
  tni[2]  = IntVector(ix+nnx,iy,      iz);
  tni[3]  = IntVector(ix,    iy+1,    iz);
  tni[4]  = IntVector(ix+1,  iy+1,    iz);
  tni[5]  = IntVector(ix+nnx,iy+1,    iz);
  tni[6]  = IntVector(ix,    iy+nny,  iz);
  tni[7]  = IntVector(ix+1,  iy+nny,  iz);
  tni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  tni[9]  = IntVector(ix,    iy,      iz+1);
  tni[10] = IntVector(ix+1,  iy,      iz+1);
  tni[11] = IntVector(ix+nnx,iy,      iz+1);
  tni[12] = IntVector(ix,    iy+1,    iz+1);
  tni[13] = IntVector(ix+1,  iy+1,    iz+1);
  tni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  tni[15] = IntVector(ix,    iy+nny,  iz+1);
  tni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  tni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  tni[18] = IntVector(ix,    iy,      iz+nnz);
  tni[19] = IntVector(ix+1,  iy,      iz+nnz);
  tni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  tni[21] = IntVector(ix,    iy+1,    iz+nnz);
  tni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  tni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  tni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  tni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  tni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
 
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
  double fx[3]  = {NAN,NAN,NAN};;
  double fy[3]  = {NAN,NAN,NAN};;
  double fz[3]  = {NAN,NAN,NAN};;
  double dfx[3] = {NAN,NAN,NAN};;
  double dfy[3] = {NAN,NAN,NAN};;
  double dfz[3] = {NAN,NAN,NAN};;

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
  
  tS[0]  = fx[0]*fy[0]*fz[0];
  tS[1]  = fx[1]*fy[0]*fz[0];
  tS[2]  = fx[2]*fy[0]*fz[0];
  tS[3]  = fx[0]*fy[1]*fz[0];
  tS[4]  = fx[1]*fy[1]*fz[0];
  tS[5]  = fx[2]*fy[1]*fz[0];
  tS[6]  = fx[0]*fy[2]*fz[0];
  tS[7]  = fx[1]*fy[2]*fz[0];
  tS[8]  = fx[2]*fy[2]*fz[0];
  tS[9]  = fx[0]*fy[0]*fz[1];
  tS[10] = fx[1]*fy[0]*fz[1];
  tS[11] = fx[2]*fy[0]*fz[1];
  tS[12] = fx[0]*fy[1]*fz[1];
  tS[13] = fx[1]*fy[1]*fz[1];
  tS[14] = fx[2]*fy[1]*fz[1];
  tS[15] = fx[0]*fy[2]*fz[1];
  tS[16] = fx[1]*fy[2]*fz[1];
  tS[17] = fx[2]*fy[2]*fz[1];
  tS[18] = fx[0]*fy[0]*fz[2];
  tS[19] = fx[1]*fy[0]*fz[2];
  tS[20] = fx[2]*fy[0]*fz[2];
  tS[21] = fx[0]*fy[1]*fz[2];
  tS[22] = fx[1]*fy[1]*fz[2];
  tS[23] = fx[2]*fy[1]*fz[2];
  tS[24] = fx[0]*fy[2]*fz[2];
  tS[25] = fx[1]*fy[2]*fz[2];
  tS[26] = fx[2]*fy[2]*fz[2];
  
  td_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
  td_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
  td_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
  td_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
  td_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
  td_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
  td_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
  td_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
  td_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);
 
  td_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
  td_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
  td_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
  td_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
  td_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
  td_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
  td_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
  td_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
  td_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);
 
  td_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
  td_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
  td_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
  td_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
  td_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
  td_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
  td_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
  td_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
  td_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

  int count = 0;
  for(int i=0;i<27;i++){
   if(tS[i]>0.0){
    //S[count]   =tS[i];
    d_S[count] =td_S[i];
    ni[count]  =tni[i];
    count++;
   }
  }

  return count;
}

int 
GIMPInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                        vector<IntVector>& ni,
                                                        vector<double>& S,
                                                        vector<Vector>& d_S,
                                                        const Matrix3& size)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  double lz = size(2,2)/2.;

  IntVector tni[27];
  double tS[27];
  Vector td_S[27];
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }
  
  tni[0]  = IntVector(ix,    iy,      iz);
  tni[1]  = IntVector(ix+1,  iy,      iz);
  tni[2]  = IntVector(ix+nnx,iy,      iz);
  tni[3]  = IntVector(ix,    iy+1,    iz);
  tni[4]  = IntVector(ix+1,  iy+1,    iz);
  tni[5]  = IntVector(ix+nnx,iy+1,    iz);
  tni[6]  = IntVector(ix,    iy+nny,  iz);
  tni[7]  = IntVector(ix+1,  iy+nny,  iz);
  tni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  tni[9]  = IntVector(ix,    iy,      iz+1);
  tni[10] = IntVector(ix+1,  iy,      iz+1);
  tni[11] = IntVector(ix+nnx,iy,      iz+1);
  tni[12] = IntVector(ix,    iy+1,    iz+1);
  tni[13] = IntVector(ix+1,  iy+1,    iz+1);
  tni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  tni[15] = IntVector(ix,    iy+nny,  iz+1);
  tni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  tni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  tni[18] = IntVector(ix,    iy,      iz+nnz);
  tni[19] = IntVector(ix+1,  iy,      iz+nnz);
  tni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  tni[21] = IntVector(ix,    iy+1,    iz+nnz);
  tni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  tni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  tni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  tni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  tni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
  
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
  double fx[3]  = {NAN,NAN,NAN};
  double fy[3]  = {NAN,NAN,NAN};
  double fz[3]  = {NAN,NAN,NAN};
  double dfx[3] = {NAN,NAN,NAN};
  double dfy[3] = {NAN,NAN,NAN};
  double dfz[3] = {NAN,NAN,NAN};

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
  
  tS[0]  = fx[0]*fy[0]*fz[0];
  tS[1]  = fx[1]*fy[0]*fz[0];
  tS[2]  = fx[2]*fy[0]*fz[0];
  tS[3]  = fx[0]*fy[1]*fz[0];
  tS[4]  = fx[1]*fy[1]*fz[0];
  tS[5]  = fx[2]*fy[1]*fz[0];
  tS[6]  = fx[0]*fy[2]*fz[0];
  tS[7]  = fx[1]*fy[2]*fz[0];
  tS[8]  = fx[2]*fy[2]*fz[0];
  tS[9]  = fx[0]*fy[0]*fz[1];
  tS[10] = fx[1]*fy[0]*fz[1];
  tS[11] = fx[2]*fy[0]*fz[1];
  tS[12] = fx[0]*fy[1]*fz[1];
  tS[13] = fx[1]*fy[1]*fz[1];
  tS[14] = fx[2]*fy[1]*fz[1];
  tS[15] = fx[0]*fy[2]*fz[1];
  tS[16] = fx[1]*fy[2]*fz[1];
  tS[17] = fx[2]*fy[2]*fz[1];
  tS[18] = fx[0]*fy[0]*fz[2];
  tS[19] = fx[1]*fy[0]*fz[2];
  tS[20] = fx[2]*fy[0]*fz[2];
  tS[21] = fx[0]*fy[1]*fz[2];
  tS[22] = fx[1]*fy[1]*fz[2];
  tS[23] = fx[2]*fy[1]*fz[2];
  tS[24] = fx[0]*fy[2]*fz[2];
  tS[25] = fx[1]*fy[2]*fz[2];
  tS[26] = fx[2]*fy[2]*fz[2];
  
  td_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
  td_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
  td_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
  td_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
  td_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
  td_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
  td_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
  td_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
  td_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);
  
  td_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
  td_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
  td_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
  td_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
  td_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
  td_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
  td_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
  td_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
  td_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);
  
  td_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
  td_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
  td_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
  td_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
  td_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
  td_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
  td_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
  td_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
  td_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

  int count = 0;
  for(int i=0;i<27;i++){
   if(tS[i]>0.0){
    S[count]   =tS[i];
    d_S[count] =td_S[i];
    ni[count]  =tni[i];
    count++;
   }
  }

  return count;
}


//______________________________________________________________________
//  This interpolation function from equation 14 of 
//  Jin Ma, Hongbind Lu and Ranga Komanduri
// "Structured Mesh Refinement in Generalized Interpolation Material Point Method
//  for Simulation of Dynamic Problems" CMES, vol 12, no 3, pp. 213-227 2006
//  This function is only called when coarse level particles, in the pseudo
//  extra cells are interpolating information to the CFI nodes.

void GIMPInterpolator::findCellAndWeights_CFI(const Point& pos,
                                             vector<IntVector>& CFI_ni,
                                             vector<double>& S,
                                             constNCVariable<Stencil7>& zoi)
{
  const Level* level = d_patch->getLevel();
  IntVector refineRatio(level->getRefinementRatio());

  //__________________________________
  // Identify the nodes that are along the coarse fine interface (*)
  //
  //             |           |
  //             |           |
  //  ___________*__o__o__o__o________
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  . 0 . |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*__o__o__o__o________
  //             |           |        
  //             |           |         
  //             |           |        
  //  Coarse fine interface nodes: *
  //  ExtraCell nodes on the fine level: o  (technically these don't exist)
  //  Particle postition on the coarse level: 0

  const int ngn = 0;
  IntVector finePatch_lo = d_patch->getNodeLowIndex(ngn);
  IntVector finePatch_hi = d_patch->getNodeHighIndex(ngn) - IntVector(1,1,1);
  
  // Find node index of coarse cell and then map that node to fine level
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  IntVector ni_c = coarseLevel->getCellIndex(pos);
  IntVector ni_f = coarseLevel->mapNodeToFiner(ni_c);
  
  int ix = ni_f.x();
  int iy = ni_f.y();
  int iz = ni_f.z();

  // loop over all (o) nodes and find which lie on edge of the patch or the CFI
  for(int x = 0; x<= refineRatio.x(); x++){
    for(int y = 0; y<= refineRatio.y(); y++){
      for(int z = 0; z<= refineRatio.z(); z++){
      
        IntVector extraCell_node = IntVector(ix + x, iy + y, iz + z);
         // this is an inside test
         if(extraCell_node == Max(extraCell_node, finePatch_lo) && 
            extraCell_node == Min(extraCell_node, finePatch_hi) ) {  
            CFI_ni.push_back(extraCell_node);
          //cout << "    ni " << extraCell_node << endl;
        } 
      }
    }
  }

  //__________________________________
  // Reference Nomenclature: Stencil7 Mapping
  // Lx- :  L.w
  // Lx+ :  L.e
  // Ly- :  L.s
  // Ly+ :  L.n
  // Lz- :  L.b
  // Lz+ :  L.t
   
  for (int i = 0; i< (int) CFI_ni.size(); i++){
    Point nodepos = level->getNodePosition(CFI_ni[i]);
    double dx = pos.x() - nodepos.x();
    double dy = pos.y() - nodepos.y();
    double dz = pos.z() - nodepos.z();
    double fx = -9, fy = -9, fz = -9;
    
    Stencil7 L = zoi[CFI_ni[i]];  // fine level zoi
    
/*`==========TESTING==========*/
#if 0
  if(ni[i].x() == 100 && (ni[i].z() == 1 || ni[i].z() == 2)){
    cout << "  findCellAndWeights " << ni[i] << endl;
    cout << "    dx " << dx << " L.w " << L.w << " L.e " << L.e << endl;
    cout << "    dy " << dy << " L.n " << L.n << " L.s " << L.s << endl;
    
   if(dx <= -L.w){                       // Lx-
      cout << "     fx = 0;" << endl; 
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
     cout << "     fx = 1 + dx/L.w; " << endl;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      cout << "     fx = 1 - dx/L.e; " << endl;
    }
    else if (L.e <= dx){                  // Lx+
      cout << "     fx = 0; " << endl;
    }
    
    if(dy <= -L.s){                       // Ly-
      cout << "     fy = 0; " << endl;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      cout << "     fy = 1 + dy/L.s; " << endl;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      cout << "     fy = 1 - dy/L.n; " << endl;
    }
    else if (L.n <= dy){                 // Ly+
      cout << "     fy = 0; " << endl;
    } 
  } 
#endif
/*===========TESTING==========`*/
  
    if(dx <= -L.w){                       // Lx-
      fx = 0; 
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
      fx = 1 + dx/L.w;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      fx = 1 - dx/L.e;
    }
    else if (L.e <= dx){                  // Lx+
      fx = 0;
    }

    if(dy <= -L.s){                       // Ly-
      fy = 0;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      fy = 1 + dy/L.s;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      fy = 1 - dy/L.n;
    }
    else if (L.n <= dy){                 // Ly+
      fy = 0;
    }

    if(dz <= -L.b){                       // Lz-
      fz = 0;
    }
    else if ( -L.b <= dz && dz <= 0 ){    // Lz-
      fz = 1 + dz/L.b;
    }
    else if ( 0 <= dz && dz <= L.t ){    // Lz+
      fz = 1 - dz/L.t;
    }
    else if (L.t <= dz){                 // Lz+
      fz = 0;
    }

    double s = fx * fy * fz;
    
    S.push_back(s);
    
/*`==========TESTING==========*/
#if 0
    if(s < 0 ) {
      cout << CFI_ni[i] << "  fx " << fx << " fy " << fy <<  " fz " << fz << "    S[i] "<< s<< endl;
    }
#endif 
/*===========TESTING==========`*/
    ASSERT(s>=0);
  }
}
 
 
 //______________________________________________________________________
//  This interpolation function from equation 14 of 
//  Jin Ma, Hongbind Lu and Ranga Komanduri
// "Structured Mesh Refinement in Generalized Interpolation Material Point Method
//  for Simulation of Dynamic Problems" CMES, vol 12, no 3, pp. 213-227 2006
//  This function is only called when coarse level particles, in the pseudo
//  extra cells are interpolating information to the CFI nodes.

void GIMPInterpolator::findCellAndWeightsAndShapeDerivatives_CFI(
                                            const Point& pos,
                                            vector<IntVector>& CFI_ni,
                                            vector<double>& S,
                                            vector<Vector>& d_S,
                                            constNCVariable<Stencil7>& zoi)
{
  const Level* level = d_patch->getLevel();
  IntVector refineRatio(level->getRefinementRatio());

  //__________________________________
  // Identify the nodes that are along the coarse fine interface (*)
  //
  //             |           |
  //             |           |
  //  ___________*__o__o__o__o________
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  . 0 . |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*__o__o__o__o________
  //             |           |        
  //             |           |         
  //             |           |        
  //  Coarse fine interface nodes: *
  //  ExtraCell nodes on the fine level: o  (technically these don't exist)
  //  Particle postition on the coarse level: 0

  const int ngn = 0;
  IntVector finePatch_lo = d_patch->getNodeLowIndex(ngn);
  IntVector finePatch_hi = d_patch->getNodeHighIndex(ngn) - IntVector(1,1,1);
  
  // Find node index of coarse cell and then map that node to fine level
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  IntVector ni_c = coarseLevel->getCellIndex(pos);
  IntVector ni_f = coarseLevel->mapNodeToFiner(ni_c);
  
  int ix = ni_f.x();
  int iy = ni_f.y();
  int iz = ni_f.z();

  // loop over all (o) nodes and find which lie on edge of the patch or the CFI
  for(int x = 0; x<= refineRatio.x(); x++){
    for(int y = 0; y<= refineRatio.y(); y++){
      for(int z = 0; z<= refineRatio.z(); z++){
      
        IntVector extraCell_node = IntVector(ix + x, iy + y, iz + z);
         // this is an inside test
         if(extraCell_node == Max(extraCell_node, finePatch_lo) && extraCell_node == Min(extraCell_node, finePatch_hi) ) {  
          CFI_ni.push_back(extraCell_node);
          //cout << "    ni " << extraCell_node << endl;
        } 
      }
    }
  }
  
  //__________________________________
  // Reference Nomenclature: Stencil7 Mapping
  // Lx- :  L.w
  // Lx+ :  L.e
  // Ly- :  L.s
  // Ly+ :  L.n
  // Lz- :  L.b
  // Lz+ :  L.t
   
  for (int i = 0; i< (int) CFI_ni.size(); i++){
    Point nodepos = level->getNodePosition(CFI_ni[i]);
    double dx = pos.x() - nodepos.x();
    double dy = pos.y() - nodepos.y();
    double dz = pos.z() - nodepos.z();
    double fx = -9, fy = -9, fz = -9;

    Stencil7 L = zoi[CFI_ni[i]];  // fine level zoi

    double d_Lx = -9;
    double d_Ly = -9;
    double d_Lz = -9;
  
    if(dx <= -L.w){                       // Lx-
      fx   = 0; 
      d_Lx = 0;
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
      fx   = 1 + dx/L.w;
      d_Lx = 1/L.w;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      fx   = 1 - dx/L.e;
      d_Lx = -1.0/L.e;
    }
    else if (L.e <= dx){                  // Lx+
      fx   = 0;
      d_Lx = 0;
    }

    if(dy <= -L.s){                       // Ly-
      fy   = 0;
      d_Ly = 0;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      fy   = 1 + dy/L.s;
      d_Ly = 1/L.s;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      fy   =  1 - dy/L.n;
      d_Ly = -1.0/L.n;
    }
    else if (L.n <= dy){                 // Ly+
      fy   = 0;
      d_Ly = 0;
    }

    if(dz <= -L.b){                       // Lz-
      fz   = 0;
      d_Lz = 0;
    }
    else if ( -L.b <= dz && dz <= 0 ){    // Lz-
      fz   = 1 + dz/L.b;
      d_Lz = 1.0/L.b;
    }
    else if ( 0 <= dz && dz <= L.t ){    // Lz+
      fz   =  1 - dz/L.t;
      d_Lz = -1/L.t;
    }
    else if (L.t <= dz){                 // Lz+
      fz   = 0;
      d_Lz = 0;
    }

    double s = fx * fy * fz;
    
    double Gx = d_Lx * fy   * fz;
    double Gy = fx   * d_Ly * fz;
    double Gz = fx   * fy   * d_Lz;
    
    S.push_back(s);
    d_S.push_back( Vector( Gx, Gy, Gz) );
    
    ASSERT(s>=0);
  }
}

int GIMPInterpolator::size()
{
  return d_size;
}
