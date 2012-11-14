/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Grid/AxiGIMPInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace SCIRun;
using namespace Uintah;

    
AxiGIMPInterpolator::AxiGIMPInterpolator()
{
  d_size = 18;
  d_patch = 0;
}

AxiGIMPInterpolator::AxiGIMPInterpolator(const Patch* patch)
{
  d_size = 18;
  d_patch = patch;
}
    
AxiGIMPInterpolator::~AxiGIMPInterpolator()
{
}

AxiGIMPInterpolator* AxiGIMPInterpolator::clone(const Patch* patch)
{
  return scinew AxiGIMPInterpolator(patch);
}
    
void AxiGIMPInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni, 
                                            vector<double>& S,
                                            const Matrix3& size,
                                            const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int nnx,nny;
  IntVector OneZ(0,0,1);
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  
  ni[0]  = IntVector(ix,    iy,      0);
  ni[1]  = IntVector(ix+1,  iy,      0);
  ni[2]  = IntVector(ix+nnx,iy,      0);
  ni[3]  = IntVector(ix,    iy+1,    0);
  ni[4]  = IntVector(ix+1,  iy+1,    0);
  ni[5]  = IntVector(ix+nnx,iy+1,    0);
  ni[6]  = IntVector(ix,    iy+nny,  0);
  ni[7]  = IntVector(ix+1,  iy+nny,  0);
  ni[8]  = IntVector(ix+nnx,iy+nny,  0);

  ni[9]  = ni[0]+OneZ;
  ni[10] = ni[1]+OneZ;
  ni[11] = ni[2]+OneZ;
  ni[12] = ni[3]+OneZ;
  ni[13] = ni[4]+OneZ;
  ni[14] = ni[5]+OneZ;
  ni[15] = ni[6]+OneZ;
  ni[16] = ni[7]+OneZ;
  ni[17] = ni[8]+OneZ;

  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX},
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
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

  S[0]  = fx[0]*fy[0]*0.5;
  S[1]  = fx[1]*fy[0]*0.5;
  S[2]  = fx[2]*fy[0]*0.5;
  S[3]  = fx[0]*fy[1]*0.5;
  S[4]  = fx[1]*fy[1]*0.5;
  S[5]  = fx[2]*fy[1]*0.5;
  S[6]  = fx[0]*fy[2]*0.5;
  S[7]  = fx[1]*fy[2]*0.5;
  S[8]  = fx[2]*fy[2]*0.5;
  S[9]  = S[0];
  S[10] = S[1];
  S[11] = S[2];
  S[12] = S[3];
  S[13] = S[4];
  S[14] = S[5];
  S[15] = S[6];
  S[16] = S[7];
  S[17] = S[8];
}
 
void AxiGIMPInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int nnx,nny;
  IntVector OneZ(0,0,1);
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  
  ni[0]  = IntVector(ix,    iy,      0);
  ni[1]  = IntVector(ix+1,  iy,      0);
  ni[2]  = IntVector(ix+nnx,iy,      0);
  ni[3]  = IntVector(ix,    iy+1,    0);
  ni[4]  = IntVector(ix+1,  iy+1,    0);
  ni[5]  = IntVector(ix+nnx,iy+1,    0);
  ni[6]  = IntVector(ix,    iy+nny,  0);
  ni[7]  = IntVector(ix+1,  iy+nny,  0);
  ni[8]  = IntVector(ix+nnx,iy+nny,  0);

  ni[9]  = ni[0]+OneZ;
  ni[10] = ni[1]+OneZ;
  ni[11] = ni[2]+OneZ;
  ni[12] = ni[3]+OneZ;
  ni[13] = ni[4]+OneZ;
  ni[14] = ni[5]+OneZ;
  ni[15] = ni[6]+OneZ;
  ni[16] = ni[7]+OneZ;
  ni[17] = ni[8]+OneZ;

  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfy[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
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

  d_S[0]  = Vector(dfx[0]*fy[0]*0.5,fx[0]*dfy[0]*0.5,0.0);
  d_S[1]  = Vector(dfx[1]*fy[0]*0.5,fx[1]*dfy[0]*0.5,0.0);
  d_S[2]  = Vector(dfx[2]*fy[0]*0.5,fx[2]*dfy[0]*0.5,0.0);
  d_S[3]  = Vector(dfx[0]*fy[1]*0.5,fx[0]*dfy[1]*0.5,0.0);
  d_S[4]  = Vector(dfx[1]*fy[1]*0.5,fx[1]*dfy[1]*0.5,0.0);
  d_S[5]  = Vector(dfx[2]*fy[1]*0.5,fx[2]*dfy[1]*0.5,0.0);
  d_S[6]  = Vector(dfx[0]*fy[2]*0.5,fx[0]*dfy[2]*0.5,0.0);
  d_S[7]  = Vector(dfx[1]*fy[2]*0.5,fx[1]*dfy[2]*0.5,0.0);
  d_S[8]  = Vector(dfx[2]*fy[2]*0.5,fx[2]*dfy[2]*0.5,0.0);
  
  d_S[9]  = d_S[0];
  d_S[10] = d_S[1];
  d_S[11] = d_S[2];
  d_S[12] = d_S[3];
  d_S[13] = d_S[4];
  d_S[14] = d_S[5];
  d_S[15] = d_S[6];
  d_S[16] = d_S[7];
  d_S[17] = d_S[8];
}

void 
AxiGIMPInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int nnx,nny;
  IntVector OneZ(0,0,1);
  double lx = size(0,0)/2.;
  double ly = size(1,1)/2.;
  
  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  
  ni[0]  = IntVector(ix,    iy,      0);
  ni[1]  = IntVector(ix+1,  iy,      0);
  ni[2]  = IntVector(ix+nnx,iy,      0);
  ni[3]  = IntVector(ix,    iy+1,    0);
  ni[4]  = IntVector(ix+1,  iy+1,    0);
  ni[5]  = IntVector(ix+nnx,iy+1,    0);
  ni[6]  = IntVector(ix,    iy+nny,  0);
  ni[7]  = IntVector(ix+1,  iy+nny,  0);
  ni[8]  = IntVector(ix+nnx,iy+nny,  0);
  ni[9]  = ni[0]+OneZ;
  ni[10] = ni[1]+OneZ;
  ni[11] = ni[2]+OneZ;
  ni[12] = ni[3]+OneZ;
  ni[13] = ni[4]+OneZ;
  ni[14] = ni[5]+OneZ;
  ni[15] = ni[6]+OneZ;
  ni[16] = ni[7]+OneZ;
  ni[17] = ni[8]+OneZ;

  // (x_p - x_v)/L
  double px0 = cellpos.x() - (ix);
  double px1 = cellpos.x() - (ix+1);
  double px2 = cellpos.x() - (ix + nnx);
  double py0 = cellpos.y() - (iy);
  double py1 = cellpos.y() - (iy+1);
  double py2 = cellpos.y() - (iy + nny);
  double fx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         fy[3] = {DBL_MAX,DBL_MAX,DBL_MAX},
         dfx[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, 
         dfy[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  
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

  S[0]  = fx[0]*fy[0]*0.5;
  S[1]  = fx[1]*fy[0]*0.5;
  S[2]  = fx[2]*fy[0]*0.5;
  S[3]  = fx[0]*fy[1]*0.5;
  S[4]  = fx[1]*fy[1]*0.5;
  S[5]  = fx[2]*fy[1]*0.5;
  S[6]  = fx[0]*fy[2]*0.5;
  S[7]  = fx[1]*fy[2]*0.5;
  S[8]  = fx[2]*fy[2]*0.5;
  S[9]  = S[0];
  S[10] = S[1];
  S[11] = S[2];
  S[12] = S[3];
  S[13] = S[4];
  S[14] = S[5];
  S[15] = S[6];
  S[16] = S[7];
  S[17] = S[8];

  d_S[0]  = Vector(dfx[0]*fy[0]*0.5,fx[0]*dfy[0]*0.5,0.0);
  d_S[1]  = Vector(dfx[1]*fy[0]*0.5,fx[1]*dfy[0]*0.5,0.0);
  d_S[2]  = Vector(dfx[2]*fy[0]*0.5,fx[2]*dfy[0]*0.5,0.0);
  d_S[3]  = Vector(dfx[0]*fy[1]*0.5,fx[0]*dfy[1]*0.5,0.0);
  d_S[4]  = Vector(dfx[1]*fy[1]*0.5,fx[1]*dfy[1]*0.5,0.0);
  d_S[5]  = Vector(dfx[2]*fy[1]*0.5,fx[2]*dfy[1]*0.5,0.0);
  d_S[6]  = Vector(dfx[0]*fy[2]*0.5,fx[0]*dfy[2]*0.5,0.0);
  d_S[7]  = Vector(dfx[1]*fy[2]*0.5,fx[1]*dfy[2]*0.5,0.0);
  d_S[8]  = Vector(dfx[2]*fy[2]*0.5,fx[2]*dfy[2]*0.5,0.0);

  d_S[9]  = d_S[0];
  d_S[10] = d_S[1];
  d_S[11] = d_S[2];
  d_S[12] = d_S[3];
  d_S[13] = d_S[4];
  d_S[14] = d_S[5];
  d_S[15] = d_S[6];
  d_S[16] = d_S[7];
  d_S[17] = d_S[8];
}

int AxiGIMPInterpolator::size()
{
  return d_size;
}
