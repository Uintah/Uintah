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

#include <Core/Grid/BSplineInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace SCIRun;
using namespace Uintah;

    
BSplineInterpolator::BSplineInterpolator()
{
  d_size = 64;
  d_patch = 0;
}

BSplineInterpolator::BSplineInterpolator(const Patch* patch)
{
  d_size = 64;
  d_patch = patch;
}
    
BSplineInterpolator::~BSplineInterpolator()
{
}

BSplineInterpolator* BSplineInterpolator::clone(const Patch* patch)
{
  return scinew BSplineInterpolator(patch);
}
 
void BSplineInterpolator::findNodeComponents(const int& ix, int* xn, int& count,
                                             const int& low, const int& hi)
{
  xn[0] = ix;
  xn[1] = ix+1;

  if(xn[0] > low){       // lowest node is not on the lower boundary
    xn[count] = ix-1;
    count++; 
  }
  if(xn[1] < hi){     // highest node is not on the upper boundary
    xn[count] = ix+2;
    count++;
  }
}

void BSplineInterpolator::getBSplineWeights(double* Sd, const int* xn,
                                            const int& count,
                                            const int& low, const int& hi,
                                            const double& cellpos)
{

  for(int n=0;n<count;n++){
     if(xn[n]==low){
       Sd[n]=evalType3BSpline(cellpos-xn[n]);
     }
     else if(xn[n]==hi-1){
       Sd[n]=evalType3BSpline(xn[n]-cellpos);
     }
     else if(xn[n]==low+1){
       Sd[n]=evalType2BSpline(cellpos-xn[n]);
     }
     else if(xn[n]==hi-2){
       Sd[n]=evalType2BSpline(xn[n]-cellpos);
     }
     else{
       Sd[n]=evalType1BSpline(cellpos-xn[n]);
     }
  }
}

double BSplineInterpolator::evalType1BSpline(const double& dx)
{
  if(dx < -2.) // shouldn't happen
    return -10.0;
  else if(dx < -1.)                                  // region (1)
    return ((1./6. * dx + 1.) * dx + 2.) * dx + 4./3.;
  else if(dx < 0.)                                   // region (2)
    return (-.5 * dx - 1.) * dx * dx + 2./3.;
  else if(dx < 1.)                                   // region (3)
    return ( .5 * dx - 1.) * dx * dx + 2./3.;
  else if(dx < 2.)                                   // region (4)
    return ((-1./6. * dx + 1.) * dx - 2.) * dx + 4./3.;
                                                                                
  // if we got here, we are > 2. Shouldn't happen.
  return 10.0;
}

double BSplineInterpolator::evalType2BSpline(const double& dx)
{
  if(dx < -1.) // shouldn't happen
    return -20.0;
  else if(dx < 0.)                                  // region (1)
    return ((-11./12. * dx - 1.25) * dx + .25) * dx + 7./12.;
  else if(dx < 1.)                                  // region (2)
    return ((7./12. * dx - 1.25) * dx + .25) * dx + 7./12.;
  else if(dx < 2.)                                  // region (3)
    return ((-1./6. * dx + 1.) * dx - 2.) * dx + 4./3.;

  // if we got here, we are > 2. Shouldn't happen.
  return 20.0;
}

double BSplineInterpolator::evalType3BSpline(const double& dx)
{
  if(dx < 0.) // shouldn't happen
    return -30.0;
  else if(dx < 1.0)                                 // region (1)
    return (.75 * dx - 1.5) * dx * dx + 1.;
  else if(dx < 2.0)                                 // region (2)
    return ((-.25 * dx + 1.5) * dx - 3.) * dx + 2.;
                                                                                
  // if we got here, we are > 2. Shouldn't happen
  return 30.0;

}

void BSplineInterpolator::getBSplineGrads(double* dSd, const int* xn,
                                          const int& count,
                                          const int& low, const int& hi,
                                          const double& cellpos)
{
  for(int n=0;n<count;n++){
     if(xn[n]==low){
       dSd[n]=evalType3BSplineGrad(cellpos-xn[n]);
     }
     else if(xn[n]==hi-1){
       dSd[n]=-evalType3BSplineGrad(xn[n]-cellpos);
     }
     else if(xn[n]==low+1){
       dSd[n]=evalType2BSplineGrad(cellpos-xn[n]);
     }
     else if(xn[n]==hi-2){
       dSd[n]=-evalType2BSplineGrad(xn[n]-cellpos);
     }
     else{
       dSd[n]=evalType1BSplineGrad(cellpos-xn[n]);
     }
  }
}

double BSplineInterpolator::evalType1BSplineGrad(const double& dx)
// internal nodes
{
  if(dx < -2.) // shouldn't happen
    return 11.0;
  else if(dx < -1.)                                  // region (1)
    return (.5 * dx + 2.) * dx + 2.;
  else if(dx < 0.)                                   // region (2)
    return (-1.5 * dx - 2.) * dx;
  else if(dx < 1.)                                   // region (3)
    return (1.5 * dx - 2.) * dx;
  else if(dx < 2.)                                   // region (4)
    return (-.5 * dx + 2.) * dx - 2.;
                                                                                
  // if we got here, we are > 2. Shouldn't happen.
  return -11.0;
}

double BSplineInterpolator::evalType2BSplineGrad(const double& dx)
// nodes 1 away from boundary
{
  if(dx < -1.) // shouldn't happen
    return 22.0;
  else if(dx < 0.)                                  // region (1)
    return (-11./4. * dx - 2.5) * dx + .25;
  else if(dx < 1.)                                  // region (2)
    return (7./4. * dx - 2.5) * dx + .25;
  else if(dx < 2.)                                  // region (3)
    return (-.5 * dx + 2.) * dx - 2.;
                                                                                
  // if we got here, we are > 2. Shouldn't happen.
  return -22.0;
}

double BSplineInterpolator::evalType3BSplineGrad(const double& dx) 
// boundary nodes
{
  if(dx < 0.) // shouldn't happen
    return 33.0;
  else if(dx < 1.0)                                 // region (1)
    return (2.25 * dx - 3.) * dx;
  else if(dx < 2.0)                                 // region (2)
    return (-.75 * dx + 3.) * dx - 3.;
                                                                                
  // if we got here, we are > 2. Shouldn't happen
  return -33.0;
}


void BSplineInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni, 
                                            vector<double>& S,
                                            const Matrix3& size,
                                            const Matrix3& defgrad)
{
  IntVector low,hi;
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  d_patch->getLevel()->findInteriorNodeIndexRange(low,hi);

  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());

  int xn[4], yn[4], zn[4];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[4],Sy[4],Sz[4];

  findNodeComponents(ix,xn,countx,low.x(),hi.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z());

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());

  int n=0;
  for(int i=0;i<countx;i++){
    for(int j=0;j<county;j++){
      for(int k=0;k<countz;k++){
        ni[n]=IntVector(xn[i],yn[j],zn[k]);
        S[n] =Sx[i]*Sy[j]*Sz[k];
        n++;
      }
    }
  }
  for(int i=n;i<64;i++){
    ni[i]=ni[0];
    S[i]=0.;
  }
}
 
void BSplineInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  IntVector low,hi;
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  Vector dx = d_patch->dCell();
  d_patch->getLevel()->findInteriorNodeIndexRange(low,hi);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());

  int xn[4], yn[4], zn[4];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[4],Sy[4],Sz[4];
  double dSx[4],dSy[4],dSz[4];

  findNodeComponents(ix,xn,countx,low.x(),hi.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z());

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());

  getBSplineGrads(dSx,  xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineGrads(dSy,  yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineGrads(dSz,  zn, countz, low.z(), hi.z(), cellpos.z());

  int n=0;
  for(int i=0;i<countx;i++){
    for(int j=0;j<county;j++){
      for(int k=0;k<countz;k++){
        ni[n]=IntVector(xn[i],yn[j],zn[k]);
        double xcomp=dSx[i]*Sy[j]*Sz[k];
        double ycomp=Sx[i]*dSy[j]*Sz[k];
        double zcomp=Sx[i]*Sy[j]*dSz[k];
        d_S[n]=Vector(xcomp,ycomp,zcomp);
        n++;
      }
    }
  }
  for(int i=n;i<64;i++){
    ni[i]=ni[0];
    d_S[i]=Vector(0.,0.,0.);
  }
}

void 
BSplineInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  IntVector low,hi;
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  Vector dx = d_patch->dCell();
  d_patch->getLevel()->findInteriorNodeIndexRange(low,hi);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());

  int xn[4], yn[4], zn[4];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[4],Sy[4],Sz[4];
  double dSx[4],dSy[4],dSz[4];

  findNodeComponents(ix,xn,countx,low.x(),hi.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z());

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());

  getBSplineGrads(dSx,  xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineGrads(dSy,  yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineGrads(dSz,  zn, countz, low.z(), hi.z(), cellpos.z());

  int n=0;
  for(int i=0;i<countx;i++){
    for(int j=0;j<county;j++){
      for(int k=0;k<countz;k++){
        ni[n]=IntVector(xn[i],yn[j],zn[k]);
        double xcomp=dSx[i]*Sy[j]*Sz[k];
        double ycomp=Sx[i]*dSy[j]*Sz[k];
        double zcomp=Sx[i]*Sy[j]*dSz[k];
        d_S[n]=Vector(xcomp,ycomp,zcomp);
        S[n]  =Sx[i]*Sy[j]*Sz[k];
        n++;
      }
    }
  }
  for(int i=n;i<64;i++){
    ni[i]=ni[0];
    d_S[i]=Vector(0.,0.,0.);
    S[i]=0.0;
  }
}

int BSplineInterpolator::size()
{
  return d_size;
}
