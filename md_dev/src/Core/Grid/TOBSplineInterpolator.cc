/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Grid/TOBSplineInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace SCIRun;
using namespace Uintah;

    
TOBSplineInterpolator::TOBSplineInterpolator()
{
  d_size = 27;
  d_patch = 0;
}

TOBSplineInterpolator::TOBSplineInterpolator(const Patch* patch)
{
  d_size = 27;
  d_patch = patch;
}
    
TOBSplineInterpolator::~TOBSplineInterpolator()
{
}

TOBSplineInterpolator* TOBSplineInterpolator::clone(const Patch* patch)
{
  return scinew TOBSplineInterpolator(patch);
}

void TOBSplineInterpolator::findNodeComponents(const int& ix, int* xn, int& count,
                                             const int& low, const int& hi,
                                             const double& cellpos)
{
  xn[0] = ix;
  xn[1] = ix+1;

  if(cellpos - xn[0] < 0.5){ // lowest node is not on the lower boundary
    xn[count] = ix-1;
    count++; 
  }
  else{ // highest node is not on the upper boundary
    xn[count] = ix+2;
    count++;
  }
}

void TOBSplineInterpolator::getBSplineWeights(double* Sd, const int* xn,
                                            const int& count,
                                            const int& low, const int& hi,
                                            const double& cellpos)
{
  for(int n=0;n<count;n++){
    Sd[n]=evalType1BSpline(cellpos-xn[n]);
  }
}

double TOBSplineInterpolator::evalType1BSpline(const double& dx)   // internal nodes
{
  // make fractions constants?

  if(dx < -1.5) // shouldn't happen
    return -10.0;
  else if(dx < -.5)
    return (dx + 1.5) * (dx + 1.5) * .5;
  else if(dx < .5)
    return (-dx * dx + .75);
  else if(dx < 1.5)
    return .5 * (1.5 - dx) * (1.5 - dx);
  
  // if we got here, we are > 1.5. Shouldn't happen.
  return 10.0;
}

double TOBSplineInterpolator::evalType2BSpline(const double& dx)    // nodes 1 away from boundary
{
  // make fractions constants?

  if(dx < -1.) // shouldn't happen
    return -10.0;
  else if(dx < -.5)                                 // region (1)
    return 4./3. * (dx + 1.) * (dx + 1.);
  else if(dx < .5)                                  // region (2)
    return (-7./6. * dx + 1./6.) * dx + 17./24.;
  else if(dx < 1.5)                                 // region (3)
    return .5 * (1.5 - dx) * (1.5 - dx);

  // if we got here, we are > 1.5 Shouldn't happen.
  return 10.0;
}

double TOBSplineInterpolator::evalType3BSpline(const double& dx)    // boundary nodes
{
  // make fractions constants?

  if(dx < 0.) // shouldn't happen
    return -10.0;
  else if(dx < 0.5)                                 // region (1)
    return -4./3. * dx * dx + 1.;
  else if(dx < 1.5)                                 // region (2)
    return 2./3. * (3./2. - dx) * (3./2. - dx);

  // if we got here, we are > 2. Shouldn't happen
  return 10.0;
}

void TOBSplineInterpolator::getBSplineGrads(double* dSd, const int* xn,
                                          const int& count,
                                          const int& low, const int& hi,
                                          const double& cellpos)
{
  for(int n=0;n<count;n++){
    dSd[n]=evalType1BSplineGrad(cellpos-xn[n]);
  }
}

double TOBSplineInterpolator::evalType1BSplineGrad(const double& dx)   // internal nodes
{
  // make fractions constants?

  if(dx < -1.5) // shouldn't happen
    return 11.0;
  else if(dx < -.5)
    return dx + 1.5;
  else if(dx < .5)
    return - 2. * dx;
  else if(dx < 1.5)
    return dx - 1.5;
  
  // if we got here, we are > 1.5. Shouldn't happen.
  return -11.0;
}

double TOBSplineInterpolator::evalType2BSplineGrad(const double& dx)    // nodes 1 away from boundary
{
  // make fractions constants?

  if(dx < -1.) // shouldn't happen
    return 22.0;
  else if(dx < -.5)                                 // region (1)
    return 8./3. * (dx + 1.);
  else if(dx < .5)                                  // region (2)
    return -7./3. * dx + 1./6.;
  else if(dx < 1.5)                                 // region (3)
    return dx - 1.5;

  // if we got here, we are > 1.5 Shouldn't happen.
  return -22.0;
}

double TOBSplineInterpolator::evalType3BSplineGrad(const double& dx)    // boundary nodes
{
  // make fractions constants?

  if(dx < 0.) // shouldn't happen
    return 33.0;
  else if(dx < 0.5)                                 // region (1)
    return -8./3. * dx;
  else if(dx < 1.5)                                 // region (2)
    return 4./3. * dx - 2.;

  // if we got here, we are > 2. Shouldn't happen
  return -33.0;
}

void TOBSplineInterpolator::findCellAndWeights(const Point& pos,
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

  int xn[3], yn[3], zn[3];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[3],Sy[3],Sz[3];

  findNodeComponents(ix,xn,countx,low.x(),hi.x(),cellpos.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y(),cellpos.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z(),cellpos.z());
//  zn[0]=iz;
//  zn[1]=iz+1;

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());
//  Sz[0]=0.5;
//  Sz[1]=0.5;

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
  for(int i=n;i<27;i++){
    ni[i]=ni[0];
    S[i]=0.;
  }
}
 
void TOBSplineInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  IntVector low,hi;
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  d_patch->getLevel()->findInteriorNodeIndexRange(low,hi);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());

  int xn[3], yn[3], zn[3];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[3],Sy[3],Sz[3];
  double dSx[3],dSy[3],dSz[3];

  findNodeComponents(ix,xn,countx,low.x(),hi.x(),cellpos.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y(),cellpos.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z(),cellpos.z());
//  zn[0]=iz;
//  zn[1]=iz+1;

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());
//  Sz[0]=0.5;
//  Sz[1]=0.5;

  getBSplineGrads(dSx,  xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineGrads(dSy,  yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineGrads(dSz,  zn, countz, low.z(), hi.z(), cellpos.z());
//  dSz[0]=0.0;
//  dSz[1]=0.0;

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
  for(int i=n;i<27;i++){
    ni[i]=ni[0];
    d_S[i]=Vector(0.,0.,0.);
  }
}

void 
TOBSplineInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  IntVector low,hi;
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  d_patch->getLevel()->findInteriorNodeIndexRange(low,hi);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());

  int xn[3], yn[3], zn[3];
  int countx = 2;
  int county = 2;
  int countz = 2;
  double Sx[3],Sy[3],Sz[3];
  double dSx[3],dSy[3],dSz[3];

  findNodeComponents(ix,xn,countx,low.x(),hi.x(),cellpos.x());
  findNodeComponents(iy,yn,county,low.y(),hi.y(),cellpos.y());
  findNodeComponents(iz,zn,countz,low.z(),hi.z(),cellpos.z());
//  zn[0]=iz;
//  zn[1]=iz+1;

  getBSplineWeights(Sx, xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineWeights(Sy, yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineWeights(Sz, zn, countz, low.z(), hi.z(), cellpos.z());
//  Sz[0]=0.5;
//  Sz[1]=0.5;

  getBSplineGrads(dSx,  xn, countx, low.x(), hi.x(), cellpos.x());
  getBSplineGrads(dSy,  yn, county, low.y(), hi.y(), cellpos.y());
  getBSplineGrads(dSz,  zn, countz, low.z(), hi.z(), cellpos.z());
//  dSz[0]=0.0;
//  dSz[1]=0.0;

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
  for(int i=n;i<27;i++){
    ni[i]=ni[0];
    d_S[i]=Vector(0.,0.,0.);
    S[i]=0.0;
  }
}

int TOBSplineInterpolator::size()
{
  return d_size;
}
