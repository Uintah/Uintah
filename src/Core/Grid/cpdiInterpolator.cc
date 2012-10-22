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
#include <Core/Grid/cpdiInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

using namespace SCIRun;
using namespace Uintah;
using namespace std;
    
cpdiInterpolator::cpdiInterpolator()
{
  d_size = 64;
  d_patch = 0;
}

cpdiInterpolator::cpdiInterpolator(const Patch* patch)
{
  d_size = 64;
  d_patch = patch;
}
    
cpdiInterpolator::~cpdiInterpolator()
{
}

cpdiInterpolator* cpdiInterpolator::clone(const Patch* patch)
{
  return scinew cpdiInterpolator(patch);
}
    
void cpdiInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni, 
                                            vector<double>& S,
                                            const Matrix3& size,
                                            const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Matrix3 dsize=defgrad*size;
  vector<Vector> relative_node_location(8,Vector(0.0,0.0,0.0));
  relative_node_location[0]=Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[1]=Vector( dsize(0,0)-dsize(0,1)-dsize(0,2),
                                    dsize(1,0)-dsize(1,1)-dsize(1,2),
                                    dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[2]=Vector( dsize(0,0)+dsize(0,1)-dsize(0,2),
                                    dsize(1,0)+dsize(1,1)-dsize(1,2),
                                    dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[3]=Vector(-dsize(0,0)+dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[4]=Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[5]=Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                                    dsize(1,0)-dsize(1,1)+dsize(1,2),
                                    dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[6]=Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                                    dsize(1,0)+dsize(1,1)+dsize(1,2),
                                    dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[7]=Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;

  Vector current_corner_pos;
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;
  int ix,iy,iz;

  double one_over_8 = 1.0/(8.0);
  vector<double> phi(8);

 // now  we will loop over each of these "nodes" or corners and use the deformation gradient to find the current location: 
  for(int i=0;i<8;i++){
    int i8  = i*8;
    int i81 = i*8+1;
    int i82 = i*8+2;
    int i83 = i*8+3;
    int i84 = i*8+4;
    int i85 = i*8+5;
    int i86 = i*8+6;
    int i87 = i*8+7;
    //    first we need to find the position vector of the ith corner of the particle with respect to the particle center:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    ix = Floor(current_corner_pos.x());
    iy = Floor(current_corner_pos.y());
    iz = Floor(current_corner_pos.z());

    ni[i8]  = IntVector(ix  , iy  , iz  ); // x1    , y1    , z1
    ni[i81] = IntVector(ix+1, iy  , iz  ); // x1+r1x, y1    , z1
    ni[i82] = IntVector(ix+1, iy+1, iz  ); // x1+r1x, y1+r2y, z1
    ni[i83] = IntVector(ix  , iy+1, iz  ); // x1    , y1+r2y, z1
    ni[i84] = IntVector(ix  , iy  , iz+1); // x1    , y1    , z1+r3z
    ni[i85] = IntVector(ix+1, iy  , iz+1); // x1+r1x, y1    , z1+r3z
    ni[i86] = IntVector(ix+1, iy+1, iz+1); // x1+r1x, y1+r2y, z1+r3z
    ni[i87] = IntVector(ix  , iy+1, iz+1); // x1    , y1+r2y, z1+r3z

    fx = current_corner_pos.x()-ix;
    fy = current_corner_pos.y()-iy;
    fz = current_corner_pos.z()-iz;
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;

    phi[0] = fx1*fy1*fz1; // x1    , y1    , z1
    phi[1] = fx *fy1*fz1; // x1+r1x, y1    , z1
    phi[2] = fx *fy *fz1; // x1+r1x, y1+r2y, z1
    phi[3] = fx1*fy *fz1; // x1    , y1+r2y, z1
    phi[4] = fx1*fy1*fz;  // x1    , y1    , z1+r3z
    phi[5] = fx *fy1*fz;  // x1+r1x, y1    , z1+r3z
    phi[6] = fx *fy *fz;  // x1+r1x, y1+r2y, z1+r3z
    phi[7] = fx1*fy *fz;  // x1    , y1+r2y, z1+r3z

    S[i8]   = one_over_8*phi[0];
    S[i81] = one_over_8*phi[1];
    S[i82] = one_over_8*phi[2];
    S[i83] = one_over_8*phi[3];
    S[i84] = one_over_8*phi[4];
    S[i85] = one_over_8*phi[5];
    S[i86] = one_over_8*phi[6];
    S[i87] = one_over_8*phi[7];
  }
}
 
void cpdiInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                   vector<IntVector>& ni,
                                                   vector<Vector>& d_S,
                                                   const Matrix3& size,
                                                   const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Matrix3 dsize=defgrad*size;
  vector<Vector> relative_node_location(8,Vector(0.0,0.0,0.0));
  relative_node_location[0]=Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[1]=Vector( dsize(0,0)-dsize(0,1)-dsize(0,2),
                                    dsize(1,0)-dsize(1,1)-dsize(1,2),
                                    dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[2]=Vector( dsize(0,0)+dsize(0,1)-dsize(0,2),
                                    dsize(1,0)+dsize(1,1)-dsize(1,2),
                                    dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[3]=Vector(-dsize(0,0)+dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[4]=Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[5]=Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                                    dsize(1,0)-dsize(1,1)+dsize(1,2),
                                    dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[6]=Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                                    dsize(1,0)+dsize(1,1)+dsize(1,2),
                                    dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[7]=Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;

  int i;
  Vector current_corner_pos;
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;
  int ix,iy,iz;

  Vector r1=Vector(dsize(0,0),dsize(1,0),dsize(2,0));
  Vector r2=Vector(dsize(0,1),dsize(1,1),dsize(2,1));
  Vector r3=Vector(dsize(0,2),dsize(1,2),dsize(2,2));

  double volume = dsize.Determinant();

  double one_over_4V = 1.0/(4.0*volume);
  vector<Vector> alpha(8,Vector(0.0,0.0,0.0));
  vector<double> phi(8);
  // conw we construct the vectors necessary for the gradient calculation:
  alpha[0][0]   =  one_over_4V* (-r2[1]*r3[2]+r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[0][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[0][2]   =  one_over_4V* (-r2[0]*r3[1]+r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[1][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[1][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[1][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[2][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[2][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[2][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[3][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[3][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[3][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[4][0]   = one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[4][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[4][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[5][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[5][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[5][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[6][0]   = one_over_4V* (r2[1]*r3[2]-r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[6][1]   = one_over_4V* (-r2[0]*r3[2]+r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[6][2]   = one_over_4V* (r2[0]*r3[1]-r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[7][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[7][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[7][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

 // now  we will loop over each of these "nodes" or corners and use the deformation gradient to find the current location: 
  for(i=0;i<8;i++){
    int i8  = i*8;
    int i81 = i*8+1;
    int i82 = i*8+2;
    int i83 = i*8+3;
    int i84 = i*8+4;
    int i85 = i*8+5;
    int i86 = i*8+6;
    int i87 = i*8+7;
    //    first we need to find the position vector of the ith corner of the particle with respect to the particle center:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    ix = Floor(current_corner_pos.x());
    iy = Floor(current_corner_pos.y());
    iz = Floor(current_corner_pos.z());

    ni[i8]  = IntVector(ix  , iy  , iz  ); // x1    , y1    , z1
    ni[i81] = IntVector(ix+1, iy  , iz  ); // x1+r1x, y1    , z1
    ni[i82] = IntVector(ix+1, iy+1, iz  ); // x1+r1x, y1+r2y, z1
    ni[i83] = IntVector(ix  , iy+1, iz  ); // x1    , y1+r2y, z1
    ni[i84] = IntVector(ix  , iy  , iz+1); // x1    , y1    , z1+r3z
    ni[i85] = IntVector(ix+1, iy  , iz+1); // x1+r1x, y1    , z1+r3z
    ni[i86] = IntVector(ix+1, iy+1, iz+1); // x1+r1x, y1+r2y, z1+r3z
    ni[i87] = IntVector(ix  , iy+1, iz+1); // x1    , y1+r2y, z1+r3z

    fx = current_corner_pos.x()-ix;
    fy = current_corner_pos.y()-iy;
    fz = current_corner_pos.z()-iz;
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;

    phi[0] = fx1*fy1*fz1; // x1    , y1    , z1
    phi[1] = fx *fy1*fz1; // x1+r1x, y1    , z1
    phi[2] = fx *fy *fz1; // x1+r1x, y1+r2y, z1
    phi[3] = fx1*fy *fz1; // x1    , y1+r2y, z1
    phi[4] = fx1*fy1*fz;  // x1    , y1    , z1+r3z
    phi[5] = fx *fy1*fz;  // x1+r1x, y1    , z1+r3z
    phi[6] = fx *fy *fz;  // x1+r1x, y1+r2y, z1+r3z
    phi[7] = fx1*fy *fz;  // x1    , y1+r2y, z1+r3z

    d_S[i8][0]   = alpha[i][0]*phi[0];
    d_S[i8][1]   = alpha[i][1]*phi[0];
    d_S[i8][2]   = alpha[i][2]*phi[0];

    d_S[i81][0] = alpha[i][0]*phi[1];
    d_S[i81][1] = alpha[i][1]*phi[1];
    d_S[i81][2] = alpha[i][2]*phi[1];

    d_S[i82][0] = alpha[i][0]*phi[2];
    d_S[i82][1] = alpha[i][1]*phi[2];
    d_S[i82][2] = alpha[i][2]*phi[2];

    d_S[i83][0] = alpha[i][0]*phi[3];
    d_S[i83][1] = alpha[i][1]*phi[3];
    d_S[i83][2] = alpha[i][2]*phi[3];

    d_S[i84][0] = alpha[i][0]*phi[4];
    d_S[i84][1] = alpha[i][1]*phi[4];
    d_S[i84][2] = alpha[i][2]*phi[4];

    d_S[i85][0] = alpha[i][0]*phi[5];
    d_S[i85][1] = alpha[i][1]*phi[5];
    d_S[i85][2] = alpha[i][2]*phi[5];

    d_S[i86][0] = alpha[i][0]*phi[6];
    d_S[i86][1] = alpha[i][1]*phi[6];
    d_S[i86][2] = alpha[i][2]*phi[6];

    d_S[i87][0] = alpha[i][0]*phi[7];
    d_S[i87][1] = alpha[i][1]*phi[7];
    d_S[i87][2] = alpha[i][2]*phi[7];
  }
}

void cpdiInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Matrix3 dsize=defgrad*size;

  vector<Vector> relative_node_location(8,Vector(0.0,0.0,0.0));
  relative_node_location[0]=Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[1]=Vector( dsize(0,0)-dsize(0,1)-dsize(0,2),
                                    dsize(1,0)-dsize(1,1)-dsize(1,2),
                                    dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[2]=Vector( dsize(0,0)+dsize(0,1)-dsize(0,2),
                                    dsize(1,0)+dsize(1,1)-dsize(1,2),
                                    dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[3]=Vector(-dsize(0,0)+dsize(0,1)-dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)-dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  relative_node_location[4]=Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)-dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[5]=Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                                    dsize(1,0)-dsize(1,1)+dsize(1,2),
                                    dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[6]=Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                                    dsize(1,0)+dsize(1,1)+dsize(1,2),
                                    dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  relative_node_location[7]=Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                                   -dsize(1,0)+dsize(1,1)+dsize(1,2),
                                   -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  Vector current_corner_pos;
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;


  int ix,iy,iz;
  Vector r1=Vector(dsize(0,0),dsize(1,0),dsize(2,0));
  Vector r2=Vector(dsize(0,1),dsize(1,1),dsize(2,1));
  Vector r3=Vector(dsize(0,2),dsize(1,2),dsize(2,2));
  double volume = dsize.Determinant();
  double one_over_4V = 1.0/(4.0*volume);
  double one_over_8 = 1.0/(8.0);
  vector<Vector> alpha(8,Vector(0.0,0.0,0.0));
  vector<double> phi(8);
  // conw we construct the vectors necessary for the gradient calculation:
  alpha[0][0]   =  one_over_4V* (-r2[1]*r3[2]+r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[0][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[0][2]   =  one_over_4V* (-r2[0]*r3[1]+r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[1][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[1][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[1][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[2][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[2][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[2][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[3][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]-r1[1]*r2[2]+r1[2]*r2[1]);
  alpha[3][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]+r1[0]*r2[2]-r1[2]*r2[0]);
  alpha[3][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]-r1[0]*r2[1]+r1[1]*r2[0]);

  alpha[4][0]   = one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[4][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[4][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[5][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[5][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[5][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[6][0]   = one_over_4V* (r2[1]*r3[2]-r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[6][1]   = one_over_4V* (-r2[0]*r3[2]+r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[6][2]   = one_over_4V* (r2[0]*r3[1]-r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[7][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[7][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[7][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  // now  we will loop over each of these "nodes" and use the deformation gradient to find the current location: 
  for(int i=0;i<8;i++){
    //    first we need to find the position vector of the ith corner of the particle:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    int i8  = i*8;
    int i81 = i*8+1;
    int i82 = i*8+2;
    int i83 = i*8+3;
    int i84 = i*8+4;
    int i85 = i*8+5;
    int i86 = i*8+6;
    int i87 = i*8+7;
    ix = Floor(current_corner_pos.x());
    iy = Floor(current_corner_pos.y());
    iz = Floor(current_corner_pos.z());

    ni[i8] = IntVector(ix    , iy  , iz  ); // x1    , y1    , z1
    ni[i81] = IntVector(ix+1, iy  , iz  ); // x1+r1x, y1    , z1
    ni[i82] = IntVector(ix+1, iy+1, iz  ); // x1+r1x, y1+r2y, z1
    ni[i83] = IntVector(ix  , iy+1, iz  ); // x1    , y1+r2y, z1
    ni[i84] = IntVector(ix  , iy  , iz+1); // x1    , y1    , z1+r3z
    ni[i85] = IntVector(ix+1, iy  , iz+1); // x1+r1x, y1    , z1+r3z
    ni[i86] = IntVector(ix+1, iy+1, iz+1); // x1+r1x, y1+r2y, z1+r3z
    ni[i87] = IntVector(ix  , iy+1, iz+1); // x1    , y1+r2y, z1+r3z

    fx = current_corner_pos.x()-ix;
    fy = current_corner_pos.y()-iy;
    fz = current_corner_pos.z()-iz;
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;

    phi[0] = fx1*fy1*fz1; // x1    , y1    , z1
    phi[1] = fx *fy1*fz1; // x1+r1x, y1    , z1
    phi[2] = fx *fy *fz1; // x1+r1x, y1+r2y, z1
    phi[3] = fx1*fy *fz1; // x1    , y1+r2y, z1
    phi[4] = fx1*fy1*fz;  // x1    , y1    , z1+r3z
    phi[5] = fx *fy1*fz;  // x1+r1x, y1    , z1+r3z
    phi[6] = fx *fy *fz;  // x1+r1x, y1+r2y, z1+r3z
    phi[7] = fx1*fy *fz;  // x1    , y1+r2y, z1+r3z

    S[i8]   = one_over_8*phi[0];
    S[i81] = one_over_8*phi[1];
    S[i82] = one_over_8*phi[2];
    S[i83] = one_over_8*phi[3];
    S[i84] = one_over_8*phi[4];
    S[i85] = one_over_8*phi[5];
    S[i86] = one_over_8*phi[6];
    S[i87] = one_over_8*phi[7];

    d_S[i8][0]   = alpha[i][0]*phi[0];
    d_S[i8][1]   = alpha[i][1]*phi[0];
    d_S[i8][2]   = alpha[i][2]*phi[0];

    d_S[i81][0] = alpha[i][0]*phi[1];
    d_S[i81][1] = alpha[i][1]*phi[1];
    d_S[i81][2] = alpha[i][2]*phi[1];

    d_S[i82][0] = alpha[i][0]*phi[2];
    d_S[i82][1] = alpha[i][1]*phi[2];
    d_S[i82][2] = alpha[i][2]*phi[2];

    d_S[i83][0] = alpha[i][0]*phi[3];
    d_S[i83][1] = alpha[i][1]*phi[3];
    d_S[i83][2] = alpha[i][2]*phi[3];

    d_S[i84][0] = alpha[i][0]*phi[4];
    d_S[i84][1] = alpha[i][1]*phi[4];
    d_S[i84][2] = alpha[i][2]*phi[4];

    d_S[i85][0] = alpha[i][0]*phi[5];
    d_S[i85][1] = alpha[i][1]*phi[5];
    d_S[i85][2] = alpha[i][2]*phi[5];

    d_S[i86][0] = alpha[i][0]*phi[6];
    d_S[i86][1] = alpha[i][1]*phi[6];
    d_S[i86][2] = alpha[i][2]*phi[6];

    d_S[i87][0] = alpha[i][0]*phi[7];
    d_S[i87][1] = alpha[i][1]*phi[7];
    d_S[i87][2] = alpha[i][2]*phi[7];
  }
}

int cpdiInterpolator::size()
{
  return d_size;
}
