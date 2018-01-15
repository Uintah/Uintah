/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Grid/fastCpdiInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

using namespace Uintah;
using namespace std;

fastCpdiInterpolator::fastCpdiInterpolator()
{
  d_size = 64;
  d_patch = 0;
  d_lcrit = 1.e10;
}

fastCpdiInterpolator::fastCpdiInterpolator(const Patch* patch)
{
  d_size = 64;
  d_patch = patch;
  d_lcrit = 1.e10;
}
    
fastCpdiInterpolator::~fastCpdiInterpolator()
{
}

fastCpdiInterpolator::fastCpdiInterpolator(const Patch* patch,
                                           const double lcrit)
{
  d_size = 64;
  d_patch = patch;
  d_lcrit = lcrit;
}

fastCpdiInterpolator* fastCpdiInterpolator::clone(const Patch* patch)
{
  return scinew fastCpdiInterpolator(patch, d_lcrit);
}
    
int fastCpdiInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni, 
                                            vector<double>& S,
                                            const Matrix3& size,
                                            const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Matrix3 dsize=defgrad*size;
  Vector relative_node_location[8];

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

  double lcrit = d_lcrit;
  double lcritsq = lcrit*lcrit;
  Vector la = relative_node_location[6];
  Vector lb = relative_node_location[5];
  Vector lc = relative_node_location[7];
  Vector ld = relative_node_location[4];

// Check to see if particles need to be scaled to stay within a
// circumscribing sphere of radius d_lcrit.  la, lb, lc, and ld
// are the distances from the particle center to the particle corners
// For example, for a 1 PPC case, the lN are sqrt(.5*.5+.5*.5+.5*.5)=.866
// All measurements are normalized to cell width.

// This scaling was implemented to prevent the need for arbitrary number
// of ghost nodes in parallel calculations, but its use may also improve
// accuracy.

  int scale_flag = 0;
  if(la.length2()>lcritsq){
    la = la*(lcrit/la.length());
    scale_flag = 1;
  }
  if(lb.length2()>lcritsq){
    lb = lb*(lcrit/lb.length());
    scale_flag = 1;
  }
  if(lc.length2()>lcritsq){
    lc = lc*(lcrit/lc.length());
    scale_flag = 1;
  }
  if(ld.length2()>lcritsq){
    ld = ld*(lcrit/ld.length());
    scale_flag = 1;
  }

  if(scale_flag==1){  // Don't do these calcs if the particle isn't needing to be rescaled
    dsize(0,0)=.5*(la.x()+lb.x()-lc.x()-ld.x());
    dsize(1,0)=.5*(la.y()+lb.y()-lc.y()-ld.y());
    dsize(2,0)=.5*(la.z()+lb.z()-lc.z()-ld.z());

    dsize(0,1)=.5*(la.x()-lb.x()+lc.x()-ld.x());
    dsize(1,1)=.5*(la.y()-lb.y()+lc.y()-ld.y());
    dsize(2,1)=.5*(la.z()-lb.z()+lc.z()-ld.z());

    dsize(0,2)=.5*(la.x()+lb.x()+lc.x()+ld.x());
    dsize(1,2)=.5*(la.y()+lb.y()+lc.y()+ld.y());
    dsize(2,2)=.5*(la.z()+lb.z()+lc.z()+ld.z());


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
  }else{ // Particle wasn't scaled, need to compute the RLN for the first 4 nodes
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
  }

  // indices
  int i;
  int xM,yM,zM;
  int ix[8],iy[8],iz[8];
  int hash;
  int hashMax=0;

  Vector current_corner_pos;
  
  // Shape function contribution variables
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;
  double one_over_8 = 0.125;
  double phi;
  double ccx[8],ccy[8],ccz[8];

  // Variables to hold minimum and maximum indicies
  int minX = 100000000, minY = 100000000, minZ = 100000000;
  
  // now  we will loop over each of the corners and find the current location: 
  for(i=0;i<8;i++) {
    // first we need to find the position vector of the ith corner of
    //  the particle with respect to the particle center:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    ccx[i] = (current_corner_pos).x();
    ccy[i] = (current_corner_pos).y();
    ccz[i] = (current_corner_pos).z();
    ix[i] = Floor(ccx[i]);
    iy[i] = Floor(ccy[i]);
    iz[i] = Floor(ccz[i]);
    
    if(ix[i] < minX){ minX = ix[i];}
    if(iy[i] < minY){ minY = iy[i];}
    if(iz[i] < minZ){ minZ = iz[i];}
  }
  
  // Initialize Values
  IntVector niVec = IntVector(minX,minY,minZ);
  for(int i = 0; i < 64; i++) {
    S[i]         = 0.0;
    ni[i]        = niVec;  // this must be set after minimum indicies are found
                           //  or index out of bound error will occur
  }
  
  // Loop over nodes
  for(i=0;i<8;i++){
    fx = (ccx[i]-ix[i]);
    fy = (ccy[i]-iy[i]);
    fz = (ccz[i]-iz[i]);
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;

    // grid offset variables so we only have to iterate over 
    // two closest nodes each time we look at a corner 
    xM = (int)(ccx[i] - minX);
    yM = (int)(ccy[i] - minY);
    zM = (int)(ccz[i] - minZ);

    // Uses a array index system like:
    //       24---25--26
    //      /    /   / |
    //     15--16--17 23
    //    /   /   / | /|
    //   6---7---8  14 20
    //   |   |   | / |/
    //   3---4---5  11
    //   |   |   | /
    //   0---1---2
    // 
    for(int jx = 0; jx < 2; jx++) {
      double phiX = fx1;
      if(jx == 1)
        phiX = fx;
      
      int curX = ix[i]+jx;
      int xMjx = xM+jx;
      
      for(int jy = 0; jy < 2; jy++) {
        double phiY = fy1;
        if(jy == 1)
          phiY = fy;
        
        int curY = iy[i]+jy;
        int yMjy = 3*(yM+jy);

        for(int jz = 0; jz < 2; jz++) {
          double phiZ = fz1;
          if(jz == 1)
            phiZ = fz;        

          // Create hash to map to unique value between [0,26]
          hash = xMjx + yMjy + 9*((zM)+jz);
          hashMax=max(hash, hashMax);
//          if( hash < 0 || hash > 26 ) {
//             cerr << "hash = " << hash << endl;
//             // send an exit to the program
//             // exit(1);
//          }
          phi  = phiX * phiY * phiZ;
          
          ni[hash]        = IntVector(curX,curY,iz[i]+jz);
          S[hash]        += one_over_8  * phi;
        } // z for
      } // y for
    } // x for
  } // node for
  return hashMax+1;
}
 
int fastCpdiInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Vector zero = Vector(0.0,0.0,0.0);
  Matrix3 dsize=defgrad*size;
  Vector relative_node_location[8];

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

  double lcrit = d_lcrit;
  double lcritsq = lcrit*lcrit;
  Vector la = relative_node_location[6];
  Vector lb = relative_node_location[5];
  Vector lc = relative_node_location[7];
  Vector ld = relative_node_location[4];

  int scale_flag = 0;
  if(la.length2()>lcritsq){
    la = la*(lcrit/la.length());
    scale_flag = 1;
  }
  if(lb.length2()>lcritsq){
    lb = lb*(lcrit/lb.length());
    scale_flag = 1;
  }
  if(lc.length2()>lcritsq){
    lc = lc*(lcrit/lc.length());
    scale_flag = 1;
  }
  if(ld.length2()>lcritsq){
    ld = ld*(lcrit/ld.length());
    scale_flag = 1;
  }

  if(scale_flag==1){  // Don't do these calcs if the particle isn't needing to be rescaled
    dsize(0,0)=.5*(la.x()+lb.x()-lc.x()-ld.x());
    dsize(1,0)=.5*(la.y()+lb.y()-lc.y()-ld.y());
    dsize(2,0)=.5*(la.z()+lb.z()-lc.z()-ld.z());

    dsize(0,1)=.5*(la.x()-lb.x()+lc.x()-ld.x());
    dsize(1,1)=.5*(la.y()-lb.y()+lc.y()-ld.y());
    dsize(2,1)=.5*(la.z()-lb.z()+lc.z()-ld.z());

    dsize(0,2)=.5*(la.x()+lb.x()+lc.x()+ld.x());
    dsize(1,2)=.5*(la.y()+lb.y()+lc.y()+ld.y());
    dsize(2,2)=.5*(la.z()+lb.z()+lc.z()+ld.z());

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
  }else{ // Particle wasn't scaled, need to compute the RLN for the first 4 nodes
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
  }
  
  // Indices
  int i;
  int xM,yM,zM;
  int ix[8],iy[8],iz[8];
  int hash;
  int hashMax=0;
  
  Vector current_corner_pos;

  // Shape function contribution variables
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;
  double ccx[8],ccy[8],ccz[8];

  Vector r1=Vector(dsize(0,0),dsize(1,0),dsize(2,0));
  Vector r2=Vector(dsize(0,1),dsize(1,1),dsize(2,1));
  Vector r3=Vector(dsize(0,2),dsize(1,2),dsize(2,2));

  double volume = dsize.Determinant();

  double one_over_4V = 1.0/(4.0*volume);
  vector<Vector> alpha(8,zero);
  double phi;
  
  // construct the vectors necessary for the gradient calculation:
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


  // Variables to hold minimum and maximum indicies
  int minX = 100000000, minY = 100000000, minZ = 100000000;

 // now  we will loop over each of these "nodes" or corners and use the deformation gradient to find the current location: 
  for(i=0;i<8;i++){
    //    first we need to find the position vector of the ith corner of the particle with respect to the particle center:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    ccx[i] = (current_corner_pos).x();
    ccy[i] = (current_corner_pos).y();
    ccz[i] = (current_corner_pos).z();
    ix[i] = Floor(ccx[i]);
    iy[i] = Floor(ccy[i]);
    iz[i] = Floor(ccz[i]);

    if(ix[i] < minX){minX = ix[i];}
    if(iy[i] < minY){minY = iy[i];}
    if(iz[i] < minZ){minZ = iz[i];}
  }
    
  // Initialize Values
  IntVector niVec = IntVector(minX,minY,minZ);
  for(int i = 0; i < 64; i++) {
    d_S[i]       = zero;   // this must be set after minimum indicies are found
    ni[i]        = niVec;  //  or index out of bound error will occur
  }
  
  // Loop over nodes
  for(i=0;i<8;i++){
    fx = (ccx[i]-ix[i]);
    fy = (ccy[i]-iy[i]);
    fz = (ccz[i]-iz[i]);
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;
    
    // grid offset variables so we only have to iterate over 
    // two closes nodes each time we look at a corner 
    xM = (int)(ccx[i] - minX);
    yM = (int)(ccy[i] - minY);
    zM = (int)(ccz[i] - minZ);
   
    // Uses a array index system like:
    //       24---25--26
    //      /    /   / |
    //     15--16--17 23
    //    /   /   / | /|
    //   6---7---8  14 20
    //   |   |   | / |/
    //   3---4---5  11
    //   |   |   | /
    //   0---1---2
    //
 
    for(int jx = 0; jx < 2; jx++) {
      double phiX = fx1;
      if(jx == 1)
        phiX = fx;
      
      int curX = ix[i]+jx;
      int xMjx = xM+jx;
      
      for(int jy = 0; jy < 2; jy++) {
        double phiY = fy1;
        if(jy == 1)
          phiY = fy;
        
        int curY = iy[i]+jy;
        int yMjy = 3*(yM+jy);
        
        for(int jz = 0; jz < 2; jz++) {
          double phiZ = fz1;
          if(jz == 1)
            phiZ = fz;        
          
          // Create hash to map to unique value between [0,26]
          hash = xMjx + yMjy + 9*((zM)+jz);
          hashMax=max(hash, hashMax);
//          if( hash < 0 || hash > 26 ) {
//             cerr << "hash = " << hash << endl;
//             // send an exit to the program
//             // exit(1);
//          }
          phi = phiX * phiY * phiZ;
          
          ni[hash]        = IntVector(curX,curY,iz[i]+jz);
          d_S[hash][0]    += alpha[i][0]*phi;
          d_S[hash][1]    += alpha[i][1]*phi;
          d_S[hash][2]    += alpha[i][2]*phi;
        } // z for
      } // y for
    } // x for
  } // node for
  return hashMax+1;
}

int
fastCpdiInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                         vector<IntVector>& ni,
                                                         vector<double>& S,
                                                         vector<Vector>& d_S,
                                                         const Matrix3& size,
                                                         const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(Point(pos));

  Vector zero = Vector(0.0,0.0,0.0);
  Matrix3 dsize=defgrad*size;
  Vector relative_node_location[8];

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
  double lcrit = d_lcrit;
  double lcritsq = lcrit*lcrit;
  Vector la = relative_node_location[6];
  Vector lb = relative_node_location[5];
  Vector lc = relative_node_location[7];
  Vector ld = relative_node_location[4];

  int scale_flag = 0;
  if(la.length2()>lcritsq){
    la = la*(lcrit/la.length());
    scale_flag=1;
  }
  if(lb.length2()>lcritsq){
    lb = lb*(lcrit/lb.length());
    scale_flag=1;
  }
  if(lc.length2()>lcritsq){
    lc = lc*(lcrit/lc.length());
    scale_flag=1;
  }
  if(ld.length2()>lcritsq){
    ld = ld*(lcrit/ld.length());
    scale_flag=1;
  }

  if(scale_flag==1){  // Don't do these calcs if the particle isn't needing to be rescaled
    dsize(0,0)=.5*(la.x()+lb.x()-lc.x()-ld.x());
    dsize(1,0)=.5*(la.y()+lb.y()-lc.y()-ld.y());
    dsize(2,0)=.5*(la.z()+lb.z()-lc.z()-ld.z());

    dsize(0,1)=.5*(la.x()-lb.x()+lc.x()-ld.x());
    dsize(1,1)=.5*(la.y()-lb.y()+lc.y()-ld.y());
    dsize(2,1)=.5*(la.z()-lb.z()+lc.z()-ld.z());

    dsize(0,2)=.5*(la.x()+lb.x()+lc.x()+ld.x());
    dsize(1,2)=.5*(la.y()+lb.y()+lc.y()+ld.y());
    dsize(2,2)=.5*(la.z()+lb.z()+lc.z()+ld.z());

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
  }else{ // Particle wasn't scaled, need to compute the RLN for the first 4 nodes
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
  }

  // Indices
  int i;
  int xM,yM,zM;
  int ix[8],iy[8],iz[8];
  int hash;
  int hashMax=0;
  
  Vector current_corner_pos;
  
  // Shape function contribution variables
  double fx;
  double fy;
  double fz;
  double fx1;
  double fy1;
  double fz1;
  double phi;
  double ccx[8],ccy[8],ccz[8];
  
  Vector r1=Vector(dsize(0,0),dsize(1,0),dsize(2,0));
  Vector r2=Vector(dsize(0,1),dsize(1,1),dsize(2,1));
  Vector r3=Vector(dsize(0,2),dsize(1,2),dsize(2,2));
  double volume = dsize.Determinant();

  //deformed volume:
  double one_over_4V = 1.0/(4.0*volume);
  double one_over_8 = .125;

  vector<Vector> alpha(8,zero);
  // now we construct the vectors necessary for the gradient calculation:
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

  alpha[4][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[4][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[4][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[5][0]   =  one_over_4V*(r2[1]*r3[2]-r2[2]*r3[1]+r1[1]*r3[2]-r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[5][1]   =  one_over_4V*(-r2[0]*r3[2]+r2[2]*r3[0]-r1[0]*r3[2]+r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[5][2]   =  one_over_4V*(r2[0]*r3[1]-r2[1]*r3[0]+r1[0]*r3[1]-r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[6][0]   =  one_over_4V* (r2[1]*r3[2]-r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[6][1]   =  one_over_4V* (-r2[0]*r3[2]+r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[6][2]   =  one_over_4V* (r2[0]*r3[1]-r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  alpha[7][0]   =  one_over_4V*(-r2[1]*r3[2]+r2[2]*r3[1]-r1[1]*r3[2]+r1[2]*r3[1]+r1[1]*r2[2]-r1[2]*r2[1]);
  alpha[7][1]   =  one_over_4V*(r2[0]*r3[2]-r2[2]*r3[0]+r1[0]*r3[2]-r1[2]*r3[0]-r1[0]*r2[2]+r1[2]*r2[0]);
  alpha[7][2]   =  one_over_4V*(-r2[0]*r3[1]+r2[1]*r3[0]-r1[0]*r3[1]+r1[1]*r3[0]+r1[0]*r2[1]-r1[1]*r2[0]);

  // Variables to hold minimum and maximum indicies
  int minX = 100000000, minY = 100000000, minZ = 100000000;

  for(i=0;i<8;i++){
    //    first we need to find the position vector of the ith corner of the particle with respect to the particle center:
    current_corner_pos = Vector(cellpos) + relative_node_location[i];
    ccx[i] = (current_corner_pos).x();
    ccy[i] = (current_corner_pos).y();
    ccz[i] = (current_corner_pos).z();
    ix[i] = Floor(ccx[i]);
    iy[i] = Floor(ccy[i]);
    iz[i] = Floor(ccz[i]);
   
    // Find minimum indices of the corners in each direction 
    if(ix[i] < minX){minX = ix[i];}
    if(iy[i] < minY){minY = iy[i];}
    if(iz[i] < minZ){minZ = iz[i];}
  }
  
  // Initialize Values
  IntVector niVec = IntVector(minX,minY,minZ);
  for(int i = 0; i < 64; i++) {
    S[i]         = 0.0;    // this must be set after minimum indicies are found
    d_S[i]       = zero;   // or index out of bound error will occur
    ni[i]        = niVec;
  }
  
  // Loop over nodes
  for(i=0;i<8;i++){
    fx = (ccx[i]-ix[i]);
    fy = (ccy[i]-iy[i]);
    fz = (ccz[i]-iz[i]);
    fx1 = 1-fx;
    fy1 = 1-fy;
    fz1 = 1-fz;

    // grid offset variables so we only have to iterate over 
    // two closes nodes each time we look at a corner 
    xM = (int)(ccx[i] - minX);
    yM = (int)(ccy[i] - minY);
    zM = (int)(ccz[i] - minZ);
    
    // Uses a array index system like:
    //       24---25--26
    //      /    /   / |
    //     15--16--17 23
    //    /   /   / | /|
    //   6---7---8  14 20
    //   |   |   | / |/
    //   3---4---5  11
    //   |   |   | /
    //   0---1---2
    // 
    for(int jx = 0; jx < 2; jx++) {
      double phiX = fx1;
      if(jx == 1)
        phiX = fx;
      
      int curX = ix[i]+jx;
      int xMjx = xM+jx;
      
      for(int jy = 0; jy < 2; jy++) {
        double phiY = fy1;
        if(jy == 1)
          phiY = fy;
        
        int curY = iy[i]+jy;
        int yMjy = 3*(yM+jy);
        
        for(int jz = 0; jz < 2; jz++) {
          double phiZ = fz1;
          if(jz == 1)
            phiZ = fz;        
          
          // Create hash to map to unique value between [0,26]
          hash = xMjx + yMjy + 9*((zM)+jz);
          hashMax=max(hash,hashMax);
//          if( hash < 0 || hash > 26 ) {
//             // send an exit to the program
//              exit(1);
//          }

          phi  = phiX * phiY * phiZ;          

          ni[hash]        = IntVector(curX,curY,iz[i]+jz);
          S[hash]         += one_over_8 *phi;
          d_S[hash][0]    += alpha[i][0]*phi;
          d_S[hash][1]    += alpha[i][1]*phi;
          d_S[hash][2]    += alpha[i][2]*phi;
        } // z for
      } // y for
    } // x for
  } // node for
  return hashMax+1;
}

int fastCpdiInterpolator::size()
{
  return d_size;
}
