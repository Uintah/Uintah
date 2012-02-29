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



#ifndef Uintah_AMRInterpolate_h
#define Uintah_AMRInterpolate_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>

#include <sstream>

#include <Core/Grid/uintahshare.h>


#define is_rightFace(face) ( (face == "xminus" || face == "yminus" ||  face == "zminus" ) ?1:0  )
namespace Uintah {

/*___________________________________________________________________
 Function~ piecewiseConstantInterpolation--
 ____________________________________________________________________*/
template<class T>
  void piecewiseConstantInterpolation(constCCVariable<T>& q_CL,// course level
                           const Level* fineLevel,
                           const IntVector& fl,
                           const IntVector& fh,
                           CCVariable<T>& q_FineLevel)
{ 
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    q_FineLevel[f_cell] = q_CL[c_cell];                       
  }
}

// find the normalized distance between the coarse and the fine cell cell-center  
UINTAHSHARE void normalizedDistance_CC(const int refineRatio,vector<double>& norm_dist);  


/*___________________________________________________________________
 Function~ linearInterpolation--
 
 X-Y PLANE 1

           |       x   |
           |     |---| |
___________|___________|_______________
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |  |  |  |  |
__|__*__|__|__|__o__|x1|__|  ---o         Q_x1 = (1-x)Q  + (x)Q(i+1)
  |  |  |  |  |  |  |  |  |   |
__|__|__|__|__|__|__|__|__|   |  y
  |  |  |  |  |  |  |  |  |   |
__|__|__|__|__|__|__|__|__|___|________
  |  |  |  |  |  |  |+ |  |  ---
__|__|__|__|__|__|__|__|__|
           |           |
           |           |
     *     |     o   x2        o          Q_x2 = (1-x)Q(j-1) + (x)(Q(i+1,j-1))
           |           |
           |           |
___________|___________|_______________

* coarse cell centers                   
o cells used for interpolation          
+ fine cell cell center
(x, y) = Normalized distance between coarse cell centers


 Q_fc_plane_1 = (1-y)Q_x1 + (y)*Q_x2
              = (1-y)(1-x)Q + (1-y)(x)Q(i+1) + (1-x)(y)Q(j-1) + (x)(y)Q(i+1)(j-1)
              = (w0)Q + (w1)Q(i+1) + (w2)Q(j-1) + (w3)Q(i+1)(j-1)               

Q_fc_plane_2 is identical to Q_fc_plane_1 with except for a z offset.

Q_FC =(1-z)Q_fc_plane_1 + (z) * Q_fc_plane2
_____________________________________________________________________*/
template<class T>
  void linearInterpolation(constCCVariable<T>& q_CL,// course level
                           const Level* coarseLevel,
                           const Level* fineLevel,
                           const IntVector& refineRatio,
                           const IntVector& fl,
                           const IntVector& fh,
                           CCVariable<T>& q_FineLevel)
{
  // compute the normalized distance between the fine and coarse cell centers
  vector<double> norm_dist_x(refineRatio.x());
  vector<double> norm_dist_y(refineRatio.y());
  vector<double> norm_dist_z(refineRatio.z());
  normalizedDistance_CC(refineRatio.x(),norm_dist_x);
  normalizedDistance_CC(refineRatio.y(),norm_dist_y);
  normalizedDistance_CC(refineRatio.z(),norm_dist_z);
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    //__________________________________ 
    // compute the index of the fine cell, relative to the
    // coarse cell center.
    IntVector relativeIndx = f_cell - (c_cell * refineRatio);
    
    Vector dist;  // normalized distance
    dist.x(norm_dist_x[relativeIndx.x()]);
    dist.y(norm_dist_y[relativeIndx.y()]);
    dist.z(norm_dist_z[relativeIndx.z()]);
    
    //__________________________________
    // Offset for coarse level surrounding cells:
    // determine the direction to the surrounding interpolation cells
    int i = SCIRun::Sign(dist.x());   // returns +/- 1.0
    int j = SCIRun::Sign(dist.y());
    int k = SCIRun::Sign(dist.z());

    i *= SCIRun::RoundUp(fabs(dist.x()));  // if dist.x,y,z() = 0 then set (i,j,k) = 0
    j *= SCIRun::RoundUp(fabs(dist.y()));  // Only need surrounding coarse cell data if dist != 0
    k *= SCIRun::RoundUp(fabs(dist.z()));  // This is especially true for 1D and 2D problems

    //__________________________________
    //  Find the weights
    double x = fabs(dist.x());  // The normalized distance is always +
    double y = fabs(dist.y());
    double z = fabs(dist.z());
       
    double w0 = (1.0 - x) * (1.0 - y);
    double w1 = x * (1.0 - y);
    double w2 = y * (1.0 - x);
    double w3 = x * y;
      
    T q_XY_Plane_1   // X-Y plane closest to the fine level cell 
        = w0 * q_CL[c_cell] 
        + w1 * q_CL[c_cell + IntVector( i, 0, 0)] 
        + w2 * q_CL[c_cell + IntVector( 0, j, 0)]
        + w3 * q_CL[c_cell + IntVector( i, j, 0)];
                   
    T q_XY_Plane_2   // X-Y plane furthest from the fine level cell
        = w0 * q_CL[c_cell + IntVector( 0, 0, k)] 
        + w1 * q_CL[c_cell + IntVector( i, 0, k)]  
        + w2 * q_CL[c_cell + IntVector( 0, j, k)]  
        + w3 * q_CL[c_cell + IntVector( i, j, k)]; 

    // interpolate the two X-Y planes in the k direction
    q_FineLevel[f_cell] = (1.0 - z) * q_XY_Plane_1 + z * q_XY_Plane_2; 
                         
    //__________________________________
    //  Debugging                        
#if 0
      IntVector half  = (fh - fl )/IntVector(2,2,2) + fl;
      if ((f_cell.y() == half.y() && f_cell.z() == half.z())){
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(5);
       cout << " f_cell " << f_cell << " c_cell "<< c_cell << " offset ["<<i<<","<<j<<","<<k<<"]  " << endl;
       cout << " relative indx " << relativeIndx  << endl;
       cout << "dist "<< dist << " dir " << dir <<  endl;
       cout << " q_CL[c_cell]                       "                           << q_CL[c_cell]                       << " w0 " << w0 << endl;
       cout << " q_CL[c_cell + IntVector( " << i << ", 0, 0)] "                 << q_CL[c_cell + IntVector( i, 0, 0)] << " w1 " << w1 << endl;
       cout << " q_CL[c_cell + IntVector( 0, " << j << ", 0)] "                 << q_CL[c_cell + IntVector( 0, j, 0)] << " w2 " << w2 << endl;
       cout << " q_CL[c_cell + IntVector( "<< i << ", " << j << ", 0)] "        << q_CL[c_cell + IntVector( i, j, 0)] << " w3 " << w3 << endl;
       cout << " q_CL[c_cell + IntVector( 0, 0, "<<k<<")] "                     << q_CL[c_cell + IntVector( 0, 0, k)] << " w0 " << w0 << endl;
       cout << " q_CL[c_cell + IntVector( "<< i << ", 0, "<<k<<")] "            << q_CL[c_cell + IntVector( i, 0, k)] << " w1 " << w1 << endl;
       cout << " q_CL[c_cell + IntVector( 0, "<< j <<", "<<k<<")] "             << q_CL[c_cell + IntVector( 0, j, k)] << " w2 " << w2 << endl;
       cout << " q_CL[c_cell + IntVector( "<< i << ", " << j << ", "<< k <<")] " << q_CL[c_cell + IntVector( i, j, k)] << " w3 " << w3 <<endl;
       cout << " q_XY_Plane_1 " << q_XY_Plane_1 << " q_XY_Plane_2 " << q_XY_Plane_2 << " q_FineLevel[f_cell] "<< q_FineLevel[f_cell] << endl;
    }
#endif                      
  }
}


/*___________________________________________________________________
 Function~  QuadraticInterpolation--
 X-Y PLANE
                   x
           |     |----||
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |  |  |  |  |
__|__o__|__|__|__o__|_0|__|      o       Q_0 = (w0_x)Q(i-1)  + (w1_x)Q(i) + (w2_x)Q(i+1)
  |  |  |  |  |  |  |  |  |               (j+1)
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|____________
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |(i,j)|  |  |
__|__o__|__|__|__o__|_1|__|  ---o         Q_1 = (w0_x)Q(i-1)  + (w1_x)Q(i) + (w2_x)Q(i+1)
  |  |  |  |  |  |  |  |  |   |             (j)
__|__|__|__|__|__|__|__|__|   |  y
  |  |  |  |  |  |  | +|  |  ---
__|__|__|__|__|__|__|__|__|____________
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
           |           |
           |           |
     o     |     o    2|       o          Q_2 = (w0_x)Q(i-1)  + (w1_x)Q(i) + (w2_x)Q(i+1)
           |           |                    (j-1)
           |           |
___________|___________|_______________

o cells used for interpolation
+ fine cell center that you want to interpolate to

(z = k -1)  Q_FC_plane_0 = (w0_y) * Q_0 + (w1_y) Q_1  + (w2_y) Q_2
(z = k)     Q_FC_plane_1 = (w0_y) * Q_0 + (w1_y) Q_1  + (w2_y) Q_2
(z = k +1)  Q_FC_plane_2 = (w0_y) * Q_0 + (w1_y) Q_1  + (w2_y) Q_2

 Q_FC = (w0_z)Q_fc_plane_0 + (w1_z)Q_fc_plane_0 + (w2_z)Q_fc_plane_0


_____________________________________________________________________*/
template<class T>
  void quadraticInterpolation(constCCVariable<T>& q_CL,// course level
                             const Level* coarseLevel,
                             const Level* fineLevel,
                             const IntVector& refineRatio,
                             const IntVector& fl,
                             const IntVector& fh,
                             CCVariable<T>& q_FineLevel)
{
  IntVector gridLo, gridHi;
  coarseLevel->findCellIndexRange(gridLo,gridHi);
  
  gridHi -= IntVector(1,1,1);
  
  // compute the normalized distance between the fine and coarse cell centers
  vector<double> norm_dist_x(refineRatio.x());
  vector<double> norm_dist_y(refineRatio.y());
  vector<double> norm_dist_z(refineRatio.z());
  normalizedDistance_CC(refineRatio.x(),norm_dist_x);
  normalizedDistance_CC(refineRatio.y(),norm_dist_y);
  normalizedDistance_CC(refineRatio.z(),norm_dist_z);  
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    IntVector baseCell = c_cell;
    
    //__________________________________
    // At the edge of the computational Domain
    // shift base/origin coarse cell inward one cell
    IntVector shift(0,0,0);
    
    for (int d =0; d<3; d++){
      if( (c_cell[d] - gridLo[d]) == 0 ) {  // (x,y,z)minus
        shift[d] = 1;
      } 
      if( (gridHi[d]-c_cell[d] ) == 0) {    // (x,y,z)plus
        shift[d] = -1;
      }
    }    
    baseCell = c_cell + shift;

    //__________________________________ 
    // compute the index of the fine cell, relative to the
    // coarse cell center.  Find the distance the normalized distance between
    // the coarse and fine cell-centers
    IntVector relativeIndx = f_cell - (baseCell * refineRatio);
    
    Vector dist;
    dist.x(norm_dist_x[relativeIndx.x()]);
    dist.y(norm_dist_y[relativeIndx.y()]);
    dist.z(norm_dist_z[relativeIndx.z()]);

    //__________________________________
    //  Find the weights 
    double x = dist.x();
    double y = dist.y();
    double z = dist.z();
    
    double w0_x =  0.5 * x  * (x - 1.0);
    double w1_x = -(x + 1.0)* (x - 1.0);
    double w2_x =  0.5 * x  * (x + 1.0);
    
    double w0_y =  0.5 * y  * (y - 1.0);
    double w1_y = -(y + 1.0)* (y - 1.0);
    double w2_y =  0.5 * y  * (y + 1.0);
    
    double w0_z =  0.5 * z  * (z - 1.0);
    double w1_z = -(z + 1.0)* (z - 1.0);
    double w2_z =  0.5 * z  * (z + 1.0);
    
    FastMatrix w(3, 3);
    //  Q_CL(-1,-1,k)      Q_CL(0,-1,k)          Q_CL(1,-1,k)
    w(0,0) = w0_x * w0_y; w(1,0) = w1_x * w0_y; w(2,0) = w2_x * w0_y;
    w(0,1) = w0_x * w1_y; w(1,1) = w1_x * w1_y; w(2,1) = w2_x * w1_y;
    w(0,2) = w0_x * w2_y; w(1,2) = w1_x * w2_y; w(2,2) = w2_x * w2_y;  
    //  Q_CL(-1, 1,k)      Q_CL(0, 1,k)          Q_CL(1, 1,k)      
        
    vector<T> q_XY_Plane(3);

    int k = -2; 
    // loop over the three X-Y planes
    for(int p = 0; p < 3; p++){
      k += 1;

      q_XY_Plane[p]   // X-Y plane
        = w(0,0) * q_CL[baseCell + IntVector( -1, -1, k)]   
        + w(1,0) * q_CL[baseCell + IntVector(  0, -1, k)]           
        + w(2,0) * q_CL[baseCell + IntVector(  1, -1, k)]           
        + w(0,1) * q_CL[baseCell + IntVector( -1,  0, k)]            
        + w(1,1) * q_CL[baseCell + IntVector(  0,  0, k)]    
        + w(2,1) * q_CL[baseCell + IntVector(  1,  0, k)]     
        + w(0,2) * q_CL[baseCell + IntVector( -1,  1, k)]   
        + w(1,2) * q_CL[baseCell + IntVector(  0,  1, k)]     
        + w(2,2) * q_CL[baseCell + IntVector(  1,  1, k)]; 
    }
    
    // interpolate the 3 X-Y planes 
    q_FineLevel[f_cell] = w0_z * q_XY_Plane[0] 
                        + w1_z * q_XY_Plane[1] 
                        + w2_z * q_XY_Plane[2];

    //__________________________________
    //  debugging
#if 0
    if(true){
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(5);
      #if 0
      for (k = -1; k< 2; k++){
        std::cout << " baseCell " << baseCell << " f_cell " << f_cell << " x " << x << " y " << y << " z " << z << "\n";
        std::cout << " q_CL[baseCell + IntVector( -1, -1, k)] " << q_CL[baseCell + IntVector( -1, -1, k)]<< " w(0,0) " << w(0,0) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  0, -1, k)] " << q_CL[baseCell + IntVector(  0, -1, k)]<< " w(1,0) " << w(1,0) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  1, -1, k)] " << q_CL[baseCell + IntVector(  1, -1, k)]<< " w(2,0) " << w(2,0) << "\n";
        std::cout << " q_CL[baseCell + IntVector( -1,  0, k)] " << q_CL[baseCell + IntVector(  1, -1, k)]<< " w(0,1) " << w(0,1) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  0,  0, k)] " << q_CL[baseCell + IntVector(  0,  0, k)]<< " w(1,1) " << w(1,1) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  1,  0, k)] " << q_CL[baseCell + IntVector(  1,  0, k)]<< " w(2,1) " << w(2,1) << "\n";
        std::cout << " q_CL[baseCell + IntVector( -1,  1, k)] " << q_CL[baseCell + IntVector( -1,  1, k)]<< " w(0,2) " << w(0,2) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  0,  1, k)] " << q_CL[baseCell + IntVector(  0,  1, k)]<< " w(1,2) " << w(1,2) << "\n";
        std::cout << " q_CL[baseCell + IntVector(  1,  1, k)] " << q_CL[baseCell + IntVector(  1,  1, k)]<< " w(2,2) " << w(2,2) << "\n";
        std::cout << " q_XY_Plane " << q_XY_Plane[k+1] << "\n";
      }
      #endif
      std::cout  << " plane 1 " << q_XY_Plane[0] << " plane2 " << q_XY_Plane[1] << " plane3 "<< q_XY_Plane[2] << "\n";
      std::cout  << " w0_x " << w0_x << " w1_x " << w1_x << " w2_x "<< w2_x << "\n";
      std::cout  << " w0_y " << w0_y << " w1_y " << w1_y << " w2_y "<< w2_y << "\n";
      std::cout  << " w0_z " << w0_z << " w1_z " << w1_z << " w2_z "<< w2_z << "\n";
      std::cout << " Q " << q_FineLevel[f_cell] << "\n";
   }
#endif   
   
  } 
}


/*___________________________________________________________________
 Function~  QuadraticInterpolation_CFI--
 
 X-Y PLANE
                   x
           |     |     |
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |  |  |  |  |
__|__o__|__|__|__o__|__|__|     (o)       
  |  |  |  |  |  |  |  |  |               
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|____________
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
  |  |  |  |  |(i,j)|  |  |
__|__o__|__|__|__o__|__|__|  ---(o)         Q_1 = (w0_y)Q(j-1)  + (w1_y)Q(j) + (w2_y)Q(j+1)
  |  |  |  |  |  |  |  |  |   |             (z)
__|__|__|__|__|__|__|__|__|   |  
  |  |  |  |  |  | *| *| x|  ---(#)   Q_(#) 
__|__|__|__|__|__|__|__|__|____________
  |  |  |  |  |  |  |  |  |
__|__|__|__|__|__|__|__|__|
           |           |
           |           |
     o     |     o     |        (o)         
           |           |                    
           |           |
___________|___________|_______________

o    Coarse level cell centers
*    Fine level cell centers
(o)  Cells used for coarse level interpolation -> (#)
(#)  Interpolated coarse level value
+ CFI fine cell center that you want to interpolate to


Step 1)  Using the coarse level cell center data in the (o) plane interpolate to (#)
         Do 3 line interpolations at z+1, z-1, and z to (#,i,k+1),(#,i,k),(#,i,k-1)
Step 2)  Using the fine level cell centered data at * and (#) interpolate to x. 

Coarse Level Interpolation in Y direction
(Plane: z-1)      Q_0 = (w0_y)Q(i,j-1,k-1)  + (w1_y)Q(i,j,k-1) + (w2_y)Q(i,j+1,k-1)
(Plane: z  )      Q_1 = (w0_y)Q(i,j-1,k)    + (w1_y)Q(i,j,k)   + (w2_y)Q(i,j+1,k)
(Plane: z+1)      Q_2 = (w0_y)Q(i,j-1,k+1)  + (w1_y)Q(i,j,k+1) + (w2_y)Q(i,j+1,k+1)

With Q_0, Q_1 and Q_2 interpolate to (#)
Q_Coarse_interpolate = (w0_z) * Q_0 + (w1_z) Q_1  + (w2_z) Q_2


Reference:  Dan Martin's Dissertation  "An Adaptive Cell-Centered 
Projection Method for Incompressible Euler Equations"
_____________________________________________________________________*/
template<class T>
  void quadraticInterpolation_CFI(constCCVariable<T>& q_CL,// course level
                             const Patch* finePatch,
                             Patch::FaceType patchFace,
                             const Level* coarseLevel,
                             const Level* fineLevel,
                             const IntVector& refineRatio,
                             const IntVector& fl,
                             const IntVector& fh,
                             CCVariable<T>& q_FineLevel)
{
  IntVector gridLo, gridHi;
  coarseLevel->findCellIndexRange(gridLo,gridHi);
  gridHi -= IntVector(1,1,1);  // we need the inclusive gridHi
  
  IntVector dir = finePatch->getFaceAxes(patchFace);        // face axes
  int p_dir = dir[0];                                    // normal direction 
  int y = dir[1];             // Orthogonal to the patch face
  int z = dir[2];
  string name = finePatch->getFaceName(patchFace);

  //__________________________________
  // compute the normalized distance between the fine and coarse cell centers
  vector<double> norm_dist_y(refineRatio[y]);
  vector<double> norm_dist_z(refineRatio[z]);
  
  normalizedDistance_CC(refineRatio[y],norm_dist_y);
  normalizedDistance_CC(refineRatio[z],norm_dist_z); 
#if 0  
  cout<< " face " << name << " refineRatio "<< refineRatio
      << " FineLevel iterator" << fl << " " << fh
      << " coarseLevel " << fineLevel->mapCellToCoarser(fl) << " " << fineLevel->mapCellToCoarser(fh)<< endl;    
#endif      
  
  //__________________________________
  // define the offsets for the CC data on the coarse Level
  vector<IntVector> offset(9);
  int counter = 0;
    
  for(int j = -1; j <=1; j++){
    for(int k = -1; k <=1; k++){
      IntVector tmp(0,0,0);
      tmp[p_dir] = 0;
      tmp[y]     = j;     // -1, 0, 1
      tmp[z]     = k;     // -1, 0, 1
      offset[counter] = tmp; 
      counter += 1;
    }
  }
  
  //__________________________________
  //  Find the interpolation weights for step 2
  // keep this out of the loop
  double d_CL = 0.5 * (refineRatio[p_dir] + 1.0);
  double w0_x = (1.0 - d_CL)/(d_CL + 1);
  double w1_x = -2.0 * (1.0 - d_CL)/d_CL;
  double w2_x = 2.0/((d_CL + 1) * d_CL);
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    IntVector baseCell = c_cell;
        
    //__________________________________
    // At the edge of the computational Domain
    // shift base/origin coarse cell inward one cell
    IntVector shift(0,0,0);
    for (int d =0; d<3; d++){
      if( (c_cell[d] - gridLo[d]) == 0 ) {  // (x,y,z)minus
        shift[d] = 1;
      } 
      if( (gridHi[d]-c_cell[d] ) == 0) {    // (x,y,z)plus
        shift[d] = -1;
      }
    }      
        
    baseCell = c_cell + shift;

    //__________________________________ 
    // compute the index of the fine cell, relative to the
    // coarse cell center.  Find the distance the normalized distance between
    // the coarse and fine cell-centers
    IntVector relativeIndx = f_cell - (baseCell * refineRatio);
    
    Vector dist(-9);
    dist.y(norm_dist_y[relativeIndx[y]]);
    dist.z(norm_dist_z[relativeIndx[z]]);

    //__________________________________
    //  Find the weights for the coarse Level interpolation
    double dy = dist.y();
    double dz = dist.z();
    
    double w0_y =  0.5 * dy  * (dy - 1.0);
    double w1_y = -(dy + 1.0)* (dy - 1.0);
    double w2_y =  0.5 * dy  * (dy + 1.0);
    
    double w0_z =  0.5 * dz  * (dz - 1.0);
    double w1_z = -(dz + 1.0)* (dz - 1.0);
    double w2_z =  0.5 * dz  * (dz + 1.0);
    
    FastMatrix w(3, 3);
    //  Q_CL(i,-1,-1)      Q_CL(i,0,-1)          Q_CL(i,1,-1)
    w(0,0) = w0_y * w0_z; w(1,0) = w1_y * w0_z; w(2,0) = w2_y * w0_z;
    w(0,1) = w0_y * w1_z; w(1,1) = w1_y * w1_z; w(2,1) = w2_y * w1_z;
    w(0,2) = w0_y * w2_z; w(1,2) = w1_y * w2_z; w(2,2) = w2_y * w2_z;  
    //  Q_CL(i,-1,1)      Q_CL(i,0,1)          Q_CL(i,1,1)      
           
            
    //__________________________________
    // step 1        
    T q_CL_Interpolated;
    q_CL_Interpolated
        = w(0,0) * q_CL[baseCell + offset[0]]   
        + w(0,1) * q_CL[baseCell + offset[1]]           
        + w(0,2) * q_CL[baseCell + offset[2]]           
        + w(1,0) * q_CL[baseCell + offset[3]]            
        + w(1,1) * q_CL[baseCell + offset[4]]    
        + w(1,2) * q_CL[baseCell + offset[5]] 
        + w(2,0) * q_CL[baseCell + offset[6]]   
        + w(2,1) * q_CL[baseCell + offset[7]]     
        + w(2,2) * q_CL[baseCell + offset[8]];

    //__________________________________
    //  step 2  1-D interpolation using coarse and fine level data
    //    |       |       |       .
    //    |  (x)  |  (x)  |   o   .               x   <-- q_CL_Interpolated
    //    |       |       |       .
    //       -1       0       1     (normalized distance)
    //                |------------d_CL-----------|
    //
    //  x2=d_CL = (refine_ratio.x + 1)/2
    //  See notes dated 09/16/06 for derivation
    IntVector dir = finePatch->faceDirection(patchFace);
    IntVector x0 = f_cell;
    IntVector x1 = f_cell;
    x0[p_dir] -= 2*dir[p_dir];
    x1[p_dir] -=   dir[p_dir];
    
    q_FineLevel[f_cell] = w0_x * q_FineLevel[x0] + w1_x * q_FineLevel[x1] + w2_x * q_CL_Interpolated;
    

    //__________________________________
    //  debugging
#if 0
    IntVector half  = (fh - fl )/IntVector(2,2,2) + fl;
    half = fineLevel->mapCellToCoarser(half);
    if( (baseCell[y] == half[y] && baseCell[z] == half[z]) &&is_rightFace(name)){
    
     std::cout.setf(ios::scientific,ios::floatfield);
     std::cout.precision(5);
     std::cout << "\n relativeIndex " << relativeIndx << " dist " << dist << endl;
     std::cout << name << " baseCell " << baseCell << " f_cell " << f_cell << " dy " << dy << " dz " << dz << "\n";
     std::cout << " q_CL[baseCell + " << offset[0] << "] " << q_CL[baseCell +  offset[0]]<< " w(0,0) " << w(0,0) << "\n";
     std::cout << " q_CL[baseCell + " << offset[1] << "] " << q_CL[baseCell +  offset[1]]<< " w(0,1) " << w(0,1) << "\n";
     std::cout << " q_CL[baseCell + " << offset[2] << "] " << q_CL[baseCell +  offset[2]]<< " w(0,2) " << w(0,2) << "\n";
     std::cout << " q_CL[baseCell + " << offset[3] << "] " << q_CL[baseCell +  offset[3]]<< " w(1,0) " << w(1,0) << "\n";
     std::cout << " q_CL[baseCell + " << offset[4] << "] " << q_CL[baseCell +  offset[4]]<< " w(1,1) " << w(1,1) << "\n";
     std::cout << " q_CL[baseCell + " << offset[5] << "] " << q_CL[baseCell +  offset[5]]<< " w(1,2) " << w(1,2) << "\n";
     std::cout << " q_CL[baseCell + " << offset[6] << "] " << q_CL[baseCell +  offset[6]]<< " w(2,0) " << w(2,0) << "\n";
     std::cout << " q_CL[baseCell + " << offset[7] << "] " << q_CL[baseCell +  offset[7]]<< " w(2,1) " << w(2,1) << "\n";
     std::cout << " q_CL[baseCell + " << offset[8] << "] " << q_CL[baseCell +  offset[8]]<< " w(2,2) " << w(2,2) << "\n";
     std::cout << " q_CL_Interpolated " << q_CL_Interpolated << "\n";  
          
     std::cout  << " w0_y " << w0_y << " w1_y " << w1_y << " w2_y "<< w2_y << " sum (w_y): " << w0_y + w1_y + w2_y<<"\n";
     std::cout  << " w0_z " << w0_z << " w1_z " << w1_z << " w2_z "<< w2_z << " sum (w_z): " << w0_z + w1_z + w2_z<<"\n";
     std::cout <<"--------------------------------" << endl;
     std::cout << " f_cell " << f_cell << " x0 " << x0 << " x1 " << x1 << " dir " << dir << endl;
     std::cout << " w0_x " << w0_x << " w1_x " << w1_x << " w2_x " << w2_x << " sum(weights) " << w0_x+w1_x+w2_x<< " x2 " << d_CL << endl;
     std::cout << " q_FineLevel[x0] " << q_FineLevel[x0] << " q_FineLevel[x1] " << q_FineLevel[x1] << " q_FineLevel " << q_FineLevel[f_cell] << endl;
     
   }
#endif   
   
  } 
}

/*___________________________________________________________________
 Function~  selectInterpolator--
_____________________________________________________________________*/
template<class T>
  void selectInterpolator(constCCVariable<T>& q_CL,
                          const int orderOfInterpolation,
                          const Level* coarseLevel,
                          const Level* fineLevel,
                          const IntVector& refineRatio,
                          const IntVector& fl,
                          const IntVector& fh,
                          CCVariable<T>& q_FineLevel)
{
  switch(orderOfInterpolation){
  case 0:
    piecewiseConstantInterpolation(q_CL, fineLevel,fl, fh, q_FineLevel);
    break;
  case 1:
    linearInterpolation<T>(q_CL, coarseLevel, fineLevel,
                          refineRatio, fl,fh, q_FineLevel); 
    break;
  case 2:                             
    quadraticInterpolation<T>(q_CL, coarseLevel, fineLevel,
                              refineRatio, fl,fh, q_FineLevel);
    break;
  default:
    throw InternalError("ERROR:AMR: You're trying to use an interpolator"
                        " that doesn't exist.  <orderOfInterpolation> must be 0,1,2",__FILE__,__LINE__);
#if !WARNS_ABOUT_UNREACHABLE_STATEMENTS
  break;
#endif
  }
}

/*___________________________________________________________________
 Function~  select_CFI_Interpolator--
_____________________________________________________________________*/
template<class T>
  void select_CFI_Interpolator(constCCVariable<T>& q_CL,
                          const int orderOfInterpolation,
                          const Level* coarseLevel,
                          const Level* fineLevel,
                          const IntVector& refineRatio,
                          const IntVector& fl,
                          const IntVector& fh,
                          const Patch* finePatch,
                          Patch::FaceType patchFace,
                          CCVariable<T>& q_FineLevel)
{
  // piecewise constant
  if(orderOfInterpolation == 0){
    std::cout << " pieceWise constant Interpolator " << std::endl;
    piecewiseConstantInterpolation(q_CL, fineLevel,fl, fh, q_FineLevel);
  }
  // linear
  if(orderOfInterpolation == 1){
    linearInterpolation<T>(q_CL, coarseLevel, fineLevel,
                          refineRatio, fl,fh, q_FineLevel); 
  }
  // colella's quadratic
  if(orderOfInterpolation == 2){
  
#if 0
//    std::cout << " colella's quadratic interpolator" << std::endl;
    quadraticInterpolation_CFI<T>(q_CL, finePatch, patchFace,coarseLevel, 
                                  fineLevel, refineRatio, fl, fh,q_FineLevel);
#else
//    std::cout << " standard quadratic Interpolator" << std::endl;
    quadraticInterpolation<T>(q_CL, coarseLevel, fineLevel,
                              refineRatio, fl,fh, q_FineLevel);
#endif
  }
  // bulletproofing
  if(orderOfInterpolation > 2 || orderOfInterpolation < 0){
    throw InternalError("ERROR:AMR: You're trying to use an interpolator"
                        " that doesn't exist.  <orderOfInterpolation> must be 1 or 2",__FILE__,__LINE__);
  }
}
/*___________________________________________________________________
 Function~  interpolationTest_helper--
_____________________________________________________________________*/
template<class T>
  void interpolationTest_helper( CCVariable<T>& q_FineLevel,
                                 CCVariable<T>& q_CL,
                                 const std::string& desc,
                                 const int test,
                                 const Level* level,
                                 const IntVector& l,
                                 const IntVector& h)
{
  int ncell = 0;
  T error(0);
  for(CellIterator iter(l,h); !iter.done(); iter++){
    IntVector c = *iter;
    
    Point cell_pos = level->getCellPosition(c);
    
    double X = cell_pos.x();
    double Y = cell_pos.y();
    double Z = cell_pos.z();
    T exact(0);
    
    switch(test){
    case 0:
      exact = T(5.0);
      break;
    case 1:
      exact = T( X );
      break;
    case 2:
      exact = T( Y );
      break;
    case 3:
      exact = T( Z );
      break;
    case 4:
      exact = T( X * Y* Z  );
      break;
    case 5:
      exact = T( X * X * Y * Y * Z * Z);
      break;
    case 6:
      exact = T( X * X * X * Y* Y * Y  * Z * Z *Z );
      break;
    default:
    break;
    }

    if(desc == "initialize"){
      q_CL[c] = exact;
    }else{
      T diff(q_FineLevel[c] - exact);
      error = error + diff * diff;
      //if( fabs(diff) > 1e-3) {
      //  std::cout << c << " q_FineLevel[c] " <<  q_FineLevel[c] << " exact " << exact << " diff " << diff << endl;
      //}
      ncell += 1; 
    }
  } 
  
  if(desc == "checkError"){
    std::cout  << "test " << test <<" interpolation error^2/ncell " << error/ncell << " ncells " << ncell << "\n";
  }
}
/*___________________________________________________________________
 Function~  testInterpolators--
_____________________________________________________________________*/
template<class T>
  void testInterpolators(DataWarehouse* new_dw,
                         const int orderOfInterpolation,
                         const Level* coarseLevel,
                         const Level* fineLevel,
                         const Patch* finePatch,
                         Patch::FaceType patchFace,
                         const string& testDomain)
{
  ASSERT( testDomain == "wholeDomain" || testDomain == "CFI");

  //__________________________________
  //  define iterator
  IntVector fl, fh;
  if(testDomain == "wholeDomain"){    
    fl = finePatch->getCellLowIndex();
    fh = finePatch->getCellHighIndex();
  }
  if(testDomain == "CFI" ){                                 
    CellIterator iter_tmp = finePatch->getFaceIterator(patchFace,   Patch::ExtraMinusEdgeCells);
    fl = iter_tmp.begin();
    fh = iter_tmp.end();
  }

  IntVector refineRatio(fineLevel->getRefinementRatio());
  std::cout << "testInterpolators:Interpolation Order = " <<orderOfInterpolation << " " << testDomain<<endl;
  std::cout << "------------------------------------" <<  finePatch->getFaceName(patchFace) << endl;
  
  for(int t=0; t<= 6; t++){
    CCVariable<T> q_CoarseLevel, q_FineLevel;
    new_dw->allocateTemporary(q_FineLevel,finePatch);
    q_FineLevel.initialize(T(-9));
    
    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);
    
    if(coarsePatches.size() > 1){
      throw InternalError("ERROR:AMR: testInterpolators: this only works for 1 coarse level patch",__FILE__,__LINE__);
    }
 
    //__________________________________
    //  initialize the coarse level data
    for(int i=0;i<coarsePatches.size();i++){
      const Patch* coarsePatch = coarsePatches[i];    
      new_dw->allocateTemporary(q_CoarseLevel, coarsePatch);
      IntVector cl = coarsePatch->getExtraCellLowIndex();
      IntVector ch = coarsePatch->getExtraCellHighIndex();
      interpolationTest_helper( q_FineLevel, q_CoarseLevel, 
                                "initialize", t, coarseLevel,cl,ch);  
    }
    constCCVariable<T> q_CL_const(q_CoarseLevel);
    
    // piecewise constant
    if(orderOfInterpolation == 0){
      piecewiseConstantInterpolation(q_CL_const, fineLevel,fl, fh, q_FineLevel);
    }
    // linear
    if(orderOfInterpolation == 1){
      linearInterpolation<T>(q_CL_const, coarseLevel, fineLevel,
                            refineRatio, fl,fh, q_FineLevel); 
    }
    // quadratic over the whole computational domain
    if(orderOfInterpolation == 2 && testDomain == "wholeDomain"){                            
      quadraticInterpolation<T>(q_CL_const, coarseLevel, fineLevel,
                                refineRatio, fl,fh, q_FineLevel);
    }
    // colella's quadratic over just the CFI
    if(orderOfInterpolation == 2 && testDomain == "CFI"){    
    
      IntVector ffl = finePatch->getExtraCellLowIndex();
      IntVector ffh = finePatch->getExtraCellHighIndex();
      interpolationTest_helper( q_FineLevel, q_FineLevel, 
                              "initialize", t, fineLevel,ffl,ffh);
                            
      quadraticInterpolation_CFI<T>(q_CL_const, finePatch, patchFace,coarseLevel, 
                                    fineLevel, refineRatio, fl, fh,q_FineLevel);
    }
    // bulletproofing
    if(orderOfInterpolation > 2 || orderOfInterpolation < 0){
      throw InternalError("ERROR:AMR: You're trying to use an interpolator"
                          " that doesn't exist.  <orderOfInterpolation> must be 1 or 2",__FILE__,__LINE__);
    }
    //__________________________________
    //  Now check the error.
    interpolationTest_helper( q_FineLevel,q_CoarseLevel, 
                              "checkError", t, fineLevel,fl,fh);              
  }
}


//______________________________________________________________________
// find the range of values to get from the finePatch that coincides with coarsePatch
// (we need the finePatch, as the fine level might not entirely overlap the coarse)
// also get the coarse range to iterate over
UINTAHSHARE void getFineLevelRange(const Patch* coarsePatch, const Patch* finePatch,
                                    IntVector& cl, IntVector& ch, 
                                    IntVector& fl, IntVector& fh);

// As above, but do the same for nodes, and include fine patch padding cell requirements 
UINTAHSHARE void getFineLevelRangeNodes(const Patch* coarsePatch, 
                                        const Patch* finePatch,
                                        IntVector& cl, IntVector& ch,
                                        IntVector& fl, IntVector& fh, 
                                        IntVector padding);

// find the range of values to get from the coarseLevel that coincides with coarsePatch
// ngc is the number of ghost cells to get at the fine level
UINTAHSHARE void getCoarseLevelRange(const Patch* finePatch, const Level* coarseLevel, 
                                     IntVector& cl, IntVector& ch, 
                                     IntVector& fl, IntVector& fh, 
                                     IntVector boundaryLayer,
                                     int ngc,
                                     const bool returnExclusiveRange);


// find the range of a coarse-fine interface along a certain face
UINTAHSHARE void getCoarseFineFaceRange(const Patch* finePatch, 
                                        const Level* coarseLevel,
                                        Patch::FaceType face,
                                        Patch::FaceIteratorType domain,
                                        const int nCells, 
                                        IntVector& cl, 
                                        IntVector& ch, 
                                        IntVector& fl, 
                                        IntVector& fh);
                                  
UINTAHSHARE void coarseLevel_CFI_NodeIterator(Patch::FaceType patchFace,
                                              const Patch* coarsePatch, 
                                              const Patch* finePatch,   
                                              const Level* fineLevel,   
                                              NodeIterator& iter,       
                                              bool& isRight_CP_FP_pair);

UINTAHSHARE void coarseLevel_CFI_Iterator(Patch::FaceType patchFace,
                                          const Patch* coarsePatch,  
                                          const Patch* finePatch,    
                                          const Level* fineLevel,    
                                          CellIterator& iter,        
                                          bool& isRight_CP_FP_pair);
                                       
UINTAHSHARE void fineLevel_CFI_Iterator(Patch::FaceType patchFace,
                                        const Patch* coarsePatch,  
                                        const Patch* finePatch,    
                                        CellIterator& iter,
                                        bool& isRight_CP_FP_pair);   
                                       
UINTAHSHARE void fineLevel_CFI_NodeIterator(Patch::FaceType patchFace,
                                            const Patch* coarsePatch,  
                                            const Patch* finePatch,    
                                            NodeIterator& iter,
                                            bool& isRight_CP_FP_pair); 
                                            
                                            
UINTAHSHARE void compute_Mag_gradient( constCCVariable<double>& q_CC,
                                       CCVariable<double>& mag_grad_q_CC,                   
                                       const Patch* patch);
                               
UINTAHSHARE void compute_Mag_Divergence( constCCVariable<Vector>& q_CC,
                                         CCVariable<double>& mag_div_q_CC,                   
                                         const Patch* patch);                                
                                        
} // end namespace Uintah
#endif
