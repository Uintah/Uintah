
#ifndef Uintah_AMRInterpolate_h
#define Uintah_AMRInterpolate_h

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Math/MiscMath.h>

namespace Uintah {
/*___________________________________________________________________
 Function~  AMRICE::interpolationTest--
_____________________________________________________________________*/
template<class T>
  void interpolationTest(const Level* fineLevel,
                         const IntVector& f_cell,
                         int& ncell,
                         T& error,
                         T& q_FineLevel)
{
                      
  Vector f_dx = fineLevel->dCell();
  double X = ((f_cell.x() * f_dx.x()) + f_dx.x()/2);
  double Y = ((f_cell.y() * f_dx.y()) + f_dx.y()/2);
  double Z = ((f_cell.z() * f_dx.z()) + f_dx.z()/2);

  //T exact(5.0);
  //T exact( X );
  //T exact( Y );
  //T exact( Z );
  T exact( X * Y* Z  );
  //T exact( X * X * Y * Y * Z * Z);
  //T exact( X * X * X * Y* Y * Y  * Z * Z *Z );

  error = error + (q_FineLevel - exact) * (q_FineLevel - exact);
  ncell += 1;

#if 0  
  cout  << "f_cell \t" << f_cell 
        << " q_FineLevel[f_cell] " << q_FineLevel
        <<  " error " << error <<  endl;
#endif
}

/*___________________________________________________________________
 Function~  AMRICE::linearInterpolation--
 
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
  //int ncell = 0;  // needed by interpolation test
  //T error(0);
  
  Vector c_dx = coarseLevel->dCell();
  Vector inv_c_dx = Vector(1.0)/c_dx;
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    //__________________________________
    // Offset for coarse level surrounding cells:
    //  -find the distance between the center of the patch and fine level cell centers
    //   This tells which direction the offset should be  
    Point coarse_cell_pos = coarseLevel->getCellPosition(c_cell);
    Point fine_cell_pos   = fineLevel->getCellPosition(f_cell);
    Vector dist = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()) * inv_c_dx;
    dist = Abs(dist);
    Vector dir = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()); 
  
    // determine the direction to the surrounding interpolation cells
    int i = Sign(dir.x());
    int j = Sign(dir.y());
    int k = Sign(dir.z());
    
    i *=RoundUp(dist.x());  // if dist.x,y,z() = 0 then set (i,j,k) = 0
    j *=RoundUp(dist.y());  // Only need surrounding coarse cell data if dist != 0
    k *=RoundUp(dist.z());  // This is especially true for 1D and 2D problems
    
    #if 0
    if(f_cell == IntVector(30,39,0) ||true){
    cout << " c_cell " << c_cell << " f_cell " << f_cell << " offset ["<<i<<","<<j<<","<<k<<"]  "
         << " dist " << dist << " dir "<< dir
         << " f_cell_pos " << fine_cell_pos.asVector()<< " c_cell_pos " << coarse_cell_pos.asVector() << endl;
    }
    #endif

    //__________________________________
    //  Find the weights      
    double w0 = (1. - dist.x()) * (1. - dist.y());
    double w1 = dist.x() * (1. - dist.y());
    double w2 = dist.y() * (1. - dist.x());
    double w3 = dist.x() * dist.y(); 
      
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
    q_FineLevel[f_cell] = (1.0 - dist.z()) * q_XY_Plane_1 
                        + dist.z() * q_XY_Plane_2;
    
    //interpolationTest(fineLevel,f_cell, ncell, error, q_FineLevel[f_cell]);  
  }
  
  //interpolation test
  //error = error/(double)ncell;
  //cout << "Linear interpolation    error^2/ncell " << error << " Ncells " << ncell<< endl;
}

/*___________________________________________________________________
 Function~  AMRICE::QuadraticInterpolation--
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
  //int ncell = 0;  // needed by interpolation test
  //T error(0);
  
  Vector c_dx = coarseLevel->dCell();
  Vector inv_c_dx = Vector(1.0)/c_dx;
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    
    //__________________________________
    // Offset for coarse level surrounding cells:
    //  -find the normalized distance between the coarse and fine level cell centers  
    Point coarse_cell_pos = coarseLevel->getCellPosition(c_cell);
    Point fine_cell_pos   = fineLevel->getCellPosition(f_cell);
    Vector dist = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()) * inv_c_dx;
    
    //dist = Abs(dist); 
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
        = w(0,0) * q_CL[c_cell + IntVector( -1, -1, k)]   
        + w(1,0) * q_CL[c_cell + IntVector(  0, -1, k)]           
        + w(2,0) * q_CL[c_cell + IntVector(  1, -1, k)]           
        + w(0,1) * q_CL[c_cell + IntVector( -1,  0, k)]            
        + w(1,1) * q_CL[c_cell + IntVector(  0,  0, k)]    
        + w(2,1) * q_CL[c_cell + IntVector(  1,  0, k)]     
        + w(0,2) * q_CL[c_cell + IntVector( -1,  1, k)]   
        + w(1,2) * q_CL[c_cell + IntVector(  0,  1, k)]     
        + w(2,2) * q_CL[c_cell + IntVector(  1,  1, k)]; 
    }
    
    // interpolate the 3 X-Y planes 
    q_FineLevel[f_cell] = w0_z * q_XY_Plane[0] 
                        + w1_z * q_XY_Plane[1] 
                        + w2_z * q_XY_Plane[2];

   //interpolationTest(fineLevel,f_cell, ncell, error, q_FineLevel[f_cell]); 
   // cout  << " plane 1 " << q_XY_Plane[0] << " plane2 " << q_XY_Plane[1] << " plane3 "<< q_XY_Plane[3]<< endl;
  } 
  
  // interpolation test
  //error = error/(double)ncell;
  //cout << "Quadratic interpolation error^2/ncell " << error << " ncells " << ncell<< endl;
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
                        " that doesn't exist.  <orderOfInterpolation> must be 1 or 2",__FILE__,__LINE__);
  break;
  }
}
#endif
