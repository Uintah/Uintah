
#ifndef Uintah_AMRInterpolate_h
#define Uintah_AMRInterpolate_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>

#include <sstream>

#include <Packages/Uintah/Core/Grid/share.h>
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
SCISHARE void normalizedDistance_CC(const int refineRatio,vector<double>& norm_dist);  

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
    // coarse cell center.  Find the distance the normalized distance between
    // the coarse and fine cell-centers
    
    IntVector relativeIndx = f_cell - (c_cell * refineRatio);
    
    Vector dist;
    dist.x(norm_dist_x[relativeIndx.x()]);
    dist.y(norm_dist_y[relativeIndx.y()]);
    dist.z(norm_dist_z[relativeIndx.z()]);
    
    //__________________________________
    // Offset for coarse level surrounding cells:
    // determine the direction to the surrounding interpolation cells
    Point coarse_cell_pos = coarseLevel->getCellPosition(c_cell);
    Point fine_cell_pos   = fineLevel->getCellPosition(f_cell);
    Vector dir = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()); 
    int i = SCIRun::Sign(dir.x());
    int j = SCIRun::Sign(dir.y());
    int k = SCIRun::Sign(dir.z());
    
    i *= SCIRun::RoundUp(dist.x());  // if dist.x,y,z() = 0 then set (i,j,k) = 0
    j *= SCIRun::RoundUp(dist.y());  // Only need surrounding coarse cell data if dist != 0
    k *= SCIRun::RoundUp(dist.z());  // This is especially true for 1D and 2D problems

    //__________________________________
    //  Find the weights
    double one_minus_dx = 1.0 - dist.x();
    double one_minus_dy = 1.0 - dist.y();
    double one_minus_dz = 1.0 - dist.z();
       
    double w0 = one_minus_dx * one_minus_dy;
    double w1 = dist.x() * one_minus_dy;
    double w2 = dist.y() * one_minus_dx;
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
    q_FineLevel[f_cell] = one_minus_dz * q_XY_Plane_1 
                        + dist.z() * q_XY_Plane_2; 
                         
    //__________________________________
    //  Debugging                        
#if 0
//    if (dist.x() != 0.25 || dist.y() != 0.25){
      if (f_cell == IntVector(15,32,1) || f_cell == IntVector(15,33,1)){
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(16);
       cout << " f_cell " << f_cell << " c_cell "<< c_cell << " offset ["<<i<<","<<j<<","<<k<<"]  " << endl;
       cout << " relative indx " << relativeIndx  << endl;
       cout << "                w0 " << w0 << " w1 " << w1 << " w2 " << w2 << " w3 " << w3<< endl;
       cout << "                dist "<< dist << " dir " << dir <<  endl;
       cout << "                f_pos " <<   fine_cell_pos.asVector() << " c_pos " <<   coarse_cell_pos.asVector() << endl;
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
  Vector c_dx = coarseLevel->dCell();
  Vector inv_c_dx = Vector(1.0)/c_dx;

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
    if(f_cell == IntVector(2, 5, 4)|| true){
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(16);
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
  break;
  }
}
/*___________________________________________________________________
 Function~  interpolationTest_helper--
_____________________________________________________________________*/
template<class T>
  void interpolationTest_helper( CCVariable<T>& q_FineLevel,
                                 CCVariable<T>& q_CL,
                                 const string& desc,
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
                         const Patch* finePatch)
{

  IntVector fl = finePatch->getCellLowIndex();
  IntVector fh = finePatch->getCellHighIndex();
  
#if 0
  if (orderOfInterpolation == 2){  // keep away from the edge of the domain
    fl += IntVector(2,2,2);
    fh -= IntVector(2,2,2);
  }
#endif
  
  IntVector refineRatio(fineLevel->getRefinementRatio());
  
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
      IntVector cl = coarsePatch->getCellLowIndex();
      IntVector ch = coarsePatch->getCellHighIndex();
      interpolationTest_helper( q_FineLevel, q_CoarseLevel, 
                                "initialize", t, coarseLevel,cl,ch);  
    }

    constCCVariable<T> q_CL_const(q_CoarseLevel);
    
    switch(orderOfInterpolation){
    case 0:
      piecewiseConstantInterpolation(q_CL_const, fineLevel,fl, fh, q_FineLevel);
      break;
    case 1:
      linearInterpolation<T>(q_CL_const, coarseLevel, fineLevel,
                            refineRatio, fl,fh, q_FineLevel); 
      break;
    case 2:                             
      quadraticInterpolation<T>(q_CL_const, coarseLevel, fineLevel,
                                refineRatio, fl,fh, q_FineLevel);
      break;
    default:
      throw InternalError("ERROR:AMR: You're trying to use an interpolator"
                          " that doesn't exist.  <orderOfInterpolation> must be 1 or 2",__FILE__,__LINE__);
    break;
    }
    //__________________________________
    //  compute the interpolation error
    interpolationTest_helper( q_FineLevel,q_CoarseLevel, 
                              "checkError", t, fineLevel,fl,fh);
  }
}

// find the range of values to get from the finePatch that coincides with coarsePatch
// (we need the finePatch, as the fine level might not entirely overlap the coarse)
// also get the coarse range to iterate over
SCISHARE void getFineLevelRange(const Patch* coarsePatch, const Patch* finePatch,
                             IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh);

// As above, but do the same for nodes, and include ghost data requirements
SCISHARE void getFineLevelRangeNodes(const Patch* coarsePatch, const Patch* finePatch,
                            IntVector& cl, IntVector& ch,
                            IntVector& fl, IntVector& fh, IntVector ghost);

// find the range of values to get from the coarseLevel that coincides with coarsePatch
// ngc is the number of ghost cells to get at the fine level
SCISHARE void getCoarseLevelRange(const Patch* finePatch, const Level* coarseLevel, 
                               IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh, int ngc);

// find the range of a coarse-fine interface along a certain face
SCISHARE void getCoarseFineFaceRange(const Patch* finePatch, const Level* coarseLevel, Patch::FaceType face,
                                  int interOrder, IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh);
} // end namespace Uintah
#endif
