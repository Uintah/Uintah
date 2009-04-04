#ifndef PartVel_h
#define PartVel_h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Scheduler.h>

#include <map>
#include <string>
#include <iostream>

#define YDIM
//#define ZDIM

//===========================================================================

namespace Uintah {
class Fields; 
class  PartVel {
   
public:

  PartVel(Fields* fieldLabels);
 
  ~PartVel();
  /** @brief Interface to the input file */
  void problemSetup( const ProblemSpecP& inputdb ); 
  /** @brief Schedules the calculation of the particle velocities */
  void schedComputePartVel( const LevelP& level, SchedulerP& sched, const int rkStep );
  /** @brief Actually computes the particle velocities */ 
  void ComputePartVel( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, const int rkStep );

private:

  Fields* d_fieldLabels; 
  
  // velocity model paramters
  double eta; 
  double rhoRatio; 
  double beta; 
  double eps;
  double nnew;  
  int regime; 

  vector<double> d_wlo;
  vector<double> d_wo;

  Vector cart2sph( Vector X ) {
    // converts cartesean to spherical coords
    double mag   = pow( X.x(), 2.0 );
    double magxy = mag;  
    double z = 0; 
    double y = 0;
#ifdef YDIM
    mag   += pow( X.y(), 2.0 );
    magxy = mag; 
    y = X.y(); 
#endif 
#ifdef ZDIM
    mag += pow( X.z(), 2.0 );
    z = X.z(); 
#endif

    mag   = pow(mag, 1./2.);
    magxy = pow(magxy, 1./2.);

    double elev = atan2( z, magxy );
    double az   = atan2( y, X.x() );  

    Vector answer(az, elev, mag);
    return answer; 

  };

  Vector sph2cart( Vector X ) {
    // converts spherical to cartesian coords
    double x = 0.;
    double y = 0.;
    double z = 0.;

    double rcoselev = X.z() * cos(X.y());
    x = rcoselev * cos(X.x());
#ifdef YDIM
    y = rcoselev * sin(X.x());
#endif
#ifdef ZDIM
    z = X.z()*sin(X.y());
#endif
    Vector answer(x,y,z);
    return answer; 

  };
  

 }; //end class Fields

} //end namespace Uintah
#endif 
