#ifndef PartVel_h
#define PartVel_h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

#include <map>
#include <string>
#include <iostream>

#define YDIM
#define ZDIM

//===========================================================================

namespace Uintah {
class ArchesLabel; 
class  PartVel {
   
public:

  PartVel(ArchesLabel* fieldLabels);
 
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
  void computeBCs( const Patch* patch, 
                   string varName, 
                   CCVariable<Vector>& vel ){

    d_boundaryCond->setVectorValueBC( 0, patch, vel, varName ); 

  };

private:

  ArchesLabel* d_fieldLabels; 
  
  // velocity model paramters
  double eta; // Kolmogorov scale
  double rhoRatio; // density ratio
  double beta; // beta parameter
  double epsilon; // turbulence intensity
  double kvisc; // fluid kinematic viscosity
  //int regime; // what is this??? 
  double d_upLimMult; // multiplies the upper limit of the scaling factor for upper bounds on ic. 
  bool d_gasBC; 

  vector<double> d_wlo;
  vector<double> d_wo;

  BoundaryCondition_new* d_boundaryCond; 

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

  double d_highClip; 
  double d_lowClip; 
  

 }; //end class PartVel

} //end namespace Uintah
#endif 
