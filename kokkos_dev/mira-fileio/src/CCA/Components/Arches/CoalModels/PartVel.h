#ifndef PartVel_h
#define PartVel_h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/Directives.h>
#include <iostream>

//===========================================================================

/** @class    PartVel
  * @author   Jeremy Thornock
  *
  * @brief    This model calculates a particle velocity, given information about
  *           the particle, using the fast equilibirum Eulerian approximation 
  *           detailed in Balachandar (2008) and Ferry and Balachandar (2001, 2003).
  */

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

  /** @brief  Sets the velocity vector boundary condtions */
  void computeBCs( const Patch* patch, 
                   std::string varName, 
                   CCVariable<Vector>& vel ){
    d_boundaryCond->setVectorValueBC( 0, patch, vel, varName ); 
  };

  //////////////////////////////////////////////////////
  // Velocity calculation methods 

  /** @brief  Static method to convert cartesian coordinates to spherical coordinates */
  static Vector cart2sph( Vector X ) {
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

  /** @brief  Static method to convert Cartesian coorinates to spherical coordinates */
  static Vector sph2cart( Vector X ) {
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


private:

  ArchesLabel* d_fieldLabels; 
  
  // velocity model paramters
  double d_eta;           ///< Kolmogorov scale
  double rhoRatio;        ///< Density ratio
  double beta;            ///< Beta parameter
  double epsilon;         ///< Turbulence intensity
  double kvisc;           ///< Fluid kinematic viscosity
  //int regime;           ///< Particle regime (I, II, or III - see Balachandar paper for details)
                          // Regime I is particles whose timescales are smaller than the Kolmogorov time scale
                          // Regime II is particles whose timescales are between the Kolmogorov time scale and the large eddy time scale
                          // Regime III is particles whose timescales are larger than the large eddy timescale
  double d_upLimMult;     ///< Multiplies the upper limit of the scaling factor for upper bounds on ic. 
  bool d_gasBC;           ///< Boolean: Use gas velocity boundary conditions for particle velocity boundary conditions?
  double d_min_vel_ratio; ///< Min ratio allow for the velocity difference. 

  std::vector<double> d_wlo;   ///< Initial value of weighted abscissa for length internal coordinate
  std::vector<double> d_wo;    ///< Initial value of weight

  BoundaryCondition_new* d_boundaryCond; 

  double d_highClip; 
  double d_lowClip; 
  double d_power; 
  double d_L; 
  int    d_totIter; 
  double d_tol;
  bool d_bala;
  bool d_drag; 
  bool d_unweighted;

 }; //end class PartVel

} //end namespace Uintah
#endif 
