#ifndef Uintah_Component_Arches_Balachandar_h
#define Uintah_Component_Arches_Balachandar_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    Balachandar
  * @author   Jeremy Thornock, Charles Reid
  * @date     April 2010
  *
  * @brief    A class for the Balachandar fast equilibirum Eulerian approximation
  *           for particle velocity, following Balachandar 2008 and Ferry & Balachandar
  *           2001, 2003.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class BalachandarBuilder: public ModelBuilder
{
public: 
  BalachandarBuilder( const std::string          & modelName,
                      const vector<std::string>  & reqICLabelNames,
                      const vector<std::string>  & reqScalarLabelNames,
                      const ArchesLabel          * fieldLabels,
                      SimulationStateP           & sharedState,
                      int qn );

  ~BalachandarBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class Balachandar: public ParticleVelocity {
public: 

  Balachandar( std::string modelName, 
               SimulationStateP& shared_state, 
               const ArchesLabel* fieldLabels,
               vector<std::string> reqICLabelNames, 
               vector<std::string> reqScalarLabelNames,
               int qn );

  ~Balachandar();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  void sched_computeParticleVelocity( const LevelP& level,
                                      SchedulerP& sched,
                                      int timeSubStep );

  void computeParticleVelocity( const ProcessorGroup* pc,
                                const PatchSubset*    patches,
                                const MaterialSubset* matls,
                                DataWarehouse*        old_dw,
                                DataWarehouse*        new_dw,
                                int timeSubStep );

  ///////////////////////////////////////////////////
  // Get/set methods

  /* getType method is defined in parent class... */

private:

  double d_eta;           ///< Kolmogorov scale
  double rhoRatio;        ///< Density ratio
  double beta;            ///< Beta parameter
  double epsilon;         ///< Turbulence intensity
  double kvisc;           ///< Fluid kinematic viscosity
  int regime;             ///< Particle regime (I, II, or III - see Balachandar paper for details)
                          // Regime I is particles whose timescales are smaller than the Kolmogorov time scale
                          // Regime II is particles whose timescales are between the Kolmogorov time scale and the large eddy time scale
                          // Regime III is particles whose timescales are larger than the large eddy timescale
  double d_upLimMult;     ///< Multiplies the upper limit of the scaling factor for upper bounds on ic. 
  bool d_gasBC;           ///< Boolean: Use gas velocity boundary conditions for particle velocity boundary conditions?
  double d_min_vel_ratio; ///< Min ratio allow for the velocity difference. 

  double d_highClip; 
  double d_lowClip; 
  double d_power; 
  double d_L; 
  double d_tol;
  
  int    d_totIter; 

  bool d_useLength;

}; // end ConstSrcTerm
} // end namespace Uintah
#endif

