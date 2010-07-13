#ifndef Uintah_Component_Arches_ParticleVelocity_h
#define Uintah_Component_Arches_ParticleVelocity_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

//===========================================================================

/**
  * @class    ParticleVelocity
  * @author   Charles Reid
  * @date     April 2010 
  *
  * @brief    This class cleans up the previous particle velocity 
  *           implementation, and makes it more consistent.
  * 
  * @details
  * There is one instance of the ParticleVelocity model for each 
  * environment (quad node). Each environment's particle velocity
  * is stored and calculated as a set of 3 scalars. This is done
  * because the particle velocity can be a DQMOM internal coordinate,
  * in which case it's implemented/transported as a scalar.
  *
  */

namespace Uintah{

class ParticleVelocity: public ModelBase {
public: 

  ParticleVelocity( std::string modelName, 
                    SimulationStateP& shared_state, 
                    const ArchesLabel* fieldLabels,
                    vector<std::string> reqICLabelNames, 
                    vector<std::string> reqScalarLabelNames,
                    int qn );

  virtual ~ParticleVelocity();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Grab model-independent devolatilization parameters */
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief  Schedule computation of particle velocity */
  virtual void sched_computeParticleVelocity( const LevelP& level,
                                              SchedulerP&   sched,
                                              int           timeSubStep ) = 0;

  /** @brief  Actually compute particle velocity */
  virtual void computeParticleVelocity( const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        int timeSubStep ) = 0;

  /** @brief  Set the velocity vector boundary conditions */
  void computeBCs( const Patch* patch,
                   string varName,
                   CCVariable<Vector>& vel ) {
    d_boundaryCond->setVectorValueBC( 0, patch, vel, varName );
  }

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type ("ParticleVelocity") */
  inline string getType() {
    return "ParticleVelocity"; };

  /** @brief  Return the variable label corresponding to the particle velocity for this environment */
  const VarLabel* getParticleVelocityLabel() {
    return d_velocity_label; };

protected:

  bool d_gasBC;                     ///< Boolean: Do particle velocities at boundary equal gas velocities at boundary?
  double d_visc;                    ///< [=] m^2/s : Viscosity
  Vector d_gravity;                 ///< [=] m/s^2 : Gravitational acceleration

  // Velocity and length labels 
  const VarLabel* d_velocity_label; ///< Variable label for particle velocity internal coordinate (this is an agglomerated Vector() with each component equal to the particle velocity internal coordinate value corresponding to that component)
  const VarLabel* d_length_label;   ///< Variable label for particle length internal coordinate

  // Scaling factors
  double d_length_scaling_factor;   ///< Scaling factor for particle length internal coordinate

  // Weights
  const VarLabel* d_weight_label;   ///< Variable label for weights

  double d_w_scaling_factor;        ///< Scaling factor for weights
  double d_w_small;                 ///< "small" clip value for weights; if weights < d_w_small, model value is not computed

  BoundaryCondition_new* d_boundaryCond;  ///< Boundary conditions for particle velocity

}; // end ParticleVelocity
} // end namespace Uintah
#endif

