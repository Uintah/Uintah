#ifndef Uintah_Component_Arches_DragModel_h
#define Uintah_Component_Arches_DragModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    DragModel
  * @author   Julien Pedel, Charles Reid
  * @date     September 2009, May 2010
  *
  * @brief    A class for calculating the two-way coupling between
  *           particle velocities and the gas phase velocities using
  *           Stokes' drag law.
  *
  */

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class DragModelBuilder: public ModelBuilder
{
public: 
  DragModelBuilder( const std::string          & modelName, 
                    const vector<std::string>  & reqICLabelNames,
                    const vector<std::string>  & reqScalarLabelNames,
                    const ArchesLabel          * fieldLabels,
                    SimulationStateP           & sharedState,
                    int qn );

  ~DragModelBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class DragModel: public ParticleVelocity {
public: 

  DragModel( std::string modelName, 
             SimulationStateP& shared_state, 
             const ArchesLabel* fieldLabels,
             vector<std::string> reqICLabelNames, 
             vector<std::string> reqScalarLabelNames,
             int qn );

  ~DragModel();

  ////////////////////////////////////////////////////
  // Initialization stuff

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation 

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Compute the source term (EMPTY! This method is empty but MUST be defined because it's a virtual function in the parent class.) */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  /** @brief Actually compute the source term (the time sub-step is required for this method) */ 
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
                                DataWarehouse*        new_dw );

  //////////////////////////////////////////////////
  // Access functions

  /* getType method is defined in parent class... */
  
  const VarLabel* getParticleUVelocityLabel() {
    return d_uvel_label; };

  const VarLabel* getParticleVVelocityLabel() {
    return d_vvel_label; };

  const VarLabel* getParticleWVelocityLabel() {
    return d_wvel_label; };


private:

  double pi;

  bool d_length_set;
  bool d_uvel_set;
  bool d_vvel_set;
  bool d_wvel_set;

  // Velocity internal coordinate labels
  const VarLabel* d_uvel_label;     ///< Velocity x-component (internal coordinate) label
  const VarLabel* d_vvel_label;     ///< Velocity y-component (internal coordinate) label
  const VarLabel* d_wvel_label;     ///< Velocity z-component (internal coordinate) label

  // Internal coordinate scaling factors
  double d_uvel_scaling_factor;
  double d_vvel_scaling_factor;
  double d_wvel_scaling_factor;

};
} // end namespace Uintah
#endif

