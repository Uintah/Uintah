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
  * @date     September 2009 : Initial version \n
  *           May 2010       : Cleanup
  *
  * @brief    A class for calculating the two-way coupling between
  *           particle velocities and the gas phase velocities using
  *           Stokes' drag law.
  *
  * @details
  * The particle velocity is tracked as 3 separate (scalar) internal coordinates.
  * However, a 3-component vector containing these 3 components is still computed.
  * This is because particle velocity must be treated as a vector in the code.
  *
  * For this reason, when the boundary conditions for the particle velocity are set, 
  * they are set as a vector, like this: \n
  * \n
  * <BCType id="0" label="vel_qn0" var="Dirichlet"> \n
  *   <value>[1.0,1.0,1.0]</value> \n
  * </BCType> \n
  * \n
  * and NOT set like this, \n
  * \n
  * <BCtype id="0" label="u_velocity_internal_coordinate" var="Dirichlet"> \n
  *   <value>1.0</value> \n
  * \n
  * etc...
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

  /** @brief  Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation 

  /** @brief  Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief  Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     int timeSubStep );

  /** @brief  Schedule computation of particle velocity */
  void sched_computeParticleVelocity( const LevelP& level,
                                      SchedulerP& sched,
                                      int timeSubStep );

  /** @brief  Actually compute the particle velocity */ 
  void computeParticleVelocity( const ProcessorGroup* pc,
                                const PatchSubset*    patches,
                                const MaterialSubset* matls,
                                DataWarehouse*        old_dw,
                                DataWarehouse*        new_dw,
                                int timeSubStep );

  //////////////////////////////////////////////////
  // Access functions

  /* getType method is defined in parent class... */
  
  /** @brief    Get the variable label for the u velocity internal coordinate (which environment depends on which DragModel object you ask...) */
  const VarLabel* getParticleUVelocityLabel() {
    return d_uvel_label; };

  /** @brief    Get the variable label for the v velocity internal coordinate (which environment depends on which DragModel object you ask...) */
  const VarLabel* getParticleVVelocityLabel() {
    return d_vvel_label; };

  /** @brief    Get the variable label for the w velocity internal coordinate (which environment depends on which DragModel object you ask...) */
  const VarLabel* getParticleWVelocityLabel() {
    return d_wvel_label; };


private:

  double pi;

  bool d_useLength;                 ///< Boolean: use particle length internal coordinate?
  bool d_useUVelocity;              ///< Boolean: use u velocity?
  bool d_useVVelocity;              ///< Boolean: use v velocity?
  bool d_useWVelocity;              ///< Boolean: use w velocity?

  // Velocity internal coordinate labels
  const VarLabel* d_uvel_label;     ///< Variable label for u velocity internal coordinate
  const VarLabel* d_vvel_label;     ///< Variable label for v velocity internal coordinate
  const VarLabel* d_wvel_label;     ///< Variable label for w velocity internal coordinate

  // Velocity drag model labels
  const VarLabel* d_uvel_model_label;  ///< Variable label for u velocity drag model term  
  const VarLabel* d_vvel_model_label;  ///< Variable label for v velocity drag model term
  const VarLabel* d_wvel_model_label;  ///< Variable label for w velocity drag model term

  // Internal coordinate scaling factors
  double d_uvel_scaling_factor;     ///< Scaling factor for u velocity internal coordinate
  double d_vvel_scaling_factor;     ///< Scaling factor for v velocity internal coordinate
  double d_wvel_scaling_factor;     ///< Scaling factor for w velocity internal coordinate

};
} // end namespace Uintah
#endif

