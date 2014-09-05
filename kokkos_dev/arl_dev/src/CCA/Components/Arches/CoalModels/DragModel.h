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
  * @author   Julien Pedel
  * @date     September 2009
  *
  * @brief    A class for calculating the two-way coupling between
  *           particle velocities and the gas phase velocities.
  *
  */

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class DragModelBuilder: public ModelBuilder
{
public: 
  DragModelBuilder( const std::string          & modelName, 
                    const std::vector<std::string>  & reqICLabelNames,
                    const std::vector<std::string>  & reqScalarLabelNames,
                    ArchesLabel          * fieldLabels,
                    SimulationStateP           & sharedState,
                    int qn );

  ~DragModelBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class DragModel: public ModelBase {
public: 

  DragModel( std::string modelName, 
             SimulationStateP& shared_state, 
             ArchesLabel* fieldLabels,
             std::vector<std::string> reqICLabelNames,
             std::vector<std::string> reqScalarLabelNames,
             int qn );

  ~DragModel();

  ////////////////////////////////////////////////////
  // Initialization stuff

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

   /** @brief Schedule the initialization of some special/local variables */ 
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special/local variables */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  ////////////////////////////////////////////////
  // Model computation 

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  //////////////////////////////////////////////////
  // Access functions

  inline std::string getType() {
    return "Drag"; }

private:

  std::map<std::string, std::string> LabelToRoleMap;

  const VarLabel* d_particle_length_label;
  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_particle_velocity_label;
  const VarLabel* d_gas_velocity_label;
  const VarLabel* d_weight_label;

  std::vector<double>  rc_mass_init;
  std::vector<double>  ash_mass_init;
  double d_pl_scaling_factor;
  double d_rcmass_scaling_factor;
  double d_charmass_scaling_factor;
  double d_pv_scaling_factor;
  double d_w_scaling_factor;
  double d_w_small; // "small" clip value for zero weights

  double pi;

};
} // end namespace Uintah
#endif
