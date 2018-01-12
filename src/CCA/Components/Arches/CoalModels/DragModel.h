#ifndef Uintah_Component_Arches_DragModel_h
#define Uintah_Component_Arches_DragModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    DragModel
  * @author   Jeremy, Ben, Derek
  * @date     Nov 2014
  *
  * @brief    A class for calculating the two-way coupling between
  *           particle velocities and the gas phase velocities. This is a
  *           major cleanup of Julien's earlier model.
  *
  */

//---------------------------------------------------------------------------

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

  /** @brief Interface for the inputfile and set constants */
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc,
                 const PatchSubset    * patches,
                 const MaterialSubset * matls,
                 DataWarehouse        * old_dw,
                 DataWarehouse        * new_dw );


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
                     const int timeSubStep );

  /** @brief Return the model type **/
  inline std::string getType() {
    return "Velocity";
  }


private:

  const VarLabel* _length_varlabel;
  const VarLabel* _rhop_varlabel;
  const VarLabel* _weight_varlabel;
  const VarLabel* _scaled_weight_varlabel;
  const VarLabel* _RHS_source_varlabel;
  const VarLabel* _RHS_weight_varlabel; 
  const VarLabel* _birth_label;

  std::string _density_name;

  double _vel_scaling_constant;
  double _weight_scaling_constant;
  double _weight_small;   ///< small weight

  Vector _gravity;
  double _kvisc;

  int _dir;

};
} // end namespace Uintah
#endif
