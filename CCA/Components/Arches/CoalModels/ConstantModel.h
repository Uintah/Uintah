#ifndef Uintah_Component_Arches_ConstantModel_h
#define Uintah_Component_Arches_ConstantModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class ConstantModelBuilder: public ModelBuilder
{
public: 
  ConstantModelBuilder( const std::string          & modelName, 
                        const vector<std::string>  & reqICLabelNames,
                        const vector<std::string>  & reqScalarLabelNames,
                        ArchesLabel          * fieldLabels,
                        SimulationStateP           & sharedState,
                        int qn );
  ~ConstantModelBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ConstantModel: public ModelBase {
public: 

  ConstantModel( std::string modelName, 
                 SimulationStateP& shared_state, 
                 ArchesLabel* fieldLabels,
                 vector<std::string> reqICLabelNames, 
                 vector<std::string> reqScalarLabelNames,
                 int qn );

  ~ConstantModel();

  ///////////////////////////////////////////////
  // Initialization methods

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

  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy solve (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /////////////////////////////////////////////////
  // Model computation methods

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  ///////////////////////////////////////////////
  // Access methods

  inline string getType() {
    return "Constant"; }


private:

  double d_constant; 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif

