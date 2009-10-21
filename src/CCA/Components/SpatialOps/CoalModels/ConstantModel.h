#ifndef Uintah_Component_SpatialOps_ConstantModel_h
#define Uintah_Component_SpatialOps_ConstantModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/SpatialOpsCoalModelFactory.h>


//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class ConstantModelBuilder: public ModelBuilder
{
public: 
  ConstantModelBuilder( const std::string         & modelName, 
                        const vector<std::string> & reqLabelNames, 
                        const Fields              * fieldLabels,
                        SimulationStateP          & sharedState,
                        int                         qn );
  ~ConstantModelBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ConstantModel: public ModelBase {
public: 

  ConstantModel( std::string modelName, SimulationStateP& shared_state, 
                const Fields* fieldLabels,
                vector<std::string> reqLabelNames, int qn );

  ~ConstantModel();
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);
  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

private:

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
