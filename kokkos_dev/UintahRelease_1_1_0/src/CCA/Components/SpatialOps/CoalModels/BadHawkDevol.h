#ifndef Uintah_Component_SpatialOps_BadHawkDevol_h
#define Uintah_Component_SpatialOps_BadHawkDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>


//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class BadHawkDevolBuilder: public ModelBuilder
{
public: 
  BadHawkDevolBuilder( const std::string         & modelName, 
                       const vector<std::string> & reqLabelNames, 
                       const Fields              * fieldLabels,
                       SimulationStateP          & sharedState,
                       int                         qn );
  ~BadHawkDevolBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class BadHawkDevol: public ModelBase {
public: 

  BadHawkDevol( std::string modelName, SimulationStateP& shared_state, 
                const Fields* fieldLabels,
                vector<std::string> reqLabelNames, int qn );

  ~BadHawkDevol();
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
