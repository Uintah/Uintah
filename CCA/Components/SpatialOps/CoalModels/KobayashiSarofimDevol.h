#ifndef Uintah_Component_SpatialOps_KobayashiSarofimDevol_h
#define Uintah_Component_SpatialOps_KobayashiSarofimDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class KobayashiSarofimDevolBuilder: public ModelBuilder
{
public: 
  KobayashiSarofimDevolBuilder(std::string modelName, 
                      vector<std::string> reqLabelNames, 
                      SimulationStateP& sharedState, int qn);
  ~KobayashiSarofimDevolBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class KobayashiSarofimDevol: public ModelBase {
public: 

  KobayashiSarofimDevol( std::string modelName, SimulationStateP& shared_state, 
                const Fields* fieldLabels,
                vector<std::string> reqLabelNames, int qn );

  ~KobayashiSarofimDevol();
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

  const Fields* d_fieldLabels; 
  const double A1 = 1;
  const double E1 = 1;
  const double A2 = 1;
  const double E2 = 1;
  const double R = 1;

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
