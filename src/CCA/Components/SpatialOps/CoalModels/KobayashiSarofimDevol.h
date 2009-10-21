#ifndef Uintah_Component_SpatialOps_KobayashiSarofimDevol_h
#define Uintah_Component_SpatialOps_KobayashiSarofimDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/SpatialOpsCoalModelFactory.h>
#include <vector>
#include <string>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class KobayashiSarofimDevolBuilder: public ModelBuilder
{
public: 
  KobayashiSarofimDevolBuilder( const std::string          & modelName,
                                const vector<std::string>  & reqLabelNames,
                                const Fields               * fieldLabels,
                                SimulationStateP           & sharedState,
                                int qn );
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

  map<string, string> LabelToRoleMap;

  // put VarLabels in correct order to differentiate between internal coordinates:
  // 1. temperature VarLabel
  // 2. coal_mass_fraction VarLabel
  vector<const VarLabel*> orderedLabels;

  double A1;
  double E1;
  double A2;
  double E2;
  double R;

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
