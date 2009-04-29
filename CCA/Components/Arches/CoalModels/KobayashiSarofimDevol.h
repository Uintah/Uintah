#ifndef Uintah_Component_Arches_KobayashiSarofimDevol_h
#define Uintah_Component_Arches_KobayashiSarofimDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/ModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

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
                                const ArchesLabel          * fieldLabels,
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
                const ArchesLabel* fieldLabels,
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

  const ArchesLabel* d_fieldLabels; 
  
  map<string, string> LabelToRoleMap;

  //const VarLabel* d_temperature_label;
  const VarLabel* d_raw_coal_mass_fraction_label;
  const VarLabel* d_weight_label;

  double A1;
  double A2;
  
  double E1;
  double E2;
  
  double R;

  double c_o;      // initial mass of raw coal
  double alpha_o;  // initial mass fraction of raw coal

  int d_quad_node;   // store which quad node this model is for

  double d_lowClip; 
  double d_highClip; 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
