#ifndef Uintah_Component_Arches_WestbrookDryer_h
#define Uintah_Component_Arches_WestbrookDryer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

//===========================================================================

/**
  * @class    WestbrookDyer Source Term
  * @author   Jeremy Thornock
  * @date     
  *           
  * @brief    
  * Computes a global reaction rate for a hydrocarbon.
  * See Turns, equation 5.1, 5.2
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class WestbrookDryerBuilder: public SourceTermBuilder
{
public: 
  WestbrookDryerBuilder(std::string srcName, 
                      vector<std::string> reqLabelNames, 
                      SimulationStateP& sharedState);
  ~WestbrookDryerBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class WestbrookDryer: public SourceTermBase {
public: 

  WestbrookDryer( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~WestbrookDryer();


  /** @brief  Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief  Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief  Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief  Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief  Return a string with the model type */
  string getType() {
    return "WestbrookDryer";
  };

private:

  double d_A;     // Pre-exponential fractor
  double d_ER;    // Activation temperature (E/R)
  double d_m;     // [C_xH_y]^m 
  double d_n;     // [O_2]^n
  double d_MW_HC; // Molecular weight of the hydrocarbon
  double d_MW_O2;  // Molecular weight of O2 (hard set) 
  double d_MF_HC_f1; // Mass fraction of hydrocarbon with f=1
  double d_MF_O2_f0; // Mass fraction of O2 with f=0
  double d_R;     // Universal gas constant ( R [=] J/mol/K )
  double d_Press; // Atmospheric pressure (set to atmospheric P for now ( 101,325 Pa )

  int d_X;      // C_xH_Y
  int d_Y;      // C_xH_y

  const VarLabel* d_WDstrippingLabel; // kg C stripped / kg C available for old timestep*
  const VarLabel* d_WDextentLabel;    // kg C reacted  / kg C available for old timestep*
  const VarLabel* d_WDO2Label;        // kg O2 / total kg -- consistent with the model
  // * but stored in the new_Dw

}; // end WestbrookDryer
} // end namespace Uintah
#endif
