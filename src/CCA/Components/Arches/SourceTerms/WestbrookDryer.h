#ifndef Uintah_Component_Arches_WestbrookDryer_h
#define Uintah_Component_Arches_WestbrookDryer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

/** 
* @class  Westbrook and Dryer Hydrocarbon Chemistry Model
* @author Jeremy Thornock
* @date   Aug 2010
* 
* @brief Computes the reaction rate source term for any CxHy hydrocarbon using a 
*         one step mechanism. 
*
* @details This class computes the reaction rate source terms for any hydrocarbon defined 
*          as CxHy where x and y are integers.  Details on the model can be found here: 
*
*          Westbrook C, Dryer F. Simplified Reaction Mechanisms for the Oxidation of Hydrocarbon 
*          Fuels in Flames, Combust. Sci. Technol. 1981;27:31-43. 
*
*          Required input has units corresponding with Table 5.1 in Turns, An Introduction to 
*          Combustion: Concepts and Applications.  This model defaults to methane if no input is specified. 
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <Sources>
*     <src label="my_source" type="westbrook_dryer">
*       <X>INTEGER</X>                  <!-- number of carbon atoms --> 
*       <Y>INTEGER</Y>                  <!-- number of hydrogen atoms --> 
*       <A>DOUBLE</A>                   <!-- Pre-exponential factor [(gmol/cm^3)^{1-m-n}/s]--> 
*       <E_R>DOUBLE</E_R>               <!-- "Activiation Temperature" (E/R, [kcal/gmol]) -->
*       <fuel_mass_fraction>DOUBLE</fuel_mass_fraction>          <!-- mass fraction of hydrocarbon at f=1 --> 
*       <oxidizer_O2_mass_fraction>DOUBLE</oxidizer_O2_mass_fraction> <!-- mass fraction of o2 when f=0 --> 
*       <m>DOUBLE</m>                   <!-- [C_xH_y]^m (per the model -- see table in turns or paper) --> 
*       <n>DOUBLE</n>                   <!-- [O_2]^n (per the model -- see table in turns or paper) --> 
*     </src>
*   </Sources>
* \endcode 
*  
*/ 
namespace Uintah{

class WestbrookDryer: public SourceTermBase {
public: 

  WestbrookDryer( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~WestbrookDryer();
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);
  /** @brief Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  /** @brief Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );
  /** @brief Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, vector<std::string> required_label_names, SimulationStateP& shared_state ) 
        : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){}; 

      WestbrookDryer* build()
      { return scinew WestbrookDryer( _name, _shared_state, _required_label_names ); };

    private: 

      std::string _name; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 

  }; // Builder

private:

  double d_A;        ///< Pre-exponential fractor
  double d_ER;       ///< Activation temperature (E/R)
  double d_m;        ///< [C_xH_y]^m 
  double d_n;        ///< [O_2]^n
  double d_MW_HC;    ///< Molecular weight of the hydrocarbon
  double d_MW_O2;    ///< Molecular weight of O2 (hard set) 
  double d_MF_HC_f1; ///< Mass fraction of hydrocarbon with f=1
  double d_MF_O2_f0; ///< Mass fraction of O2 with f=0
  double d_R;        ///< Universal gas constant ( R [=] J/mol/K )
  double d_Press;    ///< Atmospheric pressure (set to atmospheric P for now ( 101,325 Pa )

  int d_X;           ///< C_xH_Y
  int d_Y;           ///< C_xH_y

  const VarLabel* d_WDstrippingLabel; ///< kg C stripped / kg C available for old timestep*
  const VarLabel* d_WDextentLabel;    ///< kg C reacted  / kg C available for old timestep*
  const VarLabel* d_WDO2Label;        ///< kg O2 / total kg -- consistent with the model
  const VarLabel* d_WDverLabel; 
  // * but stored in the new_Dw

  std::string d_scalar_label; 
  std::string d_mw_label; 

}; // end WestbrookDryer
} // end namespace Uintah
#endif
