#ifndef Uintah_Component_Arches_WestbrookDryer_h
#define Uintah_Component_Arches_WestbrookDryer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

/** 
* @class  Westbrook and Dryer Hydrocarbon Chemistry Model
* @author Jeremy Thornock
* @date   Aug 2011
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
*     <src label = "my_source" type = "westbrook_dryer" > 
        <!-- Westbrook Dryer Global Hydrocarbon reaction rate model -->
	      <!-- see Turns, pg. 156-157 -->
        <A                          spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- Pre-exponential factor --> 
        <E_R                        spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- Activation temperature --> 
        <X                          spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- C_xH_y --> 
        <Y                          spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- C_xH_y --> 
        <m                          spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- [C_xH_y]^m --> 
        <n                          spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- [O_2]^n --> 
        <fuel_mass_fraction         spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- Mass fraction of hydrocarbon in the fuel stream --> 
        <oxidizer_O2_mass_fraction  spec="REQUIRED DOUBLE" need_applies_to="type westbrook_dryer"/> <!-- Mass fraction of O2 in the oxidizer stream --> 
        <cstar_fraction_label       spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- C*H mass fraction label --> 
        <equil_fraction_label       spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- CH mass fraction label (equilibrium calc) --> 
        <mw_label                   spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- mixture molecular weight label --> 
        <o2_label                   spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- o2 label --> 
        <temperature_label          spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- temperature label, default = "temperature" --> 
        <density_label              spec="REQUIRED STRING" need_applies_to="type westbrook_dryer"/> <!-- density label, default = "density" --> 
        <pos                        spec="OPTIONAL NO_DATA" need_applies_to="type westbrook_dryer"/><!-- source term is positive --> 
        <hot_spot                   spec="OPTIONAL NO_DATA" need_applies_to="type westbrook_dryer"> <!-- pilot light --> 
          <geom_object/>                                                                            <!-- defines the location of the pilot --> 
          <max_time                 spec="REQUIRED DOUBLE 'positive'"/>                             <!-- defines how long does the pilot last -->  
          <temperature              spec="REQUIRED DOUBLE 'positive'"/>                             <!-- defines the temperature of the pilot --> 
        </hot_spot>
      </src>
    </Sources>
* \endcode 
*  
*/ 
namespace Uintah{

class WestbrookDryer: public SourceTermBase {
public: 

  WestbrookDryer( std::string srcName, ArchesLabel* field_labels, 
                vector<std::string> reqLabelNames, std::string type );

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

      Builder( std::string name, vector<std::string> required_label_names, ArchesLabel* field_labels ) 
        : _name(name), _field_labels(field_labels), _required_label_names(required_label_names){
          _type = "westbrook_dryer"; 
        };
      ~Builder(){}; 

      WestbrookDryer* build()
      { return scinew WestbrookDryer( _name, _field_labels, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      //SimulationStateP& _shared_state; 
      ArchesLabel* _field_labels; 
      vector<std::string> _required_label_names; 

  }; // Builder

  inline double getRate( double T, double CxHy, double O2, double mix_mw, double den, double dt, double vol ) {

    double rate = 0.0; 

    if ( O2 > 0.0 && CxHy > 0.0 ) { 

      double small = 1e-16; 

      double c_O2 = O2 * 1.0/ ( mix_mw * d_MW_O2 ) * d_Press / ( d_R * T ); 
      c_O2 *= 1.0e-6; // to convert to gmol/cm^3

      double c_HC = CxHy * 1.0/ ( mix_mw * d_MW_HC ) * d_Press / ( d_R * T ); 
      c_HC *= 1.0e-6; // to convert to gmol/cm^3

      double my_exp = -1.0 * d_ER / T; 

      double p_HC = 0.0; 
      if ( c_HC > small ) {
        p_HC = pow( c_HC, d_m ); 
      }

      rate = d_A * exp( my_exp ) * p_HC * pow(c_O2, d_n); // gmol/cm^3/s

      rate *= d_MW_HC * mix_mw * d_R * T / d_Press; 
      rate *= den * 1.0e6; // to get [kg HC/s/vol]
      rate *= d_sign; // picking the sign.

      // now check the rate based on local reactants: 
      double constant = dt / den; 

      // check limiting reactant
      if ( std::abs( constant*rate ) > CxHy ){ 
        rate = d_sign * den / dt * CxHy; 
      } 

      // check for nan
      if ( rate != rate ){ 
        rate = 0.0; 
      } 

    }

    return rate; 

  };

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

  std::string d_cstar_label; 
  std::string d_ceq_label; 
  std::string d_mw_label; 
  std::string d_rho_label; 
  std::string d_T_label; 
  std::string d_o2_label; 

  const VarLabel* _temperatureLabel; 
  const VarLabel* _fLabel;           
  const VarLabel* _mixMWLabel;       
  const VarLabel* _denLabel;         
  const VarLabel* _CstarMassFracLabel;  
  const VarLabel* _CEqMassFracLabel; 
  const VarLabel* _O2MassFracLabel; 

  std::vector<GeometryPieceP> _geom_hot_spot;    ///< Geometric locations of pilot light
  double _T_hot_spot;                            ///< Temperature of the pilot light
  double _max_time_hot_spot;                     ///< How long the pilot light is on
  bool _hot_spot;                                ///< Logical on/off switch for the pilot

  ArchesLabel* _field_labels;

  double d_sign; 

}; // end WestbrookDryer
} // end namespace Uintah
#endif
