#ifndef Uintah_Component_Arches_WestbrookDryer_h
#define Uintah_Component_Arches_WestbrookDryer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
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
                std::vector<std::string> reqLabelNames, std::string type );

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
  /** @brief Schedule initialization */
  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void extraSetup( GridP& grid, BoundaryCondition* bc, TableLookup* table_lookup );

  class Builder
    : public SourceTermBase::Builder {

    public:

      Builder( std::string name, std::vector<std::string> required_label_names, ArchesLabel* field_labels )
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
      std::vector<std::string> _required_label_names;

  }; // Builder

  inline double getRate( double T, double Fp, double O2, double diluent, double f_tot_fuel, double den, double dt, double vol ) {

    double rate = 0.0;
    bool compute_rate = false;

    if ( T > d_T_clip && _use_T_clip ) {

      // USING TEMPERATURE CLIP:
      compute_rate = true;

    } else if ( _use_flam_limits ){

      // USING FLAMMABILITY LIMITS:
      // mass percent:
      double fuel_low =0.0;
      double fuel_high =0.0;
      if ( _use_flam_limits_T_dependence){
        // these equations are from the report "Flammability Characteristics of Combustible Gases
        //  and Vapors", Zabetakis 1965.
        fuel_low  = std::max(0.0,_flam_low_m * diluent + _flam_low_b*(1-_flam_T_low_m*(T-298.15)));
        fuel_high = std::min(1.0,_flam_up_m  * diluent + _flam_up_b*(1+_flam_T_up_m*(T-298.15)));
      } else {
        fuel_low  = _flam_low_m * diluent + _flam_low_b;
        fuel_high = _flam_up_m  * diluent + _flam_up_b;
      }
      double loc_fuel = f_tot_fuel*d_MF_HC_f1;

      //using a premix concept here which is why we need the
      //total mixture fraction (fp + eta) and not only fp. The fuel
      //may contain some inert, so we must multiply by the
      //d_MF_HC_f1 variable to get only the hydrocarbon.
      if ( loc_fuel > fuel_low && loc_fuel < fuel_high ) {
        compute_rate = true;
      }

    }

    if ( O2 > 0.0 && Fp > 0.0 && compute_rate ) {

      double small = 1e-16;

      double c_O2 = O2 * 1.0/d_MW_O2 * den * 1.0e-3; //gmol/cm^3

      double c_HC = Fp*d_MF_HC_f1 * 1.0/d_fuel_MW * den * 1.0e-3; //gmol/cm^3

      double my_exp = -1.0 * d_ER / T;

      double p_HC = 0.0;
      if ( c_HC > small ) {
        p_HC = pow( c_HC, d_m );
      }

      rate = d_A * exp( my_exp ) * p_HC * pow(c_O2, d_n); // gmol/cm^3/s

      rate *= 1.0e3 * d_fuel_MW; // kg/m^3/s

      // check if the rate is oxygen or fuel limited based on available O2 given stoichimetry
      // with the reactable fuel or available burnable fuel.
      rate = std::min( den / dt * std::min( O2 * _stoich_massratio * d_MF_HC_f1 , Fp * d_MF_HC_f1 ), rate );
      // Even though the inert fraction in the fuel isn't "reacting"
      // It is being converted from inert in Fp to inert in eta, so the
      // rate for eta is corrected here to incorporate the conversion of the
      // inert from Fp to eta.
      rate /= d_MF_HC_f1;

      // check for nan
      if ( rate != rate ){
        rate = 0.0;
      }
    }
    return rate;

  };

private:

  double d_fuel_MW;       ///< molucular weight of reactive fuel in the fuel stream - this is found in the table
  double d_A;        ///< Pre-exponential fractor
  double d_ER;       ///< Activation temperature (E/R)
  double d_m;        ///< [C_xH_y]^m
  double d_n;        ///< [O_2]^n
  double d_MW_O2;    ///< Molecular weight of O2 (hard set)
  double d_MF_HC_f1; ///< Mass fraction of hydrocarbon with f=1
  double d_R;        ///< Universal gas constant ( R [=] J/mol/K )
  double d_Press;    ///< Atmospheric pressure (set to atmospheric P for now ( 101,325 Pa )
  double d_T_clip;   ///< Temperature limit on the rate. Below this value, the rate turns off.
  bool   _use_T_clip;      ///< Use clip or not
  bool   _use_flam_limits; ///< Use flamibility limits or not
  bool   _use_flam_limits_T_dependence; ///< Add temperature dependence to the flamibility limits
  bool   _const_diluent;   ///< Indicates a constant diluent as specified in the input file
  bool   _using_xi;
  double _flam_low_m; ///< Lower flammability slope as defined by y=mx+b;
  double _flam_low_b; ///< Lower flammability intercept
  double _flam_up_m;  ///< Upper flammability slope
  double _flam_up_b;  ///< Upper flammability intercept
  double _flam_T_low_m; ///< Lower flammability slope for temperature dependence
  double _flam_T_up_m; ///< Lower flammability slope for temperature dependence
  double _diluent_mw; ///< molecular weight of the duluent
  double _const_diluent_mass_fraction;  ///< use a constant diluent mass fraction everywhere

  const VarLabel* d_WDstrippingLabel; ///< kg C stripped / kg C available for old timestep*
  const VarLabel* d_WDextentLabel;    ///< kg C reacted  / kg C available for old timestep*
  const VarLabel* d_WDO2Label;        ///< kg O2 / total kg -- consistent with the model
  const VarLabel* d_WDverLabel;
  // * but stored in the new_Dw

  std::string d_cstar_strip_label;
  std::string d_eta_label;
  std::string d_xi_label;
  std::string d_fp_label;
  std::string d_mw_label;
  std::string d_rho_label;
  std::string d_T_label;
  std::string d_o2_label;
  std::string _diluent_label_name;

  const VarLabel* _temperatureLabel;
  const VarLabel* _mixMWLabel;
  const VarLabel* _denLabel;
  const VarLabel* _EtaLabel;
  const VarLabel* _FpLabel;
  const VarLabel* _XiLabel;
  const VarLabel* _O2MassFracLabel;
  const VarLabel* _CstarStripLabel;
  const VarLabel* _diluentLabel;

  std::vector<GeometryPieceP> _geom_hot_spot;    ///< Geometric locations of pilot light
  double _T_hot_spot;                            ///< Temperature of the pilot light
  double _start_time_hot_spot;                   ///< Starting time for hot spot
  double _stop_time_hot_spot;                    ///< Ending time for hot spot
  double _stoich_massratio;                      ///< mass fuel/mass o2

  ArchesLabel* _field_labels;

}; // end WestbrookDryer
} // end namespace Uintah
#endif
