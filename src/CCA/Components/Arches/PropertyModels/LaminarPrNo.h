#ifndef Uintah_Component_Arches_LaminarPrNo_h
#define Uintah_Component_Arches_LaminarPrNo_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

/** 
* @class  LaminarPrNo
* @author Jeremy T. and Weston E. 
* @date   August 2011
* 
* @brief Computes the laminar Pr number by evaluating a laminar diffusion coefficient 
*        and viscosity based on local concentrations of species and local temperature. 
*
* @details This code is currently only supporting binary mixtures.  It assumes that the 
*          mixture fraction is computed as: 
*           \f[ 
*           f = \frac{kg \; fuel}{kg \; fuel + kg \; air} 
*           \f]
*
*           The input parameters for this property model include:
*             The system pressure in units of Bar.
*             The molar masses of the fuel and oxidizer in units of grams/mole,
*             The critical Temperature for fuel and oxidizer in units of Kelvins,
*             The dipole moment for both fuel and oxidizer in units of debyes
*
*             The last parameters need for the model are the pure component viscosities for fuel and oxidizer in kg/m/s.
*             For now the model takes the pure viscosities as direct inputs to the model.  This is done with the assumption
*             that the system modelled does not vary greatly in temperature as pure viscosity is a relatively weak function of 
*             temperature over short temperature ranges.  Eventually, to improve this model, a function will need to be written
*             to estimate the viscosities of various species as a function of temperature.  For Air and Helium the viscosity can 
*             be estimated by the following polynomial fits:
*
*                  Air:    Vis =  5.0477e-7 + 7.2825e-8*T + -4.8524e-11*T^2 + 1.7778e-14*T^3
*
*                  Helium: Vis = -3.09667e-5 + 4.06341e-7*T + -1.15833e-9*T^2 + 1.23016e-12*T^3      
*    
*             Viscosity in units of (kg/m/s) and T in units of (Kelvins).
*
*           Explanation of Method: getVisc(double f, double T):
*                 This method estimates the viscosity of a gas mixture.  The viscosity is in units of Kg/m/s or Pa-s (equivalent).  The
*                 inputs to the method are mixture fraction (f) (or mass fraction of fuel), and system Temperature (T) in Kelvins.
*                 This method for estimating mixture viscosity is developed by Reichenberg (NPL Rept. Chem. 29, National Physical Laboratory
*                 , Teddington, England, May, 1974), (Natl. Eng. Lab. Rept. Chem. 53, East Kilbride, Glasgow, Scotland, May 1977),
*                 (Symp. Transp. Prop. Fluids and Fluid Mixtures, Natl. Eng. Lab., East Kilbride, Glasgow, Scotland, 1979).  The method
*                 is described on page 9.15 in the 5th edition of Poling et al.  
*
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <model label = "label_name" type = "laminar_pr">
*       <atm_pressure>DOUBLE</atm_pressure>                     <!-- Atmospheric pressure [bar] --> 
*       <mix_frac_label>STRING</mix_frac_label>                 <!-- Label name of the variable representing mixture fraction --> 
*       <fuel>
*         <molar_mass>DOUBLE</molar_mass>                       <!-- [g/mol] -->
*         <critical_temperature>DOUBLE<critical_temperature>    <!-- [K] -->
*         <dipole_moment>DOUBLE</dipole_moment>                 <!-- [debyes] --> 
*         <lennard_jones_energy>DOUBLE</lennard_jones_energy>   <!-- [Angstroms] -->
*         <lennard_jones_length>DOUBLE</lennard_jones_length>   <!-- [K] -->
*         <viscosity>DOUBLE</viscosity>                         <!-- [kg/m/s] -->
*       </fuel>
*       <oxidizer>
*         <!-- SAME INPUTS AS FUEL --> 
*       </oxidizer>
*     </model>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class LaminarPrNo : public PropertyModelBase {

    public: 

      LaminarPrNo( std::string prop_name, SimulationStateP& shared_state );
      ~LaminarPrNo(); 

      void problemSetup( const ProblemSpecP& db ); 

      void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
      void computeProp(const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, 
                       int time_substep );

      void sched_dummyInit( const LevelP& level, SchedulerP& sched );
      void dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw );

      void sched_initialize( const LevelP& level, SchedulerP& sched );
      void initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw );

      class Builder
        : public PropertyModelBase::Builder { 

        public: 

          Builder( std::string name, SimulationStateP& shared_state ) : _name(name), _shared_state(shared_state){};
          ~Builder(){}; 

          LaminarPrNo* build()
          { return scinew LaminarPrNo( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      // extra local labels:
      const VarLabel* _mu_label;            ///< viscosity label 

      // input information
      std::string _mix_frac_label_name;     ///< label name for the mixture fraction used for this model 

      bool        _binary_mixture;          ///< if binary mixture 
      // -- for binary mixtures: 
      // mixture faction is defined as: 
      // f = [kg of a] / [kg of a + kg of b]
      //
      double      _pressure;                  ///< atmospheric pressure
      double      _D;                         ///< diffusion coefficient
      double      _molar_mass_a;              ///< molar mass for species (g/mol) a (fuel)  
      double      _molar_mass_b;              ///< molar mass for species (g/mol) b (air)
      double      _crit_temperature_a;        ///< critical temperature (Kelvins) species a
      double      _crit_temperature_b;        ///< critical temperature (Kelvins) species b
      double      _crit_pressure_a;           ///< critical pressure (Bar) species a
      double      _crit_pressure_b;           ///< critical pressure (Bar) species b 
      double      _dipole_moment_a;           ///< dipole moment (debye) species a
      double      _dipole_moment_b;           ///< dipole moment (debye) species b 
      double      _viscosity_a;               ///< viscosity (kg/m/s) for pure species a;
      double      _viscosity_b;               ///< viscosity (kg/m/s) for pure species b; 

      //  -------------------->>> Inline property evaluators <<<--------------------- 
      //
      inline double getVisc( double f, double T ){

        //  compute viscosity here 
        if ( _binary_mixture ) {
        
        double vis_a,vis_b;
        double MW_a,MW_b;
        double TC_a,TC_b;
        double PC_a,PC_b;
        double X_a,X_b,Y_a,Y_b;
        double DP_a,DP_b; 


        vis_a = _viscosity_a;
        MW_a  = _molar_mass_a;
        TC_a  = _crit_temperature_a;
        PC_a  = _crit_pressure_a;
        DP_a  = _dipole_moment_a;

        vis_b = _viscosity_b;
        MW_b  = _molar_mass_b;
        TC_b  = _crit_temperature_b;
        PC_b  = _crit_pressure_b;
        DP_b  = _dipole_moment_b;

        // Conversion to mole fraction from mixture fraction:
        X_a = f;
        X_b = (1 - f);

        double M_a = X_a/MW_a;
        double M_b = X_b/MW_b;

        Y_a = M_a/(M_a + M_b);
        Y_b = 1.000 - Y_a;

        //reduced temperatures:
        double TR_ab = T/(sqrt(TC_a*TC_b));
        
        double TR_a = T/TC_a;
        double TR_b = T/TC_b;

        //reduced dipoles:
        double DPR_a  = 52.46*DP_a*PC_a/(pow(TC_a,2.0));
        double DPR_b  = 52.46*DP_b*PC_b/(pow(TC_b,2.0));
        double DPR_ab = sqrt(DPR_a*DPR_b);
        double FR_a   = (pow(TR_a,3.5) + pow((10*DPR_a),7.0))/(pow(TR_a,3.5)*(1.0 + pow((10*DPR_a),7.0)));
        double FR_b   = (pow(TR_b,3.5) + pow((10*DPR_b),7.0))/(pow(TR_b,3.5)*(1.0 + pow((10*DPR_b),7.0)));
        double FR_ab  = (pow(TR_ab,3.5) + pow((10*DPR_ab),7.0))/(pow(TR_ab,3.5)*(1.0 + pow((10*DPR_ab),7.0)));

        double U_a = (pow((1.0 + 0.36*TR_a*(TR_a - 1.0)),0.16667))/(sqrt(TR_a))*FR_a;
        double U_b = (pow((1.0 + 0.36*TR_b*(TR_b - 1.0)),0.16667))/(sqrt(TR_b))*FR_b;

        double C_a = (pow((MW_a),0.25))/(sqrt(vis_a*U_a));
        double C_b = (pow((MW_b),0.25))/(sqrt(vis_b*U_b));


        double H1   = (sqrt(MW_a*MW_b/32.0))/(pow((MW_a + MW_b),1.5));
        double H2   = (pow((1.0 + 0.36*TR_ab*(TR_ab - 1.0)),0.16667))/(sqrt(TR_ab));
        double H3   = (pow((C_a + C_b),2.0))*FR_ab;
        double H_ab = H1*H2*H3;

        double K_a  = (Y_a*vis_a)/(Y_a + vis_a*(Y_b*H_ab*(3.0 + (2.0*MW_b/MW_a))));
        double K_b  = (Y_b*vis_b)/(Y_b + vis_b*(Y_a*H_ab*(3.0 + (2.0*MW_a/MW_b))));


        double viscosity = K_a*(1.0 + pow(H_ab,2.0)*pow(K_b,2.0)) + K_b*(1.0 + 2*H_ab*K_a + pow(H_ab,2.0)*pow(K_a,2.0));

        return viscosity;  ///< units of kg/m/s

        } else {

          throw InvalidValue("Error: For laminar Pr number property, only binary mixtures are currently supported", __FILE__, __LINE__); 
        } 

      }; 

  }; // class LaminarPrNo
}   // namespace Uintah

#endif
