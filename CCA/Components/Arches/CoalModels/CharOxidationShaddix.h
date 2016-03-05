#ifndef Uintah_Component_Arches_CharOxidationShaddix_h
#define Uintah_Component_Arches_CharOxidationShaddix_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>
#include <vector>
#include <string>

#include <Core/Containers/StaticArray.h>
#define USE_FUNCTOR 1
#undef  USE_FUNCTOR 
namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class CharOxidationShaddixBuilder: public ModelBuilder
{
public: 
  CharOxidationShaddixBuilder( const std::string          & modelName,
                               const std::vector<std::string>  & reqICLabelNames,
                               const std::vector<std::string>  & reqScalarLabelNames,
                               ArchesLabel          * fieldLabels,
                               SimulationStateP           & sharedState,
                               int qn );

  ~CharOxidationShaddixBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class CharOxidationShaddix: public CharOxidation {

struct computeCharOxidation{
 computeCharOxidation(double _dt, 
                      double _vol, 
                      bool _add_rawcoal_birth, 
                      bool _add_char_birth, 
                      constCCVariable<double>& _den,
                      constCCVariable<double>& _temperature,
                      constCCVariable<double>& _particle_temperature,
                      SCIRun::StaticArray< constCCVariable<double> >& _length,
                      SCIRun::StaticArray< constCCVariable<double> >& _weight,
                      constCCVariable<double>& _rawcoal_mass,
                      constCCVariable<double>& _char_mass,
                      constCCVariable<double>& _rawcoal_weighted_scaled, 
                      constCCVariable<double>& _char_weighted_scaled, 
                      constCCVariable<double>& _RHS_source, 
                      constCCVariable<double>& _RC_RHS_source, 
                      constCCVariable<double>& _number_density, 
                      constCCVariable<double>& _O2,
                      constCCVariable<double>& _CO2,
                      constCCVariable<double>& _H2O,
                      constCCVariable<double>& _N2,
                      constCCVariable<double>& _MWmix,
                      constCCVariable<double>& _devolChar,
                      constCCVariable<double>& _devolRC,
                      constCCVariable<double>& _rawcoal_birth, 
                      constCCVariable<double>& _char_birth, 
                      CCVariable<double>& _char_rate,
                      CCVariable<double>& _gas_char_rate, 
                      CCVariable<double>& _particle_temp_rate,
                      CCVariable<double>& _surface_rate,
                      CCVariable<double>& _PO2surf_,
                      CharOxidationShaddix* theClassAbove) :
                      dt(_dt), 
                      vol(_vol), 
                      add_rawcoal_birth(_add_rawcoal_birth), 
                      add_char_birth(_add_char_birth), 
                      den(_den),
                      temperature(_temperature),
                      particle_temperature(_particle_temperature),
                      length(_length),
                      weight(_weight),
                      rawcoal_mass(_rawcoal_mass),
                      char_mass(_char_mass),
                      rawcoal_weighted_scaled(_rawcoal_weighted_scaled), 
                      char_weighted_scaled(_char_weighted_scaled), 
                      RHS_source(_RHS_source), 
                      RC_RHS_source(_RC_RHS_source), 
                      number_density(_number_density), 
                      O2(_O2),
                      CO2(_CO2),
                      H2O(_H2O),
                      N2(_N2),
                      MWmix(_MWmix),
                      devolChar(_devolChar),
                      devolRC(_devolRC),
                      rawcoal_birth(_rawcoal_birth), 
                      char_birth(_char_birth), 
                      char_rate(_char_rate),
                      gas_char_rate(_gas_char_rate), 
                      particle_temp_rate(_particle_temp_rate),
                      surface_rate(_surface_rate),
                      PO2surf_(_PO2surf_),
                      TCA(theClassAbove) { } 


       void operator()(int i , int j, int k ) const {
         double max_char_reaction_rate_O2_;
         double char_reaction_rate_;
         double char_production_rate_;
         double particle_temp_rate_;
         int NIter;
         double rc_destruction_rate_;
         double PO2_surf=0.0;
         double PO2_surf_guess;
         double PO2_surf_tmp;
         double PO2_surf_new;
         double PO2_surf_old;
         double CO2CO;
         double OF;
         double ks;
         double q;

         double d_tol;
         double delta;
         double Conc;
         double DO2;
         double gamma;
         double f0;
         double f1;
         int icount;

         if (weight[TCA->d_quadNode](i,j,k)/TCA->_weight_scaling_constant < TCA->_weight_small) {
           char_production_rate_ = 0.0;
           char_rate(i,j,k) = 0.0;
           gas_char_rate(i,j,k) = 0.0;
           particle_temp_rate(i,j,k) = 0.0;
           surface_rate(i,j,k) = 0.0;

         } else {
           double denph=den(i,j,k);
           double temperatureph=temperature(i,j,k);
           double particle_temperatureph=particle_temperature(i,j,k);
           double lengthph=length[TCA->d_quadNode](i,j,k);
           double rawcoal_massph=rawcoal_mass(i,j,k);
           double char_massph=char_mass(i,j,k);
           double weightph=weight[TCA->d_quadNode](i,j,k);
           double O2ph=O2(i,j,k);
           double CO2ph=CO2(i,j,k);
           double H2Oph=H2O(i,j,k);
           double N2ph=N2(i,j,k);
           double MWmixph=MWmix(i,j,k);
           double devolCharph=devolChar(i,j,k);
           double devolRCph=devolRC(i,j,k);
           double RHS_sourceph=RHS_source(i,j,k);

           double PO2_inf = O2ph/TCA->_WO2/MWmixph;
           double AreaSum =0;
           for (int ix=0; ix<TCA->_nQn_part;ix++ ){ 
             AreaSum+=  weight[ix](i,j,k)*length[ix](i,j,k)*length[ix](i,j,k);
           }
           double surfaceAreaFraction=weightph*lengthph*lengthph/AreaSum;
        


           if((PO2_inf < 1e-12) || (rawcoal_massph+char_massph) < TCA->_small) {
             PO2_surf = 0.0;
             CO2CO = 0.0;
             q = 0.0;
           } else {
             char_reaction_rate_ = 0.0;
             char_production_rate_ = 0.0;
             particle_temp_rate_ = 0.0;
             NIter = 15;
             delta = PO2_inf/4.0;
             d_tol = 1e-15;
             PO2_surf_old = 0.0;
             PO2_surf_new = 0.0;
             PO2_surf_tmp = 0.0;
             PO2_surf_guess = 0.0;
             f0 = 0.0;
             f1 = 0.0;
             PO2_surf = 0.0;
             CO2CO = 0.0;
             q = 0.0;
             icount = 0;
          
          // Calculate O2 diffusion coefficient
             DO2 = (CO2ph/TCA->_WCO2 + H2Oph/TCA->_WH2O + N2ph/TCA->_WN2)/(CO2ph/(TCA->_WCO2*TCA->_D1) + 
                   H2Oph/(TCA->_WH2O*TCA->_D2) + 
                   N2ph/(TCA->_WN2*TCA->_D3))*(std::pow((temperatureph/TCA->_T0),1.5));
          // Concentration C = P/RT
             Conc = MWmixph*denph*1000.0;
             ks = TCA->_As*exp(-TCA->_Es/(TCA->_R*particle_temperatureph));

             PO2_surf_guess = PO2_inf/2.0;
             PO2_surf_old = PO2_surf_guess-delta;
             CO2CO = 0.02*(std::pow(PO2_surf_old,0.21))*exp(3070.0/particle_temperatureph);
             OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
             gamma = -(1.0-OF);
             q = ks*(std::pow(PO2_surf_old,TCA->_n));
             f0 = PO2_surf_old - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));

             PO2_surf_new = PO2_surf_guess+delta;
             CO2CO = 0.02*(std::pow(PO2_surf_new,0.21))*exp(3070.0/particle_temperatureph);
             OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
             gamma = -(1.0-OF);
             q = ks*(std::pow(PO2_surf_new,TCA->_n));
             f1 = PO2_surf_new - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));

             for ( int iter=0; iter < NIter; iter++) {
               icount++;
               PO2_surf_tmp = PO2_surf_old;
               PO2_surf_old=PO2_surf_new;
               PO2_surf_new=PO2_surf_tmp - (PO2_surf_new - PO2_surf_tmp)/(f1-f0) * f0;
               PO2_surf_new = std::max(0.0, std::min(PO2_inf, PO2_surf_new));            
               if (std::abs(PO2_surf_new-PO2_surf_old) < d_tol){
                 PO2_surf=PO2_surf_new;
                 CO2CO = 0.02*(std::pow(PO2_surf,0.21))*exp(3070.0/particle_temperatureph);
                 OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                 gamma = -(1.0-OF);
                 q = ks*(std::pow(PO2_surf,TCA->_n));
                 break;
               }
               f0 = f1;
               CO2CO = 0.02*(std::pow(PO2_surf_new,0.21))*exp(3070.0/particle_temperatureph);
               OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
               gamma = -(1.0-OF);
               q = ks*(std::pow(PO2_surf_new,TCA->_n));
               f1 = PO2_surf_new - gamma - (PO2_inf-gamma)*exp(-(q*lengthph)/(2*Conc*DO2));
               PO2_surf=PO2_surf_new; // This is needed to assign PO2_surf if we don't converge.
             }
           }

           char_production_rate_ = devolCharph;
           rc_destruction_rate_ = devolRCph;
           double gamma1=(TCA->_WC/TCA->_WO2)*((CO2CO+1.0)/(CO2CO+0.5)); 
           max_char_reaction_rate_O2_ = std::max( (O2ph*denph*gamma1*surfaceAreaFraction)/(dt*weightph) , 0.0 );

           double max_char_reaction_rate_ = 0.0;

           if ( add_rawcoal_birth && add_char_birth ){ 
             max_char_reaction_rate_ = std::max((rawcoal_massph+char_massph)/(dt) 
                    +( (RHS_sourceph + RC_RHS_source(i,j,k)) / (vol*weightph) + (char_production_rate_ + rc_destruction_rate_
                        +   char_birth(i,j,k) + rawcoal_birth(i,j,k) )/ weightph )
                    *TCA->_char_scaling_constant*TCA->_weight_scaling_constant, 0.0); // equation assumes RC_scaling=Char_scaling
           } else { 
             max_char_reaction_rate_ = std::max((rawcoal_massph+char_massph)/(dt) 
                  +((RHS_sourceph + RC_RHS_source(i,j,k)) / (vol*weightph) + (char_production_rate_ + rc_destruction_rate_
                      )/ weightph )
                  *TCA->_char_scaling_constant*TCA->_weight_scaling_constant, 0.0); // equation assumes RC_scaling=Char_scaling
           }


           max_char_reaction_rate_ = std::min( max_char_reaction_rate_ ,max_char_reaction_rate_O2_ );
           char_reaction_rate_ = std::min(TCA->_pi*(std::pow(lengthph,2.0))*TCA->_WC*q , max_char_reaction_rate_); // kg/(s.#)    

           particle_temp_rate_ = -char_reaction_rate_/TCA->_WC/(1.0+CO2CO)*(CO2CO*TCA->_HF_CO2 + TCA->_HF_CO); // J/(s.#)
           char_rate(i,j,k) = (-char_reaction_rate_*weightph+char_production_rate_)/(TCA->_char_scaling_constant*TCA->_weight_scaling_constant);
           gas_char_rate(i,j,k) = char_reaction_rate_*weightph;// kg/(m^3.s)
           particle_temp_rate(i,j,k) = particle_temp_rate_*weightph; // J/(s.m^3)
           surface_rate(i,j,k) = -TCA->_WC*q;  // in kg/s/m^2
           PO2surf_(i,j,k) = PO2_surf;
        //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
           if( char_rate(i,j,k)>0.0 ) {
             char_rate(i,j,k) = 0;
             gas_char_rate(i,j,k) = 0;
             particle_temp_rate(i,j,k) = 0;
             surface_rate(i,j,k) = 0;  // in kg/s/m^2
             PO2surf_(i,j,k) = PO2_surf;
           }
         }
       }

  private:
       double dt;
       double vol;
       bool add_rawcoal_birth; 
       bool add_char_birth; 
       constCCVariable<double>& den;
       constCCVariable<double>& temperature;
       constCCVariable<double>& particle_temperature;
       SCIRun::StaticArray< constCCVariable<double> >& length; 
       SCIRun::StaticArray< constCCVariable<double> >& weight;
       constCCVariable<double>& rawcoal_mass;
       constCCVariable<double>& char_mass;
       constCCVariable<double>& rawcoal_weighted_scaled; 
       constCCVariable<double>& char_weighted_scaled; 
       constCCVariable<double>& RHS_source; 
       constCCVariable<double>& RC_RHS_source; 
       constCCVariable<double>& number_density; 
       constCCVariable<double>& O2;
       constCCVariable<double>& CO2;
       constCCVariable<double>& H2O;
       constCCVariable<double>& N2;
       constCCVariable<double>& MWmix;
       constCCVariable<double>& devolChar;
       constCCVariable<double>& devolRC;
       constCCVariable<double>& rawcoal_birth; 
       constCCVariable<double>& char_birth; 
       CCVariable<double>& char_rate;
       CCVariable<double>& gas_char_rate; 
       CCVariable<double>& particle_temp_rate;
       CCVariable<double>& surface_rate;
       CCVariable<double>& PO2surf_;
       CharOxidationShaddix* TCA;
};
          
public: 

  CharOxidationShaddix( std::string modelName, 
                        SimulationStateP& shared_state,
                        ArchesLabel* fieldLabels,
                        std::vector<std::string> reqICLabelNames,
                        std::vector<std::string> reqScalarLabelNames,
                        int qn );

  ~CharOxidationShaddix();

  typedef std::map< std::string, ModelBase*> ModelMap;
  typedef std::map< std::string, Devolatilization*> DevolModelMap; 
  /////////////////////////////////////////
  // Initialization methods

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of some special/local variables */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local variables */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /////////////////////////////////////////////
  // Model computation methods

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                           int timeSubStep );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     const int timeSubStep );

  /////////////////////////////////////////////////
  // Access methods

private:

  const VarLabel* _devolCharLabel;
  const VarLabel* _devolRCLabel;
  const VarLabel* _rcmass_varlabel;
  const VarLabel* _rcmass_weighted_scaled_varlabel; 
  const VarLabel* _charmass_weighted_scaled_varlabel; 
  const VarLabel* _char_varlabel;
  const VarLabel* _RHS_source_varlabel;
  const VarLabel* _RC_RHS_source_varlabel;
  std::vector< const VarLabel*> _length_varlabel;
  const VarLabel* _particle_temperature_varlabel;
  const VarLabel* _number_density_varlabel;
  std::vector< const VarLabel*> _weight_varlabel;
  const VarLabel* _gas_temperature_varlabel;
  const VarLabel* _O2_varlabel;
  const VarLabel* _CO2_varlabel;
  const VarLabel* _H2O_varlabel;
  const VarLabel* _N2_varlabel;
  const VarLabel* _MW_varlabel;
  const VarLabel* _rawcoal_birth_label; 
  const VarLabel* _char_birth_label; 
  
  int _nQn_part;
  double _As;
  double _Es;
  double _n;
  double _R;
  double _HF_CO2;
  double _HF_CO;
  double _small;
  double _WO2;
  double _WCO2;
  double _WH2O;
  double _WN2;
  double _WC;
  double _D1;
  double _D2;
  double _D3;
  double _T0;
  double _pi;
  double _RC_scaling_constant;   ///< Scaling factor for raw coal internal coordinate
  double _char_scaling_constant;   ///< Scaling factor for char internal coordinate
  double _weight_scaling_constant;   ///< Scaling factor for weight 
  double _weight_small;   ///< small weight 
  
}; // end CharOxidationShaddix
} // end namespace Uintah
#endif
