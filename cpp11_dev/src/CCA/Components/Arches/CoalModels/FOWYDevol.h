#ifndef Uintah_Component_Arches_FOWYDevol_h
#define Uintah_Component_Arches_FOWYDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>



#include <CCA/Components/Arches/FunctorSwitch.h>

#ifdef USE_FUNCTOR
#include <boost/math/special_functions/erf.hpp>
#endif

//===========================================================================

/**
  * @class    FOWYDevol
  * @author   Jeremy Thornock, Julien Pedel, Charles Reid
  * @date     May 2009        Check-in of initial version
  *           November 2009   Verification
  *
  * @brief    A class for calculating the DQMOM model term for the 
  *           Kobayashi-Sarofim coal devolatilization model.
  *
  * The Builder is required because of the Model Factory; the Factory needs
  * some way to create the model term and register it.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ArchesLabel;
class FOWYDevolBuilder: public ModelBuilder 
{
public: 
  FOWYDevolBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                SimulationStateP                & sharedState,
                                int qn );

  ~FOWYDevolBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class FOWYDevol: public Devolatilization {




#ifdef USE_FUNCTOR

   struct computeDevolSource{  
     computeDevolSource( double _dt,
                         double _vol,
                         bool _add_birth,
                         constCCVariable<double> &_temperature, 
                         constCCVariable<double> &_rcmass, 
                         constCCVariable<double> &_charmass, 
                         constCCVariable<double> &_weight, 
                         constCCVariable<double> &_RHS_source, 
                         constCCVariable<double> &_char_RHS_source, 
                         constCCVariable<double> &_rc_weighted_scaled, 
                         constCCVariable<double> &_char_weighted_scaled, 
                         constCCVariable<double> &_rawcoal_birth,
                         CCVariable<double> &_devol_rate,
                         CCVariable<double> &_gas_devol_rate, 
                         CCVariable<double> &_char_rate,
                         CCVariable<double> &_v_inf,
                         FOWYDevol* theClassAbove) :
                       dt(_dt),
                       vol(_vol),
                       add_birth(_add_birth),
                       temperature(_temperature), 
                       rcmass(_rcmass), 
                       charmass(_charmass), 
                       weight(_weight), 
                       RHS_source(_RHS_source), 
                       char_RHS_source(_char_RHS_source), 
                       rc_weighted_scaled(_rc_weighted_scaled), 
                       char_weighted_scaled(_char_weighted_scaled), 
                       rawcoal_birth(_rawcoal_birth),
                       devol_rate(_devol_rate),
                       gas_devol_rate(_gas_devol_rate), 
                       char_rate(_char_rate),
                       v_inf(_v_inf),
                       TCA(theClassAbove) { }
   
   
     void operator()(int i , int j, int k ) const {
   
       double rcmass_init = TCA->rc_mass_init[TCA->d_quadNode];
       double Z=0;
   
       if (weight(i,j,k)/TCA->_weight_scaling_constant > TCA->_weight_small) {
   
         double rcmassph=rcmass(i,j,k);
         double RHS_sourceph=RHS_source(i,j,k);
         double temperatureph=temperature(i,j,k);
         double charmassph=charmass(i,j,k);
         double weightph=weight(i,j,k);
   
         //VERIFICATION
         //rcmassph=1;
         //temperatureph=300;
         //charmassph=0.0;
         //weightph=_rc_scaling_constant*_weight_scaling_constant;
         //rcmass_init = 1;
   
   
         // m_init = m_residual_solid + m_h_off_gas + m_vol
         // m_vol = m_init - m_residual_solid - m_h_off_gas
         // but m_h_off_gas = - m_char
         // m_vol = m_init - m_residual_solid + m_char
   
         double m_vol = rcmass_init - (rcmassph+charmassph);
   
         double v_inf_local = 0.5*TCA->_v_hiT*(1.0 - tanh(TCA->_C1*(TCA->_Tig-temperatureph)/temperatureph + TCA->_C2));
         v_inf(i,j,k) = v_inf_local; 
         double f_drive = std::max((rcmass_init*v_inf_local - m_vol) , 0.0);
         double zFact =std::min(std::max(f_drive/rcmass_init/TCA->_v_hiT,2.5e-5 ),1.0-2.5e-5  );
   
         double rateMax = 0.0; 
         if ( add_birth ){ 
           rateMax = std::max(f_drive/dt 
               + (  (RHS_sourceph+char_RHS_source(i,j,k)) /vol + rawcoal_birth(i,j,k) ) / weightph
               * TCA->_rc_scaling_constant*TCA->_weight_scaling_constant , 0.0 );
         } else { 
           rateMax = std::max(f_drive/dt 
               + (  (RHS_sourceph+char_RHS_source(i,j,k)) /vol ) / weightph
               * TCA->_rc_scaling_constant*TCA->_weight_scaling_constant , 0.0 );
         }
   
         Z = sqrt(2.0) * boost::math::erf_inv(1.0-2.0*zFact );
   
         double rate = std::min(TCA->_A*exp(-(TCA->_Ta + Z *TCA->_sigma)/temperatureph)*f_drive , rateMax);
         devol_rate(i,j,k) = -rate*weightph/(TCA->_rc_scaling_constant*TCA->_weight_scaling_constant); //rate of consumption of raw coal mass
         gas_devol_rate(i,j,k) = rate*weightph; // rate of creation of coal off gas
         char_rate(i,j,k) = 0; // rate of creation of char
   
         //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
         if( devol_rate(i,j,k)>0.0 || ( rc_weighted_scaled(i,j,k) + char_weighted_scaled(i,j,k) )<1e-16) {
           devol_rate(i,j,k) = 0;
           gas_devol_rate(i,j,k) = 0;
           char_rate(i,j,k) = 0;
         }
   
       } else {
         devol_rate(i,j,k) = 0;
         gas_devol_rate(i,j,k) = 0;
         char_rate(i,j,k) = 0;
       }
     }//end cell loop
     private:
   
     double dt;
     double vol;
     bool add_birth;
     constCCVariable<double> &temperature; 
     constCCVariable<double> &rcmass; 
     constCCVariable<double> &charmass; 
     constCCVariable<double> &weight; 
     constCCVariable<double> &RHS_source; 
     constCCVariable<double> &char_RHS_source; 
     constCCVariable<double> &rc_weighted_scaled; 
     constCCVariable<double> &char_weighted_scaled; 
     constCCVariable<double> &rawcoal_birth;
     CCVariable<double> &devol_rate;
     CCVariable<double> &gas_devol_rate; 
     CCVariable<double> &char_rate;
     CCVariable<double> &v_inf;
     FOWYDevol* TCA;
   };
#endif

public: 

  FOWYDevol( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~FOWYDevol();

  ////////////////////////////////////////////////
  // Initialization method

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

  ////////////////////////////////////////////////
  // Model computation method

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw, 
                     const int timeSubStep );

private:

  const VarLabel* _rcmass_varlabel;
  const VarLabel* _RHS_source_varlabel;
  const VarLabel* _char_RHS_source_varlabel;
  const VarLabel* _char_varlabel;
  const VarLabel* _weight_varlabel;
  const VarLabel* _particle_temperature_varlabel;
  const VarLabel* _v_inf_label; 
  const VarLabel* _charmass_weighted_scaled_varlabel; 
  const VarLabel* _rcmass_weighted_scaled_varlabel; 
  const VarLabel* _rawcoal_birth_label; 

  std::vector<double>  particle_sizes;
  std::vector<double>  ash_mass_init;
  std::vector<double>  char_mass_init;
  std::vector<double>  vol_dry;
  std::vector<double>  mass_dry;
  std::vector<double>  rc_mass_init;
  double _v_hiT;
  double _Tig;
  double _Ta;
  double _A;
  double _sigma;
  double _C1;
  double _C2;

  double rhop;
  double total_rc;
  double total_dry;
  double rc_mass_frac;
  double char_mass_frac;
  double ash_mass_frac;
  
  double pi;
  
  double _rc_scaling_constant;   ///< Scaling factor for raw coal internal coordinate
  double _weight_scaling_constant;   ///< Scaling factor for weight 
  double _weight_small;   ///< small weight 
  struct CoalAnalysis{ 
    double C;
    double H; 
    double O; 
    double N; 
    double S; 
    double CHAR; 
    double ASH; 
    double H2O; 
  };
}; // end ConstSrcTerm
} // end namespace Uintah
#endif
