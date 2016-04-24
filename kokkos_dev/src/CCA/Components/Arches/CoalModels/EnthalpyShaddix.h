#ifndef Uintah_Component_Arches_EnthalpyShaddix_h
#define Uintah_Component_Arches_EnthalpyShaddix_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/Properties.h>
#include <vector>
#include <string>

// FIXME: more descriptive name
/**
  * @class    EnthalpyShaddix
  * @author   Julien Pedel
  * @date     Feb. 2011
  *
  * @brief    A heat transfer model for coal paticles
  *          based on Shaddix's experimental work
  *
  */

#include <CCA/Components/Arches/FunctorSwitch.h>

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class EnthalpyShaddixBuilder: public ModelBuilder
{
public:
  EnthalpyShaddixBuilder( const std::string          & modelName,
                          const std::vector<std::string>  & reqICLabelNames,
                          const std::vector<std::string>  & reqScalarLabelNames,
                          ArchesLabel                * fieldLabels,
                          SimulationStateP           & sharedState,
                          Properties                 * props,
                          int qn );

  ~EnthalpyShaddixBuilder();

  ModelBase* build();

private:

  Properties* d_props;

};

// End Builder
//---------------------------------------------------------------------------

class EnthalpyShaddix: public HeatTransfer {

struct computeEnthalpySource{
       computeEnthalpySource(  double _dt,
                               constCCVariable<double> &_weight,
                               constCCVariable<double> &_rawcoal_mass,
                               constCCVariable<double> &_char_mass,
                               constCCVariable<double> &_particle_temperature,
                               constCCVariable<double> &_temperature,
                               constCCVariable<double> &_specific_heat,
                               constCCVariable<double> &_radiationVolqIN,
                               constCCVariable<double> &_abskp,
                               constCCVariable<double> &_rad_particle_temperature,
                               constCCVariable<double> &_den,
                               constCCVariable<double> &_devol_gas_source,
                               constCCVariable<double> &_chargas_source,
                               constCCVariable<double> &_length,
                               constCCVariable<double> &_charoxi_temp_source,
                               constCCVariable<double> &_surface_rate,
                               constCCVariable<Vector> &_gasVel,
                               constCCVariable<Vector> &_partVel,
                               CCVariable<double> &_heat_rate,
                               CCVariable<double> &_gas_heat_rate,
                               CCVariable<double> &_qconv,
                               CCVariable<double> &_qrad,
                               EnthalpyShaddix* theClassAbove ):
                                dt(_dt),
                                weight(_weight),
                                rawcoal_mass(_rawcoal_mass),
                                char_mass(_char_mass),
                                particle_temperature(_particle_temperature),
                                temperature(_temperature),
                                specific_heat(_specific_heat),
                                radiationVolqIN(_radiationVolqIN),
                                abskp(_abskp),
                                rad_particle_temperature(_rad_particle_temperature),
                                den(_den),
                                devol_gas_source(_devol_gas_source),
                                chargas_source(_chargas_source),
                                length(_length),
                                charoxi_temp_source(_charoxi_temp_source),
                                surface_rate(_surface_rate),
                                gasVel(_gasVel),
                                partVel(_partVel),
                                heat_rate(_heat_rate),
                                gas_heat_rate(_gas_heat_rate),
                                qconv(_qconv),
                                qrad(_qrad),
                                TCA(theClassAbove) {  }


       void operator()(int i , int j, int k ) const {
         double max_Q_convection;
         double heat_rate_;
         double gas_heat_rate_;
         double Q_convection;
         double Q_radiation;
         double Q_reaction;
         double blow;
         double kappa;

         if (weight(i,j,k)/TCA->_weight_scaling_constant < TCA->_weight_small) {
         heat_rate_ = 0.0;
         gas_heat_rate_ = 0.0;
         Q_convection = 0.0;
         Q_radiation = 0.0;
         } else {

         double rawcoal_massph=rawcoal_mass(i,j,k);
         double char_massph=char_mass(i,j,k);
         double temperatureph=temperature(i,j,k);
         double specific_heatph=specific_heat(i,j,k);
         double denph=den(i,j,k);
         double devol_gas_sourceph=devol_gas_source(i,j,k);
         double chargas_sourceph=chargas_source(i,j,k);
         double lengthph=length(i,j,k);
         double weightph=weight(i,j,k);
         double particle_temperatureph=particle_temperature(i,j,k);
         double charoxi_temp_sourceph=charoxi_temp_source(i,j,k);
         double surface_rateph=surface_rate(i,j,k);

         // velocities
         Vector gas_velocity = gasVel(i,j,k);
         Vector particle_velocity = partVel(i,j,k);

         double FSum = 0.0;

         // intermediate calculation values
         double Re;
         double Nu;
         double rkg;
         // Convection part: -----------------------
         // Reynolds number
         double delta_V =sqrt(std::pow(gas_velocity.x() - particle_velocity.x(),2.0) + std::pow(gas_velocity.y() - particle_velocity.y(),2.0)+std::pow(gas_velocity.z() - particle_velocity.z(),2.0));
         Re = delta_V*lengthph*denph/TCA->_visc;

         // Nusselt number
         Nu = 2.0 + 0.65*std::pow(Re,0.50)*std::pow(TCA->_Pr,(1.0/3.0));

         // Gas thermal conductivity
         rkg = TCA->props(temperatureph, particle_temperatureph); // [=] J/s/m/K

         // A BLOWING CORRECTION TO THE HEAT TRANSFER MODEL IS EMPLOYED
         kappa =  -surface_rateph*lengthph*specific_heatph/(2.0*rkg);
         if(std::abs(exp(kappa)-1.0) < 1e-16){
         blow = 1.0;
         } else {
         blow = kappa/(exp(kappa)-1.0);
         }

         Q_convection = Nu*TCA->_pi*blow*rkg*lengthph*(temperatureph - particle_temperatureph); // J/(#.s)
         //clip convection term if timesteps are too large
         double deltaT=temperatureph-particle_temperatureph;
         double alpha_rc=(rawcoal_massph+char_massph);
         double alpha_cp=TCA->cp_c(particle_temperatureph)*alpha_rc+TCA->cp_ash(particle_temperatureph)*TCA->_init_ash[TCA->_nQuadNode];
         max_Q_convection=alpha_cp*(deltaT/dt);
         if (std::abs(Q_convection) > std::abs(max_Q_convection)){
         Q_convection = max_Q_convection;
         }
         // Radiation part: -------------------------
         Q_radiation = 0.0;
         if ( TCA->_radiationOn) {
         double Eb;
         Eb = 4.0*TCA->_sigma*std::pow(rad_particle_temperature(i,j,k),4.0);
         FSum = radiationVolqIN(i,j,k);
         Q_radiation = abskp(i,j,k)*(FSum - Eb);
         double Q_radMax=(std::pow( radiationVolqIN(i,j,k) / (4.0 * TCA->_sigma )  , 0.25)-rad_particle_temperature(i,j,k))/(dt)*alpha_cp;
         if (std::abs(Q_radMax) < std::abs(Q_radiation)){
         Q_radiation=Q_radMax;
         }
         }
         double hint = -156.076 + 380/(-1 + exp(380 / particle_temperatureph)) + 3600/(-1 + exp(1800 / particle_temperatureph));
         double hc = TCA->_Hc0 + hint * TCA->_RdMW;
         Q_reaction = charoxi_temp_sourceph;
         // This needs to be made consistant with lagrangian particles!!! - derek 12/14
         heat_rate_ = (Q_convection*weightph + Q_radiation + TCA->_ksi*Q_reaction - (devol_gas_sourceph + chargas_sourceph)*hc)/
         (TCA->_enthalpy_scaling_constant*TCA->_weight_scaling_constant);
         gas_heat_rate_ = -weightph*Q_convection - Q_radiation - TCA->_ksi*Q_reaction + (devol_gas_sourceph+chargas_sourceph)*hc;
         }
         heat_rate(i,j,k) = heat_rate_;
         gas_heat_rate(i,j,k) = gas_heat_rate_;
         qconv(i,j,k) = Q_convection;
         qrad(i,j,k) = Q_radiation;
       }

  private:
                               double dt;
                               constCCVariable<double>& weight;
                               constCCVariable<double>& rawcoal_mass;
                               constCCVariable<double>& char_mass;
                               constCCVariable<double>& particle_temperature;
                               constCCVariable<double>& temperature;
                               constCCVariable<double>& specific_heat;
                               constCCVariable<double>& radiationVolqIN;
                               constCCVariable<double>& abskp;
                               constCCVariable<double>& rad_particle_temperature;
                               constCCVariable<double>& den;
                               constCCVariable<double>& devol_gas_source;
                               constCCVariable<double>& chargas_source;
                               constCCVariable<double>& length;
                               constCCVariable<double>& charoxi_temp_source;
                               constCCVariable<double>& surface_rate;
                               constCCVariable<Vector> &gasVel;
                               constCCVariable<Vector> &partVel;
                               CCVariable<double>& heat_rate;
                               CCVariable<double>& gas_heat_rate;
                               CCVariable<double>& qconv;
                               CCVariable<double>& qrad;
                               EnthalpyShaddix* TCA;

};

public:



  friend struct computeEnthalpySource;

  typedef std::map< std::string, CharOxidation*> CharOxiModelMap;
  typedef std::map< std::string, Devolatilization*> DevolModelMap;

  EnthalpyShaddix( std::string modelName,
                   SimulationStateP& shared_state,
                   ArchesLabel* fieldLabels,
                   std::vector<std::string> reqICLabelNames,
                   std::vector<std::string> reqScalarLabelNames,
                   Properties* props,
                   int qn );

  ~EnthalpyShaddix();



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

  // FIXME: add Glacier computation methods

  /** @brief  Get the particle heating rate (see Glacier) */
  double calcParticleHeatingRate() {
    return 0; }

  /** @brief  Get the gas heating rate (see Glacier) */
  double calcGasHeatingRate() {
    return 0; }

  /** @brief  Get the particle temperature (see Glacier) */
  double calcParticleTemperature() {
    return 0; }

  /** @brief  Get the particle heat capacity (see Glacier) */
  double calcParticleHeatCapacity() {
    return 0; }

  /** @brief  Get the convective heat transfer coefficient (see Glacier) */
  double calcConvectiveHeatXferCoeff() {
    return 0; }

  /** @brief  Calculate enthalpy of coal off-gas (see Glacier) */
  double calcEnthalpyCoalOffGas() {
    return 0; }

  /** @brief  Calculate enthalpy change of the particle (see Glacier) */
  double calcEnthalpyChangeParticle() {
    return 0; }


private:

  //////////////////////////////////////////////////
  // Private methods for calculation


  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman, p. 505) */
  double props(double Tg, double Tp);

  /** @brief  Compute cp using Merricks method (1982) */
  double cp_c(double Tp);
  double cp_ash(double Tp);
  double cp_h(double Tp);
  double g2(double z);

  // labels used for getting required variables later on in the calculation
  const VarLabel* _particle_temperature_varlabel;
  const VarLabel* _gas_temperature_varlabel;
  const VarLabel* _gas_cp_varlabel;
  const VarLabel* _volq_varlabel;
  const VarLabel* _length_varlabel;
  const VarLabel* _weight_varlabel;
  const VarLabel* _rcmass_varlabel;
  const VarLabel* _char_varlabel;
  const VarLabel* _abskg_varlabel;
  const VarLabel* _abskp_varlabel;
  const VarLabel* _charoxiTemp_varlabel;
  const VarLabel* _surfacerate_varlabel;
  const VarLabel* _chargas_varlabel;
  const VarLabel* _devolgas_varlabel;

  Properties* d_props;

  // variables used in problem setup
  double yelem[5];              ///< Fractions of each element in coal (C, H, N, O, S respectively)
  double total_rc;
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


  double _Pr;
  double _sigma;
  double _pi;
  double _Rgas;
  double _RdC;
  double _RdMW;
  double _visc;
  double _MW_avg;
  double _ksi;
  double _Hc0;
  double _Hh0;
  double _rhop_o;
  double _enthalpy_scaling_constant;
  double _weight_scaling_constant;
  double _weight_small;   ///< small weight
  bool   _radiationOn;
  int   _nQuadNode;
  std::string _weight_name;
  std::vector<double> _init_ash;
  std::vector<double> _sizes;


  //bool _radiation;                ///< Radiation flag

}; // end EnthalpyShaddix
} // end namespace Uintah
#endif
