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

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class EnthalpyShaddixBuilder: public ModelBuilder
{
public: 
  EnthalpyShaddixBuilder( const std::string          & modelName,
                             const vector<std::string>  & reqICLabelNames,
                             const vector<std::string>  & reqScalarLabelNames,
                             ArchesLabel          * fieldLabels,
                             SimulationStateP           & sharedState,
                             int qn );

  ~EnthalpyShaddixBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class EnthalpyShaddix: public HeatTransfer {
public: 

  typedef std::map< std::string, CharOxidation*> CharOxiModelMap;
  typedef std::map< std::string, Devolatilization*> DevolModelMap;

  EnthalpyShaddix( std::string modelName, 
                SimulationStateP& shared_state, 
                ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
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
                     DataWarehouse* new_dw );

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


  /////////////////////////////////////////////////
  // Access methods

  /** @brief  Access function for thermal conductivity of particles */
  inline const VarLabel* getabskp(){
    return d_abskp; };  
  
private:

  //////////////////////////////////////////////////
  // Private methods for calculation

  /** @brief  Funtion for calculation of heat capacity (from Merrick) */
  double g1( double z1, double z2);

  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman, p. 505) */
  double props(double Tg, double Tp);

  double calc_enthalpy(double particle_temperature, double rc_mass,
                               double char_mass, double ash_mass);
  double calc_hcint(double Tp);
  double calc_hhint(double Tp);
  double calc_haint(double Tp);

  double get_hc(double Tp);
  double get_hh(double Tp);
  double get_ha(double Tp);
 
  const VarLabel* d_charoxiTempLabel; 
  const VarLabel* d_surfacerateLabel;
  const VarLabel* d_chargasLabel;
  const VarLabel* d_devolgasLabel;
  const VarLabel* d_raw_coal_mass_label;        ///< Label for raw coal mass
  const VarLabel* d_char_mass_label;
  const VarLabel* d_particle_enthalpy_label; ///< Label for particle enthalpy
  const VarLabel* d_particle_length_label;      ///< Label for particle length
  const VarLabel* d_weight_label;               ///< Weight label

  const VarLabel* d_abskp;  ///< Label for thermal conductivity (of the particles, I think???)
  const VarLabel* d_volq_label;
  const VarLabel* d_abskg_label;

  double visc;
  double yelem[5];              ///< Fractions of each element in coal (C, H, N, O, S respectively)
  vector<double>  ash_mass_init;         ///< Initial ash mass
  double rhop;                  ///< Density of particle 

  double Pr;
  double blow;
  double kappa;
  double sigma;
  double pi;
  double ksi;
  double Hc0;
  double Hh0;
  double Ha0;
  double Rgas;
  double MW_avg;

  double ai;
  double bi;
  double hi;
  double Cpci;
  double Cphi;
  double Cpai;
  double xi;
  double Hc;
  double Hh;
  double Ha;

  double d_rc_scaling_constant;   ///< Scaling factor for raw coal
  double d_rh_scaling_constant;
  double d_ash_scaling_constant;  ///< Scaling factor for ash mass
  double d_pl_scaling_constant;   ///< Scaling factor for particle size (length)
  double d_pe_scaling_constant;   ///< Scaling factor for particle temperature

  //bool _radiation;                ///< Radiation flag

}; // end EnthalpyShaddix
} // end namespace Uintah
#endif
