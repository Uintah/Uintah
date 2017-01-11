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
