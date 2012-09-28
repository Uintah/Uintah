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

// FIXME: more descriptive name
/**
  * @class    CharOxidationShaddix
  * @author   Julien Pedel
  * @date     Feb 2011
  *
  * @brief    A char oxidation model for coal paticles.
  *           (This needs a more descriptive name)
  *
  */

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class CharOxidationShaddixBuilder: public ModelBuilder
{
public: 
  CharOxidationShaddixBuilder( const std::string          & modelName,
                             const vector<std::string>  & reqICLabelNames,
                             const vector<std::string>  & reqScalarLabelNames,
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
public: 

  CharOxidationShaddix( std::string modelName, 
                SimulationStateP& shared_state, 
                ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
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
                     DataWarehouse* new_dw );

  /////////////////////////////////////////////////
  // Access methods

private:

  const VarLabel* d_devolCharLabel;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_raw_coal_mass_label;        ///< Label for raw coal mass
  const VarLabel* d_particle_temperature_label; ///< Label for particle temperature
  const VarLabel* d_particle_length_label;      ///< Label for particle length
  const VarLabel* d_weight_label;               ///< Weight label
 
  double As;
  double Es;
  double n;
  double A1;
  double A2;
  double E1;
  double E2;
  double k1;
  double k2;
  double R;
  double HF_CO2;
  double HF_CO;
  double char_reaction_rate_;
  double char_production_rate_;
  double gas_char_rate_;
  double particle_temp_rate_;
  double PO2_inf;
  double PO2_surf;
  double CO2CO;
  double OF;
  double ks;
  double q;
  double WO2;
  double WCO2;
  double WH2O;
  double WN2;
  double WC;
  double D1;
  double D2;
  double D3;
  double T0;
  double d_tol;
  double delta;
  double Conc;
  double DO2;
  double gamma;
  int d_totIter;
  double f1;
  double f2;
  double f3;
  double lower_bound;
  double upper_bound;
  int icount;
  double pi;
  double rateMax;
  double d_rh_scaling_constant;
  double d_rc_scaling_constant;   ///< Scaling factor for raw coal
  double d_pl_scaling_constant;   ///< Scaling factor for particle size (length)
  double d_pt_scaling_constant;   ///< Scaling factor for particle temperature
  bool   part_temp_from_enth;

}; // end CharOxidationShaddix
} // end namespace Uintah
#endif
