#ifndef Uintah_Component_Arches_CharOxidationSmith_h
#define Uintah_Component_Arches_CharOxidationSmith_h
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
#include <Core/Datatypes/DenseMatrix.h>

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class CharOxidationSmithBuilder: public ModelBuilder
{
public: 
  CharOxidationSmithBuilder( const std::string          & modelName,
                               const std::vector<std::string>  & reqICLabelNames,
                               const std::vector<std::string>  & reqScalarLabelNames,
                               ArchesLabel          * fieldLabels,
                               SimulationStateP           & sharedState,
                               int qn );

  ~CharOxidationSmithBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class CharOxidationSmith: public CharOxidation {
public: 

  CharOxidationSmith( std::string modelName, 
                        SimulationStateP& shared_state,
                        ArchesLabel* fieldLabels,
                        std::vector<std::string> reqICLabelNames,
                        std::vector<std::string> reqScalarLabelNames,
                        int qn );

  ~CharOxidationSmith();

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

  /** @brief   */
  inline void root_function( std::vector<double> &F, DenseMatrix* &dfdrh, std::vector<double> &rh_l, double &p_T, double &cg, std::vector<double> &oxid_mass_frac, double &MW, double &r_devol, double &gas_rho, double &p_diam, std::vector<double> &Sh, double &w, double &p_area );
  inline void invert_2_2( DenseMatrix* &dfdrh );

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

  // new stuff
  std::vector< const VarLabel*> _oxidizer_varlabels;
  std::vector< const VarLabel*> _reaction_rate_varlabels;
  bool _use_simple_invert;  
  double oxidizer_MW; // 
  std::string oxidizer_name; // 
  double a; // 
  double e; // 
  double phi; // 
  std::vector<std::string> _oxid_l;
  std::vector<double> _MW_l;
  std::vector<double> _a_l;
  std::vector<double> _e_l;
  std::vector<double> _phi_l;
  std::vector<double> _D_oxid_mix_l;
  int _NUM_reactions; // 
  double _R_cal; // [cal/ (K mol) ]
  double _R; // [J/ (K mol) ]
  double _Mh; // 12 kg carbon / kmole carbon
  double _S;

  DenseMatrix* dfdrh;
  
}; // end CharOxidationSmith
} // end namespace Uintah
#endif
