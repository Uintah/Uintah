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
  const VarLabel* _rcmass_varlabel;
  const VarLabel* _char_varlabel;
  const VarLabel* _length_varlabel;
  const VarLabel* _particle_temperature_varlabel;
  const VarLabel* _weight_varlabel;
  const VarLabel* _gas_temperature_varlabel;
  const VarLabel* _O2_varlabel;
  const VarLabel* _CO2_varlabel;
  const VarLabel* _H2O_varlabel;
  const VarLabel* _N2_varlabel;
  const VarLabel* _MW_varlabel;
  
  std::vector<double>  Shaddix_char_coefficients;
  double _As;
  double _Es;
  double _n;
  double _R;
  double _HF_CO2;
  double _HF_CO;
  double char_reaction_rate_;
  double char_production_rate_;
  double particle_temp_rate_;
  double PO2_inf;
  double PO2_surf;
  double CO2CO;
  double OF;
  double ks;
  double q;
  double _WO2;
  double _WCO2;
  double _WH2O;
  double _WN2;
  double _WC;
  double _D1;
  double _D2;
  double _D3;
  double _T0;
  double d_tol;
  double delta;
  double Conc;
  double DO2;
  double gamma;
  int d_totIter;
  int NIter;
  double f1;
  double f2;
  double f3;
  double lower_bound;
  double upper_bound;
  int icount;
  double _pi;
  double _char_scaling_constant;   ///< Scaling factor for char internal coordinate
  double _weight_scaling_constant;   ///< Scaling factor for weight 
  double _weight_small;   ///< small weight 

}; // end CharOxidationShaddix
} // end namespace Uintah
#endif
