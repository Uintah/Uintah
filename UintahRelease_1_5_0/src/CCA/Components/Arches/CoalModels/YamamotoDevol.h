#ifndef Uintah_Component_Arches_YamamotoDevol_h
#define Uintah_Component_Arches_YamamotoDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    YamamotoDevol
  * @author   Julien Pedel
  * @date     June 2011
  *
  * @brief    A class for calculating the DQMOM model term for the 
  *           Yamamoto coal devolatilization model.
  *
  * Empirical devolatilization model which can reproduce more complex models
  * such as CPD or DAEM. Model based on the work of K. Yamamoto:
  * K. Yamamoto, T. Murota, T. Okazaki, M. Tanigushi, Proceedings of the
  * Combustion Institute 33 (2011), 1771-1778
  * 
  * The Builder is required because of the Model Factory; the Factory needs
  * some way to create the model term and register it.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{

class ArchesLabel;
class YamamotoDevolBuilder: public ModelBuilder 
{
public: 
  YamamotoDevolBuilder( const std::string          & modelName,
                                const vector<std::string>  & reqICLabelNames,
                                const vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel          * fieldLabels,
                                SimulationStateP           & sharedState,
                                int qn );

  ~YamamotoDevolBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class YamamotoDevol: public Devolatilization {
public: 

  YamamotoDevol( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  ~YamamotoDevol();

  ////////////////////////////////////////////////
  // Initialization method

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  // No initVars() method because no special variables needed

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
                     DataWarehouse* new_dw );

private:

  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_weight_label;
  const VarLabel* d_particle_temperature_label;

  vector<double>  rc_mass_init;
  double Av;
  double Ev;
  double Yv;
  double c0;
  double c1;
  double c2;
  double c3;
  double c4;
  double c5;
  double kv;
  double Fv;
  double Xv;
  
  double R;         ///< Ideal gas constant
  
  bool compute_part_temp; ///< Boolean: is particle temperature computed? 
                          //   (if not, gas temp = particle temp)
  bool part_temp_from_enth;
  bool compute_char_mass;

  double d_rc_scaling_factor;   ///< Scaling factor for raw coal internal coordinate
  double d_rh_scaling_factor;
  double d_pt_scaling_factor;   ///< Scaling factor for particle temperature internal coordinate
  double rateMax;
  double testVal_part;
  double testVal_gas;
  double testVal_char;

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
