#ifndef Uintah_Component_Arches_KobayashiSarofimDevol_h
#define Uintah_Component_Arches_KobayashiSarofimDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    KobayashiSarofimDevol
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
class KobayashiSarofimDevolBuilder: public ModelBuilder 
{
public: 
  KobayashiSarofimDevolBuilder( const std::string          & modelName,
                                const vector<std::string>  & reqICLabelNames,
                                const vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel          * fieldLabels,
                                SimulationStateP           & sharedState,
                                int qn );

  ~KobayashiSarofimDevolBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class KobayashiSarofimDevol: public Devolatilization {
public: 

  KobayashiSarofimDevol( std::string modelName, 
                         SimulationStateP& shared_state, 
                         ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  ~KobayashiSarofimDevol();

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

  double A1;        ///< Pre-exponential factors for devolatilization rate constants
  double A2;        ///< Pre-exponential factors for devolatilization rate constants
  double E1;        ///< Activation energy for devolatilization rate constant
  double E2;        ///< Activation energy for devolatilization rate constant
  double Y1_;       ///< Volatile fraction from proximate analysis
  double Y2_;       ///< Fraction devolatilized at higher temperatures (often near unity)
  double k1;        ///< Rate constant for devolatilization reaction 1
  double k2;        ///< Rate constant for devolatilization reaction 2
  
  double R;         ///< Ideal gas constant
  
  bool compute_part_temp; ///< Boolean: is particle temperature computed? 
                          //   (if not, gas temp = particle temp)
  bool compute_char_mass;

  double rateMax;
  double d_rc_scaling_factor;   ///< Scaling factor for raw coal internal coordinate
  double d_rh_scaling_factor;
  double d_pt_scaling_factor;   ///< Scaling factor for particle temperature internal coordinate
  double testVal_part;
  double testVal_gas;
  double testVal_char;
}; // end ConstSrcTerm
} // end namespace Uintah
#endif
