#ifndef Uintah_Component_Arches_YamamotoDevol_h
#define Uintah_Component_Arches_YamamotoDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    YamamotoDevol
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
class YamamotoDevolBuilder: public ModelBuilder 
{
public: 
  YamamotoDevolBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                MaterialManagerP                & materialManager,
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
                         MaterialManagerP& materialManager, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~YamamotoDevol();

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
  const VarLabel* _char_varlabel;
  const VarLabel* _weight_varlabel;
  const VarLabel* _particle_temperature_varlabel;

  std::vector<double>  Yamamoto_coefficients;  
  std::vector<double>  particle_sizes;
  std::vector<double>  ash_mass_init;
  std::vector<double>  char_mass_init;
  std::vector<double>  vol_dry;
  std::vector<double>  mass_dry;
  std::vector<double>  rc_mass_init;
  double _Av;        ///< Pre-exponential factors for devolatilization rate constants
  double _Ev;        ///< Activation energy for devolatilization rate constant
  double _Yv;       ///< Volatile fraction from proximate analysis
  double _c0;
  double _c1;
  double _c2;
  double _c3;
  double _c4;
  double _c5;
  double rhop;
  double total_rc;
  double total_dry;
  double rc_mass_frac;
  double char_mass_frac;
  double ash_mass_frac;
  
  double _pi;
  double _R;
  
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
