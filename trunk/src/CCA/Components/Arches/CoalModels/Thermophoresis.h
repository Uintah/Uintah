#ifndef Uintah_Component_Arches_Thermophoresis_h
#define Uintah_Component_Arches_Thermophoresis_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
  * @class    Thermophoresis
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

class ThermophoresisBuilder: public ModelBuilder 
{
public: 
  ThermophoresisBuilder( const std::string               & modelName,
                                const std::vector<std::string>  & reqICLabelNames,
                                const std::vector<std::string>  & reqScalarLabelNames,
                                ArchesLabel                     * fieldLabels,
                                MaterialManagerP                & materialManager,
                                int qn );

  ~ThermophoresisBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class Thermophoresis: public ModelBase {
public: 

  Thermophoresis( std::string modelName, 
                         MaterialManagerP& materialManager, 
                         ArchesLabel* fieldLabels,
                         std::vector<std::string> reqICLabelNames,
                         std::vector<std::string> reqScalarLabelNames,
                         int qn );

  ~Thermophoresis();

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
  
  inline std::string getType() {
    return "Constant"; }

private:

  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman, p. 505) */
  double props(double Tg, double Tp);
  
  const VarLabel* _particle_density_varlabel;
  const VarLabel* _gas_temperature_varlabel;
  const VarLabel* _length_varlabel;
  const VarLabel* _volFraction_varlabel;
  const VarLabel* _thp_weighted_scaled_varlabel;
  const VarLabel* _weight_scaled_varlabel;
  const VarLabel* _particle_temperature_varlabel;

  
  double _vel_scaling_constant;   ///< Scaling factor for raw coal internal coordinate
  double _weight_scaling_constant;   ///< Scaling factor for weight 
  double _weight_small;   ///< small weight 
  
  int _dir; 
  double _visc; 
  double _rkp; 
  double _pi; 
  double _C_tm; 
  double _C_t; 
  double _C_m; 
  double _Adep; 
  IntVector _cell_minus; 
  IntVector _cell_plus; 

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
