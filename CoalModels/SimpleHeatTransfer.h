#ifndef Uintah_Component_Arches_SimpleHeatTransfer_h
#define Uintah_Component_Arches_SimpleHeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    SimpleHeatTransfer
  * @author   Julien Pedel, Jeremy Thornock, Charles Reid
  * @date     October 2009
  *
  * @brief    A simple heat transfer model for coal paticles.
  *           (This needs a more descriptive name)
  *
  */

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class SimpleHeatTransferBuilder: public ModelBuilder
{
public: 
  SimpleHeatTransferBuilder( const std::string          & modelName,
                             const vector<std::string>  & reqICLabelNames,
                             const vector<std::string>  & reqScalarLabelNames,
                             const ArchesLabel          * fieldLabels,
                             SimulationStateP           & sharedState,
                             int qn );

  ~SimpleHeatTransferBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class SimpleHeatTransfer: public HeatTransfer {
public: 

  SimpleHeatTransfer( std::string modelName, 
                SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
                int qn );

  ~SimpleHeatTransfer();

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief Schedule the initialization of some special/local variables */ 
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local variables */
  void initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  /** @brief  Schedule the dummy solve for MPMArches - see ExplicitSolver::noSolve */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy solve */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

// use getGasSourceLabel() instead (defined in ModelBase)
//  inline const VarLabel* getGasHeatLabel(){
//    return d_gasLabel; };

  /** @brief  Access function for thermal conductivity (of particles, I think???) */
  inline const VarLabel* getabskp(){
    return d_abskp; };  
  
  /** @brief  Access function for radiation flag (on/off) */
  inline const bool getRadiationFlag(){
    return d_radiation; };   

  /** @brief  What does this do? The name isn't descriptive */
  double g1( double z);

  /** @brief  Calculate heat capacity of particle (I think?) */
  double heatcp(double Tp);
  
  /** @brief  Calculate heat capacity of ash */
  double heatap(double Tp);

  /** @brief  Calculate gas properties of N2 at atmospheric pressure (see Holman p. 505) */
  double props(double Tg, double Tp);

private:

  const VarLabel* d_raw_coal_mass_label;        ///< Label for raw coal mass
  const VarLabel* d_ash_mass_label;             ///< Label for ash mass
  const VarLabel* d_particle_temperature_label; ///< Label for particle temperature
  const VarLabel* d_particle_length_label;      ///< Label for particle length
  const VarLabel* d_weight_label;               ///< Weight label
  const VarLabel* smoothTfield;                 ///< Gas-particle temperature field
                                                //   (Could be improved if it used moment 0 for weighting)

  const VarLabel* d_abskp; ///< Label for thermal conductivity (of the particles, I think???)
  bool d_ash;

  double d_lowModelClip; 
  double d_highModelClip; 

  double visc;
  double yelem[5];
  double rhop;
  double d_rc_scaling_factor;   ///< Scaling factor for raw coal
  double d_ash_scaling_factor;  ///< Scaling factor for ash mass
  double d_pl_scaling_factor;   ///< Scaling factor for particle size (length)
  double d_pt_scaling_factor;   ///< Scaling factor for particle temperature

}; // end SimpleHeatTransfer
} // end namespace Uintah
#endif
