#ifndef Uintah_Component_Arches_YDragModel_h
#define Uintah_Component_Arches_YDragModel_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

#define YDIM
#define ZDIM

//===========================================================================

/**
  * @class    YDragModel
  * @author   Julien Pedel
  * @date     September 2009
  *
  * @brief    A class for calculating the two-way coupling between
  *           particle velocities and the gas phase velocities.
  *
  */

//---------------------------------------------------------------------------

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class YDragModelBuilder: public ModelBuilder
{
public: 
  YDragModelBuilder( const std::string          & modelName, 
                        const std::vector<std::string>  & reqICLabelNames,
                        const std::vector<std::string>  & reqScalarLabelNames,
                        ArchesLabel          * fieldLabels,
                        SimulationStateP           & sharedState,
                        int qn );
  ~YDragModelBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class YDragModel: public ModelBase {
public: 

  YDragModel( std::string modelName, 
                 SimulationStateP& shared_state, 
                 ArchesLabel* fieldLabels,
                 std::vector<std::string> reqICLabelNames,
                 std::vector<std::string> reqScalarLabelNames,
                 int qn );

  ~YDragModel();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /////////////////////////////////////////////////
  // Model computation methods

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

  ///////////////////////////////////////////////
  // Access methods

  inline std::string getType() {
    return "Constant"; }


private:

  const VarLabel* d_particle_length_label;
  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_char_mass_label;
  const VarLabel* d_particle_velocity_label;
  const VarLabel* d_gas_velocity_label;
  const VarLabel* d_weight_label;

  std::vector<double>  as_received;
  std::vector<double>  particle_sizes;
  std::vector<double>  rc_mass_init;
  std::vector<double>  ash_mass_init;
  std::vector<double>  char_mass_init;
  std::vector<double>  vol_dry;
  std::vector<double>  mass_dry;
  Vector gravity;
  double kvisc;
  double rhop;
  double total_rc;
  double total_dry;
  double rc_mass_frac;
  double char_mass_frac;
  double ash_mass_frac;
  double d_lowModelClip;
  double d_highModelClip;
  double d_pl_scaling_factor;
  double d_rcmass_scaling_factor;
  double d_charmass_scaling_factor;
  double d_pv_scaling_factor;
  double d_w_scaling_factor;
  double d_yvel_scaling_factor;
  double d_w_small; // "small" clip value for zero weights

  double pi;

  double velMag( Vector& X ) {
    double mag = X.x() * X.x();
#ifdef YDIM
    mag += X.y() * X.y();
#endif
#ifdef ZDIM
    mag += X.z() * X.z();
#endif
    mag = sqrt( mag );
    return mag;
  }

};
} // end namespace Uintah
#endif

