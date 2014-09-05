#ifndef Uintah_Component_Arches_ParticleDensity_h
#define Uintah_Component_Arches_ParticleDensity_h
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/Parallel.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    ParticleDensity
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    A parent class for particle density models.  The two most
  *           simple examples would be constant density and constant size.
  *           However, more may be added later.
  *
  */

namespace Uintah{

class ParticleDensity: public ModelBase {
public: 

  ParticleDensity( std::string modelName, 
          SimulationStateP& shared_state, 
          ArchesLabel* fieldLabels,
          vector<std::string> reqICLabelNames, 
          vector<std::string> reqScalarLabelNames,
          int qn );

  virtual ~ParticleDensity();

  ///////////////////////////////////////////////
  // Initialization stuff

  /** @brief  Grab model-independent length parameters */
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */ 
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief Schedule the dummy initialization required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy initialization */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation

  virtual void sched_computeParticleDensity( const LevelP& level,
                                             SchedulerP&   sched,
                                             int           timeSubStep ) = 0;

  virtual void computeParticleDensity( const ProcessorGroup* pc,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw, 
                                       int timeSubStep ) = 0;

  /** @brief  Calculate the new particle size */
  virtual double calcSize() = 0;

  /** @brief  Calculate the particle surface area */
  virtual double calcArea() = 0;

  /** @brief  Calculate the particle density */
  virtual double calcParticleDensity() = 0;

  ///////////////////////////////////////////////////
  // Access functions
  
  std::string getType() {
    return "ParticleDensity"; }

  const VarLabel* getParticleDensityLabel() {
    return d_density_label; };

protected:

  double d_lowModelClip; 
  double d_highModelClip; 
  int numQuadNodes;
  double pi;

  const VarLabel* d_density_label;  ///< Variable label for particle density (this is the quantity the the model calculates)
  const VarLabel* d_weight_label;   ///< Variable label for weight
  const VarLabel* d_length_label;   ///< Label for particle length internal coordinate

  double d_w_small;                 ///< "small" clip value for weights; if weight < d_w_small, no model value is calculated
  double d_w_scaling_constant;      ///< Scaling constant for weight
  double d_length_scaling_constant; ///< Scaling constant for particle length internal coordinate
  double d_length_low;              ///< Low clip value for length (if applicable)
  double d_length_hi;               ///< High clip value for length (if applicable)

  bool d_doLengthLowClip;  ///< Boolean: do low clipping for length?
  bool d_doLengthHighClip; ///< Boolean: do high clipping for length?
  bool d_useLength;        ///< Boolean: use particle length internal coordinate?

  // Constant value (if user specifies value of length should be constant)
  double d_length_constant_value;

  // Constant bool
  bool d_constantLength; ///< Boolean: is the length a constant fixed value? (as opposed to an internal coordinate)

}; // end ParticleDensity
} // end namespace Uintah
#endif

