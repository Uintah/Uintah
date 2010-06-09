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
          const ArchesLabel* fieldLabels,
          vector<std::string> reqICLabelNames, 
          vector<std::string> reqScalarLabelNames,
          int qn );

  virtual ~ParticleDensity();

  ///////////////////////////////////////////////
  // Initialization stuff

  /** @brief  Grab model-independent length parameters */
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for ParticleDensity parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for ParticleDensity parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Actually do dummy initialization (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  virtual void sched_computeParticleDensity( const LevelP& level,
                                             SchedulerP&   sched,
                                             int           timeSubStep ) = 0;

  virtual void computeParticleDensity( const ProcessorGroup* pc,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw ) = 0;

  ////////////////////////////////////////////////
  // Model computation

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

  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

  const VarLabel* d_density_label;

  vector<double> ash_mass;

  int numQuadNodes;
  double pi;

}; // end ParticleDensity
} // end namespace Uintah
#endif

