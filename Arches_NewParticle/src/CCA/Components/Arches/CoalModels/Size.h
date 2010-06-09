#ifndef Uintah_Component_Arches_Size_h
#define Uintah_Component_Arches_Size_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    Size
  * @author   Charles Reid
  * @date     April 2010
  *
  * @brief    A parent class for models related to length internal coordinate.
  *           As size models increase the amount of things they're doing, we'll
  *           have to make more Size-derived classes, which contain these effects. 
  *           e.g. Size derived classes for:
  *           - Breakage
  *           - Swelling
  *           - etc.
  *
  */

namespace Uintah{

class Size: public ModelBase {
public: 

  Size( std::string modelName, 
          SimulationStateP& shared_state, 
          const ArchesLabel* fieldLabels,
          vector<std::string> reqICLabelNames, 
          vector<std::string> reqScalarLabelNames,
          int qn );

  virtual ~Size();

  ///////////////////////////////////////////////
  // Initialization stuff

  /** @brief  Grab model-independent length parameters */
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for Size parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for Size parent class, intended to be re-defined by child classes if needed. */
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

  ////////////////////////////////////////////////
  // Model computation

  /** @brief  Calculate the new particle size */
  virtual double calcSize() {
    proc0cout << "Size::calcSize() - returning new particle size" << endl;
    return 0.0;
  };

  /** @brief  Calculate the particle surface area */
  virtual double calcArea() {
    proc0cout << "Size::calcArea() - calculating particle area" << endl;
    return 0.0;
  };

  ///////////////////////////////////////////////////
  // Access functions
  
  /** @brief  Return a string containing the model type ("Size") */
  inline string getType() {
    return "Size"; }

protected:

  double d_lowModelClip; 
  double d_highModelClip; 

  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

}; // end Size
} // end namespace Uintah
#endif
