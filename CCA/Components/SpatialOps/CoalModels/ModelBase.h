#ifndef Uintah_Component_SpatialOps_ModelBase_h
#define Uintah_Component_SpatialOps_ModelBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationStateP.h>

//===============================================================

/** 
* @class  ModelBase
* @author Jeremy Thornock
* @date   Nov, 5 2008
* 
* @brief A base class for models for a transport 
*        equation. 
* 
*/ 

namespace Uintah {

class ModelBase{ 

public: 

  ModelBase( std::string modelName, SimulationStateP& sharedState, 
             vector<std::string> reqLabelNames, int qn );
  virtual ~ModelBase();

  /** @brief Input file interface */
  virtual void problemSetup(const ProblemSpecP& db, int qn) = 0;  

  /** @brief Returns a list of required variables from the DW for scheduling */
  //virtual void getDwVariableList() = 0;

  /** @brief Schedule the source for computation. */
  virtual void sched_computeModel(const LevelP& level, SchedulerP& sched, int timeSubStep ) = 0;

  /** @brief Actually compute the source. */
  virtual void computeModel( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw ) = 0;

  /** @brief reinitialize the flags that tells the scheduler if the varLabel needs a compute or a modifies. */
  // Note I need two of these flags; 1 for scheduling and 1 for actual execution.
  inline void reinitializeLabel(){ 
    d_labelSchedInit  = false; };

  inline const VarLabel* getModelLabel(){
    return d_modelLabel; };

protected:
  std::string d_modelName; 
  vector<string> d_icLabels; //All internal coordinate labels (from DQMOM factory) needed to compute this model
  vector<string> d_scalarLabels; // All scalar labels (from scalarFactory) needed to compute this model
  const VarLabel* d_modelLabel; //The label storing the value of this model
  int d_timeSubStep;
  SimulationStateP& d_sharedState; 

  bool d_labelSchedInit;
  bool d_labelActualInit;   

  int d_quadNode; 

}; // end ModelBase
}  // end namespace Uintah

#endif
