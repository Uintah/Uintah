#ifndef Uintah_Component_Arches_SourceTermBase_h
#define Uintah_Component_Arches_SourceTermBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/ArchesMaterial.h>

//===============================================================

/** 
* @class  SourceTermBase
* @author Jeremy Thornock
* @date   Nov, 5 2008
* 
* @brief A base class for source terms for a transport 
*        equation. 
* 
*/ 

namespace Uintah {

class SourceTermBase{ 

public: 

  SourceTermBase( std::string srcName, SimulationStateP& sharedState, 
                  vector<std::string> reqLabelNames );
  virtual ~SourceTermBase();

  /** @brief Input file interface */
  virtual void problemSetup(const ProblemSpecP& db) = 0;  

  /** @brief Returns a list of required variables from the DW for scheduling */
  //virtual void getDwVariableList() = 0;

  /** @brief Schedule the source for computation. */
  virtual void sched_computeSource(const LevelP& level, SchedulerP& sched, int timeSubStep ) = 0;

  /** @brief Actually compute the source. */
  virtual void computeSource( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw, 
                              int timeSubStep ) = 0;

  /** @brief Get the labels for the MPMARCHES dummy solve. */
  virtual void sched_dummyInit( const LevelP& level, SchedulerP& sched ) = 0;

  /** @brief reinitialize the flags that tells the scheduler if the varLabel needs a compute or a modifies. */
  // Note I need two of these flags; 1 for scheduling and 1 for actual execution.
  inline void reinitializeLabel(){ 
    d_labelSchedInit  = false; };

  inline const VarLabel* getSrcLabel(){
    return d_srcLabel; };

  inline const vector<const VarLabel*> getExtraLocalLabels(){
    return d_extraLocalLabels; }; 

protected:
  std::string d_srcName; 
  const VarLabel* d_srcLabel; //The label storing the value of this source term
  SimulationStateP& d_sharedState; 
  vector<std::string> d_requiredLabels;   //All labels needed to compute this source term  
  vector<const VarLabel*> d_extraLocalLabels; //This array will hold local labels to the specific source term 
                                          // and will be used to obtain vars from the DW for initialization 

  bool d_labelSchedInit;

}; // end SourceTermBase
}  // end namespace Uintah

#endif
