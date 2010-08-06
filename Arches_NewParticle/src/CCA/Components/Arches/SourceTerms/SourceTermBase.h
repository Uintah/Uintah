#ifndef Uintah_Component_Arches_SourceTermBase_h
#define Uintah_Component_Arches_SourceTermBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <typeinfo>

namespace Uintah {

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

class ArchesLabel;
class SourceTermBase{ 

public: 

  SourceTermBase( std::string srcName, 
                  SimulationStateP& sharedState, 
                  vector<std::string> reqLabelNames );

  SourceTermBase( std::string srcName, 
                  SimulationStateP& sharedState, 
                  vector<std::string> reqLabelNames,
                  ArchesLabel* fieldLabels );

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

  /** @brief Builder class containing instructions on how to build the property model */ 
  class Builder { 

    public: 

      virtual ~Builder() {}

      virtual SourceTermBase* build() = 0; 

    protected: 

      std::string _name;
  }; 

  /* @brief   Return a string containing the model type (pure virtual) */
  virtual string getType() = 0;


protected:

  std::string d_srcName;                             ///< User assigned source name 
  std::string _init_type;                            ///< Initialization type. 
  const VarLabel* d_srcLabel;                        ///< Source varlabel
  bool d_labelSchedInit;                             ///< Boolean to clarify if a "computes" or "requires" is needed
  SimulationStateP& d_sharedState;                   ///< Local copy of sharedState
  vector<std::string> d_requiredLabels;              ///< Vector of required labels
  vector<const VarLabel*> d_extraLocalLabels;        ///< Extra labels that might be useful for storage
  ArchesLabel* d_fieldLabels;                        ///< Field labels (not used by default, only used when source term needs them)


}; // end SourceTermBase
}  // end namespace Uintah

#endif
