#ifndef Uintah_Component_Arches_PropertyModelBase_h
#define Uintah_Component_Arches_PropertyModelBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <typeinfo>

/** 
* @class  PropertyModelBase
* @author Jeremy Thornock
* @date   Aug. 2011
* 
* @brief A base class for property models. 
*        
* 
*/ 

namespace Uintah {

class PropertyModelBase{ 

public: 

  PropertyModelBase( std::string prop_name, SimulationStateP& shared_state ); 
  virtual ~PropertyModelBase();

  /** @brief Interface to the input file */ 
  virtual void problemSetup( const ProblemSpecP& db ) = 0; 

  /** @brief Scheduler for the actual property calculation */ 
  virtual void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ) = 0; 

  /** @brief Scheduler for the dummy initialization as required by MPMArches */ 
  virtual void sched_dummyInit( const LevelP& level, SchedulerP& sched ) = 0; 

  /** @brief Scheduler for the initialization of the property */ 
  virtual void sched_initialize( const LevelP& level, SchedulerP& sched ) = 0; 

  /** @brief Returns the property label */ 
  inline const VarLabel* getPropLabel(){
    return _prop_label; };

  /** @brief Returns a vector of extra labels stored for this specific property */ 
  inline const vector<const VarLabel*> getExtraLocalLabels(){
    return _extra_local_labels; }; 

  /** @brief A method for cleaning up property values */ 
  inline void cleanUp() { _has_been_computed = false; };

  /** @brief Returns the boolean to indicate if the model is to be evaluated before or after the table lookup */
  inline bool beforeTableLookUp() { return _before_table_lookup; }; 

  /** @brief Builder class containing instructions on how to build the property model */ 
  class Builder { 

    public: 

      virtual ~Builder() {}

      virtual PropertyModelBase* build() = 0; 

    protected: 

      std::string _name;
  }; 


protected:

  std::string _prop_name;                             ///< User assigned property name
  std::vector<const VarLabel*> _extra_local_labels;   ///< Vector of extra local labels
  std::string _init_type;                             ///< Initialization type
  
  const VarLabel* _prop_label;                        ///< Property varlabel

  bool _has_been_computed;                            ///< To determine if the property has been computed (to avoid computing twice)
  bool _before_table_lookup;                          ///< To determine if the property model is evaluated before the table look up or after. 

  SimulationStateP& _shared_state;                    ///< Uintah shared state

  /** @brief A common intialization proceedure that can be used by all derived types */ 
  template <class phiT > 
  void base_initialize( const Patch* patch, phiT& phi ); 


  // Constant initialization
  double _const_init;                                 ///< Constant for intialization

}; // end PropertyModelBase

template <class phiT > 
void PropertyModelBase::base_initialize( const Patch* patch, phiT& phi ){

  proc0cout << "Initializing property models. " << endl;
  
  if ( _init_type == "constant" ) {

    phi.initialize( _const_init ); 

  } else {
    proc0cout << " For property model: " << _prop_name << endl;
    throw InvalidValue("Initialization type for property model not recognized or supported!", __FILE__, __LINE__); 
  }
}

}  // end namespace Uintah

#endif
