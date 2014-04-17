#ifndef Uintah_Component_Arches_TaskInterface_h
#define Uintah_Component_Arches_TaskInterface_h

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <string>
#include <vector>
#include <boost/foreach.hpp>

//===============================================================

/** 
* @class  Task Interface for Arches
* @author Jeremy Thornock
* @date   2014
* 
* @brief Serves as the interface to a standard uintah task. 
* 
**/ 

//===============================================================

namespace Uintah{ 

  class Task; 
  class VarLabel; 
  class Level;  

  class TaskInterface{ 

public: 

    enum VAR_DEPEND { COMPUTES, MODIFIES, REQUIRES, LOCAL_COMPUTES, LOCAL_MODIFIES, LOCAL_REQUIRES };
    enum WHICH_DW { OLDDW, NEWDW, LATEST };
    enum VAR_TYPE { CC_INT, CC_DOUBLE, CC_VEC, FACEX, FACEY, FACEZ, SUM, MAX, MIN };

    struct VariableInformation { 

      std::string name;
      VAR_TYPE    type; 
      VAR_DEPEND  depend; 
      WHICH_DW    dw;
      int         nGhost;
      bool        dw_inquire; 
      const VarLabel* label; 
      Task::WhichDW uintah_task_dw; 
      Ghost::GhostType ghost_type; 
    
    };

    /** @brief Default constructor **/ 
    TaskInterface( std::string take_name, int matl_index ); 

    /** @brief Default destructor **/ 
    virtual ~TaskInterface();

    /** @brief Print task name. **/ 
    void print_task_name(){ 
      std::cout << "Task: " << _task_name << std::endl; 
    }

    /** @brief Registers all variables with pertinent information for the 
     *         uintah dw interface **/ 
    virtual void register_all_variables( std::vector<VariableInformation>& variable_registry ) = 0; 

    /** @brief Matches labels to variables in the registry **/ 
    void resolve_labels( std::vector<VariableInformation>& variable_registry ); 


    /** @brief Add this task to the Uintah task scheduler. (Overloaded)
     * Start with a predefined variable registry **/ 
    void schedule_task( const LevelP& level, 
                        SchedulerP& sched, 
                        const MaterialSet* matls,
                        std::vector<VariableInformation>& variable_registry,
                        int time_substep );

    /** @brief Add this task to the Uintah task scheduler **/ 
    void schedule_task( const LevelP& level, 
                        SchedulerP& sched, 
                        const MaterialSet* matls,
                        int time_substep );

    /** @brief The actual task interface function that references the 
     *         derived class implementation **/ 
    void do_task( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw, 
                  std::vector<VariableInformation> variable_registry, 
                  int time_substep );


protected: 

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/ 
    void register_variable( std::string name, 
                            VAR_TYPE type, 
                            VAR_DEPEND dep, 
                            int nGhost, 
                            WHICH_DW dw, 
                            std::vector<VariableInformation>& var_reg );

    /** @brief Builds a struct for each variable containing all pertinent uintah
     * DW information **/ 
    void register_variable_work( std::string name, 
                                 VAR_TYPE type, 
                                 VAR_DEPEND dep, 
                                 int nGhost, 
                                 WHICH_DW dw, 
                                 std::vector<VariableInformation>& var_reg );

    /** ()()()() Task grid variable storage ()()()() **/ 
    template<class T>
    struct VarContainer{ 
      T* variable; 
    };

    typedef std::map<std::string, VarContainer<CCVariable<double> > > CCDoubleVarMap; 
    typedef std::map<std::string, GridVariableBase* > UintahVarMap; 
    typedef std::map<std::string, constVariableBase<GridVariableBase>* > ConstUintahVarMap; 

    /** @brief Resolves the DW fields with the dependency **/ 
    void resolve_fields( DataWarehouse* old_dw, 
                         DataWarehouse* new_dw, 
                         const Patch* patch, 
                         std::vector<VariableInformation>& variable_registry, 
                         UintahVarMap& var_map, 
                         ConstUintahVarMap& const_var_map,
                         const int time_substep );

    /** @brief The actual work done within the derived class **/ 
    virtual void eval( const Patch* patch, UintahVarMap& var_map, ConstUintahVarMap& const_var_map ) = 0; 

    /** @brief Return the grid variable by string name for non-const computing **/ 
    template<class T>
    T* get_var(std::string name, UintahVarMap& var_map); 

    /** @brief Return the grid variable by string name for non-const computing **/ 
    template<class T>
    T* get_const_var(std::string name, ConstUintahVarMap& const_var_map); 

    std::string _task_name; 
    const int _matl_index; 
   
private: 

    /** @brief Performs all DW get*,allocateAndPut, etc.. for all variables for this 
     *         task. **/
    template<class T>
    void resolve_field_modifycompute( DataWarehouse* old_dw, DataWarehouse* new_dw, T* field, VariableInformation& info, const Patch* patch, const int time_substep );

    /** @brief Performs all DW get*,allocateAndPut, etc.. for all variables for this 
     *         task. **/
    template<class T>
    void resolve_field_requires( DataWarehouse* old_dw, DataWarehouse* new_dw, T& field, VariableInformation& info, const Patch* patch, const int time_substep );

  
  };

  //====================================================================================
  // GRID VARIABLE ACCESS
  //====================================================================================
  template<class T>
  inline T* TaskInterface::get_var(std::string name, UintahVarMap& var_map ){ };

  template<class T>
  inline T* TaskInterface::get_const_var(std::string name, ConstUintahVarMap& const_var_map){ }; 

  //CCVARIABLE DOUBLE 
  template<>
  inline CCVariable<double>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<CCVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CC_DOUBLE) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constCCVariable<double>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constCCVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST CC_DOUBLE) "+name, __FILE__, __LINE__); 

  }
  //CCVARIABLE INT
  template<>
  inline CCVariable<int>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<CCVariable<int>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CC_INT) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constCCVariable<int>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constCCVariable<int>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST CC_DOUBLE) "+name, __FILE__, __LINE__); 

  }
  //CCVARIABLE VECTOR
  template<>
  inline CCVariable<Vector>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<CCVariable<Vector>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CC_VECTOR) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constCCVariable<Vector>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constCCVariable<Vector>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST CC_VECTOR) "+name, __FILE__, __LINE__); 

  }
  //SFCXVARIABLE DOUBLE
  template<>
  inline SFCXVariable<double>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<SFCXVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (FACEX) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constSFCXVariable<double>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constSFCXVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST FACEX) "+name, __FILE__, __LINE__); 

  }
  //SFCYVARIABLE DOUBLE
  template<>
  inline SFCYVariable<double>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<SFCYVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (FACEY) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constSFCYVariable<double>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constSFCYVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST FACEY) "+name, __FILE__, __LINE__); 

  }
  //SFCZVARIABLE DOUBLE
  template<>
  inline SFCZVariable<double>* TaskInterface::get_var( std::string name, UintahVarMap& var_map ){ 

    UintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<SFCZVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (FACEZ) "+name, __FILE__, __LINE__); 

  }

  template<>
  inline constSFCZVariable<double>* TaskInterface::get_const_var( std::string name, ConstUintahVarMap& var_map ){ 

    ConstUintahVarMap::iterator itr = var_map.find(name);    

    if ( itr != var_map.end() ) return dynamic_cast<constSFCZVariable<double>* >(itr->second); 

    throw InvalidValue("Arches Task Error: Cannot resolve grid variable (CONST FACEZ) "+name, __FILE__, __LINE__); 

  }

}

#endif 
