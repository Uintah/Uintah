#ifndef Uintah_Component_Arches_SampleTask_h
#define Uintah_Component_Arches_SampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class SampleTask : public TaskInterface { 

public: 

    SampleTask( std::string task_name, int matl_index ); 
    ~SampleTask(); 

    void register_all_variables( std::vector<VariableInformation>& variable_registry ); 


    void eval( const Patch* patch, UintahVarMap& var_map, ConstUintahVarMap& const_var_map ); 
  
  };
}
#endif 
