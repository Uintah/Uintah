#ifndef Uintah_Component_Arches_TemplatedSampleTask_h
#define Uintah_Component_Arches_TemplatedSampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/Nebo.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

namespace Uintah{ 

  template <typename T>
  class TemplatedSampleTask : public TaskInterface { 

public: 

    TemplatedSampleTask<T>( std::string task_name, int matl_index ); 
    ~TemplatedSampleTask<T>(); 

    void problemSetup( ProblemSpecP& db ); 

    //Build instructions for this (TemplatedSampleTask) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      TemplatedSampleTask* build()
      { return scinew TemplatedSampleTask<T>( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

protected: 

    void register_all_variables( std::vector<VariableInformation>& variable_registry ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );
  

    void eval( const Patch* patch, UintahVarMap& var_map, 
               ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr, const int time_substep ); 

    void initialize( const Patch* patch, UintahVarMap& var_map, 
                     ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr );

private:

  
  };

  //Function definitions: 

  template <typename T>
  TemplatedSampleTask<T>::TemplatedSampleTask( std::string task_name, int matl_index ) : 
  TaskInterface( task_name, matl_index ){

    // This needs to be done to set the variable type 
    // for this function. All templated tasks should do this. 
    set_type<T>(); 
  
  }

  template <typename T>
  TemplatedSampleTask<T>::~TemplatedSampleTask()
  {
    const VarLabel* V1 = VarLabel::find("templated_variable"); 
    VarLabel::destroy(V1); 

  }

  template <typename T>
  void TemplatedSampleTask<T>::problemSetup( ProblemSpecP& db ){ 
  }


  template <typename T>
  void TemplatedSampleTask<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 
  
  }
  
  //This is the work for the task.  First, get the variables. Second, do the work! 
  template <typename T> 
  void TemplatedSampleTask<T>::initialize( const Patch* patch, UintahVarMap& var_map, 
                          ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr ){ 
  }


  template <typename T> 
  void TemplatedSampleTask<T>::register_all_variables( std::vector<VariableInformation>& variable_registry ){
   
    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", _mytype, LOCAL_COMPUTES, 0, NEWDW, variable_registry ); 
  
  }

  template <typename T>
  void TemplatedSampleTask<T>::eval(const Patch* patch, UintahVarMap& var_map, 
                                    ConstUintahVarMap& const_var_map, SpatialOps::OperatorDatabase& opr, const int time_substep ){

    using namespace SpatialOps;
    using SpatialOps::operator *; 

    T* const field = get_so_field<T>( "templated_variable", var_map, patch, 0, *this ); 

    *field <<= 24.0;
  
  }

}
#endif 
