#ifndef Uintah_Component_Arches_TemplatedSampleTask_h
#define Uintah_Component_Arches_TemplatedSampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/Nebo.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

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

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ){} 

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ); 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr ); 

    void create_local_labels(); 

private:

    VAR_TYPE _mytype; 

  
  };

  //Function definitions: 

  template <typename T>
  TemplatedSampleTask<T>::TemplatedSampleTask( std::string task_name, int matl_index ) : 
  TaskInterface( task_name, matl_index ){

    VarTypeHelper<T> helper; 
    _mytype = helper.get_vartype(); 
  
  }

  template <typename T>
  TemplatedSampleTask<T>::~TemplatedSampleTask()
  {
    const VarLabel* V1 = VarLabel::find("templated_variable"); 
    VarLabel::destroy(V1); 

  }

  template <typename T>
  void TemplatedSampleTask<T>::problemSetup( ProblemSpecP& db ){ 

    _do_ts_init_task = false; 
    _do_bcs_task = false; 

  }

  template <typename T>
  void TemplatedSampleTask<T>::create_local_labels(){ 

    register_new_variable( "templated_variable", _mytype ); 

  }


  template <typename T>
  void TemplatedSampleTask<T>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", _mytype, LOCAL_COMPUTES, 0, NEWDW, variable_registry ); 
  
  }
  
  //This is the work for the task.  First, get the variables. Second, do the work! 
  template <typename T> 
  void TemplatedSampleTask<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                           SpatialOps::OperatorDatabase& opr ){ 

    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef SpatialOps::SpatFldPtr<T> SVolFP;

    SVolFP field = tsk_info->get_so_field<T>( "templated_variable" ); 
    *field <<= 3.2;

  }


  template <typename T> 
  void TemplatedSampleTask<T>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){
   
    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", _mytype, COMPUTES, 0, NEWDW, variable_registry, time_substep ); 
  
  }

  template <typename T>
  void TemplatedSampleTask<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                     SpatialOps::OperatorDatabase& opr ){ 

    using namespace SpatialOps;
    using SpatialOps::operator *; 
    typedef SpatialOps::SpatFldPtr<T> SVolFP;

    SVolFP field = tsk_info->get_so_field<T>( "templated_variable" ); 

    *field <<= 24.0;
  
  }

  template <typename T>
  void TemplatedSampleTask<T>::register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 
  }
  
  template <typename T>
  void TemplatedSampleTask<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                                            SpatialOps::OperatorDatabase& opr ){ 
  
  }
}
#endif 
