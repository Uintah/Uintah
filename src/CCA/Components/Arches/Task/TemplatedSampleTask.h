#ifndef Uintah_Component_Arches_TemplatedSampleTask_h
#define Uintah_Component_Arches_TemplatedSampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

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

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

  };

  //Function definitions:

  template <typename T>
  TemplatedSampleTask<T>::TemplatedSampleTask( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
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
  void TemplatedSampleTask<T>::create_local_labels(){

    register_new_variable<T>( "templated_variable" );

  }


  template <typename T>
  void TemplatedSampleTask<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
                                                    const bool packed_tasks ){

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void TemplatedSampleTask<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& field = *(tsk_info->get_uintah_field<T>( "templated_variable" ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      field(i,j,k) = 3.2;
    });

  }


  template <typename T>
  void TemplatedSampleTask<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    //FUNCITON CALL     STRING NAME(VL)     TYPE       DEPENDENCY    GHOST DW     VR
    register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  }

  template <typename T>
  void TemplatedSampleTask<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& field = *(tsk_info->get_uintah_field<T>( "templated_variable" ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      field(i,j,k) = 23.4;
    });

  }

  template <typename T>
  void TemplatedSampleTask<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){
  }

  template <typename T>
  void TemplatedSampleTask<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

}
#endif
