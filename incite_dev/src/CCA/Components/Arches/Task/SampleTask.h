#ifndef Uintah_Component_Arches_SampleTask_h
#define Uintah_Component_Arches_SampleTask_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class SampleTask : public TaskInterface {

public:

    SampleTask( std::string task_name, int matl_index );
    ~SampleTask();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ); 

    void create_local_labels(){

      register_new_variable<CCVariable<double> >("a_sample_field");
      register_new_variable<CCVariable<double> >("a_result_field");

    };

    //Build instructions for this (SampleTask) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      SampleTask* build()
      { return scinew SampleTask( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    double _value;

  };
}
#endif
