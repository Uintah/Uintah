#ifndef Uintah_Component_Arches_ContinuityPredictor_h
#define Uintah_Component_Arches_ContinuityPredictor_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class ContinuityPredictor : public TaskInterface {

public:

    ContinuityPredictor( std::string task_name, int matl_index );
    ~ContinuityPredictor();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    template <typename EXECUTION_SPACE, typename MEMORY_SPACE>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, void* stream );

    void create_local_labels();

    //Build instructions for this (ContinuityPredictor) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      ContinuityPredictor* build()
      { return scinew ContinuityPredictor( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::string m_label_drhodt; 
    std::string m_label_balance; 

    //void compute_density(  const Patch* patch, ArchesTaskInfoManager* tsk_info);


  };
}
#endif
