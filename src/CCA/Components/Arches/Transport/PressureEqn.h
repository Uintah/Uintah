#ifndef Uintah_Component_Arches_PressureEqn_h
#define Uintah_Component_Arches_PressureEqn_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class PressureEqn : public TaskInterface {

public:

    PressureEqn( std::string task_name, int matl_index );
    ~PressureEqn();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (PressureEqn) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name),
               m_matl_index(matl_index){}
      ~Builder(){}

      PressureEqn* build()
      { return scinew PressureEqn( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    std::string m_eps_name;
    std::string m_xmom_name;
    std::string m_ymom_name;
    std::string m_zmom_name;

  };
}
#endif
