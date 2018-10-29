#ifndef Uintah_Component_Arches_PressureEqn_h
#define Uintah_Component_Arches_PressureEqn_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class SolverInterface;

  class PressureEqn : public TaskInterface {

public:

    PressureEqn( std::string task_name, int matl_index, MaterialManagerP materialManager );
    ~PressureEqn();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (PressureEqn) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) : m_task_name(task_name),
               m_matl_index(matl_index), m_materialManager(materialManager){}
      ~Builder(){}

      PressureEqn* build()
      { return scinew PressureEqn( m_task_name, m_matl_index, m_materialManager ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      MaterialManagerP m_materialManager;

    };

    void solve( const LevelP& level, SchedulerP& sched, const int time_substep );

    void set_solver( SolverInterface* solver ){ m_hypreSolver = solver; }

    void setup_solver( ProblemSpecP& db );
    
    
    void sched_Initialize( const LevelP& level, SchedulerP& sched );
    
    void sched_restartInitialize( const LevelP& level, SchedulerP& sched );

private:

    std::string m_eps_name;
    std::string m_xmom_name;
    std::string m_ymom_name;
    std::string m_zmom_name;
    std::string m_pressure_name;
    std::string m_density_name;
    std::string m_drhodt_name;

    bool m_enforceSolvability;
    bool m_use_mms_drhodt;

    SolverInterface* m_hypreSolver;

    MaterialManagerP m_materialManager;

    IntVector m_periodic_vector;

  };
}
#endif
