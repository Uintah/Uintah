#ifndef Uintah_Component_Arches_PressureEqn_h
#define Uintah_Component_Arches_PressureEqn_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class SolverInterface;
  class SolverParameters;

  class PressureEqn : public TaskInterface {

public:

    PressureEqn( std::string task_name, int matl_index, SimulationStateP shared_state );
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

      Builder( std::string task_name, int matl_index, SimulationStateP shared_state ) : m_task_name(task_name),
               m_matl_index(matl_index), m_state(shared_state){}
      ~Builder(){}

      PressureEqn* build()
      { return scinew PressureEqn( m_task_name, m_matl_index, m_state ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      SimulationStateP m_state;

    };

    void solve( const LevelP& level, SchedulerP& sched, const int time_substep );

    void set_solver( SolverInterface* solver ){ m_hypreSolver = solver; }

    void setup_solver( ProblemSpecP& db );

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
    SolverParameters* m_hypreSolver_parameters;

    SimulationStateP m_sharedState;

    IntVector m_periodic_vector;

  };
}
#endif
