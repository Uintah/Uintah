#ifndef Uintah_Component_Arches_GasKineticEnergy_h
#define Uintah_Component_Arches_GasKineticEnergy_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class GasKineticEnergy : public TaskInterface {

public:

    GasKineticEnergy( std::string task_name, int matl_index );
    ~GasKineticEnergy();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (GasKineticEnergy) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      GasKineticEnergy* build()
      { return scinew GasKineticEnergy( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    std::string m_u_vel_name; 
    std::string m_v_vel_name; 
    std::string m_w_vel_name; 
    std::string m_kinetic_energy; 
    double m_max_ke; 
    //void compute_density(  const Patch* patch, ArchesTaskInfoManager* tsk_info);


  };
}
#endif
