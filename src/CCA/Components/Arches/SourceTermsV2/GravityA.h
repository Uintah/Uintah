#ifndef Uintah_Component_Arches_GravityA_h
#define Uintah_Component_Arches_GravityA_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah{

  class GravityA : public TaskInterface {

public:

    GravityA( std::string task_name, int matl_index );
    ~GravityA();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();
          

    //Build instructions for this (GravityA) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index)      {}
      ~Builder(){}

      GravityA* build()
      { return scinew GravityA( m_task_name, m_matl_index  ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:
    Vector m_gravity;
    std::string m_gx_label;
    std::string m_gy_label;
    std::string m_gz_label;
    std::string m_density_label;
    double m_ref_density;
  };
}
#endif
