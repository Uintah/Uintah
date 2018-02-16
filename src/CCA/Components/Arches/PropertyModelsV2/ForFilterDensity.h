#ifndef Uintah_Component_Arches_ForFilterDensity_h
#define Uintah_Component_Arches_ForFilterDensity_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  class ForFilterDensity : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    ForFilterDensity( std::string task_name, int matl_index );
    ~ForFilterDensity();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){};

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    //Build instructions for this (ForFilterDensity) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      ForFilterDensity* build()
      { return scinew ForFilterDensity( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    std::string m_density_name;
    std::string m_ff_density_name;
    std::string m_volfraction_name;
    std::string m_fm_name;
    void compute_ff_density( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  };
}

#endif
