#ifndef Uintah_Component_Arches_DensityStar_h
#define Uintah_Component_Arches_DensityStar_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class DensityStar : public TaskInterface {

public:

    DensityStar( std::string task_name, int matl_index );
    ~DensityStar();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (DensityStar) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      DensityStar* build()
      { return scinew DensityStar( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::string m_label_density;
    std::string m_label_densityStar;
    int _time_order;
    std::vector<double> _alpha;
    std::vector<double> _beta;
    std::vector<double> _time_factor;

  };
}
#endif
