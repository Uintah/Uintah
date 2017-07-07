#ifndef Uintah_Component_Arches_OneDWallHT_h
#define Uintah_Component_Arches_OneDWallHT_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class OneDWallHT : public TaskInterface {

public:

    OneDWallHT( std::string task_name, int matl_index );
    ~OneDWallHT();

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
          
    inline void newton_solve( double&, const double&, const double&, const double&, const double&);

    //Build instructions for this (OneDWallHT) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      OneDWallHT* build()
      { return scinew OneDWallHT( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    double _sigma_constant;
    std::string _incident_hf_label;
    std::string _emissivity_label;
    std::string _Tshell_label;
    std::string _wall_resistance_label;

  };
}
#endif
