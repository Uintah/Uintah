#ifndef Uintah_Component_Arches_Burnout_h
#define Uintah_Component_Arches_Burnout_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class Burnout : public TaskInterface {

public:

    Burnout( std::string task_name, int matl_index, const int N ) :
      TaskInterface(task_name, matl_index), _Nenv(N) {
        //_pi = std::acos(-1.0);
        //_Rgas = 8314.3;
    }
    ~Burnout(){}

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();


    //Build instructions for this (Burnout) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, const int N ) : m_task_name(task_name), m_matl_index(matl_index), _Nenv(N){}
      ~Builder(){}

      Burnout* build()
      { return scinew Burnout( m_task_name, m_matl_index, _Nenv ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      int _Nenv;

    };

private:

    int _Nenv;
    std::vector<double> m_weight_scaling_constants; 
    std::vector<double> m_rc_scaling_constants; 
    std::vector<double> m_char_scaling_constants; 
    std::string m_vol_fraction_name;
    std::vector<std::string > m_weight_names;
    std::vector<std::string > m_rc_names;
    std::vector<std::string > m_char_names;
  };
}
#endif
