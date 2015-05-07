#ifndef Uintah_Component_Arches_CoalMassClip_h
#define Uintah_Component_Arches_CoalMassClip_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class CoalMassClip : public TaskInterface { 

public: 

    CoalMassClip( std::string task_name, int matl_index, const int N ); 
    ~CoalMassClip(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ); 

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){}; 

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}; 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels(); 

    const std::string get_env_name( const int i, const std::string base_name ){ 
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
               

    //Build instructions for this (CoalMassClip) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, const int N ) : _task_name(task_name), _matl_index(matl_index), _N(N){}
      ~Builder(){}

      CoalMassClip* build()
      { return scinew CoalMassClip( _task_name, _matl_index, _N ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      const int _N; 

    };

private: 

    int _Nenv; 

    std::string _raw_coal_base; 
    std::string _char_base; 

  };
}
#endif 
