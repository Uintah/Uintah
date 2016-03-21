#ifndef Uintah_Component_Arches_DepositionVelocity_h
#define Uintah_Component_Arches_DepositionVelocity_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/SimulationState.h>

namespace Uintah{ 
  
  class Operators; 
  class DepositionVelocity : public TaskInterface { 

public: 

    DepositionVelocity( std::string task_name, int matl_index, const int N, SimulationStateP shared_state ); 
    ~DepositionVelocity(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ); 

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){}; 

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
               

    //Build instructions for this (DepositionVelocity) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, const int N, SimulationStateP shared_state ) : _task_name(task_name), _matl_index(matl_index), _Nenv(N), _shared_state(shared_state){}
      ~Builder(){}

      DepositionVelocity* build()
      { return new DepositionVelocity( _task_name, _matl_index, _Nenv, _shared_state ); }

      private: 

      std::string _task_name; 
      int _matl_index;
      int _Nenv;
      SimulationStateP _shared_state;

    };

private: 

      int _Nenv;
      SimulationStateP _shared_state;
      std::vector<IntVector> _d; 
      std::vector<IntVector> _fd; 
      std::string _cellType_name; 
      std::string _ratedepx_name;
      std::string _ratedepy_name;
      std::string _ratedepz_name;
      std::string _rhoP_name;
      std::string _dep_vel_rs_name;
      std::string _dep_vel_rs_start_name;
      std::string _new_time_name;
      double _t_interval; // the time interval required for a steady-state thermal profile.
      double _new_time;

  };
}
#endif 
