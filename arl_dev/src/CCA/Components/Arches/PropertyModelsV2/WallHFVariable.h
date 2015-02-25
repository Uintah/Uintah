#ifndef Uintah_Component_Arches_WallHFVariable_h
#define Uintah_Component_Arches_WallHFVariable_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/SimulationState.h>

namespace Uintah{ 

  class Operators; 
  class WallHFVariable : public TaskInterface { 

public: 

    WallHFVariable( std::string task_name, int matl_index, SimulationStateP& shared_state ); 
    ~WallHFVariable(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels(); 


    //Build instructions for this (WallHFVariable) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, SimulationStateP& shared_state ) 
        : _task_name(task_name), _matl_index(matl_index), _shared_state(shared_state){}
      ~Builder(){}

      WallHFVariable* build()
      { return scinew WallHFVariable( _task_name, _matl_index, _shared_state ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      SimulationStateP _shared_state; 

    };

private: 

    std::string _flux_x; 
    std::string _flux_y; 
    std::string _flux_z; 

    int _f;

    SimulationStateP _shared_state; 


  };
}
#endif 
