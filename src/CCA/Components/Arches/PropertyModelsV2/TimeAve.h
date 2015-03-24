#ifndef Uintah_Component_Arches_TimeAve_h
#define Uintah_Component_Arches_TimeAve_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Grid/SimulationState.h>

namespace Uintah{ 

  class Operators; 
  class TimeAve : public TaskInterface { 

public: 

    TimeAve( std::string task_name, int matl_index, SimulationStateP& shared_state ); 
    ~TimeAve(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels(); 


    //Build instructions for this (TimeAve) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, SimulationStateP& shared_state ) 
        : _task_name(task_name), _matl_index(matl_index), _shared_state(shared_state){}
      ~Builder(){}

      TimeAve* build()
      { return scinew TimeAve( _task_name, _matl_index, _shared_state ); }

      private: 

      std::string _task_name; 
      int _matl_index; 
      SimulationStateP _shared_state; 

    };

private: 

    std::vector<const VarLabel*> ave_sum_labels; 
    std::vector<const VarLabel*> ave_flux_sum_labels; 

    //single variables
    std::vector<std::string> ave_sum_names; 
    std::vector<std::string> base_var_names; 

    std::string rho_name;

    //fluxes
    struct FluxInfo{ 
      std::string phi; 
      bool do_phi;
    };
    std::vector<std::string> ave_x_flux_sum_names; 
    std::vector<std::string> ave_y_flux_sum_names; 
    std::vector<std::string> ave_z_flux_sum_names; 
    std::vector<FluxInfo>    flux_sum_info; 

    SimulationStateP _shared_state; 


  };
}
#endif 
