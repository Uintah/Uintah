#ifndef Uintah_Component_Arches_UpdateParticlePosition_h
#define Uintah_Component_Arches_UpdateParticlePosition_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class UpdateParticlePosition : public TaskInterface { 

public: 

    UpdateParticlePosition( std::string task_name, int matl_index ); 
    ~UpdateParticlePosition(); 

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

    void create_local_labels(){}

    //Build instructions for this (UpdateParticlePosition) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      UpdateParticlePosition* build()
      { return scinew UpdateParticlePosition( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    std::string _px_name; 
    std::string _py_name; 
    std::string _pz_name; 

    std::string _u_name; 
    std::string _v_name; 
    std::string _w_name; 
  
  };
}
#endif 
