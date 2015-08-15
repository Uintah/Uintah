#ifndef Uintah_Component_Arches_DensityPredictor_h
#define Uintah_Component_Arches_DensityPredictor_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class DensityPredictor : public TaskInterface { 

public: 

    DensityPredictor( std::string task_name, int matl_index ); 
    ~DensityPredictor(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){} 

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){} 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr ); 

    void create_local_labels(); 

    //Build instructions for this (DensityPredictor) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      DensityPredictor* build()
      { return new DensityPredictor( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    bool _use_exact_guess; 
    double _rho0; 
    double _rho1; 
    std::string _f_name; 
    std::vector<std::string> _mass_sources; 

  
  };
}
#endif 
