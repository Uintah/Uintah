#ifndef Uintah_Component_Arches_ScalarRHS_h
#define Uintah_Component_Arches_ScalarRHS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class Discretization_new; 
  class ScalarRHS : public TaskInterface { 

public: 

    ScalarRHS( std::string task_name, int matl_index ); 
    ~ScalarRHS(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry ); 

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr ); 
         

    //Build instructions for this (ScalarRHS) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      ScalarRHS* build()
      { return scinew ScalarRHS( _task_name, _matl_index ); }

      private: 

      std::string _task_name; 
      int _matl_index; 

    };

private: 

    std::string _rhs_name; 
    std::string _D_name; 
    std::string _Fconv_name; 
    std::string _Fdiff_name; 
    Discretization_new* _disc; 
    std::string _conv_scheme; 

    bool _do_conv; 
    bool _do_diff; 
    bool _do_clip; 

    double _low_clip; 
    double _high_clip; 


    struct SourceInfo{ 
      std::string name; 
      double weight; 
    };
    std::vector<SourceInfo> _source_info; 
  
  };
}
#endif 
