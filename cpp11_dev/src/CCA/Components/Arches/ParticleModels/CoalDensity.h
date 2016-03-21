#ifndef Uintah_Component_Arches_CoalDensity_h
#define Uintah_Component_Arches_CoalDensity_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{ 

  class Operators; 
  class CoalDensity : public TaskInterface { 

public: 

    CoalDensity( std::string task_name, int matl_index, const int N );
    ~CoalDensity(); 

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
               

    //Build instructions for this (CoalDensity) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, const int N ) : _task_name(task_name), _matl_index(matl_index), _Nenv(N){}
      ~Builder(){}

      CoalDensity* build()
      { return new CoalDensity( _task_name, _matl_index, _Nenv ); }

      private: 

      std::string _task_name; 
      int _matl_index;
      int _Nenv;

    };

private: 

    bool _const_size;
    int _Nenv; 
    double _value; 
    double _rhop_o;
    double _pi;
    double _raw_coal_mf;
    double _char_mf;
    double _ash_mf;

    std::vector<double> _init_ash;
    std::vector<double> _init_rawcoal;
    std::vector<double> _init_char;
    std::vector<double> _sizes;
    std::vector<double> _denom; 

    std::string _diameter_base_name;
    std::string _rawcoal_base_name;
    std::string _char_base_name; 

    struct CoalAnalysis{ 
      double C;
      double H; 
      double O; 
      double N; 
      double S; 
      double CHAR; 
      double ASH; 
      double H2O; 
    };
  };
}
#endif 
