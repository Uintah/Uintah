#ifndef Uintah_Component_Arches_SurfaceNormals_h
#define Uintah_Component_Arches_SurfaceNormals_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class Operators;
  class SurfaceNormals : public TaskInterface {

public:

    SurfaceNormals( std::string task_name, int matl_index );
    ~SurfaceNormals();

    void problemSetup( ProblemSpecP& db );

    void register_initialize(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep ){};

    void register_compute_bcs(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
              SpatialOps::OperatorDatabase& opr ){};

    void create_local_labels();


    //Build instructions for this (SurfaceNormals) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      SurfaceNormals* build()
      { return new SurfaceNormals( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    double _value;

  };
}
#endif
