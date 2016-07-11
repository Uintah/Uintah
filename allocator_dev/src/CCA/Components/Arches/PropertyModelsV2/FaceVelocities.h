#ifndef Uintah_Component_Arches_FaceVelocities_h
#define Uintah_Component_Arches_FaceVelocities_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  class FaceVelocities : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    FaceVelocities( std::string task_name, int matl_index );
    ~FaceVelocities();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    void register_initialize( VIVec& variable_registry );

    void register_timestep_init( VIVec& variable_registry ){};

    void register_restart_initialize( VIVec& variable_registry ){};

    void register_timestep_eval( VIVec& variable_registry, const int time_substep );

    void register_compute_bcs( VIVec& variable_registry, const int time_substep ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                             SpatialOps::OperatorDatabase& opr ){};

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){};

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

    //Build instructions for this (FaceVelocities) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      FaceVelocities* build()
      { return new FaceVelocities( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;

    std::vector<std::string> m_vel_names;

  };
}

#endif
