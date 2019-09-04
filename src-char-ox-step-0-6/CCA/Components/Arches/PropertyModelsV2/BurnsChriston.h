#ifndef Uintah_Component_Arches_PropertyModelV2_BurnsChriston_h
#define Uintah_Component_Arches_PropertyModelV2_BurnsChriston_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

  class BoundaryCondition_new;
  class BurnsChriston : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    BurnsChriston( std::string task_name, int matl_index ) :
    TaskInterface( task_name, matl_index ){
      m_notSetMin = Point(SHRT_MAX, SHRT_MAX, SHRT_MAX);
      m_notSetMax = Point(SHRT_MIN, SHRT_MIN, SHRT_MIN);
    }
    ~BurnsChriston(){}

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_inf ){}

    void create_local_labels();


    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      BurnsChriston* build()
      { return scinew BurnsChriston( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::string m_abskg_name;

    Point  m_min;
    Point  m_max;
    Point  m_notSetMin;
    Point  m_notSetMax;

  };
}
#endif
