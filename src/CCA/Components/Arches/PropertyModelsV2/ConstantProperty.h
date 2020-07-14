#ifndef Uintah_Component_Arches_ConstantProperty_h
#define Uintah_Component_Arches_ConstantProperty_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

namespace Uintah{

  template <typename T>
  class ConstantProperty : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    ConstantProperty<T>( std::string task_name, int matl_index );
    ~ConstantProperty<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){
      register_new_variable<T>( m_task_name );
    }

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){};

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    //Build instructions for this (ConstantProperty) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      ConstantProperty* build()
      { return scinew ConstantProperty( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    double m_constant;
    bool m_has_regions;

    struct ConstantGeomContainer{
      double constant;
      bool inverted;
      std::vector<GeometryPieceP> geometry;
    };

    std::vector<ConstantGeomContainer> m_region_info;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ConstantProperty<T>::ConstantProperty( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ConstantProperty<T>::~ConstantProperty(){}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ConstantProperty<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ConstantProperty<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &ConstantProperty<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &ConstantProperty<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &ConstantProperty<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ConstantProperty<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ConstantProperty<T>::loadTaskTimestepInitFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                       , &ConstantProperty<T>::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &ConstantProperty<T>::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &ConstantProperty<T>::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace ConstantProperty<T>::loadTaskRestartInitFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::RESTART_INITIALIZE>( this
                                       , &ConstantProperty<T>::restart_initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &ConstantProperty<T>::restart_initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &ConstantProperty<T>::restart_initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ConstantProperty<T>::problemSetup( ProblemSpecP& db ){


    if ( db->findBlock("regions") ){

      ProblemSpecP db_regions = db->findBlock("regions");

      //geometrically bound constants
      m_has_regions = true;

      for ( ProblemSpecP db_r = db_regions->findBlock("region"); db_r != nullptr;
            db_r = db_r->findNextBlock("region") ){

        ConstantGeomContainer region_info;

        if ( db_r->findBlock("inverted")){
          region_info.inverted = true;
        } else {
          region_info.inverted = false;
        }

        ProblemSpecP db_geom = db_r->findBlock("geom_object");
        GeometryPieceFactory::create( db_geom, region_info.geometry );
        db_r->require("value", region_info.constant);
        m_region_info.push_back( region_info );

      }

    } else {

      //just one global constant
      m_has_regions = false;

      db->require("value", m_constant );

    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ConstantProperty<T>::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ConstantProperty<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    T& property = tsk_info->get_field<T>( m_task_name );
    property.initialize(0.0);

    if ( m_has_regions ){

      for ( auto region_iter = m_region_info.begin(); region_iter != m_region_info.end();
            region_iter++){

        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          //not worried about efficiency since this is only at startup:
          Point p = patch->cellPosition(IntVector(i,j,k));


          for ( auto geom_iter = region_iter->geometry.begin();
                geom_iter != region_iter->geometry.end(); geom_iter++ ){

            GeometryPieceP geom = *geom_iter;

            if ( !region_iter->inverted ){
              if ( geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            } else {
              if ( !geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            }
          }

        });

      }

    } else {

      property.initialize(m_constant);

    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ConstantProperty<T>::register_restart_initialize( AVarInfo& variable_registry , const bool packed_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ConstantProperty<T>::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    T& property = tsk_info->get_field<T>( m_task_name );
    property.initialize(0.0);

    if ( m_has_regions ){

      for ( auto region_iter = m_region_info.begin(); region_iter != m_region_info.end();
            region_iter++){

        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          //not worried about efficiency since this is only at startup:
          Point p = patch->cellPosition(IntVector(i,j,k));


          for ( auto geom_iter = region_iter->geometry.begin();
                geom_iter != region_iter->geometry.end(); geom_iter++ ){

            GeometryPieceP geom = *geom_iter;

            if ( !region_iter->inverted ){
              if ( geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            } else {
              if ( !geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            }
          }

        });

      }

    } else {

      property.initialize(m_constant);

    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ConstantProperty<T>::register_timestep_init( AVarInfo& variable_registry , const bool packed_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_task_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,
                      variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace> void
  ConstantProperty<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    auto property = tsk_info->get_field<T, double, MemSpace>( m_task_name );
    auto old_property = tsk_info->get_field<CT, const double, MemSpace>( m_task_name );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA( int i, int j, int k){
      property(i,j,k) = old_property(i,j,k);
    });

  }
}
#endif
