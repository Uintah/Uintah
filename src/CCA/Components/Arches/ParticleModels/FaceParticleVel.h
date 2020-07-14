#ifndef Uintah_Component_Arches_FaceParticleVel_h
#define Uintah_Component_Arches_FaceParticleVel_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

namespace Uintah{

  //IT is the independent variable type
  template <typename T>
  class FaceParticleVel : public TaskInterface {

public:

    FaceParticleVel<T>( std::string task_name, int matl_index, const std::string var_name );
    ~FaceParticleVel<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );



    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string base_var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name){}
      ~Builder(){}

      FaceParticleVel* build()
      { return scinew FaceParticleVel<T>( m_task_name, m_matl_index, _base_var_name ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

private:

    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    const std::string _base_var_name;
    std::string _temperature_var_name;
    std::string _conc_var_name;
    ArchesCore::INTERPOLANT m_int_scheme;

    int m_N;                 //<<< The number of "environments"
    int m_ghost_cells;
    std::string up_root;
    std::string vp_root;
    std::string wp_root;
    std::string up_face;
    std::string vp_face;
    std::string wp_face;

  };

  //Function definitions:

  template <typename T>
  void FaceParticleVel<T>::create_local_labels(){

    for ( int i = 0; i < m_N; i++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,i);
      std::string vp_face_i = ArchesCore::append_env(vp_face,i);
      std::string wp_face_i = ArchesCore::append_env(wp_face,i);
      register_new_variable<FXT>( up_face_i );
      register_new_variable<FYT>( vp_face_i );
      register_new_variable<FZT>( wp_face_i );
    }

  }


  template <typename T>
  FaceParticleVel<T>::FaceParticleVel( std::string task_name, int matl_index,
                                                      const std::string base_var_name ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name){
  }

  template <typename T>
  FaceParticleVel<T>::~FaceParticleVel()
  {}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FaceParticleVel<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FaceParticleVel<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &FaceParticleVel<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &FaceParticleVel<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &FaceParticleVel<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FaceParticleVel<T>::loadTaskEvalFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                       , &FaceParticleVel<T>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &FaceParticleVel<T>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &FaceParticleVel<T>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FaceParticleVel<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FaceParticleVel<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  void FaceParticleVel<T>::problemSetup( ProblemSpecP& db ){

    m_N = ArchesCore::get_num_env(db,ArchesCore::DQMOM_METHOD);
    up_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL );
    vp_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_YVEL );
    wp_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ZVEL );
    up_face = "face_pvel_x";
    vp_face = "face_pvel_y";
    wp_face = "face_pvel_z";

    std::string scheme = "second";
    m_int_scheme = ArchesCore::get_interpolant_from_string( scheme );

    m_ghost_cells = 1;
    if (m_int_scheme== ArchesCore::FOURTHCENTRAL){
      m_ghost_cells = 2;
    }
  }

  //======INITIALIZATION:
  template <typename T>
  void FaceParticleVel<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < m_N; i++ ){
      std::string up_face_i = ArchesCore::append_env(up_face,i);
      std::string vp_face_i = ArchesCore::append_env(vp_face,i);
      std::string wp_face_i = ArchesCore::append_env(wp_face,i);
      register_variable( up_face_i, ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( vp_face_i, ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( wp_face_i, ArchesFieldContainer::COMPUTES, variable_registry );

      std::string up_i = ArchesCore::append_env(up_root,i);
      std::string vp_i = ArchesCore::append_env(vp_root,i);
      std::string wp_i = ArchesCore::append_env(wp_root,i);

      register_variable( up_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( vp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( wp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry );

    }

  }

  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void FaceParticleVel<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){


    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );


    for ( int ienv = 0; ienv < m_N; ienv++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,ienv);
      std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
      std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);

      auto up_f = tsk_info->get_field<FXT, double, MemSpace>(up_face_i);
      auto vp_f = tsk_info->get_field<FYT, double, MemSpace>(vp_face_i);
      auto wp_f = tsk_info->get_field<FZT, double, MemSpace>(wp_face_i);

      std::string up_i = ArchesCore::append_env(up_root,ienv);
      std::string vp_i = ArchesCore::append_env(vp_root,ienv);
      std::string wp_i = ArchesCore::append_env(wp_root,ienv);

      auto up = tsk_info->get_field<CT, const double, MemSpace>(up_i);
      auto vp = tsk_info->get_field<CT, const double, MemSpace>(vp_i);
      auto wp = tsk_info->get_field<CT, const double, MemSpace>(wp_i);

      Uintah::parallel_initialize( execObj, 0.0, up_f, vp_f, wp_f );

      ArchesCore::doInterpolation( execObj, range, up_f, up, -1,  0,  0, m_int_scheme );
      ArchesCore::doInterpolation( execObj, range, vp_f, vp,  0, -1,  0, m_int_scheme );
      ArchesCore::doInterpolation( execObj, range, wp_f, wp,  0,  0, -1, m_int_scheme );

    }
  }

  //======TIME STEP INITIALIZATION:
  template <typename T>
  void FaceParticleVel<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename T>
  template <typename ExecSpace, typename MemSpace> void
  FaceParticleVel<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

  //======TIME STEP EVALUATION:
  template <typename T>
  void FaceParticleVel<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    for ( int ienv = 0; ienv < m_N; ienv++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,ienv);
      std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
      std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);
      register_variable( up_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );
      register_variable( vp_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );
      register_variable( wp_face_i, ArchesFieldContainer::COMPUTES, variable_registry , m_task_name  );

      std::string up_i = ArchesCore::append_env(up_root,ienv);
      std::string vp_i = ArchesCore::append_env(vp_root,ienv);
      std::string wp_i = ArchesCore::append_env(wp_root,ienv);

      register_variable( up_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( vp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( wp_i, ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }

  }

  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void FaceParticleVel<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

    for ( int ienv = 0; ienv < m_N; ienv++ ){

      std::string up_face_i = ArchesCore::append_env(up_face,ienv);
      std::string vp_face_i = ArchesCore::append_env(vp_face,ienv);
      std::string wp_face_i = ArchesCore::append_env(wp_face,ienv);

      auto up_f = tsk_info->get_field<FXT, double, MemSpace>(up_face_i);
      auto vp_f = tsk_info->get_field<FYT, double, MemSpace>(vp_face_i);
      auto wp_f = tsk_info->get_field<FZT, double, MemSpace>(wp_face_i);

      std::string up_i = ArchesCore::append_env(up_root,ienv);
      std::string vp_i = ArchesCore::append_env(vp_root,ienv);
      std::string wp_i = ArchesCore::append_env(wp_root,ienv);

      auto up = tsk_info->get_field<CT, const double, MemSpace>(up_i);
      auto vp = tsk_info->get_field<CT, const double, MemSpace>(vp_i);
      auto wp = tsk_info->get_field<CT, const double, MemSpace>(wp_i);

      GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(0, 1);
      GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(0, 1);
      GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(0, 1);

      Uintah::BlockRange range_x( low_fx_patch_range, high_fx_patch_range );
      Uintah::BlockRange range_y( low_fy_patch_range, high_fy_patch_range );
      Uintah::BlockRange range_z( low_fz_patch_range, high_fz_patch_range );

      Uintah::parallel_initialize( execObj, 0.0, up_f, vp_f, wp_f );

      ArchesCore::doInterpolation( execObj, range_x, up_f, up, -1,  0,  0, m_int_scheme );
      ArchesCore::doInterpolation( execObj, range_y, vp_f, vp,  0, -1,  0, m_int_scheme );
      ArchesCore::doInterpolation( execObj, range_z, wp_f, wp,  0,  0, -1, m_int_scheme );

    }
  }
}
#endif
