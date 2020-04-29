#include <CCA/Components/Arches/TurbulenceModels/SGSforTransport.h>

namespace Uintah{

  //--------------------------------------------------------------------------------------------------
  SGSforTransport::SGSforTransport( std::string task_name, int matl_index ) :
    TaskInterface( task_name, matl_index ) {
      // Create SGS stress
      m_SgsStress_names.resize(9);
      m_SgsStress_names[0] = "ucell_xSgsStress";
      m_SgsStress_names[1] = "ucell_ySgsStress";
      m_SgsStress_names[2] = "ucell_zSgsStress";
      m_SgsStress_names[3] = "vcell_xSgsStress";
      m_SgsStress_names[4] = "vcell_ySgsStress";
      m_SgsStress_names[5] = "vcell_zSgsStress";
      m_SgsStress_names[6] = "wcell_xSgsStress";
      m_SgsStress_names[7] = "wcell_ySgsStress";
      m_SgsStress_names[8] = "wcell_zSgsStress";
      // register source terms
      m_fmom_source_names.resize(3);
      m_fmom_source_names[0]="FractalXSrc";
      m_fmom_source_names[1]="FractalYSrc";
      m_fmom_source_names[2]="FractalZSrc";

    }

  //--------------------------------------------------------------------------------------------------
  SGSforTransport::~SGSforTransport(){
  }

  //--------------------------------------------------------------------------------------------------
  TaskAssignedExecutionSpace SGSforTransport::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  TaskAssignedExecutionSpace SGSforTransport::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &SGSforTransport::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &SGSforTransport::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &SGSforTransport::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  TaskAssignedExecutionSpace SGSforTransport::loadTaskEvalFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                       , &SGSforTransport::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &SGSforTransport::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &SGSforTransport::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  TaskAssignedExecutionSpace SGSforTransport::loadTaskTimestepInitFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                       , &SGSforTransport::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       //, &SGSforTransport::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &SGSforTransport::timestep_init<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  TaskAssignedExecutionSpace SGSforTransport::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::problemSetup( ProblemSpecP& db ){

      using namespace Uintah::ArchesCore;
      // u, v , w velocities
      m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, "uVelocitySPBC" );
      m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, "vVelocitySPBC" );
      m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, "wVelocitySPBC" );
      m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

      m_rhou_vel_name = ArchesCore::default_uMom_name;
      m_rhov_vel_name = ArchesCore::default_vMom_name;
      m_rhow_vel_name = ArchesCore::default_wMom_name;

      m_cc_u_vel_name = m_u_vel_name + "_cc";
      m_cc_v_vel_name = m_v_vel_name + "_cc";
      m_cc_w_vel_name = m_w_vel_name + "_cc";


    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::create_local_labels(){
      // U-cell labels
      register_new_variable<SFCXVariable<double> >("FractalXSrc");
      //V-CELL LABELS:
      register_new_variable<SFCYVariable<double> >("FractalYSrc");
      //W-CELL LABELS:
      register_new_variable<SFCZVariable<double> >("FractalZSrc");


    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry , const bool packed_tasks){
      register_variable("FractalXSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      register_variable("FractalYSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      register_variable("FractalZSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

    }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void SGSforTransport::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
      SFCXVariable<double>& FractalXSrc= tsk_info->get_field<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>& FractalYSrc= tsk_info->get_field<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>& FractalZSrc= tsk_info->get_field<SFCZVariable<double> >("FractalZSrc");

      FractalXSrc.initialize(0.0);
      FractalYSrc.initialize(0.0);
      FractalZSrc.initialize(0.0);
    }
  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry , const bool packed_tasks){
      register_variable("FractalXSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      register_variable("FractalYSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      register_variable("FractalZSrc", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

    }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void SGSforTransport::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
      SFCXVariable<double>& FractalXSrc= tsk_info->get_field<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>& FractalYSrc= tsk_info->get_field<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>& FractalZSrc= tsk_info->get_field<SFCZVariable<double> >("FractalZSrc");

      FractalXSrc.initialize(0.0);
      FractalYSrc.initialize(0.0);
      FractalZSrc.initialize(0.0);
    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry, const int time_substep , const bool packed_tasks){

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry, m_task_name );
      }
      for (auto iter = m_fmom_source_names.begin(); iter != m_fmom_source_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
      }

    }

  //--------------------------------------------------------------------------------------------------
  template <typename ExecSpace, typename MemSpace>
  void SGSforTransport::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
      Vector Dx=patch->dCell();
      //  double dx=Dx.x(); double dy=Dx.y(); double dz=Dx.z();
      // double vol = Dx.x()*Dx.y()*Dx.z();
      double Area_NS =Dx.x()*Dx.z();
      double Area_EW =Dx.y()*Dx.z();
      double Area_TB =Dx.x()*Dx.y();
      double densitygas=1.0;//kg/m3
      double cellvol=Dx.x()*Dx.y()*Dx.z();
      // Subgrid stress
      constSFCXVariable<double>& ucell_xSgsStress = tsk_info->get_field<constSFCXVariable<double> >("ucell_xSgsStress");
      constSFCXVariable<double>& ucell_ySgsStress = tsk_info->get_field<constSFCXVariable<double> >("ucell_ySgsStress");
      constSFCXVariable<double>& ucell_zSgsStress = tsk_info->get_field<constSFCXVariable<double> >("ucell_zSgsStress");
      constSFCYVariable<double>& vcell_xSgsStress = tsk_info->get_field<constSFCYVariable<double> >("vcell_xSgsStress");
      constSFCYVariable<double>& vcell_ySgsStress = tsk_info->get_field<constSFCYVariable<double> >("vcell_ySgsStress");
      constSFCYVariable<double>& vcell_zSgsStress = tsk_info->get_field<constSFCYVariable<double> >("vcell_zSgsStress");
      constSFCZVariable<double>& wcell_xSgsStress = tsk_info->get_field<constSFCZVariable<double> >("wcell_xSgsStress");
      constSFCZVariable<double>& wcell_ySgsStress = tsk_info->get_field<constSFCZVariable<double> >("wcell_ySgsStress");
      constSFCZVariable<double>& wcell_zSgsStress = tsk_info->get_field<constSFCZVariable<double> >("wcell_zSgsStress");
      SFCXVariable<double>& FractalXSrc= tsk_info->get_field<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>& FractalYSrc= tsk_info->get_field<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>& FractalZSrc= tsk_info->get_field<SFCZVariable<double> >("FractalZSrc");

      Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

      Uintah::parallel_for( range, [&](int i, int j, int k){

          // compute the source term for the finite volume method.
          //   X-momentum
          FractalXSrc(i,j,k)=-densitygas/cellvol*((ucell_xSgsStress(i+1,j,k)-ucell_xSgsStress(i,j,k))*Area_EW+(ucell_ySgsStress(i,j+1,k) -ucell_ySgsStress(i,j,k))*Area_NS+(ucell_zSgsStress(i,j,k+1)-ucell_zSgsStress(i,j,k))*Area_TB);
          //// Y-momentum
          FractalYSrc(i,j,k)=-densitygas/cellvol*((vcell_xSgsStress(i+1,j,k)-vcell_xSgsStress(i,j,k))*Area_EW+(vcell_ySgsStress(i,j+1,k) -vcell_ySgsStress(i,j,k))*Area_NS+(vcell_zSgsStress(i,j,k+1)-vcell_zSgsStress(i,j,k))*Area_TB);
          //// Z-momentum
          FractalZSrc(i,j,k)=-densitygas/cellvol*((wcell_xSgsStress(i+1,j,k)-wcell_xSgsStress(i,j,k))*Area_EW+(wcell_ySgsStress(i,j+1,k) -wcell_ySgsStress(i,j,k))*Area_NS+(wcell_zSgsStress(i,j,k+1)-wcell_zSgsStress(i,j,k))*Area_TB);

          });

    }

} //namespace Uintah
