#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Ports/SolverInterface.h>

#include <Core/Grid/MaterialManager.h>

using namespace Uintah;

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
PressureEqn::PressureEqn( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
TaskInterface( task_name, matl_index ) {

  m_materialManager = materialManager;
  m_pressure_name = "pressure";

}

//--------------------------------------------------------------------------------------------------
PressureEqn::~PressureEqn(){}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureEqn::loadTaskComputeBCsFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::BC>( this
                                     , &PressureEqn::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureEqn::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureEqn::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureEqn::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &PressureEqn::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureEqn::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureEqn::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureEqn::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &PressureEqn::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureEqn::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureEqn::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace PressureEqn::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &PressureEqn::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureEqn::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureEqn::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::OpenMP builds
                                     );
}

TaskAssignedExecutionSpace PressureEqn::loadTaskRestartInitFunctionPointers()
{
  return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void PressureEqn::create_local_labels(){

  register_new_variable<CCVariable<Stencil7> >( "A_press" );
  register_new_variable<CCVariable<double> >( "b_press" );
  register_new_variable<CCVariable<double> >( m_pressure_name );
  register_new_variable<CCVariable<double> >( "guess_press");

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::problemSetup( ProblemSpecP& db ){

  ArchesCore::GridVarMap<CCVariable<double> > var_map;
  var_map.problemSetup( db );
  m_eps_name = var_map.vol_frac_name;

  m_xmom_name = "x-mom";
  m_ymom_name = "y-mom";
  m_zmom_name = "z-mom";

  m_drhodt_name = "drhodt";

  if (db->findBlock("drhodt")){
    db->findBlock("drhodt")->getAttribute("label",m_drhodt_name);
  }
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::sched_Initialize( const LevelP& level, SchedulerP& sched ){
  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );
  m_hypreSolver->scheduleInitialize( level, sched, matls );
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::sched_restartInitialize( const LevelP& level, SchedulerP& sched ){
  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );
  m_hypreSolver->scheduleRestartInitialize( level, sched, matls );
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::setup_solver( ProblemSpecP& db ){

  ProblemSpecP db_pressure = db->findBlock("KMomentum")->findBlock("PressureSolver");

  if ( !db_pressure ){
    throw ProblemSetupException("Error: You must specify a <PressureSolver> block in the UPS file.",__FILE__,__LINE__);
  }

  m_hypreSolver->readParameters(db_pressure, "pressure" );

  m_hypreSolver->getParameters()->setSolveOnExtraCells(false);

  //force a zero setup frequency since nothing else
  //makes any sense at the moment.
  m_hypreSolver->getParameters()->setSetupFrequency(0.0);

  m_enforceSolvability = false;
  if ( db->findBlock("enforce_solvability")){
    m_enforceSolvability = true;
  }

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_initialize(
  std::vector<AFC::VariableInformation>& variable_registry,
  const bool pack_tasks ){

  register_variable( "A_press", AFC::COMPUTES, variable_registry );
  register_variable( "b_press", AFC::COMPUTES, variable_registry );
  register_variable( m_pressure_name, AFC::COMPUTES, variable_registry );
  register_variable( "guess_press", AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void PressureEqn::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();

  auto Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7>, Stencil7, MemSpace >("A_press");
  auto b = tsk_info->get_uintah_field_add<CCVariable<double>,double , MemSpace>("b_press");
  auto x = tsk_info->get_uintah_field_add<CCVariable<double>,double , MemSpace>(m_pressure_name);
  auto guess = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >("guess_press");





  //const double dt = tsk_info->get_dt();
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( exObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    Stencil7& A = Apress(i,j,k);
    A.e = 0.0;
    A.w = 0.0;
    A.n = 0.0;
    A.s = 0.0;
    A.t = 0.0;
    A.b = 0.0;
    b(i,j,k)=0.0;
    x(i,j,k)=0.0;
    guess(i,j,k)=0.0;

  });

   Uintah::BlockRange range2(patch->getCellLowIndex(), patch->getCellHighIndex() );
   Uintah::parallel_for(exObj, range2, KOKKOS_LAMBDA(int i, int j, int k){

   Stencil7& A = Apress(i,j,k);

    A.e = -area_EW/DX.x();
    A.w = -area_EW/DX.x();
    A.n = -area_NS/DX.y();
    A.s = -area_NS/DX.y();
    A.t = -area_TB/DX.z();
    A.b = -area_TB/DX.z();

    A.p = A.e + A.w + A.n + A.s + A.t + A.b;
    A.p *= -1;

   });
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_init(
  std::vector<AFC::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  register_variable( "A_press", AFC::COMPUTES, variable_registry );
  register_variable( "A_press", AFC::REQUIRES, 0, AFC::OLDDW, variable_registry );
  register_variable( "b_press", AFC::COMPUTES, variable_registry );
  register_variable( m_pressure_name, AFC::COMPUTES, variable_registry );
  register_variable( "guess_press", AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace> void
PressureEqn::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

 auto  Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7>, Stencil7, MemSpace >("A_press");
 auto  old_Apress = tsk_info->get_const_uintah_field_add<constCCVariable<Stencil7>, const Stencil7, MemSpace >("A_press");
 auto  b = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >("b_press");
 auto  x = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >(m_pressure_name);
 auto  guess = tsk_info->get_uintah_field_add<CCVariable<double>,double,MemSpace >("guess_press");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
 parallel_for(exObj,range, KOKKOS_LAMBDA (int i, int j, int k){
  b(i,j,k)=0.0;
  x(i,j,k)=0.0;
  guess(i,j,k)=0.0;
  Apress(i,j,k)=old_Apress(i,j,k);
});

}


//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_eval(
  std::vector<AFC::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( "b_press", AFC::MODIFIES, variable_registry, time_substep );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep );
  register_variable( "x-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep );
  register_variable( "y-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep );
  register_variable( "z-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep );
  register_variable( m_drhodt_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void PressureEqn::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double V       = DX.x()*DX.y()*DX.z();

  auto b = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >("b_press");
  auto eps = tsk_info->get_const_uintah_field_add<constCCVariable<double>,const double, MemSpace >(m_eps_name);
  auto xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double>,const double, MemSpace >("x-mom");
  auto ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double>, const double, MemSpace >("y-mom");
  auto zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double>, const double, MemSpace  >("z-mom");
  auto drhodt = tsk_info->get_const_uintah_field_add<constCCVariable<double>, const double, MemSpace  >(m_drhodt_name);

  const double dt = tsk_info->get_dt();
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for(exObj, range, KOKKOS_LAMBDA(int i, int j, int k){

    b(i,j,k) = ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                 area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) ) +
                 area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) ) +
                 V*drhodt(i,j,k)  ) / dt ;
    b(i,j,k)  *= -eps(i,j,k) ;

  });
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_compute_bcs(
  std::vector<AFC::VariableInformation>& variable_registry, const int time_substep,
  const bool packed_tasks ){

  register_variable( "b_press", AFC::MODIFIES, variable_registry );
  register_variable( "A_press", AFC::MODIFIES, variable_registry );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemSpace>
void PressureEqn::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

  //This only applies BCs to A. Boundary conditions to the RHS are handled upstream in RhoUHatBC
  auto A = tsk_info->get_uintah_field_add<CCVariable<Stencil7>, Stencil7, MemSpace >("A_press");
  auto b = tsk_info->get_uintah_field_add<CCVariable<double>, double, MemSpace >("b_press");
  auto eps = tsk_info->get_const_uintah_field_add<constCCVariable<double>, const double, MemSpace >(m_eps_name);

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    Patch::FaceType face = i_bc->second.face;
    BndTypeEnum my_type = i_bc->second.type;

    double sign;

    if ( my_type == OUTLET ||
         my_type == PRESSURE ){
      // Dirichlet
      // P = 0
      sign = -1.0;
    } else {
      // Applies to Inlets, walls where
      // P satisfies a Neumann condition
      // dP/dX = 0
      sign = 1.0;
    }

    parallel_for_unstructured(exObj,cell_iter.get_ref_to_iterator<MemSpace>(),cell_iter.size(), KOKKOS_LAMBDA(const int i,const int j,const int k) {

      const int im=i- iDir[0];
      const int jm=j- iDir[1];
      const int km=k- iDir[2];

      A(im,jm,km).p = A(im,jm,km).p + sign * A(im,jm,km)[face];
      A(im,jm,km)[face] = 0.;

    });
  }

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  parallel_for(exObj,range, KOKKOS_LAMBDA (const int i, const int j, const int k){ 
  //Now take care of intrusions:
    A(i,j,k).e *= eps(i,j,k);
    A(i,j,k).w *= eps(i,j,k);
    A(i,j,k).n *= eps(i,j,k);
    A(i,j,k).s *= eps(i,j,k);
    A(i,j,k).t *= eps(i,j,k);
    A(i,j,k).b *= eps(i,j,k);

    if ( eps(i,j,k) < 1.e-10 ){
      A(i,j,k).p = 1.;
      b(i,j,k) = 0.0;
    }

    //east:
    if ( eps(i+1,j,k) < 1.e-10 ){
      A(i,j,k).p += A(i,j,k).e;
      A(i,j,k).e = 0.0;
    }
    //west:
    if ( eps(i-1,j,k) < 1.e-10 ){
      A(i,j,k).p += A(i,j,k).w;
      A(i,j,k).w = 0.0;
    }
    //north:
    if ( eps(i,j+1,k)< 1.e-10 ){
      A(i,j,k).p += A(i,j,k).n;
      A(i,j,k).n = 0.0;
    }
    //south:
    if ( eps(i,j-1,k) < 1.e-10 ){
      A(i,j,k).p += A(i,j,k).s;
      A(i,j,k).s = 0.0;
    }
    //top:
    if ( eps(i,j,k+1) < 1.e-10 ){
      A(i,j,k).p += A(i,j,k).t;
      A(i,j,k).t = 0.0;
    }
    //bottom:
    if ( eps(i,j,k-1) < 1.e-10 ){
      A(i,j,k).p += A(i,j,k).b;
      A(i,j,k).b = 0.0;
    }

  });
}

void
PressureEqn::solve( const LevelP& level, SchedulerP& sched, const int time_substep ){

  const VarLabel* A = NULL;
  const VarLabel* b = NULL;
  const VarLabel* x = NULL;
  const VarLabel* guess = NULL;

  for ( auto i = _local_labels.begin(); i != _local_labels.end(); i++ ){
    if ( (*i)->getName() == "A_press" ){
      A = *i;
    } else if ( (*i)->getName() == "b_press" ){
      b = *i;
    } else if ( (*i)->getName() == m_pressure_name ){
      x = *i;
    } else if ( (*i)->getName() == "guess_press"){
      guess = *i;
    }
  }

  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );
  IntVector m_periodic_vector = level->getPeriodicBoundaries();

  const bool isPeriodic = m_periodic_vector.x() == 1 && m_periodic_vector.y() == 1 && m_periodic_vector.z() ==1;
  if ( isPeriodic || m_enforceSolvability ) {
    m_hypreSolver->scheduleEnforceSolvability<CCVariable<double> >(level, sched, matls, b, time_substep);
  }

  bool isFirstSolve = true;

  bool modifies_x = true; //because x was computed upstream

  if ( time_substep > 0 ) {
    isFirstSolve = false;
  }

  m_hypreSolver->scheduleSolve(level, sched,  matls,
                               A,      Task::NewDW,
                               x,      modifies_x,
                               b,      Task::NewDW,
                               guess,  Task::NewDW,
                               isFirstSolve);

}
