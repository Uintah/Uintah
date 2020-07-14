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

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureEqn::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &PressureEqn::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &PressureEqn::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &PressureEqn::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureEqn::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
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

  m_xmom_name = ArchesCore::default_uMom_name;
  m_ymom_name = ArchesCore::default_vMom_name;
  m_zmom_name = ArchesCore::default_wMom_name;

  m_drhodt_name = "drhodt";

  if (db->findBlock("drhodt")){
    db->findBlock("drhodt")->getAttribute("label",m_drhodt_name);
  }
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::sched_Initialize( const LevelP& level, SchedulerP& sched ){
  const MaterialSet* matls = m_materialManager->allMaterials( "Arches" );

  if(!do_custom_arches_linear_solve){
    m_hypreSolver->scheduleInitialize( level, sched, matls );
  }
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

  //do_custom_arches_linear_solve = true; // Use the custom Arches linear solve instead of hypre

  if(!do_custom_arches_linear_solve){
    //);
    //custom_solver->sched_PreconditionerConstruction( sched, matls, level );// hard coded for level 0 
    ProblemSpecP db_pressure{nullptr};

    //Pressure Solve as part of the momentum solver
    if ( db->findBlock("KMomentum") != nullptr ){

      db_pressure = db->findBlock("KMomentum")->findBlock("PressureSolver");
      if ( !db_pressure ){
        throw ProblemSetupException("Error: You must specify a <PressureSolver> block in the UPS file.",__FILE__,__LINE__);
      }

    } else {

      //Part of the utility_factory?
      ProblemSpecP db_all_util = db->findBlock("Utilities");
      if ( db_all_util != nullptr ){
        for ( ProblemSpecP db_util = db_all_util->findBlock("utility"); db_util != nullptr;
              db_util = db_util->findNextBlock("utility")){

          std::string type;
          db_util->getAttribute("type", type);
          if ( type == "poisson" ){
            db_pressure = db_util->findBlock("PoissonSolver");
          }
        }
      }
    }

    m_hypreSolver->readParameters(db_pressure, "pressure" );

    m_hypreSolver->getParameters()->setSolveOnExtraCells(false);

    //force a zero setup frequency since nothing else
    //makes any sense at the moment.
    m_hypreSolver->getParameters()->setSetupFrequency(0.0);

    if ( db_pressure != nullptr ){
      if ( db_pressure->findBlock("enforce_solvability") != nullptr ){
        m_enforceSolvability = true;
      }
    }
  } else {
    proc0cout << "\n     WARNING: Using custom Arches linear solve instead of hypre!" << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_initialize(
  std::vector<AFC::VariableInformation>& variable_registry,
  const bool pack_tasks ){

  register_variable( "A_press", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "b_press", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_pressure_name, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "guess_press", AFC::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void PressureEqn::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();

  auto Apress = tsk_info->get_field<CCVariable<Stencil7>, Stencil7, MemSpace>("A_press");
  auto b = tsk_info->get_field<CCVariable<double>, double, MemSpace>("b_press");
  auto x = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_pressure_name);
  auto guess = tsk_info->get_field<CCVariable<double>, double, MemSpace>("guess_press");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    Stencil7& A = Apress(i,j,k);
    A.e = 0.0;
    A.w = 0.0;
    A.n = 0.0;
    A.s = 0.0;
    A.t = 0.0;
    A.b = 0.0;
    b(i,j,k) = 0.0;
    x(i,j,k) = 0.0;
    guess(i,j,k) = 0.0;

  });

   Uintah::BlockRange range2(patch->getCellLowIndex(), patch->getCellHighIndex() );
   Uintah::parallel_for(execObj, range2, KOKKOS_LAMBDA(int i, int j, int k){

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

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());
    if ( !on_this_patch ) continue;

    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    Patch::FaceType face = i_bc->second.face;
    BndTypeEnum my_type = i_bc->second.type;

    double sign;

    if ( my_type == OUTLET_BC ||
         my_type == PRESSURE_BC ){
      // Dirichlet
      // P = 0
      sign = -1.0;
    } else {
      // Applies to Inlets, walls where
      // P satisfies a Neumann condition
      // dP/dX = 0
      sign = 1.0;
    }

    parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), KOKKOS_LAMBDA(const int i,const int j,const int k) {

      const int im=i- iDir[0];
      const int jm=j- iDir[1];
      const int km=k- iDir[2];

      Apress(im,jm,km).p = Apress(im,jm,km).p + sign * Apress(im,jm,km)[face];
      Apress(im,jm,km)[face] = 0.;

    });
  }

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_init(
  std::vector<AFC::VariableInformation>& variable_registry,
  const bool packed_tasks ){

  register_variable( m_pressure_name, AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "A_press", AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "A_press", AFC::REQUIRES, 0, AFC::OLDDW, variable_registry, m_task_name );
  register_variable( "guess_press", AFC::COMPUTES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
PressureEqn::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  auto Apress = tsk_info->get_field<CCVariable<Stencil7>, Stencil7, MemSpace>("A_press");
  auto old_Apress = tsk_info->get_field<constCCVariable<Stencil7>, const Stencil7, MemSpace>("A_press");
  auto x = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_pressure_name);
  auto guess = tsk_info->get_field<CCVariable<double>, double, MemSpace>("guess_press");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    Apress(i,j,k) = old_Apress(i,j,k);
    x(i,j,k) = 0.0;
    guess(i,j,k) = 0.0;
  });
}


//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_eval(
  std::vector<AFC::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( "b_press", AFC::COMPUTES, variable_registry, time_substep, m_task_name );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( ArchesCore::default_uMom_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( ArchesCore::default_vMom_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( ArchesCore::default_wMom_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_drhodt_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void PressureEqn::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();
  const double V       = DX.x()*DX.y()*DX.z();

  auto b = tsk_info->get_field<CCVariable<double>, double, MemSpace>("b_press");
  auto eps = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_eps_name);
  auto xmom = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>(ArchesCore::default_uMom_name);
  auto ymom = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>(ArchesCore::default_vMom_name);
  auto zmom = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>(ArchesCore::default_wMom_name);
  auto drhodt = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_drhodt_name);

  Uintah::BlockRange range2(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  parallel_for(execObj, range2, KOKKOS_LAMBDA (int i, int j, int k){
    b(i,j,k) = 0.0;
  });

  const double dt = tsk_info->get_dt();
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){

    b(i,j,k) = -eps(i,j,k) * ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                 area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) ) +
                 area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) ) +
                 V*drhodt(i,j,k)  ) / dt ;

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
template <typename ExecSpace, typename MemSpace>
void PressureEqn::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  //This only applies BCs to A. Boundary conditions to the RHS are handled upstream in RhoUHatBC
  auto A = tsk_info->get_field<CCVariable<Stencil7>, Stencil7, MemSpace>("A_press");
  auto b = tsk_info->get_field<CCVariable<double>, double, MemSpace>("b_press");
  auto eps = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_eps_name);

  //Now take care of intrusions:
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  parallel_for(execObj,range, KOKKOS_LAMBDA (const int i, const int j, const int k){ 

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

  for ( auto i = m_local_labels.begin(); i != m_local_labels.end(); i++ ){
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

  bool isFirstSolve = true;

  bool modifies_x = true; //because x was computed upstream

  if(do_custom_arches_linear_solve){
    sched_custom(level, sched,  matls,
        A,      Task::NewDW,
        x,      modifies_x,
        b,      Task::NewDW,
        guess,  Task::NewDW,
        time_substep);
  } else { // use hypre
   const bool isPeriodic = m_periodic_vector.x() == 1 && m_periodic_vector.y() == 1 && m_periodic_vector.z() ==1;
   if ( isPeriodic || m_enforceSolvability ) {
     m_hypreSolver->scheduleEnforceSolvability<CCVariable<double> >(level, sched, matls, b, time_substep);
    }

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
}

 void
 PressureEqn::sched_custom( const LevelP           & level
                          ,       SchedulerP       & sched
                          , const MaterialSet      * matls
                          , const VarLabel         * A_label
                          ,       Task::WhichDW      which_A_dw
                          , const VarLabel         * x_label
                          ,       bool               modifies_X
                          , const VarLabel         * b_label
                          ,       Task::WhichDW      which_b_dw
                          , const VarLabel         * guess_label
                          ,       Task::WhichDW      which_guess_dw
                          ,       int       rk_step         
                          ){

    xLabel=x_label;

    auto taskDependencies = [&](Task* tsk) {
      tsk->modifies(xLabel);
    };

    create_portable_tasks(taskDependencies, this,
                          "PressureEqn::blindGuessToLinearSystem",
                          &PressureEqn::blindGuessToLinearSystem<UINTAH_CPU_TAG>,
                          &PressureEqn::blindGuessToLinearSystem<KOKKOS_OPENMP_TAG>,
                          //&PressureEqn::blindGuessToLinearSystem<KOKKOS_CUDA_TAG>,
                          sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT);
   }

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::blindGuessToLinearSystem(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj){

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

      auto x_values   = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>(  xLabel    , indx , patch, Ghost::None, 0 );
    parallel_initialize(execObj,0.0,x_values);
  }
}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_init1(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj, int rk_step){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
//
  bool getModifiable=true;
  int matl = indx ;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    auto A_m   = new_dw->getConstGridVariable<constCCVariable<Stencil7>, Stencil7, MemSpace>( ALabel , indx , patch, Ghost::AroundCells, 1 );
    auto g_v   = new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>    (  guess , indx , patch, Ghost::AroundCells, 1 );
    auto b_v   = new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>    (  bLabel, indx , patch, Ghost::None, 0 );
    auto x_v   = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>    (  xLabel, indx , patch, Ghost::AroundCells, 1 , getModifiable);
    auto residual = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>    ( d_residualLabel, indx , patch, Ghost::AroundCells, cg_ghost );

    auto BJmat=createContainer<CCVariable<double>, double, num_prec_elem, MemSpace>(num_prec_elem); 
    for (unsigned int i=0;i<d_precMLabel.size();i++){
        new_dw->assignGridVariable<CCVariable<double>, double, MemSpace>(BJmat[i],d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
    }

    if (rk_step==0){
      parallel_initialize(execObj,0.0,residual,BJmat);  
    }

    Uintah::parallel_for(execObj, range,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
      residual(i,j,k) = b_v(i,j,k) - ( A_m(i,j,k)[6] * g_v(i,j,k) +
                                       A_m(i,j,k)[0] * g_v(i-1,j,k) +
                                       A_m(i,j,k)[1] * g_v(i+1,j,k) +
                                       A_m(i,j,k)[2] * g_v(i,j-1,k) +
                                       A_m(i,j,k)[3] * g_v(i,j+1,k) +
                                       A_m(i,j,k)[4] * g_v(i,j,k-1) +
                                       A_m(i,j,k)[5] * g_v(i,j,k+1) );

      BJmat[0](i,j,k) = A_m(i,j,k)[6];
      BJmat[1](i,j,k) = A_m(i,j,k)[0];
      BJmat[2](i,j,k) = A_m(i,j,k)[2];
      BJmat[3](i,j,k) = A_m(i,j,k)[4];

      x_v(i,j,k)=g_v(i,j,k);
  });

//if(true){   // for debugging
  //const IntVector idxLo = patch->getCellLowIndex();
  //const IntVector idxHi = patch->getCellHighIndex();
  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              //int dy = idxHi.y()-idxLo.y();
              //int dx = idxHi.x()-idxLo.x();
             //for (int ii=1; ii<4; ii++){
               ////if (A_m[ii](i,j,k) >-1e-85){
                 ////A_m[ii](i,j,k) =-A_m[ii](i,j,k) ; // for clarity of output
               ////}
             //}

              //for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                //for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  //for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    //if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    ////std::cout <<  A_m[0](i,j,k) << " ";
                    //std::cout <<  A_m(i,j,k)[6] << " ";
                    //}else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      ////std::cout << A_m[1](i,j,k)<< " " ;
                    //std::cout <<  A_m(i,j,k)[0] << " ";
                    //}else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                     ////std::cout << A_m[1](i+1,j,k)<< " " ;
                    //std::cout <<  A_m(i,j,k)[1] << " ";
                    //}else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                     ////std::cout << A_m[2](i,j,k)<< " " ;
                    //std::cout <<  A_m(i,j,k)[2] << " ";
                    //}else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                     ////std::cout << a_m[2](i,j+1,k)<< " " ;
                    //std::cout <<  A_m(i,j,k)[3] << " ";
                    //}else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      ////std::cout << A_m[3](i,j,k)<< " " ;
                    //std::cout <<  A_m(i,j,k)[4] << " ";
                    //}else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      ////std::cout << A_m[3](i,j,k+1)<< " " ;
                    //std::cout <<  A_m(i,j,k)[5] << " ";
                    //}else{
                      //std::cout <<" " <<"0" << " "; 
                    //}
                  //}
                //}
              //}
              //std::cout << "\n";

          //} // end i loop
        //} // end j loop
      //} // end k loop

  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
        //std::cout <<    b_v(i,j,k) << " \n";
      //}
    //} 
  //}

//}

} // end patch loop
}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_init2(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj,int iter,  int rk_step){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
//
  double R_squared=0.0;
//
    int matl = indx ;
  for (int p = 0; p < patches->size(); p++) {
   const Patch* patch = patches->get(p);

   const IntVector idxLo = patch->getCellLowIndex();
   const IntVector idxHi = patch->getCellHighIndex();

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    auto BJmat=createConstContainer<constCCVariable<double>,const double,num_prec_elem, MemSpace>(num_prec_elem); 

    int final_iter=total_rb_switch-1;

    auto residual= new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace> (d_residualLabel,matl,patch,Ghost::AroundCells,cg_ghost);
    auto smallP= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_smallPLabel  ,matl,patch,Ghost::AroundCells,cg_ghost);
    if (iter==0){
      parallel_initialize(execObj,0.0,smallP);
    }
  
    if (iter == final_iter){
    auto littleQ= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_littleQLabel  ,matl,patch,Ghost::AroundCells,1); // computes with scratch ghosts
    auto bigZ= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_bigZLabel  ,matl,patch,Ghost::AroundCells,cg_ghost);
    parallel_initialize(execObj,0.0,bigZ,littleQ);
    }

    for (unsigned int i=0;i<d_precMLabel.size();i++){
      BJmat[i] =  new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>(d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
    }

       //               A      b       x0
      precondition_relax(execObj,BJmat,residual,smallP,idxLo, idxHi,iter, patch);
      
  if (iter==final_iter){
        Uintah::parallel_reduce_sum(execObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED
           sum=residual(i,j,k)*smallP(i,j,k);                                                                                                                        
          }, R_squared);
  
    
  
  if ( rk_step==0 ){
    new_dw->put(sum_vartype(R_squared),d_resSumLabel[0]);
  } else{
    //new_dw->put(sum_vartype(R_squared),d_resSumLabel_rk2[0]); // need for RK
  }

} // final_iter if
} // patch loop

}

template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
void
PressureEqn::precondition_relax(ExecutionObject<ExecSpace, MemSpace>& execObj,
                                struct1DArray<grid_CT,num_prec_elem>& precMatrix,
                                grid_CT& residual, grid_T& bigZ,const IntVector &idxLo,const IntVector &idxHi, int rb_switch,const Patch* patch ){
       // precMatrix is the inverted precondition matrix or the A matrix.  Depending on the relaxation type. 
  if (d_custom_relax_type==jacobBlock){
    int offset=cg_ghost;
    int inside_buffer=max(cg_ghost-1,0);
    for (int kk=0;kk<d_stencilWidth;kk++){
      int kr=kk-offset;
      for (int jj=0;jj<d_stencilWidth;jj++){
        int jr=jj-offset;
        for (int ii=0;ii<d_stencilWidth;ii++){
          int ir=ii-offset;

          //residual may only has 1 layer of wall cells, so we have to check for walls, but not for ghosts...........
          int wallCheckzp= kk==4? idxHi.z()-inside_buffer: idxHi.z();
          int wallCheckzm= kk==0? idxLo.z()+inside_buffer: idxLo.z();
          int wallCheckyp= jj==4? idxHi.y()-inside_buffer: idxHi.y();
          int wallCheckym= jj==0? idxLo.y()+inside_buffer: idxLo.y();
          int wallCheckxp= ii==4? idxHi.x()-inside_buffer: idxHi.x(); 
          int wallCheckxm= ii==0? idxLo.x()+inside_buffer: idxLo.x(); 

          Uintah::BlockRange rangetemp(IntVector(wallCheckxm,wallCheckym,wallCheckzm) , IntVector(wallCheckxp,wallCheckyp,wallCheckzp) );

          Uintah::parallel_for(execObj, rangetemp,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
              bigZ(i,j,k)+= precMatrix[d_stencilWidth*d_stencilWidth*(kk)+d_stencilWidth*(jj)+ii](i,j,k)*residual(i+ir,j+jr,k+kr); /// JACOBI 3x3 BLOCK
              });
        }
      }
    }
  } else if(d_custom_relax_type==redBlack){ 
              //cout << "  REDBLACK\n";
    int niter = cg_ghost; 
    if(rb_switch < 0 ){ // HACK DEREKX DEBUG FIX
      rb_switch=0;
      niter=3;
    }
    if ((cg_ghost%2)==0){
      rb_switch=0;
    }
    for (int rb_i = 0 ; rb_i <niter ; rb_i++) {
      IntVector  iter_on_ghosts(max(cg_ghost-1-rb_i,0),max(cg_ghost-1-rb_i,0),max(cg_ghost-1-rb_i,0));
      Uintah::BlockRange rangedynamic(idxLo-iter_on_ghosts*patch->neighborsLow(),idxHi +iter_on_ghosts*patch->neighborsHigh() ); // assumes 1 ghost cell
      Uintah::parallel_for(execObj, rangedynamic,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
          if ( (i + j +k + rb_i +rb_switch )% 2 ==0){    
          bigZ(i,j,k)= (residual(i,j,k) - precMatrix[1](i,j,k)*bigZ(i-1,j,k)-precMatrix[1](i+1,j,k)*bigZ(i+1,j,k)  // SYMMTRIC APPROXIMATION
                                        - precMatrix[2](i,j,k)*bigZ(i,j-1,k)-precMatrix[2](i,j+1,k)*bigZ(i,j+1,k) 
                                        - precMatrix[3](i,j,k)*bigZ(i,j,k-1)-precMatrix[3](i,j,k+1)*bigZ(i,j,k+1) ) / precMatrix[0](i,j,k) ; //red_black
          } 
      });
    } // this for loop exists to try and reduce mpi communication costs 
  } else{
    Uintah::BlockRange jrange(idxLo,idxHi); // assumes 1 ghost cell
    Uintah::parallel_for(execObj, jrange,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
        bigZ(i,j,k)=residual(i,j,k)/precMatrix[0](i,j,k) ;// SYMMTRIC APPROXIMATION
        });
  }
}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_task1(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj,int iter){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
  double correction_sum=0.0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
 
    auto A_m   = new_dw->getConstGridVariable<constCCVariable<Stencil7>, Stencil7, MemSpace>( ALabel , indx , patch, Ghost::None, 0 );
    auto smallP   = new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace> (d_smallPLabel , indx , patch, Ghost::AroundCells, 1 );
    auto littleQ   = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>  (d_littleQLabel, indx , patch, Ghost::AroundCells, 1 );

    Uintah::parallel_reduce_sum(execObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED

         //cout << "  " << littleQ(i,j,k) <<  "  " << smallP(i,j,k) << "  " << smallP(i,j,k) << " \n";
        // NON-symmetric 
        littleQ(i,j,k)=A_m(i,j,k)[6]*smallP(i,j,k)+
        A_m(i,j,k)[0]*smallP(i-1,j,k)+
        A_m(i,j,k)[1]*smallP(i+1,j,k)+
        A_m(i,j,k)[2]*smallP(i,j-1,k)+
        A_m(i,j,k)[3]*smallP(i,j+1,k)+
        A_m(i,j,k)[4]*smallP(i,j,k-1)+
        A_m(i,j,k)[5]*smallP(i,j,k+1);

        // ASSUME SYMMETRIC
        //littleQ(i,j,k)=A_m(i,j,k)[6]*smallP(i,j,k)+  // THIS MAKES NANS because A_m is not defined in extra cells, which is required for symmetric
        //A_m(i,j,k)[0]  *smallP(i-1,j,k)+
        //A_m(i+1,j,k)[0]*smallP(i+1,j,k)+ 
        //A_m(i,j,k)[2]  *smallP(i,j-1,k)+
        //A_m(i,j+1,k)[2]*smallP(i,j+1,k)+
        //A_m(i,j,k)[4]  *smallP(i,j,k-1)+
        //A_m(i,j,k+1)[4]*smallP(i,j,k+1);
        //cout << littleQ(i,j,k) << " \n";
        //
        sum=littleQ(i,j,k)*smallP(i,j,k);   // REDUCTION
        }, correction_sum);

  }
  new_dw->put(sum_vartype(correction_sum),d_corrSumLabel[iter]);
  
}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_task2(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj,int iter){
/////////////////////////////////// TASK 2 ///////////////////////////////
///////   apply correction to x_v as well as the residual vector"   //////
//////////////////////////////////////////////////////////////////////////

    sum_vartype R_squared_old;
    new_dw->get(R_squared_old,d_resSumLabel[iter]); 
    sum_vartype correction_sum;
    new_dw->get(correction_sum,d_corrSumLabel[iter]);

    const double correction_factor= (std::abs(correction_sum) < 1e-100) ? 0.0 :R_squared_old/correction_sum; // ternary may not be needed when we switch to do-while loop

    bool getModifiable=true;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    auto residual = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>    ( d_residualLabel, indx , patch, Ghost::AroundCells, max(cg_ghost,1),getModifiable );
    auto x_v   = new_dw->getGridVariable<CCVariable<double>, double, MemSpace>    (  xLabel, indx , patch, Ghost::AroundCells, 1 , getModifiable);
    auto smallP   = new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>(d_smallPLabel , indx , patch, Ghost::None, 0 );
    auto littleQ   = new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>(d_littleQLabel , indx , patch, Ghost::None, 0 );

    Uintah::parallel_for( execObj,range,  KOKKOS_LAMBDA(int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                               x_v(i,j,k)=x_v(i,j,k)+correction_factor*smallP(i,j,k);
                               residual(i,j,k)=residual(i,j,k)-correction_factor*littleQ(i,j,k);
                                });
  } // patch loop
}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_task3(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj,int iter){
  int matl = indx;  
  double  R_squared=0.0;
  double max_residual=0.0;
  const bool getModifiable=true;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    IntVector idxLo(patch->getCellLowIndex());
    IntVector idxHi(patch->getCellHighIndex());

    auto BJmat=createConstContainer<constCCVariable<double>,const double,num_prec_elem, MemSpace>(num_prec_elem); 
    for (unsigned int i=0;i<d_precMLabel.size();i++){
      BJmat[i] =  new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>(d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
    }

    auto bigZ= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_bigZLabel  ,matl,patch,Ghost::None,0,getModifiable);
    auto residual= new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace> (d_residualLabel,matl,patch,Ghost::None,0);
    parallel_initialize(execObj, 0.0, bigZ);

    precondition_relax(execObj,BJmat,residual,bigZ,idxLo, idxHi,-1, patch);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
         
     Uintah::parallel_reduce_sum(execObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED
          sum=residual(i,j,k)*bigZ(i,j,k); // reduction
       }, R_squared);

     Uintah::parallel_reduce_min(execObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& lmin){  //compute correction, GHOST CELLS REQUIRED
          lmin=-fabs(residual(i,j,k));  // presumably most efficcient to comptue here.......could be computed earlier
       }, max_residual);
          
  }

  new_dw->put(sum_vartype(R_squared),d_resSumLabel[iter+1]);
  new_dw->put(max_vartype(-max_residual),d_convMaxLabel[iter]);

}

template <typename ExecSpace, typename MemSpace>
void
PressureEqn::cg_task4(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecSpace, MemSpace>& execObj,int iter){

/////////////////////////////////// TASK 4 ////////////////////////////////
//          apply preconditioner to residual vector           ////////////
//          requires no ghost cells, but a reduction          ///////////
//////////////////////////////////////////////////////////////////////

    sum_vartype R_squared;
    new_dw->get(R_squared,d_resSumLabel[iter+1]);
    sum_vartype R_squared_old;
    new_dw->get(R_squared_old,d_resSumLabel[iter]);


    max_vartype convergence;
    new_dw->get(convergence,d_convMaxLabel[iter]);
    //proc0cout << " R_squared " << R_squared << "  " <<R_squared_old << "  " << convergence << " \n"; 
    proc0cout << "MAX RESIDUAL VALUE:::: " << convergence << " \n"; 

    const double  beta=R_squared/R_squared_old;  
    bool getModifiable=true;
  int matl = indx ;  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());


    auto smallP= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_smallPLabel  ,matl,patch,Ghost::None,0,getModifiable);
    auto bigZ=new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace> (d_bigZLabel  ,matl,patch,Ghost::None,0);


       Uintah::parallel_for( execObj, range,   KOKKOS_LAMBDA (int i, int j, int k){
                             smallP(i,j,k)=bigZ(i,j,k)+beta*smallP(i,j,k);
                            });

}
}
