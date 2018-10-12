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
   //do_custom_arches_linear_solve=true;
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

  if(!do_custom_arches_linear_solve){
    //);
    //custom_solver->sched_PreconditionerConstruction( sched, matls, level );// hard coded for level 0 
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

  bool modifies_x = true; //because x was computed upstream
  bool isFirstSolve = true;
  if(do_custom_arches_linear_solve){
    sched_custom(level, sched,  matls,
        A,      Task::NewDW,
        x,      modifies_x,
        b,      Task::NewDW,
        guess,  Task::NewDW,
        time_substep);
  }else{ // use hypre
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
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  if (rk_step ==0){
    d_residualLabel = VarLabel::create("cg_residual",  CC_double);
    d_littleQLabel = VarLabel::create("littleQ",  CC_double);
    d_bigZLabel    = VarLabel::create("bigZ",  CC_double);
    d_smallPLabel  = VarLabel::create("smallP",  CC_double);

    ALabel=A_label;
    bLabel=b_label;
    guess=guess_label;

    d_blockSize=1; d_stencilWidth=(d_blockSize-1)+d_blockSize  ; // NOT TRUE FOR EVEN BLOCK SIZE 
    cg_ghost=3;


    cg_n_iter=150;
    for (int i=0 ; i < cg_n_iter ; i++){
      d_convMaxLabel.push_back(  VarLabel::create("convergence_check"+ std::to_string(i),   max_vartype::getTypeDescription()));
      d_corrSumLabel.push_back(  VarLabel::create("correctionSum"+ std::to_string(i),   sum_vartype::getTypeDescription()));
      d_resSumLabel.push_back(  VarLabel::create("residualSum"+ std::to_string(i),   sum_vartype::getTypeDescription()));
    }
    d_resSumLabel.push_back(  VarLabel::create("residualSum_"+ std::to_string(cg_n_iter),   sum_vartype::getTypeDescription()));

    d_precMLabel.push_back(VarLabel::create("mg_A_p",  CC_double)); // mutigrid A -cetnral
     // THese 3 are UNUSED FOR jacobi!!!!!!!!!! 
    d_precMLabel.push_back(VarLabel::create("mg_A_w",  CC_double)); // mutigrid A - west 
    d_precMLabel.push_back(VarLabel::create("mg_A_s",  CC_double)); // mutigrid A - north
    d_precMLabel.push_back(VarLabel::create("mg_A_b",  CC_double)); // mutigrid A - south


    d_custom_relax_type=jacobi;
    if (d_custom_relax_type==redBlack){
      cg_ghost=1;   // number of rb iterations
    }else if (d_custom_relax_type==jacobBlock){
      cg_ghost=d_blockSize-1;   // Not true for even block size
    } else{ // jacobi
      cg_ghost=0;   
    }


    }else{
        throw ProblemSetupException("Error: only RK 1 time integration is supported when using the custom linear solver.",__FILE__,__LINE__);
    }

////------------------ set up CG solver---------------------//
    auto task_i1 = [&](Task* tsk) {
      tsk->requires(Task::NewDW, guess, Ghost::AroundCells, 1);
      tsk->requires(Task::NewDW, bLabel, Ghost::None, 0);
      tsk->requires(Task::NewDW, ALabel, Ghost::AroundCells, 1);
      tsk->modifies(xLabel);
      if (rk_step==0){
      tsk->computesWithScratchGhost
          (d_residualLabel, nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,cg_ghost);
      }else{
      tsk->modifies(d_residualLabel);
      }



     for (unsigned int i=0;i<d_precMLabel.size();i++){
       tsk->computesWithScratchGhost //  possibly not needed...
          (d_precMLabel[i], nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,cg_ghost);
     }

    };

    create_portable_tasks(task_i1, this,
                          "PressureEqn::cg_init1",
                          &PressureEqn::cg_init1<UINTAH_CPU_TAG>,
                          &PressureEqn::cg_init1<KOKKOS_OPENMP_TAG>,
                          &PressureEqn::cg_init1<KOKKOS_CUDA_TAG>,
                          sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, rk_step);


total_rb_switch=1;
int final_iter=total_rb_switch-1;
for (int rb_iter=0; rb_iter < total_rb_switch; rb_iter++){
  auto task_i2 = [&](Task* tsk) {
    if (rk_step==0 && rb_iter==0){
        tsk->computesWithScratchGhost
          (d_smallPLabel, nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,1);
    }else{
      tsk->modifies(d_smallPLabel);
    }

    if (rb_iter == final_iter){
      if (rk_step==0 ){
        tsk->computesWithScratchGhost
          (d_bigZLabel, nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,cg_ghost);
        tsk->computesWithScratchGhost
          (d_littleQLabel, nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,1);
        tsk->computes(d_resSumLabel[0]);
      }else{
        tsk->modifies(d_smallPLabel);
        tsk->modifies(d_bigZLabel);
        tsk->modifies(d_littleQLabel);
        //task_i2->computes(d_resSumLabel_rk2[0]);   // NEED FOR RK2
      }
    }


  tsk->requires(Task::NewDW, d_residualLabel, Ghost::AroundCells, cg_ghost); 

  for (unsigned int i=0;i<d_precMLabel.size();i++){
    tsk->requires(Task::NewDW,d_precMLabel[i],Ghost::AroundCells,cg_ghost+1); //  possibly not needed...
    //tsk->computesWithScratchGhosts(Task::NewDW,d_precMLabel[i],Ghost::AroundCells,cg_ghost+1); //  possibly not needed...
  }
  };

  create_portable_tasks(task_i2, this,
                        "PressureEqn::cg_init2",
                        &PressureEqn::cg_init2<UINTAH_CPU_TAG>,
                        &PressureEqn::cg_init2<KOKKOS_OPENMP_TAG>,
                        &PressureEqn::cg_init2<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, rb_iter,rk_step);
}


  for (int cg_iter=0 ; cg_iter < cg_n_iter ; cg_iter++){
  auto task1 = [&](Task* tsk) {
  tsk->requires(Task::NewDW,ALabel,Ghost::None, 0 );
  tsk->requires(Task::NewDW,d_smallPLabel, Ghost::AroundCells, 1);
  tsk->modifies(d_littleQLabel);
  tsk->computes(d_corrSumLabel[cg_iter]);

  };
  create_portable_tasks(task1, this,
                        "PressureEqn::cg_task1",
                        &PressureEqn::cg_task1<UINTAH_CPU_TAG>,
                        &PressureEqn::cg_task1<KOKKOS_OPENMP_TAG>,
                        &PressureEqn::cg_task1<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, cg_iter);





  auto task2 = [&](Task* tsk) {

   tsk->computes(d_corrSumLabel[cg_iter]); // GPU LIMITATION ( THIS SHOULD BE REQUIRES!!!)
   tsk->computes(d_resSumLabel[cg_iter]); 
   
   //tsk->requires(Task::NewDW,d_corrSumLabel[cg_iter]);
   //tsk->requires(Task::NewDW,d_resSumLabel[cg_iter]);


   tsk->requires(Task::NewDW,d_littleQLabel, Ghost::None, 0);
   tsk->requires(Task::NewDW,d_smallPLabel, Ghost::None, 0);
   tsk->modifies(xLabel);
   tsk->modifies(d_residualLabel);
  };

  create_portable_tasks(task2, this,
                        "PressureEqn::cg_task2",
                        &PressureEqn::cg_task2<UINTAH_CPU_TAG>,
                        &PressureEqn::cg_task2<KOKKOS_OPENMP_TAG>,
                        &PressureEqn::cg_task2<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, cg_iter);

   auto task3 = [&](Task* tsk) {
   for (unsigned int i=0;i<d_precMLabel.size();i++){
     tsk->requires(Task::NewDW,d_precMLabel[i],Ghost::AroundCells,cg_ghost+1); //  possibly not needed...
   }
   tsk->requires(Task::NewDW,d_residualLabel,Ghost::AroundCells,cg_ghost);
   tsk->requires(Task::NewDW,d_bigZLabel,Ghost::AroundCells,0);
   tsk->computes(d_resSumLabel[cg_iter+1]);
   tsk->computes(d_convMaxLabel[cg_iter]);
   };


  create_portable_tasks(task3, this,
                        "PressureEqn::cg_task3",
                        &PressureEqn::cg_task3<UINTAH_CPU_TAG>,
                        &PressureEqn::cg_task3<KOKKOS_OPENMP_TAG>,
                        &PressureEqn::cg_task3<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, cg_iter);




  auto task4 = [&](Task* tsk) {

    tsk->computes(d_convMaxLabel[cg_iter]);// GPU reduction problem with these on GPU LIMITATION!
    tsk->computes(d_resSumLabel[cg_iter]);
    tsk->computes(d_resSumLabel[cg_iter+1]);

    //tsk->requires(Task::NewDW,d_convMaxLabel[cg_iter]); 
    //tsk->requires(Task::NewDW,d_resSumLabel[cg_iter]);
    //tsk->requires(Task::NewDW,d_resSumLabel[cg_iter+1]);


    tsk->modifies(d_smallPLabel);
    tsk->requires(Task::NewDW,d_bigZLabel,Ghost::None, 0);
  };

  create_portable_tasks(task4, this,
                        "PressureEqn::cg_task4",
                        &PressureEqn::cg_task4<UINTAH_CPU_TAG>,
                        &PressureEqn::cg_task4<KOKKOS_OPENMP_TAG>,
                        &PressureEqn::cg_task4<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT, cg_iter);

    //int maxLevels=1; // MULTIGRID OFF
  //for (int l = maxLevels-2; l > -1; l--) { // Coarsen fine to coarse
    //const LevelP& coarse_level = grid->getLevel(l);

    //const PatchSet * level_patches=coarse_level->eachPatch();
    //Task* task_multigrid_up = scinew Task("linSolver::cg_coarsenResidual",
        //this, &linSolver::cg_moveResUp, cg_iter);

    //task_multigrid_up->requires( Task::NewDW ,d_residualLabel, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    //if ( cg_iter==0){
      //task_multigrid_up->computes(d_residualLabel);
    //}else{
      //task_multigrid_up->modifies(d_residualLabel);

    //}
    //sched->addTask(task_multigrid_up, level_patches,matls);
  //}



  //for (int l = 0; l < maxLevels; ++l) {
    //const LevelP& coarse_level = grid->getLevel(l);
    //const PatchSet * level_patches=coarse_level->eachPatch();

    //Task* task_multigrid_down = scinew Task("linSolver::cg_multigridDown",
        //this, &linSolver::cg_multigrid_down, cg_iter);

    //if (l<maxLevels-1 && cg_iter==0){
      //task_multigrid_down->computes(d_bigZLabel);
    //}else{
      //task_multigrid_down->modifies(d_bigZLabel);
    //}

    //if (l>0){
        ////int offset = 1;
        //task_multigrid_down->requires( Task::NewDW ,d_bigZLabel, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    //}
    
    //sched->addTask(task_multigrid_down, level_patches,matls);

    //int smoother_iter=3;// In case user didn't set up levels

    //for (int red_black_switch=smoother_iter-1; red_black_switch > -1; red_black_switch--){
      //Task* task_multigrid_smooth = scinew Task("linSolver::cg_multigridSmooth_" +  std::to_string(coarse_level->getID()),
          //this, &linSolver::cg_multigrid_smooth,red_black_switch, cg_iter);

      //task_multigrid_smooth->requires(Task::NewDW,d_residualLabel, Ghost::AroundCells, cg_ghost);
      //task_multigrid_smooth->requires(Task::NewDW,d_bigZLabel, Ghost::AroundCells,cg_ghost);
      //task_multigrid_smooth->modifies(d_bigZLabel);

      //for (unsigned int i=0;i<d_precMLabel.size();i++){
        //task_multigrid_smooth->requires(Task::NewDW,d_precMLabel[i], Ghost::AroundCells,cg_ghost+1); // NOT REALLY 0 ghost cells, we are taking shortcuts upstream
      //}


      //if (l==maxLevels-1 && red_black_switch==0){
        //task_multigrid_smooth->computes(d_resSumLabel [cg_iter+1]);
        //task_multigrid_smooth->computes(d_convMaxLabel[cg_iter]);
      //}
      //sched->addTask(task_multigrid_smooth, level_patches,matls);
    //}
    

  //}


  //Task* task4 = scinew Task("linSolver::cg_task4",
                           //this, &linSolver::cg_task4, cg_iter);

  //task4->requires(Task::NewDW,d_convMaxLabel[cg_iter]);
  //task4->requires(Task::NewDW,d_resSumLabel[cg_iter]);
  //task4->requires(Task::NewDW,d_resSumLabel[cg_iter+1]);
  //task4->modifies(d_smallPLabel);
  //task4->requires(Task::NewDW,d_bigZLabel,Ghost::None, 0);

  //sched->addTask(task4, fineLevel->eachPatch(),matls);
  } // END CG_iter



    //auto taskDependencies = [&](Task* tsk) {
      //tsk->modifies(xLabel);
    //};

    //create_portable_tasks(taskDependencies, this,
                          //"PressureEqn::blindGuessToLinearSystem",
                          //&PressureEqn::blindGuessToLinearSystem<UINTAH_CPU_TAG>,
                          //&PressureEqn::blindGuessToLinearSystem<KOKKOS_OPENMP_TAG>,
                          //&PressureEqn::blindGuessToLinearSystem<KOKKOS_CUDA_TAG>,
                          //sched, level->eachPatch(),m_materialManager->allMaterials( "Arches" ), TASKGRAPH::DEFAULT);


   }
  

template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::blindGuessToLinearSystem(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj){


  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

      auto x_values   = new_dw->getGridVariable<CCVariable<double>, double , MemSpace>(  xLabel    , indx , patch, Ghost::None, 0 );
    parallel_initialize(exObj,0.0,x_values);
  }
}



template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_init1(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj, int rk_step){
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


    auto A_m   = new_dw->getConstGridVariable<constCCVariable<Stencil7> , Stencil7 , MemSpace>( ALabel , indx , patch, Ghost::AroundCells, 1 );
    auto g_v   = new_dw->getConstGridVariable<constCCVariable<double> , double , MemSpace>    (  guess , indx , patch, Ghost::AroundCells, 1 );
    auto b_v   = new_dw->getConstGridVariable<constCCVariable<double> , double , MemSpace>    (  bLabel, indx , patch, Ghost::None, 0 );
    auto x_v   = new_dw->getGridVariable<CCVariable<double> , double , MemSpace>    (  xLabel, indx , patch, Ghost::AroundCells, 1 , getModifiable);
    auto residual = new_dw->getGridVariable<CCVariable<double> , double , MemSpace>    ( d_residualLabel, indx , patch, Ghost::AroundCells, cg_ghost );

    auto BJmat=createContainer<CCVariable<double>, double,num_prec_elem, MemSpace>(num_prec_elem); 
    for (unsigned int i=0;i<d_precMLabel.size();i++){
        new_dw->assignGridVariable<CCVariable<double>, double, MemSpace>(BJmat[i],d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
    }

    if (rk_step==0){
      parallel_initialize(exObj,0.0,residual,BJmat);  
    }



    Uintah::parallel_for(exObj, range,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
               residual(i,j,k)=b_v(i,j,k)-(A_m(i,j,k)[6]*g_v(i,j,k)+
                                           A_m(i,j,k)[0]*g_v(i-1,j,k)+
                                           A_m(i,j,k)[1]*g_v(i+1,j,k)+
                                           A_m(i,j,k)[2]*g_v(i,j-1,k)+
                                           A_m(i,j,k)[3]*g_v(i,j+1,k)+
                                           A_m(i,j,k)[4]*g_v(i,j,k-1)+
                                           A_m(i,j,k)[5]*g_v(i,j,k+1));
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

template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_init2(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj,int iter,  int rk_step){
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
      parallel_initialize(exObj,0.0,smallP);
    }
  
    if (iter == final_iter){
    auto littleQ= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_littleQLabel  ,matl,patch,Ghost::AroundCells,1); // computes with scratch ghosts
    auto bigZ= new_dw->getGridVariable<CCVariable<double>, double, MemSpace> (d_bigZLabel  ,matl,patch,Ghost::AroundCells,cg_ghost);
    parallel_initialize(exObj,0.0,bigZ,littleQ);
    }


    for (unsigned int i=0;i<d_precMLabel.size();i++){
      BJmat[i] =  new_dw->getConstGridVariable<constCCVariable<double>, double, MemSpace>(d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
    }
       
       //               A      b       x0
      precondition_relax(exObj,BJmat,residual,smallP,idxLo, idxHi,iter, patch);
      
  if (iter==final_iter){
        Uintah::parallel_reduce_sum(exObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED
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


template <typename ExecutionSpace, typename MemSpace, typename grid_T, typename grid_CT>
void
PressureEqn::precondition_relax(ExecutionObject<ExecutionSpace, MemSpace>& exObj,
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

          Uintah::parallel_for(exObj, rangetemp,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
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
      Uintah::parallel_for(exObj, rangedynamic,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
          if ( (i + j +k + rb_i +rb_switch )% 2 ==0){    
          bigZ(i,j,k)= (residual(i,j,k) - precMatrix[1](i,j,k)*bigZ(i-1,j,k)-precMatrix[1](i+1,j,k)*bigZ(i+1,j,k)  // SYMMTRIC APPROXIMATION
                                        - precMatrix[2](i,j,k)*bigZ(i,j-1,k)-precMatrix[2](i,j+1,k)*bigZ(i,j+1,k) 
                                        - precMatrix[3](i,j,k)*bigZ(i,j,k-1)-precMatrix[3](i,j,k+1)*bigZ(i,j,k+1) ) / precMatrix[0](i,j,k) ; //red_black


          } 

          });
     } // this for loop exists to try and reduce mpi communication costs 
  } else{
    Uintah::BlockRange jrange(idxLo,idxHi); // assumes 1 ghost cell
    Uintah::parallel_for(exObj, jrange,   KOKKOS_LAMBDA (int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
        bigZ(i,j,k)=residual(i,j,k)/precMatrix[0](i,j,k) ;// SYMMTRIC APPROXIMATION
        });
  }
}

template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_task1(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj,int iter){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
  double correction_sum=0.0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
 
    auto A_m   = new_dw->getConstGridVariable<constCCVariable<Stencil7> , Stencil7 , MemSpace>( ALabel , indx , patch, Ghost::None, 0 );
    auto smallP   = new_dw->getConstGridVariable<constCCVariable<double> , double , MemSpace> (d_smallPLabel , indx , patch, Ghost::AroundCells, 1 );
    auto littleQ   = new_dw->getGridVariable<CCVariable<double> , double , MemSpace>  (d_littleQLabel, indx , patch, Ghost::AroundCells, 1 );

    Uintah::parallel_reduce_sum(exObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED

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

template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_task2(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj,int iter){
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

    auto residual = new_dw->getGridVariable<CCVariable<double> , double , MemSpace>    ( d_residualLabel, indx , patch, Ghost::AroundCells, max(cg_ghost,1),getModifiable );
    auto x_v   = new_dw->getGridVariable<CCVariable<double> , double , MemSpace>    (  xLabel, indx , patch, Ghost::AroundCells, 1 , getModifiable);
    auto smallP   = new_dw->getConstGridVariable<constCCVariable<double> , double , MemSpace>(d_smallPLabel , indx , patch, Ghost::None, 0 );
    auto littleQ   = new_dw->getConstGridVariable<constCCVariable<double> , double , MemSpace>(d_littleQLabel , indx , patch, Ghost::None, 0 );

    Uintah::parallel_for( exObj,range,  KOKKOS_LAMBDA(int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                               x_v(i,j,k)=x_v(i,j,k)+correction_factor*smallP(i,j,k);
                               residual(i,j,k)=residual(i,j,k)-correction_factor*littleQ(i,j,k);
                                });
  } // patch loop
}


template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_task3(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj,int iter){
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
    parallel_initialize(exObj, 0.0, bigZ);

    precondition_relax(exObj,BJmat,residual,bigZ,idxLo, idxHi,-1, patch);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
         
     Uintah::parallel_reduce_sum(exObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& sum){  //compute correction, GHOST CELLS REQUIRED
          sum=residual(i,j,k)*bigZ(i,j,k); // reduction
       }, R_squared);

     Uintah::parallel_reduce_min(exObj, range,   KOKKOS_LAMBDA (int i, int j, int k, double& lmin){  //compute correction, GHOST CELLS REQUIRED
          lmin=-fabs(residual(i,j,k));  // presumably most efficcient to comptue here.......could be computed earlier
       }, max_residual);
          
  }


  new_dw->put(sum_vartype(R_squared),d_resSumLabel[iter+1]);
  new_dw->put(max_vartype(-max_residual),d_convMaxLabel[iter]);


}


template <typename ExecutionSpace, typename MemSpace>
void
PressureEqn::cg_task4(const PatchSubset* patches,
                           const MaterialSubset* matls,
                           OnDemandDataWarehouse* old_dw,
                           OnDemandDataWarehouse* new_dw,
                           UintahParams& uintahParams,
                           ExecutionObject<ExecutionSpace, MemSpace>& exObj,int iter){

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


       Uintah::parallel_for( exObj, range,   KOKKOS_LAMBDA (int i, int j, int k){
                             smallP(i,j,k)=bigZ(i,j,k)+beta*smallP(i,j,k);
                            });

}
}





