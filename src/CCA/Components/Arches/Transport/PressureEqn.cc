#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Ports/SolverInterface.h>

using namespace Uintah;

typedef ArchesFieldContainer AFC;
typedef ArchesTaskInfoManager ATIM;

//--------------------------------------------------------------------------------------------------
PressureEqn::PressureEqn( std::string task_name, int matl_index, SimulationStateP shared_state ) :
TaskInterface( task_name, matl_index ) {

  m_hypreSolver_parameters = NULL;
  m_sharedState = shared_state;
  m_pressure_name = "pressure";

}

//--------------------------------------------------------------------------------------------------
PressureEqn::~PressureEqn(){

  delete m_hypreSolver_parameters;

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


}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::setup_solver( ProblemSpecP& db ){

  ProblemSpecP db_pressure = db->findBlock("KMomentum")->findBlock("PressureSolver");

  if ( !db_pressure ){
    throw ProblemSetupException("Error: You must specify a <PressureSolver> block in the UPS file.",__FILE__,__LINE__);
  }

  m_hypreSolver_parameters = m_hypreSolver->readParameters(db_pressure, "pressure",
                                                           m_sharedState );
  m_hypreSolver_parameters->setSolveOnExtraCells(false);

  //force a zero setup frequency since nothing else
  //makes any sense at the moment.
  m_hypreSolver_parameters->setSetupFrequency(0.0);

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
  //register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  //register_variable( "x-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  //register_variable( "y-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  //register_variable( "z-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::initialize( const Patch* patch, ATIM* tsk_info ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();

  CCVariable<Stencil7>& Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7> >("A_press");
  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  CCVariable<double>& x = tsk_info->get_uintah_field_add<CCVariable<double> >(m_pressure_name);
  CCVariable<double>& guess = tsk_info->get_uintah_field_add<CCVariable<double> >("guess_press");
  //constCCVariable<double>& eps = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);
  //constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  //constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  //constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");


  b.initialize(0.0);
  x.initialize(0.0);
  guess.initialize(0.0);



  //const double dt = tsk_info->get_dt();
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    Stencil7& A = Apress(i,j,k);
    A.e = 0.0;
    A.w = 0.0;
    A.n = 0.0;
    A.s = 0.0;
    A.t = 0.0;
    A.b = 0.0;

  });

   Uintah::BlockRange range2(patch->getCellLowIndex(), patch->getCellHighIndex() );
   Uintah::parallel_for( range2, [&](int i, int j, int k){

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
void
PressureEqn::timestep_init( const Patch* patch, ATIM* tsk_info ){

  CCVariable<Stencil7>& Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7> >("A_press");
  constCCVariable<Stencil7>& old_Apress = tsk_info->get_const_uintah_field_add<constCCVariable<Stencil7> >("A_press");
  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  CCVariable<double>& x = tsk_info->get_uintah_field_add<CCVariable<double> >(m_pressure_name);
  CCVariable<double>& guess = tsk_info->get_uintah_field_add<CCVariable<double> >("guess_press");

  b.initialize(0.0);
  x.initialize(0.0);
  guess.initialize(0.0);
  Apress.copyData( old_Apress );

}


//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_eval(
  std::vector<AFC::VariableInformation>& variable_registry,
  const int time_substep, const bool packed_tasks ){

  register_variable( "b_press", AFC::MODIFIES, variable_registry, time_substep );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "x-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "y-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "z-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
 // register_variable( m_drhodt_name, AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  Vector DX = patch->dCell();
  const double area_EW = DX.y()*DX.z();
  const double area_NS = DX.x()*DX.z();
  const double area_TB = DX.x()*DX.y();

  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  constCCVariable<double>& eps = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);
  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");
  //constCCVariable<double>& drhodt = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_drhodt_name);

  const double dt = tsk_info->get_dt();
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for( range, [&](int i, int j, int k){

    b(i,j,k) = ( area_EW * ( xmom(i+1,j,k) - xmom(i,j,k) ) +
                 area_NS * ( ymom(i,j+1,k) - ymom(i,j,k) ) +
                 area_TB * ( zmom(i,j,k+1) - zmom(i,j,k) ) )/dt;
                 //+V*drhodt(i,j,k)  ) / dt ;
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
void
PressureEqn::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  //This only applies BCs to A. Boundary conditions to the RHS are handled upstream in RhoUHatBC
  CCVariable<Stencil7>& A = tsk_info->get_uintah_field_add<CCVariable<Stencil7> >("A_press");
  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  constCCVariable<double>& eps = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    Uintah::Iterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
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

    for (cell_iter.reset(); !cell_iter.done(); cell_iter++ ){

      IntVector c = *cell_iter - iDir;

      A[c].p = A[c].p + sign * A[c][face];
      A[c][face] = 0.;

    }
  }

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

  const MaterialSet* matls = m_sharedState->allArchesMaterials();
  IntVector m_periodic_vector = level->getPeriodicBoundaries();

  const bool isPeriodic = m_periodic_vector.x() == 1 && m_periodic_vector.y() == 1 && m_periodic_vector.z() ==1;
  if ( isPeriodic || m_enforceSolvability ) {
    m_hypreSolver->scheduleEnforceSolvability<CCVariable<double> >(level, sched, matls, b, time_substep);
  }

  bool modifies_hypre = false;

  bool modifies_x = true; //because x was computed upstream

  if ( time_substep > 0 ) {
    modifies_hypre = true;
  }

  m_hypreSolver->scheduleSolve(level, sched,  matls,
                               A,      Task::NewDW,
                               x,      modifies_x,
                               b,      Task::NewDW,
                               guess,  Task::NewDW,
                               m_hypreSolver_parameters,
                               modifies_hypre);

}
