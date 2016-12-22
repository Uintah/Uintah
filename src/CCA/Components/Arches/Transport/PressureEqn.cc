#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/GridTools.h>

using namespace Uintah;

typedef ArchesFieldContainer AFC;
typedef ArchesTaskInfoManager ATIM;

//--------------------------------------------------------------------------------------------------
PressureEqn::PressureEqn( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {
}

//--------------------------------------------------------------------------------------------------
PressureEqn::~PressureEqn(){
}

//--------------------------------------------------------------------------------------------------
void PressureEqn::create_local_labels(){

  register_new_variable<CCVariable<Stencil7> >( "A_press" );
  register_new_variable<CCVariable<double> >( "b_press" );

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
PressureEqn::register_initialize(
  std::vector<AFC::VariableInformation>& variable_registry ){

  register_variable( "A_press", AFC::COMPUTES, variable_registry );
  register_variable( "b_press", AFC::COMPUTES, variable_registry );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "x-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "y-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "z-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::initialize( const Patch* patch, ATIM* tsk_info ){

  Vector DX = patch->dCell();
  double area_EW = DX.y()*DX.z();
  double area_NS = DX.x()*DX.z();
  double area_TB = DX.x()*DX.y();

  CCVariable<Stencil7>& Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7> >("A_press");
  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  constCCVariable<double>& eps = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);
  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  for ( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){
    IntVector c = *iter;
    Stencil7& A = Apress[c];
    A.e = 0.0;
    A.w = 0.0;
    A.n = 0.0;
    A.s = 0.0;
    A.t = 0.0;
    A.b = 0.0;
  }

  b.initialize(0.0);

  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

    IntVector c = *iter;
    IntVector E  = c + IntVector(1,0,0);
    IntVector N  = c + IntVector(0,1,0);
    IntVector T  = c + IntVector(0,0,1);

    // A
    Stencil7& A = Apress[c];

    A.e = area_EW/DX.x();
    A.w = area_EW/DX.x();
    A.n = area_NS/DX.y();
    A.s = area_NS/DX.y();
    A.t = area_TB/DX.z();
    A.b = area_TB/DX.z();

    // b
    b[c] = area_EW * ( xmom[E] - xmom[c] ) +
           area_NS * ( ymom[N] - ymom[c] ) +
           area_TB * ( zmom[T] - zmom[c] );

  }
}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_init(
  std::vector<AFC::VariableInformation>& variable_registry ){

  register_variable( "A_press", AFC::COMPUTES, variable_registry );
  register_variable( "A_press", AFC::REQUIRES, 0, AFC::OLDDW, variable_registry );
  register_variable( "b_press", AFC::COMPUTES, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::timestep_init( const Patch* patch, ATIM* tsk_info ){

  CCVariable<Stencil7>& Apress = tsk_info->get_uintah_field_add<CCVariable<Stencil7> >("A_press");
  constCCVariable<Stencil7>& old_Apress = tsk_info->get_const_uintah_field_add<constCCVariable<Stencil7> >("A_press");
  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");

  b.initialize(0.0);
  Apress.copyData( Apress );

}


//--------------------------------------------------------------------------------------------------
void
PressureEqn::register_timestep_eval(
  std::vector<AFC::VariableInformation>& variable_registry,
  const int time_substep ){

  register_variable( "b_press", AFC::MODIFIES, variable_registry );
  register_variable( m_eps_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "x-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "y-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( "z-mom", AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
void
PressureEqn::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  Vector DX = patch->dCell();
  double area_EW = DX.y()*DX.z();
  double area_NS = DX.x()*DX.z();
  double area_TB = DX.x()*DX.y();

  CCVariable<double>& b = tsk_info->get_uintah_field_add<CCVariable<double> >("b_press");
  constCCVariable<double>& eps = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_eps_name);
  constSFCXVariable<double>& xmom = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("x-mom");
  constSFCYVariable<double>& ymom = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("y-mom");
  constSFCZVariable<double>& zmom = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("z-mom");

  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

    IntVector c = *iter;
    IntVector E  = c + IntVector(1,0,0);
    IntVector N  = c + IntVector(0,1,0);
    IntVector T  = c + IntVector(0,0,1);

    // b
    b[c] = area_EW * ( xmom[E] - xmom[c] ) +
           area_NS * ( ymom[N] - ymom[c] ) +
           area_TB * ( zmom[T] - zmom[c] );

    b[c] *= eps[c];

  }
}
