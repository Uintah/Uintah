#include <CCA/Components/Arches/Transport/AddPressGradient.h>

using namespace Uintah;
typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
AddPressGradient::AddPressGradient( std::string task_name, int matl_index ) :
AtomicTaskInterface( task_name, matl_index )
{
}

//--------------------------------------------------------------------------------------------------
AddPressGradient::~AddPressGradient()
{
}

//--------------------------------------------------------------------------------------------------
void AddPressGradient::problemSetup( ProblemSpecP& db ){
  m_xmom = "x-mom";
  m_ymom = "y-mom";
  m_zmom = "z-mom";
  m_press = "pressure";
}

//--------------------------------------------------------------------------------------------------
void AddPressGradient::create_local_labels(){
}

//--------------------------------------------------------------------------------------------------
void AddPressGradient::register_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                 const int time_substep ){
  register_variable( m_xmom, AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( m_ymom, AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( m_zmom, AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( m_press, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, m_task_name );
}

void AddPressGradient::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  const double dt = tsk_info->get_dt();
  Vector DX = patch->dCell();
  SFCXVariable<double>& xmom = tsk_info->get_uintah_field_add<SFCXVariable<double> >( m_xmom );
  SFCYVariable<double>& ymom = tsk_info->get_uintah_field_add<SFCYVariable<double> >( m_ymom );
  SFCZVariable<double>& zmom = tsk_info->get_uintah_field_add<SFCZVariable<double> >( m_zmom );
  constCCVariable<double>& p = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_press);

  // because the hypre solve required a positive diagonal
  // so we -1 * ( Ax = b ) requiring that we change the sign
  // back.

  // boundary conditions on the pressure fields are applied
  // post linear solve in the PressureBC.cc class.

  IntVector shift(0,0,0);
  if ( patch->getBCType(Patch::xplus) != Patch::Neighbor ) shift[0] = 1;

  Uintah::BlockRange x_range( patch->getCellLowIndex(), patch->getCellHighIndex()+shift );

  Uintah::parallel_for( x_range, [&](int i, int j, int k){

    xmom(i,j,k) += dt * ( p(i-1,j,k) - p(i,j,k) ) / DX.x();

  });

  shift[0] = 0;
  if ( patch->getBCType(Patch::yplus) != Patch::Neighbor ) shift[1] = 1;

  Uintah::BlockRange y_range( patch->getCellLowIndex(), patch->getCellHighIndex()+shift );

  Uintah::parallel_for( y_range, [&](int i, int j, int k){

    ymom(i,j,k) += dt * ( p(i,j-1,k) - p(i,j,k) ) / DX.y();

  });

  shift[1] = 0;
  if ( patch->getBCType(Patch::zplus) != Patch::Neighbor ) shift[2] = 1;

  Uintah::BlockRange z_range( patch->getCellLowIndex(), patch->getCellHighIndex()+shift );
  Uintah::parallel_for( z_range, [&](int i, int j, int k){

    zmom(i,j,k) += dt * ( p(i,j,k-1) - p(i,j,k) ) / DX.z();

  });
}
