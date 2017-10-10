#include <CCA/Components/Arches/Transport/AddPressGradient.h>
#include <CCA/Components/Arches/GridTools.h>

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

  GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1, 0)
  Uintah::BlockRange x_range( low_fx_patch_range, high_fx_patch_range );

  Uintah::parallel_for( x_range, [&](int i, int j, int k){

    xmom(i,j,k) += dt * ( p(i-1,j,k) - p(i,j,k) ) / DX.x();

  });

  GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1, 0)
  Uintah::BlockRange y_range( low_fy_patch_range, high_fy_patch_range );

  Uintah::parallel_for( y_range, [&](int i, int j, int k){

    ymom(i,j,k) += dt * ( p(i,j-1,k) - p(i,j,k) ) / DX.y();

  });

  GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1, 0)
  Uintah::BlockRange z_range( low_fz_patch_range, high_fz_patch_range );
  Uintah::parallel_for( z_range, [&](int i, int j, int k){

    zmom(i,j,k) += dt * ( p(i,j,k-1) - p(i,j,k) ) / DX.z();

  });
}
