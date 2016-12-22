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

  Vector DX = patch->dCell();
  SFCXVariable<double>& xmom = tsk_info->get_uintah_field_add<SFCXVariable<double> >( m_xmom );
  SFCYVariable<double>& ymom = tsk_info->get_uintah_field_add<SFCYVariable<double> >( m_ymom );
  SFCZVariable<double>& zmom = tsk_info->get_uintah_field_add<SFCZVariable<double> >( m_zmom );
  constCCVariable<double>& p = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_press);

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  Uintah::parallel_for( range, [&](int i, int j, int k){

    xmom(i,j,k) += ( p(i,j,k) - p(i-1,j,k) ) / DX.x();
    ymom(i,j,k) += ( p(i,j,k) - p(i,j-1,k) ) / DX.y();
    zmom(i,j,k) += ( p(i,j,k) - p(i,j,k-1) ) / DX.z();

  });
}
