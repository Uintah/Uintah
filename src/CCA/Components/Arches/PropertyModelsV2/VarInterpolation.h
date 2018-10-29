#ifndef Uintah_Component_Arches_VarInterpolation_h
#define Uintah_Component_Arches_VarInterpolation_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>


namespace Uintah{

  template <typename T, typename IT>
  class VarInterpolation : public TaskInterface {

public:

    VarInterpolation<T, IT>( std::string task_name, int matl_index );
    ~VarInterpolation<T, IT>();

    void problemSetup( ProblemSpecP& db );

    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      VarInterpolation* build()
      { return scinew VarInterpolation<T, IT>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

 protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    std::string m_var_name;         ///< The variable to be interpolated
    std::string m_inter_var_name;   ///< The interpolated variable
    std::string m_scheme;
    std::vector<int> m_ijk_off;
    ArchesCore::INTERPOLANT m_int_scheme;
    int m_dir;
    int Nghost_cells;

  };

//------------------------------------------------------------------------------------------------
// This code allows interpolation from F -> CC or CC -> F. The option FF -> FF is dissallowed.
template <typename T, typename IT>
VarInterpolation<T, IT>::VarInterpolation( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

  ArchesCore::VariableHelper<T> helper;
  ArchesCore::VariableHelper<IT> interpolated_helper;
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);

  if ( helper.dir == ArchesCore::NODIR ){
    if ( interpolated_helper.dir == ArchesCore::NODIR ){
      throw InvalidValue("Error: Cannot use variable interpolant CC -> CC.", __FILE__, __LINE__ );
    }
    m_dir = interpolated_helper.dir;
    if ( interpolated_helper.dir == ArchesCore::XDIR ){
      m_ijk_off[0] = helper.ioff;
    } else if ( interpolated_helper.dir == ArchesCore::YDIR ){
      m_ijk_off[1] = helper.joff;
    } else {
      m_ijk_off[2] = helper.koff;
    }
  }

  if ( helper.dir == ArchesCore::XDIR ||
       helper.dir == ArchesCore::YDIR ||
       helper.dir == ArchesCore::ZDIR ){
    if ( interpolated_helper.dir != ArchesCore::NODIR ){
      throw InvalidValue("Error: Cannot use variable interpolant with FC -> FC", __FILE__, __LINE__ );
    }
    m_dir = helper.dir;
    m_ijk_off[0] = helper.ioff;
    m_ijk_off[1] = helper.joff;
    m_ijk_off[2] = helper.koff;
  }

}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
VarInterpolation<T, IT>::~VarInterpolation()
{}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
void VarInterpolation<T, IT>::problemSetup( ProblemSpecP& db ){

  db->findBlock("variable")->getAttribute("label", m_var_name);
  db->findBlock("new_variable")->getAttribute("label", m_inter_var_name);
  db->findBlock("interpolation")->getAttribute("scheme", m_scheme);

  m_int_scheme = ArchesCore::get_interpolant_from_string( m_scheme );

  Nghost_cells = 1;
  if (m_int_scheme== ArchesCore::FOURTHCENTRAL){
    Nghost_cells = 2;
  }

}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
void VarInterpolation<T,IT>::create_local_labels(){
  register_new_variable< IT >( m_inter_var_name );
}

//--------------------------------------------------------------------------------------------------

template <typename T, typename IT>
void VarInterpolation<T,IT>::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>&
  variable_registry , const bool pack_tasks )
{

  register_variable( m_var_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( m_inter_var_name, ArchesFieldContainer::COMPUTES ,  variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
void VarInterpolation<T,IT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  IT& int_var = tsk_info->get_uintah_field_add<IT>(m_inter_var_name);
  int_var.initialize(0.0);

}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
void VarInterpolation<T,IT>::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>&
  variable_registry, const int time_substep , const bool packed_tasks)
{

  register_variable( m_inter_var_name, ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
  register_variable( m_var_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::LATEST, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T, typename IT>
void VarInterpolation<T,IT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  IT& int_var = tsk_info->get_uintah_field_add<IT>(m_inter_var_name);
  T& var = tsk_info->get_const_uintah_field_add<T >(m_var_name);

  const int ioff = m_ijk_off[0];
  const int joff = m_ijk_off[1];
  const int koff = m_ijk_off[2];

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );
  ArchesCore::OneDInterpolator my_interpolant( int_var, var, ioff, joff, koff );

  if ( m_int_scheme == ArchesCore::SECONDCENTRAL ) {

    ArchesCore::SecondCentral ci;
    Uintah::parallel_for( range, my_interpolant, ci );

  } else if ( m_int_scheme== ArchesCore::FOURTHCENTRAL ){

    ArchesCore::FourthCentral ci;
    Uintah::parallel_for( range, my_interpolant, ci );

  }

}
}
#endif
