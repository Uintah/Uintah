#ifndef Uintah_Component_Arches_UnweightVariable_h
#define Uintah_Component_Arches_UnweightVariable_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>


namespace Uintah{

  template <typename T>
  class UnweightVariable : public TaskInterface {

public:

    UnweightVariable<T>( std::string task_name, int matl_index );
    ~UnweightVariable<T>();

    void problemSetup( ProblemSpecP& db );

    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      UnweightVariable* build()
      { return scinew UnweightVariable<T>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

 protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    std::string m_var_name;
    std::string m_rho_name;
    std::string m_un_var_name;
    std::vector<std::string> m_eqn_names;
    std::vector<std::string> m_un_eqn_names;
    std::vector<int> m_ijk_off;
    int m_dir;
    int Nghost_cells;
    bool m_compute_mom;

  };

//------------------------------------------------------------------------------------------------
template <typename T>
UnweightVariable<T>::UnweightVariable( std::string task_name, int matl_index ) :
TaskInterface( task_name, matl_index ) {

  ArchesCore::VariableHelper<T> helper;
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);
  m_ijk_off.push_back(0);


  if ( helper.dir == ArchesCore::XDIR ||
       helper.dir == ArchesCore::YDIR ||
       helper.dir == ArchesCore::ZDIR ){
       m_dir = helper.dir;
       m_ijk_off[0] = helper.ioff;
       m_ijk_off[1] = helper.joff;
       m_ijk_off[2] = helper.koff;
  }

}

//--------------------------------------------------------------------------------------------------
template <typename T>
UnweightVariable<T>::~UnweightVariable()
{}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::problemSetup( ProblemSpecP& db ){
  // works for scalars
  //m_total_eqns = 0;

  for( ProblemSpecP db_eqn = db->findBlock("eqn"); db_eqn != nullptr; db_eqn = db_eqn->findNextBlock("eqn") ) {
    std::string eqn_name;
    db_eqn->getAttribute("label", eqn_name);
    m_var_name = "rho_"+eqn_name;
    m_un_var_name = eqn_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);

  }
  m_compute_mom = false;
  if (_task_name == "uVel"){
    m_var_name = "x-mom";
    m_un_var_name = _task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    m_compute_mom = true;

  } else if (_task_name == "vVel"){
    m_var_name = "y-mom";
    m_un_var_name = _task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    m_compute_mom = true;

  } else if (_task_name == "wVel"){
    m_var_name = "z-mom";
    m_un_var_name = _task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    m_compute_mom = true;

  }

  m_rho_name = "density";
  Nghost_cells = 0;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::create_local_labels(){
}

//--------------------------------------------------------------------------------------------------

template <typename T>
void UnweightVariable<T>::register_initialize(
  std::vector<ArchesFieldContainer::VariableInformation>&
  variable_registry , const bool pack_tasks )
{
  //ArchesCore::VariableHelper<T> helper;
  //if ( helper.dir == ArchesCore::NODIR){
    // scalar at cc
  const int istart = 0;
  const int iend = m_eqn_names.size();
  for (int ieqn = istart; ieqn < iend; ieqn++ ){
    register_variable( m_un_eqn_names[ieqn] , ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( m_eqn_names[ieqn],     ArchesFieldContainer::MODIFIES ,  variable_registry );
  }
  //} else {
  //  register_variable( m_var_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
 //   register_variable( m_un_var_name, ArchesFieldContainer::MODIFIES ,  variable_registry );
  //}
  register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  ArchesCore::VariableHelper<T> helper;
  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
  constCCVariable<double>& rho = tsk_info->get_const_uintah_field_add<constCCVariable<double>>(m_rho_name);

  const int ioff = m_ijk_off[0];
  const int joff = m_ijk_off[1];
  const int koff = m_ijk_off[2];
  IntVector cell_lo = patch->getCellLowIndex();
  IntVector cell_hi = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(cell_lo,cell_hi,ioff,0,joff,0,koff,0)
  Uintah::BlockRange range( cell_lo, cell_hi );

  //if ( helper.dir == ArchesCore::NODIR){
    //scalar
  const int istart = 0;
  const int iend = m_eqn_names.size();
  for (int ieqn = istart; ieqn < iend; ieqn++ ){
    T&  var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);
    CT& un_var = tsk_info->get_const_uintah_field_add<CT>(m_un_eqn_names[ieqn]);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double rho_inter = 0.5 * (rho(i,j,k)+rho(i-ioff,j-joff,k-koff));
      var(i,j,k) = un_var(i,j,k)*rho_inter;
    });
  }
  //}else {
  //  CT&  var = tsk_info->get_const_uintah_field_add<CT>(m_var_name);
  //  T& un_var = tsk_info->get_uintah_field_add<T>(m_un_var_name);

  //  Uintah::parallel_for( range, [&](int i, int j, int k){
  //    const double rho_inter = 0.5 * (rho(i,j,k)+rho(i-ioff,j-joff,k-koff));
  //    un_var(i,j,k) = var(i,j,k)/rho_inter;
  //  });
  //}

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::register_compute_bcs(
        std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
       const int time_substep , const bool packed_tasks)
{
  ArchesCore::VariableHelper<T> helper;
  const int istart = 0;
  const int iend = m_eqn_names.size();

  if ( helper.dir == ArchesCore::NODIR || m_compute_mom == false){
    // scalar at cc
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
      register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
  }else {

    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
  }
   register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info )
{

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  ArchesCore::VariableHelper<T> helper;
  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

  constCCVariable<double>& rho = tsk_info->get_const_uintah_field_add<constCCVariable<double>>(m_rho_name);
  const IntVector vDir(helper.ioff, helper.joff, helper.koff);

  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());

    if ( on_this_patch ){
      //Get the iterator
      Uintah::Iterator cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
      std::string facename = i_bc->second.name;
      IntVector iDir = patch->faceDirection( i_bc->second.face );

      const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

      const int istart = 0;
      const int iend = m_eqn_names.size();

      if ( helper.dir == ArchesCore::NODIR){
        //scalar
        for (int ieqn = istart; ieqn < iend; ieqn++ ){
          T&  var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);
          CT& un_var = tsk_info->get_const_uintah_field_add<CT>(m_un_eqn_names[ieqn]);

          for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
            IntVector c = *cell_iter;
            var[c] = un_var[c]*rho[c];
          }
        }
      } else if (m_compute_mom == false) {
        for (int ieqn = istart; ieqn < iend; ieqn++ ){

          T&  var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);// rho*phi
          CT& un_var = tsk_info->get_const_uintah_field_add<CT>(m_un_eqn_names[ieqn]); // phi

          if ( dot == -1 ){
            // face (-) in Staggered Variablewe set BC at 0
            for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
              IntVector c = *cell_iter;
              IntVector cp = *cell_iter - iDir;
              const double rho_inter = 0.5 * (rho[c]+rho[cp]);
              var[cp] = un_var[cp]*rho_inter; // BC
              var[c]  = var[cp]; // extra cell
            }
          } else {
         // face (+) in Staggered Variablewe set BC at extra cell
            for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
              IntVector c = *cell_iter;
              IntVector cp = *cell_iter - iDir;
              const double rho_inter = 0.5 * (rho[c]+rho[cp]);
              var[c] = un_var[c]*rho_inter; // BC and extra cell value
            }
         }
         }
      } else {
        // only works if var is mom
        for (int ieqn = istart; ieqn < iend; ieqn++ ){

          T&  un_var = tsk_info->get_uintah_field_add<T>(m_un_eqn_names[ieqn]);
          CT& var = tsk_info->get_const_uintah_field_add<CT>(m_eqn_names[ieqn]);

          if ( dot == -1 ){
            // face (-) in Staggered Variablewe set BC at 0
            for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
              IntVector c = *cell_iter;
              IntVector cp = *cell_iter - iDir;
              const double rho_inter = 0.5 * (rho[c]+rho[cp]);
              un_var[cp] = var[cp]/rho_inter; // BC
              un_var[c] = un_var[cp]; // extra cell
            }
          } else {
         // face (+) in Staggered Variablewe set BC at extra cell
            for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
              IntVector c = *cell_iter;
              IntVector cp = *cell_iter - iDir;
              const double rho_inter = 0.5 * (rho[c]+rho[cp]);
              un_var[c] = var[c]/rho_inter; // BC and extra cell value
            }
         }
         }

      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::register_timestep_eval(
  std::vector<ArchesFieldContainer::VariableInformation>&
  variable_registry, const int time_substep , const bool packed_tasks)
{
  const int istart = 0;
  const int iend = m_eqn_names.size();
  for (int ieqn = istart; ieqn < iend; ieqn++ ){
    register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
    register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  }
  register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ArchesCore::VariableHelper<T> helper;

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
  constCCVariable<double>& rho = tsk_info->get_const_uintah_field_add<constCCVariable<double>>(m_rho_name);
  const int ioff = m_ijk_off[0];
  const int joff = m_ijk_off[1];
  const int koff = m_ijk_off[2];

  IntVector cell_lo = patch->getCellLowIndex();
  IntVector cell_hi = patch->getCellHighIndex();

  GET_WALL_BUFFERED_PATCH_RANGE(cell_lo,cell_hi,ioff,0,joff,0,koff,0)
  Uintah::BlockRange range( cell_lo, cell_hi );

  const int istart = 0;
  const int iend = m_eqn_names.size();
  for (int ieqn = istart; ieqn < iend; ieqn++ ){
    T& un_var = tsk_info->get_uintah_field_add<T>(m_un_eqn_names[ieqn]);
    CT& var = tsk_info->get_const_uintah_field_add<CT>(m_eqn_names[ieqn]);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double rho_inter = 0.5 * (rho(i,j,k)+rho(i-ioff,j-joff,k-koff));
      un_var(i,j,k) = var(i,j,k)/rho_inter;
    });

  }
}
}
#endif
