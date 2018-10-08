#ifndef Uintah_Component_Arches_UnweightVariable_h
#define Uintah_Component_Arches_UnweightVariable_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Transport/TransportHelper.h>
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

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      UnweightVariable* build()
      { return scinew UnweightVariable<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

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
    ArchesCore::EQUATION_CLASS m_eqn_class;
    //std::map<std::string, double> m_scaling_info;
    struct Scaling_info {
      std::string unscaled_var; // unscaled value
      double constant; // 
    };
    std::map<std::string, Scaling_info> m_scaling_info;
    std::string m_volFraction_name{"volFraction"};

    struct Clipping_info {
      std::string var; // 
      double high; // 
      double low;
    };
    std::map<std::string, Clipping_info> m_clipping_info;
    
    
    //bool m_compute_mom;

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

  std::string eqn_class = "density_weighted";

  if ( db->findAttribute("class") ){
    db->getAttribute("class", eqn_class);
  }
  
  m_eqn_class = ArchesCore::assign_eqn_class_enum( eqn_class );

  std::string premultiplier_name  = get_premultiplier_name(m_eqn_class);
  std::string postmultiplier_name = get_postmultiplier_name(m_eqn_class);
  
  std::string env_number="NA";
  if (m_eqn_class == ArchesCore::DQMOM) {      
    db->findBlock("env_number")->getAttribute("number", env_number);    
  }
  
  for( ProblemSpecP db_eqn = db->findBlock("eqn"); db_eqn != nullptr; db_eqn = db_eqn->findNextBlock("eqn") ) {
    std::string eqn_name;
    db_eqn->getAttribute("label", eqn_name);
    
    if (m_eqn_class == ArchesCore::DQMOM){
      std::string delimiter = env_number ;
      std::string name_1    = eqn_name.substr(0, eqn_name.find(delimiter));
      m_var_name = name_1 + postmultiplier_name + env_number;
    } else {
      m_var_name    = premultiplier_name + eqn_name + postmultiplier_name;
    }

    m_un_var_name = eqn_name;

    if (db_eqn->findBlock("no_weight_factor") == nullptr ){
      m_un_eqn_names.push_back(m_un_var_name);
      m_eqn_names.push_back(m_var_name);
      //Scaling Constant
      if ( db_eqn->findBlock("scaling") ){
        double scaling_constant;
        db_eqn->findBlock("scaling")->getAttribute("value", scaling_constant);
        //m_scaling_info.insert(std::make_pair(m_var_name, scaling_constant));

        Scaling_info scaling_w ;
        scaling_w.unscaled_var = eqn_name;
        scaling_w.constant     = scaling_constant;
        m_scaling_info.insert(std::make_pair(m_var_name, scaling_w));
    
      }
      
    } else {
      // weight do not performe division 
    }  

    //Clipping
    if ( db_eqn->findBlock("clip")){
      double low; double high;
      db_eqn->findBlock("clip")->getAttribute("low", low);
      db_eqn->findBlock("clip")->getAttribute("high", high);

      Clipping_info clipping_eqn ;
      clipping_eqn.var = eqn_name;
      clipping_eqn.high = high;
      clipping_eqn.low  = low;
      m_clipping_info.insert(std::make_pair(m_var_name, clipping_eqn));
    }
    

  }
  
  //m_compute_mom = false;
  if (m_task_name == "uVel"){
    m_eqn_class = ArchesCore::MOMENTUM;
    m_var_name = "x-mom";
    m_un_var_name = m_task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    //m_compute_mom = true;

  } else if (m_task_name == "vVel"){
    m_eqn_class = ArchesCore::MOMENTUM;
    m_var_name = "y-mom";
    m_un_var_name = m_task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    //m_compute_mom = true;

  } else if (m_task_name == "wVel"){
    m_eqn_class = ArchesCore::MOMENTUM;
    m_var_name = "z-mom";
    m_un_var_name = m_task_name;
    m_un_eqn_names.push_back(m_un_var_name);
    m_eqn_names.push_back(m_var_name);
    //m_compute_mom = true;

  }
  Nghost_cells = 1;
  m_rho_name = parse_ups_for_role( ArchesCore::DENSITY, db, "density" );
  if (m_eqn_class == ArchesCore::DENSITY_WEIGHTED) {
    m_rho_name = parse_ups_for_role( ArchesCore::DENSITY, db, "density" );
  }else if (m_eqn_class == ArchesCore::DQMOM){
    db->findBlock("weight_factor")->getAttribute("label", m_rho_name);
    Nghost_cells = 1;
  }
  

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
    register_variable( m_un_eqn_names[ieqn] , ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( m_eqn_names[ieqn],     ArchesFieldContainer::MODIFIES ,  variable_registry );
  }
  //} else {
  //  register_variable( m_var_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
 //   register_variable( m_un_var_name, ArchesFieldContainer::MODIFIES ,  variable_registry );
  //}
  register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry );
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

  // scaling w*Ic
  //int eqn =0;
  for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
    Scaling_info info = ieqn->second;
    T&  var = tsk_info->get_uintah_field_add<T>(ieqn->first);
    //const double scaling_constant = ieqn->second;
    Uintah::parallel_for( range, [&](int i, int j, int k){
      var(i,j,k) /= info.constant;
    });
    //eqn += 1;
  }
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

  if ( helper.dir == ArchesCore::NODIR || m_eqn_class !=ArchesCore::MOMENTUM ){
    // scalar at cc
    
    if (m_eqn_class ==ArchesCore::DQMOM) {
      register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name  );
      for (int ieqn = istart; ieqn < iend; ieqn++ ){
        register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
        register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }
    } else {
      for (int ieqn = istart; ieqn < iend; ieqn++ ){
        register_variable( m_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
        register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }
    }
  } else {

    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
  }
  register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  
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
      Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
      //std::string facename = i_bc->second.name;
      IntVector iDir = patch->faceDirection( i_bc->second.face );

      const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

      const int istart = 0;
      const int iend = m_eqn_names.size();
      const double SMALL = 1e-20;
      if ( helper.dir == ArchesCore::NODIR){
        //scalar
        if (m_eqn_class ==ArchesCore::DQMOM) {
      
          // DQMOM : BCs are Ic_qni, then we need to compute Ic 
          for (int ieqn = istart; ieqn < iend; ieqn++ ){
            CT&  var = tsk_info->get_const_uintah_field_add<CT>(m_eqn_names[ieqn]);
            T& un_var = tsk_info->get_uintah_field_add<T>(m_un_eqn_names[ieqn]);
            constCCVariable<double>& vol_fraction = 
            tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);
          
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              un_var(i,j,k) = var(i,j,k)/(rho(i,j,k)+ SMALL)*vol_fraction(i,j,k);
            });
          }
         // unscaling only DQMOM
          for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
            Scaling_info info = ieqn->second;
            T&  un_var = tsk_info->get_uintah_field_add<T>(info.unscaled_var);
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              un_var(i,j,k) *= info.constant;
            });
          }
        } else {
          for (int ieqn = istart; ieqn < iend; ieqn++ ){
            T&  var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);
            CT& un_var = tsk_info->get_const_uintah_field_add<CT>(m_un_eqn_names[ieqn]);
          
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              int ip=i - iDir[0];
              int jp=j - iDir[1];
              int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k) + rho(ip,jp,kp));
              const double phi_inter = 0.5 * (un_var(i,j,k) + un_var(ip,jp,kp));
              var(i,j,k) = 2.0*rho_inter*phi_inter - un_var(ip,jp,kp)*rho(ip,jp,kp);
            });
          } 
        }
      
      //} else if (m_compute_mom == false) {
      } else if (m_eqn_class !=ArchesCore::MOMENTUM) {
        // variable that are transported in staggered position
        // rho_phi = phi/pho 
        for (int ieqn = istart; ieqn < iend; ieqn++ ){
            T&  var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);// rho*phi
            CT& un_var = tsk_info->get_const_uintah_field_add<CT>(m_un_eqn_names[ieqn]); // phi
      
            if ( dot == -1 ){
            // face (-) in Staggered Variablewe set BC at 0
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              const int ip=i - iDir[0];
              const int jp=j - iDir[1];
              const int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k)+rho(ip,jp,kp));
              var(ip,jp,kp) = un_var(ip,jp,kp)*rho_inter; // BC
              var(i,j,k)  = var(ip,jp,kp); // extra cell
            });
          } else {
         // face (+) in Staggered Variablewe set BC at extra cell
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              const int ip=i - iDir[0];
              const int jp=j - iDir[1];
              const int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k)+rho(ip,jp,kp));
              var(i,j,k) = un_var(i,j,k)*rho_inter; // BC and extra cell value
            });
         }
         }
      } else {
        // only works if var is mom
        for (int ieqn = istart; ieqn < iend; ieqn++ ){

          T&  un_var = tsk_info->get_uintah_field_add<T>(m_un_eqn_names[ieqn]);
          CT& var = tsk_info->get_const_uintah_field_add<CT>(m_eqn_names[ieqn]);

          if ( dot == -1 ){
            // face (-) in Staggered Variablewe set BC at 0
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              const int ip=i - iDir[0];
              const int jp=j - iDir[1];
              const int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k)+rho(ip,jp,kp));
              un_var(ip,jp,kp) = var(ip,jp,kp)/rho_inter; // BC
              un_var(i,j,k) = un_var(ip,jp,kp); // extra cell
            });
          } else if ( dot == 1 ){
            // face (+) in Staggered Variablewe set BC at 0
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              const int ip=i - iDir[0];
              const int jp=j - iDir[1];
              const int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k)+rho(ip,jp,kp));
              un_var(i,j,k) = var(i,j,k)/rho_inter; // extra cell
              // aditional extra cell for staggered variables on face (+)
              const int ie = i + iDir[0];
              const int je = j + iDir[1];
              const int ke = k + iDir[2];
              un_var(ie,je,ke) = un_var(i,j,k);  
            });
          } else {
         // other direction that are not staggered  
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
              const int ip=i - iDir[0];
              const int jp=j - iDir[1];
              const int kp=k - iDir[2];
              const double rho_inter = 0.5 * (rho(i,j,k)+rho(ip,jp,kp));
              un_var(i,j,k) = var(i,j,k)/rho_inter; // BC and extra cell value
            });
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
    register_variable( m_un_eqn_names[ieqn], ArchesFieldContainer::MODIFIES ,  variable_registry );
    register_variable( m_eqn_names[ieqn], ArchesFieldContainer::MODIFIES, variable_registry );
  }
  register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, Nghost_cells, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

}

//--------------------------------------------------------------------------------------------------
template <typename T>
void UnweightVariable<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  ArchesCore::VariableHelper<T> helper;

  //typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
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
    T& var = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      const double rho_inter = 0.5 * (rho(i,j,k)+rho(i-ioff,j-joff,k-koff));
      un_var(i,j,k) = var(i,j,k)/rho_inter;
    });

  }

  // unscaling
  //int eqn =0;
  for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
    Scaling_info info = ieqn->second;
    T& un_var = tsk_info->get_uintah_field_add<T>(info.unscaled_var);
    Uintah::parallel_for( range, [&](int i, int j, int k){
      un_var(i,j,k) *= info.constant;
    });
  }
  
  // clipping   
  for ( auto ieqn = m_clipping_info.begin(); ieqn != m_clipping_info.end(); ieqn++ ){
    Clipping_info info = ieqn->second;
    T& var = tsk_info->get_uintah_field_add<T>(info.var);
    T& rho_var = tsk_info->get_uintah_field_add<T>(ieqn->first);
    Uintah::parallel_for( range, [&](int i, int j, int k){
    if ( var(i,j,k) > info.high ) {

      var(i,j,k)     = info.high;
      rho_var(i,j,k) = rho(i,j,k)*var(i,j,k); 

    } else if ( var(i,j,k) < info.low ) {
      var(i,j,k) = info.low;
      rho_var(i,j,k) = rho(i,j,k)*var(i,j,k); 
    }
    });
  }
  
  

}
}
#endif
