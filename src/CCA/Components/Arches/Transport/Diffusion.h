#ifndef Uintah_Component_Arches_Diffusion_h
#define Uintah_Component_Arches_Diffusion_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  template <typename T>
  class Diffusion : public TaskInterface {

public:

    Diffusion<T>( std::string task_name, int matl_index );
    ~Diffusion<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (Diffusion) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      Diffusion* build()
      { return scinew Diffusion<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;

    std::vector<std::string> m_eqn_names;
    bool m_has_D;
    std::vector<bool> m_do_diff;
    std::string m_eps_name;
    std::string m_D_name;

  };

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------
  template <typename T>
  Diffusion<T>::Diffusion( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
    m_has_D = false;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  Diffusion<T>::~Diffusion()
  {
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::problemSetup( ProblemSpecP& db ){

    for( ProblemSpecP input_db = db->findBlock("eqn"); input_db != nullptr;
         input_db = input_db->findNextBlock("eqn") ) {

      //Equation name
      std::string eqn_name;
      input_db->getAttribute("label", eqn_name);
      m_eqn_names.push_back(eqn_name);

      //Check to see if this eqn has diffusion enabled
      m_has_D = false;
      if ( input_db->findBlock("diffusion")){
        m_do_diff.push_back(true);
        m_has_D = true;
      } else {
        m_do_diff.push_back(false);
      }

      ArchesCore::GridVarMap<T> var_map;
      var_map.problemSetup( input_db );
      m_eps_name = var_map.vol_frac_name;

    }
    if ( m_has_D ){
      if ( db->findBlock("diffusion_coef") ) {
        db->findBlock("diffusion_coef")->getAttribute("label",m_D_name);
      } else {
        std::stringstream msg;
        msg << "Error: Diffusion specified for task " << m_task_name << std::endl
        << "but no diffusion coefficient label specified." << std::endl;
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::create_local_labels(){

    for (int ieqn = 0; ieqn < int(m_eqn_names.size()); ieqn++ ){
      if ( m_do_diff[ieqn] ){
        register_new_variable<FXT>( m_eqn_names[ieqn]+"_x_dflux" );
        register_new_variable<FYT>( m_eqn_names[ieqn]+"_y_dflux" );
        register_new_variable<FZT>( m_eqn_names[ieqn]+"_z_dflux" );
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    for (int ieqn = 0; ieqn < int(m_eqn_names.size()); ieqn++ ){
      if ( m_do_diff[ieqn] ){
        register_variable(  m_eqn_names[ieqn]+"_x_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
        register_variable(  m_eqn_names[ieqn]+"_y_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
        register_variable(  m_eqn_names[ieqn]+"_z_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      }
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for (int ieqn = 0; ieqn < int(m_eqn_names.size()); ieqn++ ){

      if ( m_do_diff[ieqn] ){
        FXT& x_flux = tsk_info->get_uintah_field_add<FXT>(m_eqn_names[ieqn]+"_x_dflux");
        FYT& y_flux = tsk_info->get_uintah_field_add<FYT>(m_eqn_names[ieqn]+"_y_dflux");
        FZT& z_flux = tsk_info->get_uintah_field_add<FZT>(m_eqn_names[ieqn]+"_z_dflux");

        x_flux.initialize(0.0);
        y_flux.initialize(0.0);
        z_flux.initialize(0.0);
      }
    }
  }


  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::register_timestep_eval(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    for (int ieqn = 0; ieqn < int(m_eqn_names.size()); ieqn++ ){

      if ( m_do_diff[ieqn] ){
        //should this be latest?
        register_variable(  m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 1,
                            ArchesFieldContainer::LATEST, variable_registry,
                            time_substep, m_task_name );
        register_variable(  m_eqn_names[ieqn]+"_x_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
        register_variable(  m_eqn_names[ieqn]+"_y_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
        register_variable(  m_eqn_names[ieqn]+"_z_dflux", ArchesFieldContainer::COMPUTES , variable_registry, m_task_name );
      }
    }
    if ( m_has_D ){
      register_variable( m_D_name, ArchesFieldContainer::REQUIRES , 1 , ArchesFieldContainer::NEWDW  , variable_registry , time_substep, m_task_name );
    }
    register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep, m_task_name );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    CT& eps = tsk_info->get_const_uintah_field_add<CT>(m_eps_name);

    Vector Dx = patch->dCell();

    for (int ieqn = 0; ieqn < int(m_eqn_names.size()); ieqn++ ){

      if ( m_do_diff[ieqn] ){

        FXT& x_flux = tsk_info->get_uintah_field_add<FXT>(m_eqn_names[ieqn]+"_x_dflux");
        FYT& y_flux = tsk_info->get_uintah_field_add<FYT>(m_eqn_names[ieqn]+"_y_dflux");
        FZT& z_flux = tsk_info->get_uintah_field_add<FZT>(m_eqn_names[ieqn]+"_z_dflux");

        CT& phi = tsk_info->get_const_uintah_field_add<CT>(m_eqn_names[ieqn]);

        x_flux.initialize(0.0);
        y_flux.initialize(0.0);
        z_flux.initialize(0.0);

        CT& D = *(tsk_info->get_const_uintah_field<CT>(m_D_name));

        // x - Direction
        GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1,0);
        Uintah::BlockRange xrange(low_fx_patch_range, high_fx_patch_range);
        Uintah::parallel_for( xrange, [&](int i, int j, int k){

          const double afx  = ( eps(i,j,k) + eps(i-1,j,k) ) / 2. < 0.51 ? 0.0 : 1.0;

          x_flux(i,j,k) =  1./(2.*Dx.x()) * afx *
            ( D(i,j,k)   + D(i-1,j,k)) * (phi(i,j,k) - phi(i-1,j,k));

        });

        // y - Direction
        GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1,0);
        Uintah::BlockRange yrange(low_fy_patch_range, high_fy_patch_range);
        Uintah::parallel_for( yrange, [&](int i, int j, int k){

          const double afy  = ( eps(i,j,k) + eps(i,j-1,k) ) / 2. < 0.51 ? 0.0 : 1.0;

          y_flux(i,j,k) =  1./(2.*Dx.y()) * afy *
            ( D(i,j,k)   + D(i,j-1,k)) * (phi(i,j,k) - phi(i,j-1,k));

        });

        // z - Direction
        GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1,0);
        Uintah::BlockRange zrange(low_fz_patch_range, high_fz_patch_range);
        Uintah::parallel_for( zrange, [&](int i, int j, int k){

          const double afz  = ( eps(i,j,k) + eps(i,j,k-1) ) / 2. < 0.51 ? 0.0 : 1.0;

          z_flux(i,j,k) =  1./(2.*Dx.z()) * afz *
            ( D(i,j,k)   + D(i,j,k-1)) * (phi(i,j,k) - phi(i,j,k-1));

        });


      } // if do diffusion
    } // eqn loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::register_compute_bcs(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void Diffusion<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  }

}
#endif
