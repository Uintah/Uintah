#ifndef Uintah_Component_Arches_KMomentum_h
#define Uintah_Component_Arches_KMomentum_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctors.h>
#include <CCA/Components/Arches/UPSHelper.h>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

namespace Uintah{

  template<typename T>
  class KMomentum : public TaskInterface {

public:

    KMomentum<T>( std::string task_name, int matl_index );
    ~KMomentum<T>();

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry );

    void register_timestep_init( ArchesVIVector& variable_registry );

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep );

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep );

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (KMomentum) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
      : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      KMomentum* build()
      { return scinew KMomentum<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    std::string m_x_velocity_name;
    std::string m_y_velocity_name;
    std::string m_z_velocity_name;
    std::string m_mu_name;
    std::string m_rho_name;
    std::string m_eps_name;

    std::vector<std::string> m_eqn_names;
    std::vector<bool> m_do_clip;
    std::vector<double> m_low_clip;
    std::vector<double> m_high_clip;
    std::vector<double> m_init_value;

    ArchesCore::DIR my_dir;

    bool m_inviscid;

    int m_total_eqns;

    struct SourceInfo{
      std::string name;
      double weight;
    };

    std::vector<std::vector<SourceInfo> > m_source_info;

    std::vector<LIMITER> m_conv_scheme;

    ArchesCore::BCFunctors<T>* m_boundary_functors;

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  KMomentum<T>::KMomentum( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ) {

    m_boundary_functors = scinew ArchesCore::BCFunctors<T>();

    // Hard Coded defaults for the momentum eqns:
    m_total_eqns = 1;
    m_eqn_names.push_back(task_name);

    ArchesCore::VariableHelper<T> helper;
    my_dir = helper.dir;

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  KMomentum<T>::~KMomentum(){

    delete m_boundary_functors;

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::problemSetup( ProblemSpecP& input_db ){

    ProblemSpecP db = input_db;
    ConvectionHelper* conv_helper = scinew ConvectionHelper();

    //Convection
    if ( db->findBlock("convection")){
      std::string conv_scheme;
      db->findBlock("convection")->getAttribute("scheme", conv_scheme);
      m_conv_scheme.push_back(conv_helper->get_limiter_from_string(conv_scheme));
    } else {
      m_conv_scheme.push_back(NOCONV);
    }

    m_inviscid = false;
    if ( db->findBlock("inviscid")){
      m_inviscid = true;
    }

    //Clipping
    if ( db->findBlock("clip")){
      m_do_clip.push_back(true);
      double low; double high;
      db->findBlock("clip")->getAttribute("low", low);
      db->findBlock("clip")->getAttribute("high", high);
      m_low_clip.push_back(low);
      m_high_clip.push_back(high);
    } else {
      m_do_clip.push_back(false);
      m_low_clip.push_back(-99999.9);
      m_high_clip.push_back(99999.9);
    }

    //Initial Value
    if ( db->findBlock("initialize") ){
      double value;
      db->findBlock("initialize")->getAttribute("value",value);
      m_init_value.push_back(value);
    } else {
      m_init_value.push_back(0.0);
    }

    std::vector<SourceInfo> eqn_srcs;
    for ( ProblemSpecP src_db = db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src") ){

      std::string src_label;
      double weight = 1.0;

      src_db->getAttribute("label",src_label);

      if ( src_db->findBlock("weight")){
        src_db->findBlock("weight")->getAttribute("value",weight);
      }

      SourceInfo info;
      info.name = src_label;
      info.weight = weight;

      eqn_srcs.push_back(info);

    }

    m_source_info.push_back(eqn_srcs);

    // setup the boundary conditions for this eqn set
    m_boundary_functors->create_bcs( db, m_eqn_names );

    delete conv_helper;

    using namespace ArchesCore;

    ArchesCore::GridVarMap<T> var_map;
    var_map.problemSetup( input_db );
    m_eps_name = var_map.vol_frac_name;
    m_x_velocity_name = var_map.uvel_name;
    m_y_velocity_name = var_map.vvel_name;
    m_z_velocity_name = var_map.wvel_name;
    m_mu_name = var_map.mu_name;
    m_rho_name = "density";

    if ( input_db->findBlock("velocity") ){
      // can overide the global velocity space with this:
      input_db->findBlock("velocity")->getAttribute("xlabel",m_x_velocity_name);
      input_db->findBlock("velocity")->getAttribute("ylabel",m_y_velocity_name);
      input_db->findBlock("velocity")->getAttribute("zlabel",m_z_velocity_name);
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void
  KMomentum<T>::create_local_labels(){

    const int istart = 0;
    int iend = m_eqn_names.size();
    for (int i = istart; i < iend; i++ ){
      register_new_variable<T>( m_eqn_names[i] );
      register_new_variable<T>( m_eqn_names[i]+"_rhs" );
      register_new_variable<FXT>( m_eqn_names[i]+"_x_flux" );
      register_new_variable<FYT>( m_eqn_names[i]+"_y_flux" );
      register_new_variable<FZT>( m_eqn_names[i]+"_z_flux" );
      register_new_variable<T>( m_eqn_names[i]+"_div_tauij" );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable(  m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn]+"_rhs", ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES , variable_registry );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      double scalar_init_value = m_init_value[ieqn];

      T& phi    = *(tsk_info->get_uintah_field<T>(m_eqn_names[ieqn]+"_rhs"));
      T& rhs    = *(tsk_info->get_uintah_field<T>(m_eqn_names[ieqn]));
      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( range, [&](int i, int j, int k){
        phi(i,j,k) = scalar_init_value;
        rhs(i,j,k) = 0.0;
      });

      FXT& x_flux = *(tsk_info->get_uintah_field<FXT>(m_eqn_names[ieqn]+"_x_flux"));
      FYT& y_flux = *(tsk_info->get_uintah_field<FYT>(m_eqn_names[ieqn]+"_y_flux"));
      FZT& z_flux = *(tsk_info->get_uintah_field<FZT>(m_eqn_names[ieqn]+"_z_flux"));

      Uintah::parallel_for( range, [&](int i, int j, int k){
        x_flux(i,j,k) = 0.0;
        y_flux(i,j,k) = 0.0;
        z_flux(i,j,k) = 0.0;
      });

    } //eqn loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( m_eqn_names[ieqn]+"_rhs", ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      T& phi = *(tsk_info->get_uintah_field<T>( m_eqn_names[ieqn] ));
      T& rhs = *(tsk_info->get_uintah_field<T>( m_eqn_names[ieqn]+"_rhs" ));
      CT& old_phi = *(tsk_info->get_const_uintah_field<CT>( m_eqn_names[ieqn] ));

      phi.copyData(old_phi);
      rhs.initialize(0.0);

    } //equation loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_rhs", ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      if ( m_conv_scheme[ieqn] != NOCONV ){
        register_variable( m_eqn_names[ieqn]+"_x_psi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
        register_variable( m_eqn_names[ieqn]+"_y_psi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
        register_variable( m_eqn_names[ieqn]+"_z_psi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }

      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator i = m_source_info[ieqn].begin(); i != m_source_info[ieqn].end(); i++){
        register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }

    }

    //globally common variables
    if ( my_dir == ArchesCore::XDIR ){
      register_variable( "ucell_xvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "ucell_yvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "ucell_zvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    } else if ( my_dir == ArchesCore::YDIR ){
      register_variable( "vcell_xvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "vcell_yvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "vcell_zvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    } else {
      register_variable( "wcell_xvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "wcell_yvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( "wcell_zvel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
    register_variable( m_x_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_y_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_z_velocity_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    register_variable( m_mu_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::LATEST, variable_registry, time_substep );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    Vector Dx = patch->dCell();
    double V = Dx.x()*Dx.y()*Dx.z();

    CFXT& u     = *(tsk_info->get_const_uintah_field<CFXT>(m_x_velocity_name));
    CFYT& v     = *(tsk_info->get_const_uintah_field<CFYT>(m_y_velocity_name));
    CFZT& w     = *(tsk_info->get_const_uintah_field<CFZT>(m_z_velocity_name));
    CT& eps     = *(tsk_info->get_const_uintah_field<CT>(m_eps_name));
    constCCVariable<double>& mu = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_mu_name));
    constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_rho_name));

#ifdef DO_TIMINGS
    SpatialOps::TimeLogger timer("kokkos_scalar_assemble.out."+_task_name);
    timer.start("work");
#endif
    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      CT& phi     = *(tsk_info->get_const_uintah_field<CT>(m_eqn_names[ieqn]));
      T& rhs      = *(tsk_info->get_uintah_field<T>(m_eqn_names[ieqn]+"_rhs"));

      //Convection:
      FXT& x_flux = *(tsk_info->get_uintah_field<FXT>(m_eqn_names[ieqn]+"_x_flux"));
      FYT& y_flux = *(tsk_info->get_uintah_field<FYT>(m_eqn_names[ieqn]+"_y_flux"));
      FZT& z_flux = *(tsk_info->get_uintah_field<FZT>(m_eqn_names[ieqn]+"_z_flux"));

      Uintah::BlockRange init_range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( init_range, [&](int i, int j, int k){

        rhs(i,j,k) = 0.;
        x_flux(i,j,k) = 0.;
        y_flux(i,j,k) = 0.;
        z_flux(i,j,k) = 0.;

      });

      if ( m_conv_scheme[ieqn] != NOCONV ){

        CFXT& x_psi = *(tsk_info->get_const_uintah_field<CFXT>(m_eqn_names[ieqn]+"_x_psi"));
        CFYT& y_psi = *(tsk_info->get_const_uintah_field<CFYT>(m_eqn_names[ieqn]+"_y_psi"));
        CFZT& z_psi = *(tsk_info->get_const_uintah_field<CFZT>(m_eqn_names[ieqn]+"_z_psi"));

        if ( my_dir == ArchesCore::XDIR ){

          CT& u_fx = tsk_info->get_const_uintah_field_add<CT>("ucell_xvel");
          CT& v_fy = tsk_info->get_const_uintah_field_add<CT>("ucell_yvel");
          CT& w_fz = tsk_info->get_const_uintah_field_add<CT>("ucell_zvel");

          Uintah::ComputeConvectiveFlux get_flux( phi, u_fx, v_fy, w_fz, x_psi, y_psi, z_psi,
                                                  x_flux, y_flux, z_flux, eps );

          GET_FX_BUFFERED_PATCH_RANGE( 1, 0 )
          Uintah::BlockRange x_range( low_fx_patch_range, high_fx_patch_range );
          Uintah::parallel_for( x_range, get_flux );

        } else if ( my_dir == ArchesCore::YDIR ){

          CT& u_fx = tsk_info->get_const_uintah_field_add<CT>("vcell_xvel");
          CT& v_fy = tsk_info->get_const_uintah_field_add<CT>("vcell_yvel");
          CT& w_fz = tsk_info->get_const_uintah_field_add<CT>("vcell_zvel");

          Uintah::ComputeConvectiveFlux get_flux( phi, u_fx, v_fy, w_fz, x_psi, y_psi, z_psi,
                                                  x_flux, y_flux, z_flux, eps );

          GET_FY_BUFFERED_PATCH_RANGE( 1, 0 )
          Uintah::BlockRange y_range( low_fy_patch_range, high_fy_patch_range );
          Uintah::parallel_for( y_range, get_flux );

        } else {

          CT& u_fx = tsk_info->get_const_uintah_field_add<CT>("wcell_xvel");
          CT& v_fy = tsk_info->get_const_uintah_field_add<CT>("wcell_yvel");
          CT& w_fz = tsk_info->get_const_uintah_field_add<CT>("wcell_zvel");

          Uintah::ComputeConvectiveFlux get_flux( phi, u_fx, v_fy, w_fz, x_psi, y_psi, z_psi,
                                                  x_flux, y_flux, z_flux, eps );

          GET_FZ_BUFFERED_PATCH_RANGE( 1, 0 )
          Uintah::BlockRange z_range( low_fz_patch_range, high_fz_patch_range );
          Uintah::parallel_for( z_range, get_flux );

        }

      }

      //Stress
      if ( !m_inviscid ){
        const double areaEW = Dx.y()*Dx.z();
        const double areaNS = Dx.x()*Dx.z();
        const double areaTB = Dx.x()*Dx.y();

        ArchesCore::VariableHelper<T> var_help;

        if ( var_help.dir == 0 ){

          GET_FX_BUFFERED_PATCH_RANGE(1, 0)
          Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);

          Uintah::parallel_for( range, [&](int i, int j, int k){

            const double SE = (u(i+1,j,k) - u(i,j,k))/Dx.x();
            const double SW = (u(i,j,k) - u(i,j,k))/Dx.x();
            const double SN = 0.5 * (( u(i,j+1,k) - u(i,j,k) ) / Dx.y() + (v(i,j+1,k) - v(i-1,j+1,k))/Dx.x());
            const double SS = 0.5 * (( u(i,j,k) - u(i,j-1,k) ) / Dx.y() + (v(i,j,k) - v(i-1,j,k))/Dx.x());
            const double ST = 0.5 * (( u(i,j,k+1) - u(i,j,k) ) / Dx.z() + (w(i,j,k+1) - w(i-1,j,k+1))/Dx.x());
            const double SB = 0.5 * (( u(i,j,k) - u(i,j,k-1) ) / Dx.z() + (w(i,j,k) - w(i-1,j,k))/Dx.x());

            const double mu_E = mu(i,j,k);
            const double mu_W = mu(i-1,j,k);
            const double mu_N = 0.5 * ( 0.5 * (mu(i,j+1,k) + mu(i,j,k))   + 0.5 * (mu(i-1,j+1,k) + mu(i-1,j,k)) );
            const double mu_S = 0.5 * ( 0.5 * (mu(i,j,k)   + mu(i,j-1,k)) + 0.5 * (mu(i-1,j,k)   + mu(i-1,j-1,k)) );
            const double mu_T = 0.5 * ( 0.5 * (mu(i,j,k+1) + mu(i,j,k))   + 0.5 * (mu(i-1,j,k+1) + mu(i-1,j,k)) );
            const double mu_B = 0.5 * ( 0.5 * (mu(i,j,k)   + mu(i,j,k-1)) + 0.5 * (mu(i-1,j,k)   + mu(i-1,j,k-1)) );

            // add in once we have a rho
            const double rho_E = rho(i,j,k);
            const double rho_W = rho(i-1,j,k);
            const double rho_N = 0.5 * ( 0.5 * (rho(i,j+1,k) + rho(i,j,k))   + 0.5 * (rho(i-1,j+1,k) + rho(i-1,j,k)) );
            const double rho_S = 0.5 * ( 0.5 * (rho(i,j,k)   + rho(i,j-1,k)) + 0.5 * (rho(i-1,j,k)   + rho(i-1,j-1,k)) );
            const double rho_T = 0.5 * ( 0.5 * (rho(i,j,k+1) + rho(i,j,k))   + 0.5 * (rho(i-1,j,k+1) + rho(i-1,j,k)) );
            const double rho_B = 0.5 * ( 0.5 * (rho(i,j,k)   + rho(i,j,k-1)) + 0.5 * (rho(i-1,j,k)   + rho(i-1,j,k-1)) );

            double div_tauij = areaEW * ( rho_E * mu_E * SE - rho_W * mu_W * SW ) +
                               areaNS * ( rho_N * mu_N * SN - rho_S * mu_S * SS ) +
                               areaTB * ( rho_T * mu_T * ST - rho_B * mu_B * SB );

            rhs(i,j,k) += div_tauij;

          });
        } else if ( var_help.dir == 1 ){
          GET_FY_BUFFERED_PATCH_RANGE(1, 0);
        } else {
          GET_FZ_BUFFERED_PATCH_RANGE(1, 0);
        }
      }

      //Sources:
      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator isrc = m_source_info[ieqn].begin(); isrc != m_source_info[ieqn].end(); isrc++){

        CT& src = *(tsk_info->get_const_uintah_field<CT>((*isrc).name));
        double weight = (*isrc).weight;
        Uintah::BlockRange src_range(patch->getCellLowIndex(), patch->getCellHighIndex());

        Uintah::parallel_for( src_range, [&](int i, int j, int k){

          rhs(i,j,k) += weight * src(i,j,k) * V;

        });
      }
    } // equation loop
#ifdef DO_TIMINGS
    timer.stop("work");
#endif

  }

//--------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    for ( auto i = m_eqn_names.begin(); i != m_eqn_names.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
    }

    std::vector<std::string> bc_dep;
    m_boundary_functors->get_bc_dependencies( m_eqn_names, m_bcHelper, bc_dep );
    for ( auto i = bc_dep.begin(); i != bc_dep.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0 , ArchesFieldContainer::NEWDW,
                         variable_registry );
    }

  }

//--------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    m_boundary_functors->apply_bc( m_eqn_names, m_bcHelper, tsk_info, patch );

  }
}
#endif
