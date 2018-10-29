#ifndef Uintah_Component_Arches_KMomentum_h
#define Uintah_Component_Arches_KMomentum_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#define MOMCONVECTION() \
        Uintah::ComputeConvectiveFlux                                       \
          get_flux( phi, u_fx, v_fy, w_fz, x_flux, y_flux, z_flux, eps );   \
        Uintah::parallel_for( convection_range, get_flux, scheme );

namespace Uintah{

  template<typename T>
  class KMomentum : public TaskInterface {

public:

    KMomentum<T>( std::string task_name, int matl_index );
    ~KMomentum<T>();

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry , const bool pack_tasks);

    void register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks);

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep , const bool packed_tasks);

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep , const bool packed_tasks);

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
    typedef typename ArchesCore::VariableHelper<CT>::XFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<CT>::YFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<CT>::ZFaceType CFZT;

    //typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
    //typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
    //typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    int m_ghost_cells;

    ArchesCore::INTERPOLANT m_int_scheme;

    std::string m_x_velocity_name;
    std::string m_y_velocity_name;
    std::string m_z_velocity_name;
    std::string m_mu_name;
    std::string m_rho_name;
    std::string m_eps_name;

    std::string m_sigmax_name;
    std::string m_sigmay_name;
    std::string m_sigmaz_name;
    std::vector<std::string> m_vel_name;

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
    using namespace ArchesCore;

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

    //Note that this current implementation is hardwired for 1 eqn
    m_ghost_cells = 1;

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
      if ( my_dir == ArchesCore::XDIR ){
        db->findBlock("initialize")->getAttribute("u",value);
      } else if ( my_dir == ArchesCore::YDIR ){
        db->findBlock("initialize")->getAttribute("v",value);
      } else if ( my_dir == ArchesCore::ZDIR ){
        db->findBlock("initialize")->getAttribute("w",value);
      }
      m_init_value.push_back(value);
    } else {
      m_init_value.push_back(0.0);
    }

    std::vector<SourceInfo> eqn_srcs;
    if ( my_dir == ArchesCore::XDIR ){

      m_vel_name.push_back(parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" ));
      for ( ProblemSpecP src_db = db->findBlock("src_x");
            src_db != nullptr; src_db = src_db->findNextBlock("src_x") ){

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

    } else if ( my_dir == ArchesCore::YDIR ){

      m_vel_name.push_back(parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" ));
      for ( ProblemSpecP src_db = db->findBlock("src_y");
            src_db != nullptr; src_db = src_db->findNextBlock("src_y") ){

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

    } else if ( my_dir == ArchesCore::ZDIR ){

      m_vel_name.push_back(parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" ));
      for ( ProblemSpecP src_db = db->findBlock("src_z");
            src_db != nullptr; src_db = src_db->findNextBlock("src_z") ){

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
    }

    m_source_info.push_back(eqn_srcs);

    // setup the boundary conditions for this eqn set
    m_boundary_functors->create_bcs( db, m_eqn_names );

    delete conv_helper;

    ArchesCore::GridVarMap<T> var_map;
    var_map.problemSetup( input_db );
    m_eps_name = var_map.vol_frac_name;
    m_x_velocity_name = var_map.uvel_name;
    m_y_velocity_name = var_map.vvel_name;
    m_z_velocity_name = var_map.wvel_name;

    m_sigmax_name     = var_map.sigmax_name;
    m_sigmay_name     = var_map.sigmay_name;
    m_sigmaz_name     = var_map.sigmaz_name;

    m_mu_name = var_map.mu_name;
    m_rho_name = parse_ups_for_role( DENSITY, db, "density" );

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

    for ( auto i = m_conv_scheme.begin(); i != m_conv_scheme.end(); i++ ){
      if ( *i == FOURTH ){
        m_ghost_cells = 2;
      }
    }

    const int istart = 0;
    int iend = m_eqn_names.size();
    for (int i = istart; i < iend; i++ ){
      register_new_variable<T>( m_vel_name[i] );
      register_new_variable<T>( m_eqn_names[i] );
      register_new_variable<T>( m_eqn_names[i]+"_RHS" );
      register_new_variable<FXT>( m_eqn_names[i]+"_x_flux" );
      register_new_variable<FYT>( m_eqn_names[i]+"_y_flux" );
      register_new_variable<FZT>( m_eqn_names[i]+"_z_flux" );
      register_new_variable<T>( m_eqn_names[i]+"_div_tauij" );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable(  m_vel_name[ieqn] , ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry );
      register_variable(  m_eqn_names[ieqn]+"_RHS", ArchesFieldContainer::COMPUTES , variable_registry );
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

      T& u    = tsk_info->get_uintah_field_add<T>(m_vel_name[ieqn]);
      T& phi    = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]);
      T& rhs    = tsk_info->get_uintah_field_add<T>(m_eqn_names[ieqn]+"_RHS");
      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      Uintah::parallel_for( range, [&](int i, int j, int k){
        phi(i,j,k) = 0.0;
        rhs(i,j,k) = 0.0;
        u(i,j,k)   = scalar_init_value; // initial value for velocity, phi (rho_u) is computed in UnweightVariable task
      });

      FXT& x_flux = tsk_info->get_uintah_field_add<FXT>(m_eqn_names[ieqn]+"_x_flux");
      FYT& y_flux = tsk_info->get_uintah_field_add<FYT>(m_eqn_names[ieqn]+"_y_flux");
      FZT& z_flux = tsk_info->get_uintah_field_add<FZT>(m_eqn_names[ieqn]+"_z_flux");

      Uintah::parallel_for( range, [&](int i, int j, int k){
        x_flux(i,j,k) = 0.0;
        y_flux(i,j,k) = 0.0;
        z_flux(i,j,k) = 0.0;
      });

    } //eqn loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      register_variable( m_vel_name[ieqn], ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( m_eqn_names[ieqn]+"_RHS", ArchesFieldContainer::COMPUTES , variable_registry  );
      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
      register_variable( m_vel_name[ieqn], ArchesFieldContainer::REQUIRES , 0 , ArchesFieldContainer::OLDDW , variable_registry  );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){
      T& u    = tsk_info->get_uintah_field_add<T>(m_vel_name[ieqn]);
      T& phi =  tsk_info->get_uintah_field_add<T>( m_eqn_names[ieqn] );
      T& rhs =  tsk_info->get_uintah_field_add<T>( m_eqn_names[ieqn]+"_RHS" );
      CT& old_phi = tsk_info->get_const_uintah_field_add<CT>( m_eqn_names[ieqn] );
      CT& old_u    = tsk_info->get_const_uintah_field_add<CT>(m_vel_name[ieqn]);

      phi.copyData(old_phi);
      u.copyData(old_u);
      rhs.initialize(0.0);

    } //equation loop
  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
    variable_registry, const int time_substep , const bool packed_tasks){

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      register_variable( m_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, m_ghost_cells, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_RHS", ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( m_eqn_names[ieqn]+"_x_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      register_variable( m_eqn_names[ieqn]+"_y_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );
      register_variable( m_eqn_names[ieqn]+"_z_flux", ArchesFieldContainer::COMPUTES, variable_registry, time_substep, m_task_name );

      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator i = m_source_info[ieqn].begin(); i != m_source_info[ieqn].end(); i++){
        register_variable( i->name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }

    }

    //globally common variables
    if ( !m_inviscid ){
      register_variable( m_sigmax_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( m_sigmay_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( m_sigmaz_name, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }

    register_variable( m_x_velocity_name, ArchesFieldContainer::REQUIRES, m_ghost_cells , ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_y_velocity_name, ArchesFieldContainer::REQUIRES, m_ghost_cells , ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_z_velocity_name, ArchesFieldContainer::REQUIRES, m_ghost_cells , ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_eps_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    //register_variable( m_mu_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    //register_variable( m_rho_name, ArchesFieldContainer::REQUIRES, 1 , ArchesFieldContainer::NEWDW, variable_registry, time_substep );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    Vector Dx = patch->dCell();
    double V = Dx.x()*Dx.y()*Dx.z();

    Uintah::IntVector low_patch_range = patch->getCellLowIndex();
    Uintah::IntVector high_patch_range = patch->getCellHighIndex();

    // CFXT& u     = *(tsk_info->get_const_uintah_field<CFXT>(m_x_velocity_name));
    // CFYT& v     = *(tsk_info->get_const_uintah_field<CFYT>(m_y_velocity_name));
    // CFZT& w     = *(tsk_info->get_const_uintah_field<CFZT>(m_z_velocity_name));
    CT& eps     = *(tsk_info->get_const_uintah_field<CT>(m_eps_name));
    // constCCVariable<double>& mu = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_mu_name));
    // constCCVariable<double>& rho = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_rho_name));

    const int istart = 0;
    const int iend = m_eqn_names.size();
    for (int ieqn = istart; ieqn < iend; ieqn++ ){

      CT& phi     = *(tsk_info->get_const_uintah_field<CT>(m_eqn_names[ieqn]));
      T& rhs      = *(tsk_info->get_uintah_field<T>(m_eqn_names[ieqn]+"_RHS"));

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

      //convection
      if ( m_conv_scheme[ieqn] != NOCONV ){

        IntVector low_patch_range  = patch->getCellLowIndex();
        IntVector high_patch_range = patch->getCellHighIndex();

        //resolve the range:
        if ( my_dir == ArchesCore::XDIR ){
          GET_WALL_BUFFERED_PATCH_RANGE( low_patch_range, high_patch_range, 1, 1, 0, 1, 0, 1 );
        } else if ( my_dir == ArchesCore::YDIR ){
          GET_WALL_BUFFERED_PATCH_RANGE( low_patch_range, high_patch_range, 0, 1, 1, 1, 0, 1 );
        } else {
          GET_WALL_BUFFERED_PATCH_RANGE( low_patch_range, high_patch_range, 0, 1, 0, 1, 1, 1 );
        }

        Uintah::BlockRange convection_range(low_patch_range, high_patch_range);

        CT& u_fx = tsk_info->get_const_uintah_field_add<CT>(m_x_velocity_name);
        CT& v_fy = tsk_info->get_const_uintah_field_add<CT>(m_y_velocity_name);
        CT& w_fz = tsk_info->get_const_uintah_field_add<CT>(m_z_velocity_name);

        // switch between difference convection schemes:
        switch (m_conv_scheme[ieqn]){
          case CENTRAL:
            {
            CentralConvection scheme;
            MOMCONVECTION();
            }
            break;
          case FOURTH:
            {
            FourthConvection scheme;
            MOMCONVECTION();
            }
            break;
          case VANLEER:
            {
            VanLeerConvection scheme;
            MOMCONVECTION();
            }
            break;
          case SUPERBEE:
            {
            SuperBeeConvection scheme;
            MOMCONVECTION();
            }
            break;
          case ROE:
            {
            RoeConvection scheme;
            MOMCONVECTION();
            }
            break;
          case UPWIND:
            {
            UpwindConvection scheme;
            MOMCONVECTION();
            }
            break;
          default:
            throw InvalidValue("Error: Momentum convection scheme not recognized.", __FILE__, __LINE__);
        }
      }

      //Stress
      if ( !m_inviscid ){
        const double areaEW = Dx.y()*Dx.z();
        const double areaNS = Dx.x()*Dx.z();
        const double areaTB = Dx.x()*Dx.y();

        ArchesCore::VariableHelper<T> var_help;

        constCCVariable<double>& sigma1 = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_sigmax_name));
        constCCVariable<double>& sigma2 = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_sigmay_name));
        constCCVariable<double>& sigma3 = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(m_sigmaz_name));

        auto stressTensor = [&] (int i, int j, int k){
          double div_sigma1 = (sigma1(i+1,j,k) - sigma1(i,j,k))*areaEW +
                              (sigma2(i,j+1,k) - sigma2(i,j,k))*areaNS +
                              (sigma3(i,j,k+1) - sigma3(i,j,k))*areaTB;

          rhs(i,j,k) += div_sigma1;
        };


        if ( my_dir == ArchesCore::XDIR ){
          GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1, 0)
          Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);
          Uintah::parallel_for( range, stressTensor );
        } else if ( my_dir == ArchesCore::YDIR ){
          GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1, 0);
          Uintah::BlockRange range(low_fy_patch_range, high_fy_patch_range);
          Uintah::parallel_for( range, stressTensor );
        } else {
          GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1, 0);
          Uintah::BlockRange range(low_fz_patch_range, high_fz_patch_range);
          Uintah::parallel_for( range, stressTensor );
        }
      }

      //Sources:
      typedef std::vector<SourceInfo> VS;
      for (typename VS::iterator isrc = m_source_info[ieqn].begin(); isrc != m_source_info[ieqn].end(); isrc++){

        CT& src = *(tsk_info->get_const_uintah_field<CT>((*isrc).name));
        double weight = (*isrc).weight;

        if ( my_dir == ArchesCore::XDIR ){
          GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1, 0)
          Uintah::BlockRange range_x(low_fx_patch_range, high_fx_patch_range);
          Uintah::parallel_for( range_x, [&](int i, int j, int k){
            rhs(i,j,k) += weight * src(i,j,k) * V;

          });
        } else if ( my_dir == ArchesCore::YDIR ){
          GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1, 0);
          Uintah::BlockRange range_y(low_fy_patch_range, high_fy_patch_range);
          Uintah::parallel_for( range_y, [&](int i, int j, int k){
            rhs(i,j,k) += weight * src(i,j,k) * V;
          });
        } else {
          GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1, 0);
          Uintah::BlockRange range_z(low_fz_patch_range, high_fz_patch_range);
          Uintah::parallel_for( range_z, [&](int i, int j, int k){
            rhs(i,j,k) += weight * src(i,j,k) * V;
          });
        }
      }
    } // equation loop

  }

//--------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    for ( auto i = m_eqn_names.begin(); i != m_eqn_names.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
    }

    ArchesCore::FunctorDepList bc_dep;
    m_boundary_functors->get_bc_dependencies( m_eqn_names, m_bcHelper, bc_dep );
    for ( auto i = bc_dep.begin(); i != bc_dep.end(); i++ ){

      register_variable( (*i).variable_name, ArchesFieldContainer::REQUIRES, (*i).n_ghosts , (*i).dw,
                         variable_registry );
    }

    std::vector<std::string> bc_mod;
    m_boundary_functors->get_bc_modifies( m_eqn_names, m_bcHelper, bc_mod );
    for ( auto i = bc_mod.begin(); i != bc_mod.end(); i++ ){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry );
    }

  }

//--------------------------------------------------------------------------------------------------
  template <typename T> void
  KMomentum<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    m_boundary_functors->apply_bc( m_eqn_names, m_bcHelper, tsk_info, patch );

  }
}
#endif
