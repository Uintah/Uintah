#ifndef Uintah_Component_Arches_ComputePsi_h
#define Uintah_Component_Arches_ComputePsi_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Arches/UPSHelper.h>
#include <Core/Util/Timers/Timers.hpp>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

namespace Uintah{

  template <typename T>
  class ComputePsi : public TaskInterface {

public:

    typedef Uintah::ArchesFieldContainer AFC;

    ComputePsi<T>( std::string task_name, int matl_index );
    ~ComputePsi<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){

      for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

        register_new_variable<XFaceT>( *i + "_x_psi" );
        register_new_variable<YFaceT>( *i + "_y_psi" );
        register_new_variable<ZFaceT>( *i + "_z_psi" );

      }
    }

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      ComputePsi* build()
      { return scinew ComputePsi( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

protected:

    typedef std::vector<AFC::VariableInformation> AVarInfo;

    void register_initialize( AVarInfo& variable_registry , const bool pack_tasks);

    void register_timestep_init( AVarInfo& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( AVarInfo& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( AVarInfo& variable_registry, const int time_substep , const bool packed_tasks){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

private:

    std::vector<std::string> _eqn_names;
    std::map<std::string, LIMITER> _name_to_limiter_map;
    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;
    std::string m_eps_name;

    int m_boundary_int{0};
    int m_dir{0};

    typedef std::vector<std::string> SV;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType XFaceT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType YFaceT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType ZFaceT;
    typedef typename ArchesCore::VariableHelper<CT>::XFaceType ConstXFaceT;
    typedef typename ArchesCore::VariableHelper<CT>::YFaceType ConstYFaceT;
    typedef typename ArchesCore::VariableHelper<CT>::ZFaceType ConstZFaceT;

    //typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType ConstXFaceT;
    //typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType ConstYFaceT;
    //typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType ConstZFaceT;

    template<typename CPhiT, typename PsiT, typename VelT>
    struct PsiHelper{

      void computePsi( CPhiT& phi, PsiT& psi, VelT& vel, CPhiT& eps,
                       Patch::FaceType PLow, Patch::FaceType PHigh,
                       const int boundary_buffer, const int dir, LIMITER lim_type,
                       const Patch* patch, ArchesTaskInfoManager* tsk_info ){

        int buffer = 0;
        if ( tsk_info->packed_tasks() ) buffer = 1;

        IntVector low_patch_range(0,0,0), high_patch_range(0,0,0);
        IntVector lbuffer(0,0,0), hbuffer(0,0,0);

        low_patch_range = IntVector(0,0,0); high_patch_range = IntVector(0,0,0);

        lbuffer[dir] = ( patch->getBCType(PLow) != Patch::Neighbor ) ? 1 : 0;
        hbuffer[dir] = ( patch->getBCType(PHigh) != Patch::Neighbor ) ?
                        boundary_buffer :
                        buffer;

        low_patch_range = patch->getCellLowIndex() + lbuffer;
        high_patch_range = patch->getCellHighIndex() + hbuffer;

        Uintah::BlockRange range(low_patch_range, high_patch_range);

        if ( lim_type == UPWIND ){
          GetPsi<Array3<double>, UpwindConvection > get_psi( phi, psi, vel, eps, dir );
          Uintah::parallel_for(range, get_psi);
        } else if ( lim_type == CENTRAL ){
          GetPsi<Array3<double>, CentralConvection > get_psi( phi, psi, vel, eps, dir );
          Uintah::parallel_for(range, get_psi);
        } else if ( lim_type == SUPERBEE ){
          GetPsi<Array3<double>, SuperBeeConvection > get_psi( phi, psi, vel, eps, dir );
          Uintah::parallel_for(range, get_psi);
        } else if ( lim_type == ROE ){
          GetPsi<Array3<double>, RoeConvection >   get_psi( phi, psi, vel, eps, dir );
          Uintah::parallel_for(range, get_psi);
        } else if ( lim_type == VANLEER ){
          GetPsi<Array3<double>, VanLeerConvection> get_psi( phi, psi, vel, eps, dir );
          Uintah::parallel_for(range, get_psi);
        } else {

        }

      }

    };

  };

  //------------------------------------------------------------------------------------------------
  //                              Member Function Definitions
  //------------------------------------------------------------------------------------------------
  template <typename T>
  ComputePsi<T>::ComputePsi( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){

    ArchesCore::VariableHelper<T>* helper = scinew ArchesCore::VariableHelper<T>;
    if ( helper->dir != ArchesCore::NODIR ) m_boundary_int = 1;
    m_dir = helper->dir;
    delete helper;

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  ComputePsi<T>::~ComputePsi(){}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ComputePsi<T>::problemSetup( ProblemSpecP& db ){

    using namespace ArchesCore;

    _eqn_names.clear();
    for ( ProblemSpecP eqn_db = db->findBlock("eqn");
          eqn_db != nullptr; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string limiter;
      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);

      if ( eqn_db->findBlock("convection" )){

        eqn_db->findBlock("convection")->getAttribute("scheme",limiter);

        ConvectionHelper* conv_helper = scinew ConvectionHelper();

        LIMITER enum_limiter = conv_helper->get_limiter_from_string(limiter);

        _name_to_limiter_map.insert(std::make_pair(scalar_name, enum_limiter));

        _eqn_names.push_back(scalar_name);

        delete conv_helper;

      }
    }

    // FOR MOMENTUM SPECIFICALLY
    if ( _eqn_names.size() == 0 ){

      std::string limiter;

      if ( db->findBlock("convection")){

        db->findBlock("convection")->getAttribute("scheme",limiter);

        ConvectionHelper* conv_helper = scinew ConvectionHelper();

        LIMITER enum_limiter = conv_helper->get_limiter_from_string(limiter);

        std::string which_mom = m_task_name.substr(0,5);

        _name_to_limiter_map.insert(std::make_pair(which_mom, enum_limiter));

        _eqn_names.push_back(which_mom);

        delete conv_helper;

      }
    }

    ArchesCore::GridVarMap<T> var_map;
    var_map.problemSetup( db );
    m_eps_name = var_map.vol_frac_name;
    m_u_vel_name = var_map.uvel_name;
    m_v_vel_name = var_map.vvel_name;
    m_w_vel_name = var_map.wvel_name;

    if ( db->findBlock("velocity") ){
      // can overide the global velocity space with this:
      db->findBlock("velocity")->getAttribute("xlabel",m_u_vel_name);
      db->findBlock("velocity")->getAttribute("ylabel",m_v_vel_name);
      db->findBlock("velocity")->getAttribute("zlabel",m_w_vel_name);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ComputePsi<T>::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i+"_x_psi", AFC::COMPUTES, variable_registry );
      register_variable( *i+"_y_psi", AFC::COMPUTES, variable_registry );
      register_variable( *i+"_z_psi", AFC::COMPUTES, variable_registry );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ComputePsi<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      XFaceT& psi_x = tsk_info->get_field<XFaceT>(*i+"_x_psi");
      YFaceT& psi_y = tsk_info->get_field<YFaceT>(*i+"_y_psi");
      ZFaceT& psi_z = tsk_info->get_field<ZFaceT>(*i+"_z_psi");

      psi_x.initialize(0.0);
      psi_y.initialize(0.0);
      psi_z.initialize(0.0);

    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ComputePsi<T>::register_timestep_eval(
    std::vector<AFC::VariableInformation>& variable_registry,
    const int time_substep, const bool packed_tasks )
  {

    for ( auto i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i+"_x_psi", AFC::COMPUTES, variable_registry, time_substep, m_task_name, packed_tasks );
      register_variable( *i+"_y_psi", AFC::COMPUTES, variable_registry, time_substep, m_task_name, packed_tasks );
      register_variable( *i+"_z_psi", AFC::COMPUTES, variable_registry, time_substep, m_task_name, packed_tasks );
      register_variable( *i, AFC::REQUIRES, 2, AFC::LATEST, variable_registry, time_substep, m_task_name );
    }

    int nGhosts = 1;
    if ( packed_tasks ){
      nGhosts = 2;
    }

    register_variable( m_eps_name,   AFC::REQUIRES, 2, AFC::NEWDW, variable_registry, time_substep, m_task_name );
    register_variable( m_u_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, m_task_name );
    register_variable( m_v_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, m_task_name );
    register_variable( m_w_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, m_task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void ComputePsi<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    CT& eps = tsk_info->get_field<CT>(m_eps_name);

    ConstXFaceT& u = tsk_info->get_field<ConstXFaceT>(m_u_vel_name);
    ConstYFaceT& v = tsk_info->get_field<ConstYFaceT>(m_v_vel_name);
    ConstZFaceT& w = tsk_info->get_field<ConstZFaceT>(m_w_vel_name);

    for ( auto i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      std::map<std::string, LIMITER>::iterator ilim = _name_to_limiter_map.find(*i);
      LIMITER my_limiter = ilim->second;
      CT& phi = tsk_info->get_field<CT>(*i);

      int nGhosts = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
      if ( tsk_info->packed_tasks() ){
        nGhosts = 1;
      }

      XFaceT& psi_x = tsk_info->get_field<XFaceT>(*i+"_x_psi", nGhosts);
      YFaceT& psi_y = tsk_info->get_field<YFaceT>(*i+"_y_psi", nGhosts);
      ZFaceT& psi_z = tsk_info->get_field<ZFaceT>(*i+"_z_psi", nGhosts);

      psi_x.initialize(1.0);
      psi_y.initialize(1.0);
      psi_z.initialize(1.0);

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_compute_psi.out."+*i);
      timer.start("ComputePsi");
#endif

      PsiHelper<CT, XFaceT, ConstXFaceT> x_psi_helper;
      PsiHelper<CT, YFaceT, ConstYFaceT> y_psi_helper;
      PsiHelper<CT, ZFaceT, ConstZFaceT> z_psi_helper;

      int boundary_int = 0;
      if ( m_boundary_int > 0 && m_dir == 0 ) boundary_int = 1;
      x_psi_helper.computePsi( phi, psi_x, u, eps, Patch::xminus, Patch::xplus,
                               boundary_int, 0, my_limiter, patch, tsk_info );
      boundary_int = 0;
      if ( m_boundary_int > 0 && m_dir == 1 ) boundary_int = 1;
      y_psi_helper.computePsi( phi, psi_y, v, eps, Patch::yminus, Patch::yplus,
                               boundary_int, 1, my_limiter, patch, tsk_info );
      boundary_int = 0;
      if ( m_boundary_int > 0 && m_dir == 2 ) boundary_int = 1;
      z_psi_helper.computePsi( phi, psi_z, w, eps, Patch::zminus, Patch::zplus,
                               boundary_int, 2, my_limiter, patch, tsk_info );

#ifdef DO_TIMINGS
      timer.stop("ComputePsi");
#endif

    }
  }
}
#endif
