#ifndef Uintah_Component_Arches_ComputePsi_h
#define Uintah_Component_Arches_ComputePsi_h

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
#include <CCA/Components/Arches/UPSHelper.h>
#include <Core/Util/Timers/Timers.hpp>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

#define GET_PSI(my_limiter_struct) \
    GetPsi get_psi_x( phi, psi_x, u, eps, 0 ); \
    GetPsi get_psi_y( phi, psi_y, v, eps, 1 ); \
    GetPsi get_psi_z( phi, psi_z, w, eps, 2 ); \
    IntVector low_patch_range(0,0,0), high_patch_range(0,0,0); \
    IntVector lbuffer(0,0,0), hbuffer(1,0,0); \
    int buffer = 0; \
    if ( tsk_info->packed_tasks() ) buffer = 1; \
    lbuffer[0] = ( patch->getBCType(Patch::xminus) != Patch::Neighbor ) ? 1 : 0; \
    hbuffer[0] = ( patch->getBCType(Patch::xplus) != Patch::Neighbor ) ? -1 : buffer; \
    low_patch_range = patch->getCellLowIndex() + lbuffer; \
    high_patch_range = patch->getCellHighIndex() + hbuffer; \
    Uintah::BlockRange x_range(low_patch_range, high_patch_range); \
    lbuffer[0] =  0; hbuffer[0] = 0; \
    Uintah::parallel_for(x_range, get_psi_x, my_limiter_struct); \
    low_patch_range = IntVector(0,0,0); high_patch_range = IntVector(0,0,0); \
    lbuffer[1] = ( patch->getBCType(Patch::yminus) != Patch::Neighbor ) ? 1 : 0; \
    hbuffer[1] = ( patch->getBCType(Patch::yplus) != Patch::Neighbor ) ? -1 : buffer; \
    low_patch_range = patch->getCellLowIndex() + lbuffer; \
    high_patch_range = patch->getCellHighIndex() + hbuffer; \
    Uintah::BlockRange y_range(low_patch_range, high_patch_range); \
    lbuffer[1] =  0; hbuffer[1] = 0; \
    Uintah::parallel_for(y_range, get_psi_y, my_limiter_struct); \
    low_patch_range = IntVector(0,0,0); high_patch_range = IntVector(0,0,0); \
    lbuffer[2] = ( patch->getBCType(Patch::zminus) != Patch::Neighbor ) ? 1 : 0; \
    hbuffer[2] = ( patch->getBCType(Patch::zplus) != Patch::Neighbor ) ? -1 : buffer; \
    low_patch_range = patch->getCellLowIndex() + lbuffer; \
    high_patch_range = patch->getCellHighIndex() + hbuffer; \
    Uintah::BlockRange z_range(low_patch_range, high_patch_range); \
    Uintah::parallel_for(z_range, get_psi_z, my_limiter_struct);

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
        _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      ComputePsi* build()
      { return scinew ComputePsi( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

protected:

    typedef std::vector<AFC::VariableInformation> AVarInfo;

    void register_initialize( AVarInfo& variable_registry , const bool pack_tasks);

    void register_timestep_init( AVarInfo& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( AVarInfo& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( AVarInfo& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

private:

    std::vector<std::string> _eqn_names;
    std::map<std::string, LIMITER> _name_to_limiter_map;
    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;
    std::string m_eps_name;

    typedef std::vector<std::string> SV;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType XFaceT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType YFaceT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType ZFaceT;
    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType ConstXFaceT;
    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType ConstYFaceT;
    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType ConstZFaceT;

  };

  //------------------------------------------------------------------------------------------------
  //                              Member Function Definitions
  //------------------------------------------------------------------------------------------------
  template <typename T>
  ComputePsi<T>::ComputePsi( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

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

        std::string which_mom = _task_name.substr(0,5);

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
  void ComputePsi<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      XFaceT& psi_x = tsk_info->get_uintah_field_add<XFaceT>(*i+"_x_psi");
      YFaceT& psi_y = tsk_info->get_uintah_field_add<YFaceT>(*i+"_y_psi");
      ZFaceT& psi_z = tsk_info->get_uintah_field_add<ZFaceT>(*i+"_z_psi");

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
      register_variable( *i+"_x_psi", AFC::COMPUTES, variable_registry, time_substep, _task_name, packed_tasks );
      register_variable( *i+"_y_psi", AFC::COMPUTES, variable_registry, time_substep, _task_name, packed_tasks );
      register_variable( *i+"_z_psi", AFC::COMPUTES, variable_registry, time_substep, _task_name, packed_tasks );
      register_variable( *i, AFC::REQUIRES, 2, AFC::LATEST, variable_registry, time_substep, _task_name );
    }

    int nGhosts = 1;
    if ( packed_tasks ){
      nGhosts = 2;
    }

    register_variable( m_eps_name,   AFC::REQUIRES, 2, AFC::NEWDW, variable_registry, time_substep, _task_name );
    register_variable( m_u_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, _task_name );
    register_variable( m_v_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, _task_name );
    register_variable( m_w_vel_name, AFC::REQUIRES, nGhosts, AFC::NEWDW, variable_registry, time_substep, _task_name );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void ComputePsi<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    CT& eps = *(tsk_info->get_const_uintah_field<CT>(m_eps_name));

    ConstXFaceT& u = *(tsk_info->get_const_uintah_field<ConstXFaceT>(m_u_vel_name));
    ConstYFaceT& v = *(tsk_info->get_const_uintah_field<ConstYFaceT>(m_v_vel_name));
    ConstZFaceT& w = *(tsk_info->get_const_uintah_field<ConstZFaceT>(m_w_vel_name));

    for ( auto i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      std::map<std::string, LIMITER>::iterator ilim = _name_to_limiter_map.find(*i);
      LIMITER my_limiter = ilim->second;
      CT& phi = *(tsk_info->get_const_uintah_field<CT>(*i));

      int nGhosts = -1; //not using a temp field but rather the DW (ie, if nGhost < 0 then DW var)
      if ( tsk_info->packed_tasks() ){
        nGhosts = 1;
      }

      XFaceT& psi_x = tsk_info->get_uintah_field_add<XFaceT>(*i+"_x_psi", nGhosts);
      YFaceT& psi_y = tsk_info->get_uintah_field_add<YFaceT>(*i+"_y_psi", nGhosts);
      ZFaceT& psi_z = tsk_info->get_uintah_field_add<ZFaceT>(*i+"_z_psi", nGhosts);

      psi_x.initialize(1.0);
      psi_y.initialize(1.0);
      psi_z.initialize(1.0);

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_compute_psi.out."+*i);
      timer.start("ComputePsi");
#endif

      if ( my_limiter == UPWIND ){
        UpwindStruct up;
        GET_PSI(up);
      } else if ( my_limiter == CENTRAL ){
        CentralStruct central;
        GET_PSI(central);
      } else if ( my_limiter == SUPERBEE ){
        SuperBeeStruct sb;
        GET_PSI(sb);
      } else if ( my_limiter == ROE ){
        RoeStruct roe;
        GET_PSI(roe);
      } else if ( my_limiter == VANLEER ){
        VanLeerStruct vl;
        GET_PSI(vl);
      }

#ifdef DO_TIMINGS
      timer.stop("ComputePsi");
#endif

    }
  }
}
#endif
