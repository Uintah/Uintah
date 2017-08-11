#ifndef Uintah_Component_Arches_KFEUpdate_h
#define Uintah_Component_Arches_KFEUpdate_h

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
#include <CCA/Components/Arches/Directives.h>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

namespace Uintah{

  template <typename T>
  class KFEUpdate : public TaskInterface {

public:

    KFEUpdate<T>( std::string task_name, int matl_index );
    ~KFEUpdate<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){}

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        _task_name(task_name), _matl_index(matl_index) {}
      ~Builder(){}

      KFEUpdate* build()
      { return scinew KFEUpdate( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    std::vector<std::string> _eqn_names;

    int _time_order;
    std::vector<double> _alpha;
    std::vector<double> _beta;
    std::vector<double> _time_factor;

    ArchesCore::DIR m_dir;

  };

  //Function definitions:
  template <typename T>
  KFEUpdate<T>::KFEUpdate( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

  template <typename T>
  KFEUpdate<T>::~KFEUpdate()
  {
  }

  template <typename T>
  void KFEUpdate<T>::problemSetup( ProblemSpecP& db ){

    ProblemSpecP db_root = db->getRootNode();
    db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator")->getAttribute("order", _time_order);

    if ( _time_order == 1 ){

      _alpha.resize(1);
      _beta.resize(1);
      _time_factor.resize(1);

      _alpha[0] = 0.0;

      _beta[0]  = 1.0;

      _time_factor[0] = 1.0;

    } else if ( _time_order == 2 ) {

      _alpha.resize(2);
      _beta.resize(2);
      _time_factor.resize(2);

      _alpha[0]= 0.0;
      _alpha[1]= 0.5;

      _beta[0]  = 1.0;
      _beta[1]  = 0.5;

      _time_factor[0] = 1.0;
      _time_factor[1] = 1.0;

    } else if ( _time_order == 3 ) {

      _alpha.resize(3);
      _beta.resize(3);
      _time_factor.resize(3);

      _alpha[0] = 0.0;
      _alpha[1] = 0.75;
      _alpha[2] = 1.0/3.0;

      _beta[0]  = 1.0;
      _beta[1]  = 0.25;
      _beta[2]  = 2.0/3.0;

      _time_factor[0] = 1.0;
      _time_factor[1] = 0.5;
      _time_factor[2] = 1.0;

    } else {
      throw InvalidValue("Error: <TimeIntegrator> must have value: 1, 2, or 3 (representing the order).",__FILE__, __LINE__);
    }

    _eqn_names.clear();
    for (ProblemSpecP eqn_db = db->findBlock("eqn");
	 eqn_db.get_rep() != nullptr;
         eqn_db = eqn_db->findNextBlock("eqn")){

      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);
      _eqn_names.push_back(scalar_name);

    }

    ArchesCore::VariableHelper<T> varhelp;
    m_dir = varhelp.dir;

    //special momentum case
    if ( _eqn_names.size() == 0 ){
      std::string which_mom = _task_name.substr(0,5);
      _eqn_names.push_back(which_mom);
    }

  }


  template <typename T>
  void KFEUpdate<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks){
  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void KFEUpdate<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  template <typename T>
  void KFEUpdate<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      std::string rhs_name = *i + "_rhs";
      register_variable( rhs_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( *i+"_x_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_y_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_z_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    }
  }

  template <typename T>
  void KFEUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double V = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    const int time_substep = tsk_info->get_time_substep();

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      T& phi = *(tsk_info->get_uintah_field<T>(*i));
      T& rhs = *(tsk_info->get_uintah_field<T>(*i+"_rhs"));
      CT& old_phi = *(tsk_info->get_const_uintah_field<CT>(*i, ArchesFieldContainer::OLDDW));
      CFXT& x_flux = *(tsk_info->get_const_uintah_field<CFXT>(*i+"_x_flux"));
      CFYT& y_flux = *(tsk_info->get_const_uintah_field<CFYT>(*i+"_y_flux"));
      CFZT& z_flux = *(tsk_info->get_const_uintah_field<CFZT>(*i+"_z_flux"));

      Vector Dx = patch->dCell();
      double ax = Dx.y() * Dx.z();
      double ay = Dx.z() * Dx.x();
      double az = Dx.x() * Dx.y();

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_fe_update.out."+*i);
      timer.start("work");
#endif

      if ( time_substep == 0 ){
        auto fe_update = [&](int i, int j, int k){

          rhs(i,j,k) = rhs(i,j,k) - ( ax * ( x_flux(i+1,j,k) - x_flux(i,j,k) ) +
                                      ay * ( y_flux(i,j+1,k) - y_flux(i,j,k) ) +
                                      az * ( z_flux(i,j,k+1) - z_flux(i,j,k) ) );

          phi(i,j,k) = phi(i,j,k) + dt/V * rhs(i,j,k);

        };

        if ( m_dir == ArchesCore::XDIR ){
          GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else if ( m_dir == ArchesCore::YDIR ){
          GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fy_patch_range, high_fy_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else if ( m_dir == ArchesCore::ZDIR ){
          GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fz_patch_range, high_fz_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else {
          Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());
          Uintah::parallel_for( range, fe_update );
        }

      } else {

        auto fe_update = [&](int i, int j, int k){

          rhs(i,j,k) = rhs(i,j,k) - ( ax * ( x_flux(i+1,j,k) - x_flux(i,j,k) ) +
                                      ay * ( y_flux(i,j+1,k) - y_flux(i,j,k) ) +
                                      az * ( z_flux(i,j,k+1) - z_flux(i,j,k) ) );

          phi(i,j,k) = phi(i,j,k) + dt/V * rhs(i,j,k);

          phi(i,j,k) = _alpha[time_substep] * old_phi(i,j,k) + _beta[time_substep] * phi(i,j,k);

        };

        if ( m_dir == ArchesCore::XDIR ){
          GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fx_patch_range, high_fx_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else if ( m_dir == ArchesCore::YDIR ){
          GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fy_patch_range, high_fy_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else if ( m_dir == ArchesCore::ZDIR ){
          GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1,0);
          Uintah::BlockRange range(low_fz_patch_range, high_fz_patch_range);
          Uintah::parallel_for( range, fe_update );
        } else {
          Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());
          Uintah::parallel_for( range, fe_update );
        }
      }

#ifdef DO_TIMINGS
      timer.stop("work");
#endif
    }
  }
}
#endif
