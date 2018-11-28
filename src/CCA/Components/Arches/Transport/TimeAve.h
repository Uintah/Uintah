
#ifndef Uintah_Component_Arches_TimeAve_h
#define Uintah_Component_Arches_TimeAve_h

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
#include <CCA/Components/Arches/Transport/TransportHelper.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/Directives.h>
#include <iomanip>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

namespace Uintah{

  template <typename T>
  class TimeAve : public TaskInterface {

public:

    TimeAve<T>( std::string task_name, int matl_index );
    ~TimeAve<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        m_task_name(task_name), _matl_index(matl_index) {}
      ~Builder(){}

      TimeAve* build()
      { return scinew TimeAve( m_task_name, _matl_index ); }

      private:

      std::string m_task_name;
      int _matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks){}

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<CT>::XFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<CT>::YFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<CT>::ZFaceType CFZT;

    std::vector<std::string> _eqn_names;
    std::vector<std::string> m_transported_eqn_names;
    //std::map<std::string, double> m_scaling_info;

    struct Scaling_info {
      std::string unscaled_var; // unscaled value
      double constant; // 
    };
    std::map<std::string, Scaling_info> m_scaling_info;

    int _time_order;
    std::vector<double> _alpha;
    std::vector<double> _beta;
    std::vector<double> _time_factor;
    ArchesCore::EQUATION_CLASS m_eqn_class;

    ArchesCore::DIR m_dir;
    std::string m_volFraction_name;

    //std::string m_premultiplier_name;

  };

  //Function definitions:
  //------------------------------------------------------------------------------------------------
  template <typename T>
  TimeAve<T>::TimeAve( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TimeAve<T>::~TimeAve()
  {
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeAve<T>::create_local_labels(){
    for ( auto i = m_scaling_info.begin(); i != m_scaling_info.end(); i++ ){
      register_new_variable<T>( (i->second).unscaled_var);

    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeAve<T>::problemSetup( ProblemSpecP& db ){

    m_volFraction_name = "volFraction";

    std::string eqn_class = "density_weighted";
    if ( db->findAttribute("class") ){
      db->getAttribute("class", eqn_class);
    }
    m_eqn_class = ArchesCore::assign_eqn_class_enum( eqn_class );
    std::string premultiplier_name = get_premultiplier_name( m_eqn_class );
    std::string postmultiplier_name = get_postmultiplier_name( m_eqn_class );

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
    std::string env_number="NA";
    if (m_eqn_class == ArchesCore::DQMOM) {      
      db->findBlock("env_number")->getAttribute("number", env_number);    
    }
    _eqn_names.clear();
    for (ProblemSpecP eqn_db = db->findBlock("eqn");
	       eqn_db.get_rep() != nullptr;
         eqn_db = eqn_db->findNextBlock("eqn")){

      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);
      _eqn_names.push_back(scalar_name);

    if (eqn_db->findBlock("no_weight_factor") == nullptr ){
      std::string trans_variable;
      if (m_eqn_class == ArchesCore::DQMOM) {

        std::string delimiter = env_number ;
        std::string name_1    = scalar_name.substr(0, scalar_name.find(delimiter));
        trans_variable         = name_1 + postmultiplier_name + env_number;//

      } else {

        trans_variable = premultiplier_name + scalar_name + postmultiplier_name;//

      }
      //if ( eqn_db->findBlock("scaling") ){

      //  double scaling_constant;
      //  eqn_db->findBlock("scaling")->getAttribute("value", scaling_constant);

      //  Scaling_info scaling_w ;
      //  scaling_w.unscaled_var = trans_variable + "_unscaled" ;
      //  scaling_w.constant     = scaling_constant;
      //  m_scaling_info.insert(std::make_pair(trans_variable, scaling_w));
      // }
    
      m_transported_eqn_names.push_back(trans_variable);//
    } else {
      // weight:  w is transported 
          m_transported_eqn_names.push_back(scalar_name);// for weights in DQMOM
      //Scaling Constant only for weight
      if ( eqn_db->findBlock("scaling") ){

        double scaling_constant;
        eqn_db->findBlock("scaling")->getAttribute("value", scaling_constant);
        //m_scaling_info.insert(std::make_pair(scalar_name, scaling_constant));

        Scaling_info scaling_w ;
        scaling_w.unscaled_var = "w_" + env_number ;
        scaling_w.constant    = scaling_constant;
        m_scaling_info.insert(std::make_pair(scalar_name, scaling_w));
      }
    }  

    }

    ArchesCore::VariableHelper<T> varhelp;
    m_dir = varhelp.dir;

    //special momentum case
    if ( _eqn_names.size() == 0 ){
      std::string which_mom = m_task_name.substr(0,5);
      _eqn_names.push_back(which_mom);
      m_transported_eqn_names.push_back(which_mom);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeAve<T>::register_timestep_eval(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    typedef std::vector<std::string> SV;
    int ieqn =0;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
      ieqn += 1;
    }
    register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name  );

    for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
      register_variable((ieqn->second).unscaled_var, ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeAve<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    //const double dt = tsk_info->get_dt();
    //Vector DX = patch->dCell();
    //const double V = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    const int time_substep = tsk_info->get_time_substep();
    int ceqn = 0;
    for ( SV::iterator ieqn = _eqn_names.begin(); ieqn != _eqn_names.end(); ieqn++){

      T& phi = tsk_info->get_uintah_field_add<T>(m_transported_eqn_names[ceqn]);
      CT& old_phi = tsk_info->get_const_uintah_field_add<CT>(m_transported_eqn_names[ceqn], ArchesFieldContainer::OLDDW);
      ceqn +=1;

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_time_av.out."+*i);
      timer.start("work");
#endif

      auto fe_update = [&](int i, int j, int k){
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

#ifdef DO_TIMINGS
      timer.stop("work");
#endif

    }

    // unscaling
    // work in progress
    for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){

      std::string varname = ieqn->first;
      Scaling_info info = ieqn->second;
      //const double scaling_constant = ieqn->second;
      constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);

      T& phi = tsk_info->get_uintah_field_add<T>(varname);
      T& phi_unscaled = tsk_info->get_uintah_field_add<T>(info.unscaled_var);

      Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

      Uintah::parallel_for( range, [&](int i, int j, int k){

        phi_unscaled(i,j,k) = phi(i,j,k) * info.constant * vol_fraction(i,j,k);

      });
    }
  }

//--------------------------------------------------------------------------------------------------
  template <typename T> void
  TimeAve<T>::register_compute_bcs(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep, m_task_name  );
    
    for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
      register_variable((ieqn->second).unscaled_var, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
      register_variable(ieqn->first, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
  }
//--------------------------------------------------------------------------------------------------
  template <typename T > void
  TimeAve<T >::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  ArchesCore::VariableHelper<T> helper;
  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
  constCCVariable<double>& vol_fraction = tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_volFraction_name);
  

  for ( auto ieqn = m_scaling_info.begin(); ieqn != m_scaling_info.end(); ieqn++ ){
    CT& phi = tsk_info->get_const_uintah_field_add<CT>(ieqn->first);
    T& phi_unscaled = tsk_info->get_uintah_field_add<T>((ieqn->second).unscaled_var);
    
    for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){
      const bool on_this_patch = i_bc->second.has_patch(patch->getID());
      //Get the iterator

      if ( on_this_patch ){
        Uintah::Iterator cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
        
          for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
            IntVector c = *cell_iter;
            phi_unscaled[c] = phi[c] * (ieqn->second).constant*vol_fraction[c] ;
          }
        }
    }
  }
  }
}
#endif
