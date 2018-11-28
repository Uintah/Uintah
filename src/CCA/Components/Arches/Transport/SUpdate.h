
#ifndef Uintah_Component_Arches_SUpdate_h
#define Uintah_Component_Arches_SUpdate_h

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
  class SUpdate : public TaskInterface {

public:

    SUpdate<T>( std::string task_name, int matl_index );
    ~SUpdate<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        m_task_name(task_name), _matl_index(matl_index) {}
      ~Builder(){}

      SUpdate* build()
      { return scinew SUpdate( m_task_name, _matl_index ); }

      private:

      std::string m_task_name;
      int _matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks){}

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ) {}

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

    ArchesCore::EQUATION_CLASS m_eqn_class;

    ArchesCore::DIR m_dir;


  };

  //Function definitions:
  //------------------------------------------------------------------------------------------------
  template <typename T>
  SUpdate<T>::SUpdate( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  SUpdate<T>::~SUpdate()
  {
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void SUpdate<T>::create_local_labels(){
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void SUpdate<T>::problemSetup( ProblemSpecP& db ){

    std::string eqn_class = "density_weighted";
    if ( db->findAttribute("class") ){
      db->getAttribute("class", eqn_class);
    }
    m_eqn_class = ArchesCore::assign_eqn_class_enum( eqn_class );
    std::string premultiplier_name = get_premultiplier_name( m_eqn_class );
    std::string postmultiplier_name = get_postmultiplier_name( m_eqn_class );

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
    
      m_transported_eqn_names.push_back(trans_variable);//
    } else {
      // weight:  w is transported 
          m_transported_eqn_names.push_back(scalar_name);// for weights in DQMOM
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
  void SUpdate<T>::register_timestep_eval(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    typedef std::vector<std::string> SV;
    int ieqn =0;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      //register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
      std::string rhs_name = m_transported_eqn_names[ieqn] + "_RHS";
      register_variable( rhs_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( *i+"_x_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_y_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_z_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      ieqn += 1;
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void SUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double V = DX.x()*DX.y()*DX.z();

    Vector Dx = patch->dCell();
    const double ax = Dx.y() * Dx.z();
    const double ay = Dx.z() * Dx.x();
    const double az = Dx.x() * Dx.y();
    typedef std::vector<std::string> SV;
    //typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    //const int time_substep = tsk_info->get_time_substep();
    int ceqn = 0;
    for ( SV::iterator ieqn = _eqn_names.begin(); ieqn != _eqn_names.end(); ieqn++){

      T& phi = tsk_info->get_uintah_field_add<T>(m_transported_eqn_names[ceqn]);
      T& rhs = tsk_info->get_uintah_field_add<T>(m_transported_eqn_names[ceqn]+"_RHS");
      //CT& old_phi = tsk_info->get_const_uintah_field_add<CT>(m_transported_eqn_names[ceqn], ArchesFieldContainer::OLDDW);
      ceqn +=1;
      CFXT& x_flux = tsk_info->get_const_uintah_field_add<CFXT>(*ieqn+"_x_flux");
      CFYT& y_flux = tsk_info->get_const_uintah_field_add<CFYT>(*ieqn+"_y_flux");
      CFZT& z_flux = tsk_info->get_const_uintah_field_add<CFZT>(*ieqn+"_z_flux");


#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_fe_update.out."+*i);
      timer.start("work");
#endif


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


#ifdef DO_TIMINGS
      timer.stop("work");
#endif

    }
  }

}
#endif
