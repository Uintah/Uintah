
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
#include <Core/Parallel/LoopExecution.hpp>
#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

namespace Uintah{

  template <typename T>
  class SUpdate : public TaskInterface {

public:

    SUpdate<T>( std::string task_name, int matl_index );
    ~SUpdate<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        m_task_name(task_name), m_matl_index(matl_index) {}
      ~Builder(){}

      SUpdate* build()
      { return scinew SUpdate( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

    template <typename ExecutionSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& executionObject ) {}

    template <typename ExecutionSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& executionObject ){}

    template<typename ExecutionSpace, typename MemSpace> void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace,MemSpace>& exObj){}

    template <typename ExecutionSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& executionObject );

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks){}

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks) {}


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
    for ( auto i = m_scaling_info.begin(); i != m_scaling_info.end(); i++ ){
      register_new_variable<T>( (i->second).unscaled_var);

    }
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace SUpdate<T>::loadTaskComputeBCsFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::BC>( this
                                       , &SUpdate<T>::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &SUpdate<T>::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &SUpdate<T>::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace SUpdate<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &SUpdate<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &SUpdate<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &SUpdate<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace SUpdate<T>::loadTaskEvalFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                       , &SUpdate<T>::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &SUpdate<T>::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &SUpdate<T>::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace SUpdate<T>::loadTaskTimestepInitFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                       , &SUpdate<T>::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &SUpdate<T>::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       , &SUpdate<T>::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::OpenMP builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace SUpdate<T>::loadTaskRestartInitFunctionPointers()
  {
    return  TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
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
  void SUpdate<T>::register_timestep_eval(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep , const bool packed_tasks){

    typedef std::vector<std::string> SV;
    int ieqn =0;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( m_transported_eqn_names[ieqn], ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
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
  template<typename ExecutionSpace, typename MemSpace>
  void SUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemSpace>& exObj ){

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double Vol = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;

    int ceqn = 0;
    for ( SV::iterator ieqn = _eqn_names.begin(); ieqn != _eqn_names.end(); ieqn++){

      auto phi = tsk_info->get_uintah_field_add<T, double, MemSpace>(m_transported_eqn_names[ceqn]);
      auto rhs = tsk_info->get_uintah_field_add<T, double, MemSpace>(m_transported_eqn_names[ceqn]+"_RHS");

      ceqn +=1;
      auto x_flux = tsk_info->get_const_uintah_field_add<CFXT, const double, MemSpace>(*ieqn+"_x_flux");
      auto y_flux = tsk_info->get_const_uintah_field_add<CFYT, const double, MemSpace>(*ieqn+"_y_flux");
      auto z_flux = tsk_info->get_const_uintah_field_add<CFZT, const double, MemSpace>(*ieqn+"_z_flux");

      Vector Dx = patch->dCell();
      double ax = Dx.y() * Dx.z();
      double ay = Dx.z() * Dx.x();
      double az = Dx.x() * Dx.y();

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_fe_update.out."+*i);
      timer.start("work");
#endif

      BlockRange range2;
      if ( m_dir == ArchesCore::XDIR ){
        GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(1,0);
        range2=Uintah::BlockRange(low_fx_patch_range, high_fx_patch_range);
      } else if ( m_dir == ArchesCore::YDIR ){
        GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(1,0);
        range2=Uintah::BlockRange(low_fy_patch_range, high_fy_patch_range);
      } else if ( m_dir == ArchesCore::ZDIR ){
        GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(1,0);
        range2=Uintah::BlockRange(low_fz_patch_range, high_fz_patch_range);
      } else {
        range2=Uintah::BlockRange(patch->getCellLowIndex(), patch->getCellHighIndex());
      }

        Uintah::parallel_for(exObj, range2, KOKKOS_LAMBDA (int i, int j, int k){
       
        rhs(i,j,k) = rhs(i,j,k) - ( ax * ( x_flux(i+1,j,k) - x_flux(i,j,k) ) +
                                    ay * ( y_flux(i,j+1,k) - y_flux(i,j,k) ) +
                                    az * ( z_flux(i,j,k+1) - z_flux(i,j,k) ) );

        phi(i,j,k) = phi(i,j,k) + dt/Vol * rhs(i,j,k);

      });

#ifdef DO_TIMINGS
      timer.stop("work");
#endif

    }

  }

//--------------------------------------------------------------------------------------------------
}
#endif
