#ifndef Uintah_Component_Arches_ComputePsi_h
#define Uintah_Component_Arches_ComputePsi_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <spatialops/util/TimeLogger.h>

#define GET_PSI(my_limiter) \
    GetPsi<my_limiter, XFaceT> get_psi_x( phi, psi_x, u, af_x ); \
    GetPsi<my_limiter, YFaceT> get_psi_y( phi, psi_y, v, af_y ); \
    GetPsi<my_limiter, ZFaceT> get_psi_z( phi, psi_z, w, af_z ); \
    GET_FX_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange x_range(low_fx_patch_range, high_fx_patch_range); \
    Uintah::parallel_for(x_range, get_psi_x); \
    GET_FY_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange y_range(low_fy_patch_range, high_fy_patch_range); \
    Uintah::parallel_for(y_range, get_psi_y); \
    GET_FZ_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange z_range(low_fz_patch_range, high_fz_patch_range); \
    Uintah::parallel_for(z_range, get_psi_z);

namespace Uintah{

  template <typename T>
  class ComputePsi : public TaskInterface {

public:

    ComputePsi<T>( std::string task_name, int matl_index, std::vector<std::string> eqn_names );
    ~ComputePsi<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){

      typedef typename VariableHelper<T>::XFaceType XFaceT;
      typedef typename VariableHelper<T>::YFaceType YFaceT;
      typedef typename VariableHelper<T>::ZFaceType ZFaceT;

      typedef std::vector<std::string> SV;
      for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

        register_new_variable<XFaceT>( *i + "_x_psi" );
        register_new_variable<YFaceT>( *i + "_y_psi" );
        register_new_variable<ZFaceT>( *i + "_z_psi" );

      }

    }

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) :
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names){}
      ~Builder(){}

      ComputePsi* build()
      { return scinew ComputePsi( _task_name, _matl_index, _eqn_names ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::vector<std::string> _eqn_names;

    };

protected:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    void register_initialize( AVarInfo& variable_registry );

    void register_timestep_init( AVarInfo& variable_registry ){}

    void register_timestep_eval( AVarInfo& variable_registry, const int time_substep );

    void register_compute_bcs( AVarInfo& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

private:

    std::vector<std::string> _eqn_names;
    std::map<std::string, LIMITER> _name_to_limiter_map;


  };

  //---------------------------------------------------------------------------------------
  //Function definitions:
  template <typename T>
  ComputePsi<T>::ComputePsi( std::string task_name, int matl_index,
                             std::vector<std::string> eqn_names ) :
  TaskInterface( task_name, matl_index ){

    _eqn_names = eqn_names;

  }

  template <typename T>
  ComputePsi<T>::~ComputePsi(){}

  template <typename T>
  void ComputePsi<T>::problemSetup( ProblemSpecP& db ){
    for (ProblemSpecP eqn_db = db->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){
      std::string limiter;
      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);
      eqn_db->findBlock("convection")->getAttribute("scheme",limiter);

      ConvectionHelper* conv_helper;

      LIMITER enum_limiter = conv_helper->get_limiter_from_string(limiter);

      _name_to_limiter_map.insert(std::make_pair(scalar_name, enum_limiter));

    }
  }

  template <typename T>
  void ComputePsi<T>::register_initialize( AVarInfo& variable_registry ){
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i+"_x_psi", ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( *i+"_y_psi", ArchesFieldContainer::COMPUTES, variable_registry );
      register_variable( *i+"_z_psi", ArchesFieldContainer::COMPUTES, variable_registry );
    }
  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void ComputePsi<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                  SpatialOps::OperatorDatabase& opr ){
    typedef typename VariableHelper<T>::XFaceType XFaceT;
    typedef typename VariableHelper<T>::YFaceType YFaceT;
    typedef typename VariableHelper<T>::ZFaceType ZFaceT;

    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      XFaceT& psi_x = *(tsk_info->get_uintah_field<XFaceT>(*i+"_x_psi"));
      YFaceT& psi_y = *(tsk_info->get_uintah_field<YFaceT>(*i+"_y_psi"));
      ZFaceT& psi_z = *(tsk_info->get_uintah_field<ZFaceT>(*i+"_z_psi"));

      psi_x.initialize(0.0);
      psi_y.initialize(0.0);
      psi_z.initialize(0.0);
    }
  }


  template <typename T>
  void ComputePsi<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i+"_x_psi", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( *i+"_y_psi", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( *i+"_z_psi", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }
    register_variable( "areaFractionX", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( "areaFractionY", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( "areaFractionZ", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( "uVel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( "vVel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( "wVel", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  }

  template <typename T>
  void ComputePsi<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

    typedef std::vector<std::string> SV;
    typedef typename VariableHelper<T>::ConstType CT;
    typedef typename VariableHelper<T>::XFaceType XFaceT;
    typedef typename VariableHelper<T>::YFaceType YFaceT;
    typedef typename VariableHelper<T>::ZFaceType ZFaceT;
    typedef typename VariableHelper<T>::ConstXFaceType ConstXFaceT;
    typedef typename VariableHelper<T>::ConstYFaceType ConstYFaceT;
    typedef typename VariableHelper<T>::ConstZFaceType ConstZFaceT;

    ConstXFaceT& af_x = *(tsk_info->get_const_uintah_field<ConstXFaceT>("areaFractionX"));
    ConstYFaceT& af_y = *(tsk_info->get_const_uintah_field<ConstYFaceT>("areaFractionY"));
    ConstZFaceT& af_z = *(tsk_info->get_const_uintah_field<ConstZFaceT>("areaFractionZ"));

    ConstXFaceT& u = *(tsk_info->get_const_uintah_field<ConstXFaceT>("uVel"));
    ConstYFaceT& v = *(tsk_info->get_const_uintah_field<ConstYFaceT>("vVel"));
    ConstZFaceT& w = *(tsk_info->get_const_uintah_field<ConstZFaceT>("wVel"));

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      //const LIMITER my_limiter = (_name_to_limiter_map.find(*i))->second();
      std::map<std::string, LIMITER>::iterator ilim = _name_to_limiter_map.find(*i);
      LIMITER my_limiter = ilim->second;

      CT& phi = *(tsk_info->get_const_uintah_field<CT>(*i));
      XFaceT& psi_x = *(tsk_info->get_uintah_field<XFaceT>(*i+"_x_psi"));
      YFaceT& psi_y = *(tsk_info->get_uintah_field<YFaceT>(*i+"_y_psi"));
      ZFaceT& psi_z = *(tsk_info->get_uintah_field<ZFaceT>(*i+"_z_psi"));

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_compute_psi.out."+*i);
      timer.start("ComputePsi");
#endif
      if ( my_limiter == UPWIND ){
        GET_PSI(UPWIND);
      } else if ( my_limiter == CENTRAL ){
        GET_PSI(CENTRAL);
      } else if ( my_limiter == SUPERBEE ){
        GET_PSI(SUPERBEE);
      } else if ( my_limiter == ROE ){
        GET_PSI(ROE);
      } else if ( my_limiter == VANLEER ){
        GET_PSI(VANLEER);
      }
#ifdef DO_TIMINGS
      timer.stop("ComputePsi");
#endif

    }
  }
}
#endif
