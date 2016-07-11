#ifndef Uintah_Component_Arches_ComputePsi_h
#define Uintah_Component_Arches_ComputePsi_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <CCA/Components/Arches/Directives.h>
#include <spatialops/util/TimeLogger.h>

#define GET_PSI(my_limiter_struct) \
    GetPsi get_psi_x( phi, psi_x, u, af_x, 0 ); \
    GetPsi get_psi_y( phi, psi_y, v, af_y, 1 ); \
    GetPsi get_psi_z( phi, psi_z, w, af_z, 2 ); \
    GET_FX_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange x_range(low_fx_patch_range, high_fx_patch_range); \
    Uintah::parallel_for(x_range, get_psi_x, my_limiter_struct); \
    GET_FY_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange y_range(low_fy_patch_range, high_fy_patch_range); \
    Uintah::parallel_for(y_range, get_psi_y, my_limiter_struct); \
    GET_FZ_BUFFERED_PATCH_RANGE(1,0); \
    Uintah::BlockRange z_range(low_fz_patch_range, high_fz_patch_range); \
    Uintah::parallel_for(z_range, get_psi_z, my_limiter_struct);


namespace Uintah{

  template <typename T>
  class ComputePsi : public TaskInterface {

public:

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
      { return new ComputePsi( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

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

    typedef std::vector<std::string> SV;
    typedef typename GridTools::VariableHelper<T>::ConstType CT;
    typedef typename GridTools::VariableHelper<T>::XFaceType XFaceT;
    typedef typename GridTools::VariableHelper<T>::YFaceType YFaceT;
    typedef typename GridTools::VariableHelper<T>::ZFaceType ZFaceT;
    typedef typename GridTools::VariableHelper<T>::ConstXFaceType ConstXFaceT;
    typedef typename GridTools::VariableHelper<T>::ConstYFaceType ConstYFaceT;
    typedef typename GridTools::VariableHelper<T>::ConstZFaceType ConstZFaceT;



  };

  //---------------------------------------------------------------------------------------
  //Function definitions:
  template <typename T>
  ComputePsi<T>::ComputePsi( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

  template <typename T>
  ComputePsi<T>::~ComputePsi(){}

  template <typename T>
  void ComputePsi<T>::problemSetup( ProblemSpecP& db ){
    for (ProblemSpecP eqn_db = db->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){
      std::string limiter;
      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);

      if ( eqn_db->findBlock("convection" )){
        eqn_db->findBlock("convection")->getAttribute("scheme",limiter);

        ConvectionHelper* conv_helper;

        LIMITER enum_limiter = conv_helper->get_limiter_from_string(limiter);

        _name_to_limiter_map.insert(std::make_pair(scalar_name, enum_limiter));

        _eqn_names.push_back(scalar_name);
      }

    }
  }

  template <typename T>
  void ComputePsi<T>::register_initialize( AVarInfo& variable_registry ){
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
