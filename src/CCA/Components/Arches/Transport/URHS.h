#ifndef Uintah_Component_Arches_URHS
#define Uintah_Component_Arches_URHS

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Transport/MomentumHelper.h>

namespace Uintah{

  template <typename UT>
  class URHS : public TaskInterface {

public:


  URHS<UT>( std::string task_name, int matl_index, const std::string u_name, const std::string v_name, const std::string w_name );
  ~URHS<UT>();

  void problemSetup( ProblemSpecP& db );

  class Builder : public TaskInterface::TaskBuilder{

    public:

      Builder( std::string task_name, int matl_index, std::string u_name, std::string v_name, std::string w_name ) :
        _task_name(task_name), _matl_index(matl_index), _u_name(u_name), _v_name(v_name), _w_name(w_name){}
      ~Builder(){}

      URHS* build()
      { return scinew URHS<UT>(_task_name, _matl_index, _u_name, _v_name, _w_name ); }

    private:

      std::string _task_name;
      int _matl_index;
      std::string _u_name;
      std::string _v_name;
      std::string _w_name;

  };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels();

private:

    //typedefs....


    //fields
    typedef SpatialOps::SVolField   SVolF;
    typedef SpatialOps::SpatFldPtr<UT> UFieldTP;
    typedef SpatialOps::SpatFldPtr<typename MomentumHelper<UT>::VT> VFieldTP;
    typedef SpatialOps::SpatFldPtr<typename MomentumHelper<UT>::WT> WFieldTP;
    typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;

    //operators
    typedef typename SpatialOps::BasicOpTypes<UT>::GradX UGradX;
    typedef typename SpatialOps::BasicOpTypes<UT>::GradY UGradY;
    typedef typename SpatialOps::BasicOpTypes<UT>::GradZ UGradZ;

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomentumHelper<UT>::XFaceT, UT>::type UDivX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomentumHelper<UT>::YFaceT, UT>::type UDivY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomentumHelper<UT>::ZFaceT, UT>::type UDivZ;

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomentumHelper<UT>::XFaceT>::type IUFX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomentumHelper<UT>::YFaceT >::type IUFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename MomentumHelper<UT>::VT, typename MomentumHelper<UT>::YFaceT >::type IVFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomentumHelper<UT>::ZFaceT >::type IUFZ;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename MomentumHelper<UT>::WT, typename MomentumHelper<UT>::ZFaceT >::type IWFZ;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, UT >::type SVolToUVol;

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, UT, typename MomentumHelper<UT>::XFaceT>::type GradUFX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, UT, typename MomentumHelper<UT>::YFaceT>::type GradUFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, UT, typename MomentumHelper<UT>::ZFaceT>::type GradUFZ;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, typename MomentumHelper<UT>::VT, typename MomentumHelper<UT>::YFaceT >::type GradVFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, typename MomentumHelper<UT>::WT, typename MomentumHelper<UT>::ZFaceT >::type GradWFZ;

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, typename MomentumHelper<UT>::XFaceT>::type SVolToFaceX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, typename MomentumHelper<UT>::YFaceT>::type SVolToFaceY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolF, typename MomentumHelper<UT>::ZFaceT>::type SVolToFaceZ;

    std::string _rhou_name;
    std::string _Fconv_name;
    std::string _Tau_name;
    const std::string _u_name;
    const std::string _v_name;
    const std::string _w_name;

    std::string _rhs_name;
    std::string _conv_scheme;

    bool _do_conv;

    Wasatch::ConvInterpMethods _limiter_type;

  }; //class URHS


  //Functions:
  template <typename UT>
  void URHS<UT>::create_local_labels(){

    register_new_variable<UT>( _rhs_name   );
    register_new_variable<UT>( _u_name     );
    register_new_variable<UT>( _rhou_name  );
    register_new_variable<UT>( _Fconv_name );
    register_new_variable<UT>( _Tau_name );

  }

  //-------constructor-------------------
  template <typename UT>
  URHS<UT>::URHS( std::string task_name, int matl_index,
                  const std::string u_name, const std::string v_name, const std::string w_name ) :
    TaskInterface( task_name, matl_index ), _u_name(u_name), _v_name(v_name), _w_name(w_name) {

    _rhs_name = task_name + "_RHS";
    _rhou_name = task_name;
    _Fconv_name = task_name +"_Fconv";
    _Tau_name = task_name +"_Tauij";

  }

  //-------destructor--------------------
  template <typename UT>
  URHS<UT>::~URHS()
  {}

  //------problem setup-----------------
  template <typename UT>
  void URHS<UT>::problemSetup( ProblemSpecP& db ){

    _do_conv = false;
    if ( db->findBlock("convection")){
      db->findBlock("convection")->getAttribute("scheme", _conv_scheme);
      _do_conv = true;

      if ( _conv_scheme == "superbee"){
        _limiter_type = Wasatch::SUPERBEE;
      } else if ( _conv_scheme == "central"){
        _limiter_type = Wasatch::CENTRAL;
      } else if ( _conv_scheme == "upwind"){
        _limiter_type = Wasatch::UPWIND;
      } else if ( _conv_scheme == "charm"){
        _limiter_type = Wasatch::CHARM;
      } else if ( _conv_scheme == "koren"){
        _limiter_type = Wasatch::KOREN;
      } else if ( _conv_scheme == "mc"){
        _limiter_type = Wasatch::MC;
      } else if ( _conv_scheme == "ospre"){
        _limiter_type = Wasatch::OSPRE;
      } else if ( _conv_scheme == "smart"){
        _limiter_type = Wasatch::SMART;
      } else if ( _conv_scheme == "vanleer"){
        _limiter_type = Wasatch::VANLEER;
      } else if ( _conv_scheme == "hcus"){
        _limiter_type = Wasatch::HCUS;
      } else if ( _conv_scheme == "minmod"){
        _limiter_type = Wasatch::MINMOD;
      } else if ( _conv_scheme == "hquick"){
        _limiter_type = Wasatch::HQUICK;
      } else {
        throw InvalidValue("Error: Convection scheme not supported for scalar.",__FILE__,__LINE__);
      }

      //for now:
      if ( _conv_scheme != "central" ){
        _conv_scheme = "central";
      }

    }

  }

  //------t=0 initialize-----------------
  template <typename UT>
  void URHS<UT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    register_variable( _rhs_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( _u_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( _rhou_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );

  }

  template <typename UT>
  void URHS<UT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                SpatialOps::OperatorDatabase& opr ){

    using namespace SpatialOps;
    using SpatialOps::operator *;

    typedef SpatialOps::SpatFldPtr<UT> UFieldTP;

    UFieldTP uvel  = tsk_info->get_so_field<UT>(_u_name);
    UFieldTP rhs   = tsk_info->get_so_field<UT>(_rhs_name);
    UFieldTP rhou  = tsk_info->get_so_field<UT>(_rhou_name);

    *rhs <<= 0.0;
    *uvel <<= 0.0;
    *rhou <<= 0.0;

  }

  //------timestep initialization---------------
  template <typename UT>
  void URHS<UT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry){

    register_variable( _u_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( _u_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
    register_variable( _rhou_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( _rhou_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
    register_variable( _Fconv_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( _Tau_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    register_variable( "density", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry );

  }

  template <typename UT>
  void URHS<UT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                    SpatialOps::OperatorDatabase& opr ){


    using namespace SpatialOps;
    using SpatialOps::operator *;

    const SVolToUVol* const isvol_to_uvol = opr.retrieve_operator<SVolToUVol>();

    UFieldTP u_new    = tsk_info->get_so_field<UT>( _u_name );
    UFieldTP u_old    = tsk_info->get_const_so_field<UT>( _u_name );
    UFieldTP rhou_new = tsk_info->get_so_field<UT>( _rhou_name );
    UFieldTP rhou_old = tsk_info->get_const_so_field<UT>( _rhou_name );
    UFieldTP Fconv    = tsk_info->get_so_field<UT>( _Fconv_name );
    UFieldTP Tauij    = tsk_info->get_so_field<UT>( _Tau_name );
    SVolFP rho        = tsk_info->get_const_so_field<SVolF>( "density" );

    *u_new <<= *u_old;
    *rhou_new <<= *rhou_old;
    *Fconv <<= 0.0;
    *Tauij <<= 0.0;

    *u_new <<= *rhou_new / ((*isvol_to_uvol)(*rho));

  }

  //-------timestep work------------------------
  template <typename UT>
  void URHS<UT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    register_variable( _rhs_name   ,ArchesFieldContainer::COMPUTES ,0 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _Fconv_name ,ArchesFieldContainer::MODIFIES ,0 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _Tau_name   ,ArchesFieldContainer::MODIFIES ,0 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _rhou_name  ,ArchesFieldContainer::REQUIRES ,1 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _u_name     ,ArchesFieldContainer::REQUIRES ,1 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _v_name     ,ArchesFieldContainer::REQUIRES ,1 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( _w_name     ,ArchesFieldContainer::REQUIRES ,1 ,ArchesFieldContainer::NEWDW  ,variable_registry );
    register_variable( "density"   ,ArchesFieldContainer::REQUIRES ,1 ,ArchesFieldContainer::LATEST ,variable_registry );

  }

  template <typename UT>
  void URHS<UT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                           SpatialOps::OperatorDatabase& opr ){

    using namespace SpatialOps;
    using SpatialOps::operator *;

    typedef typename MomentumHelper<UT>::VT VT;
    typedef typename MomentumHelper<UT>::WT WT;

    //MomentumHelper<UT> mom_helper;

    const double dt = tsk_info->get_dt();

    //fields
    UFieldTP rhs   = tsk_info->get_so_field<UT>( _rhs_name );
    UFieldTP Fconv = tsk_info->get_so_field<UT>( _Fconv_name );
    UFieldTP Tauij   = tsk_info->get_so_field<UT>( _Tau_name );
    UFieldTP rhou  = tsk_info->get_const_so_field<UT>( _rhou_name );
    UFieldTP u     = tsk_info->get_const_so_field<UT>( _u_name );
    VFieldTP v     = tsk_info->get_const_so_field<VT>( _v_name );
    WFieldTP w     = tsk_info->get_const_so_field<WT>( _w_name );
    SVolFP rho     = tsk_info->get_const_so_field<SVolF>( "density" );

    //operators
    const IUFX* const uinterpx = opr.retrieve_operator<IUFX>();
    const IUFY* const uinterpy = opr.retrieve_operator<IUFY>();
    const IVFY* const vinterpy = opr.retrieve_operator<IVFY>();
    const IUFZ* const uinterpz = opr.retrieve_operator<IUFZ>();
    const IWFZ* const winterpz = opr.retrieve_operator<IWFZ>();
    //---
    const UDivX* const divx = opr.retrieve_operator<UDivX>();
    const UDivY* const divy = opr.retrieve_operator<UDivY>();
    const UDivZ* const divz = opr.retrieve_operator<UDivZ>();
    //---
    const GradUFX* const ugradx = opr.retrieve_operator<GradUFX>();
    const GradUFY* const ugrady = opr.retrieve_operator<GradUFY>();
    const GradUFZ* const ugradz = opr.retrieve_operator<GradUFZ>();
    const GradVFY* const vgradx = opr.retrieve_operator<GradVFY>();
    const GradWFZ* const wgradx = opr.retrieve_operator<GradWFZ>();
    //---
    const SVolToFaceX* const svol_to_facex = opr.retrieve_operator<SVolToFaceX>();
    const SVolToFaceY* const svol_to_facey = opr.retrieve_operator<SVolToFaceY>();
    const SVolToFaceZ* const svol_to_facez = opr.retrieve_operator<SVolToFaceZ>();

    //------ work -----------
    *Fconv <<= (*divx)( (*uinterpx)(*rhou) * (*uinterpx)(*u) ) +
               (*divy)( (*uinterpy)(*rhou) * (*vinterpy)(*v) ) +
               (*divz)( (*uinterpz)(*rhou) * (*winterpz)(*w) );

    //need to move viscosity into div.
    *Tauij <<= 1.0e-4 * (*divx)( (*svol_to_facex)(*rho) * 2.0 * (*ugradx)(*u) ) +
                        (*divy)( (*svol_to_facey)(*rho) * ((*ugrady)(*u) + (*vgradx)(*v)) ) +
                        (*divz)( (*svol_to_facez)(*rho) * ((*ugradz)(*u) + (*wgradx)(*w)) );

    *rhs <<= *rhou + dt * ( *Tauij - *Fconv );

  }

  //------BCs------------------------------------
  template <typename UT>
  void URHS<UT>::register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
  }

  template <typename UT>
  void URHS<UT>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                  SpatialOps::OperatorDatabase& opr ){
  }


}

    //debugging print help
    //typename UT::iterator it = utemp2->interior_begin();
    //for (;it != utemp2->interior_end();it++){
      //std::cout << *it << std::endl;
    //}

#endif
