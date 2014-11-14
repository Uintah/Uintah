#ifndef Uintah_Component_Arches_URHS
#define Uintah_Component_Arches_URHS

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Arches/Transport/ScalarRHS.h>

namespace Uintah{ 


  template< typename FT>
  struct MomHelper{
  };

  template <>
  struct MomHelper<SpatialOps::XVolField>{

    typedef typename FaceTypes<SpatialOps::XVolField>::XFace XFaceT; 
    typedef typename FaceTypes<SpatialOps::XVolField>::YFace YFaceT; 
    typedef typename FaceTypes<SpatialOps::XVolField>::ZFace ZFaceT; 

    typedef typename SpatialOps::YVolField VT; 
    typedef typename SpatialOps::ZVolField WT; 

  };

  template <>
  struct MomHelper<SpatialOps::YVolField>{

    typedef typename FaceTypes<SpatialOps::YVolField>::YFace XFaceT; 
    typedef typename FaceTypes<SpatialOps::YVolField>::ZFace YFaceT; 
    typedef typename FaceTypes<SpatialOps::YVolField>::XFace ZFaceT; 

    typedef typename SpatialOps::ZVolField VT; 
    typedef typename SpatialOps::XVolField WT; 

  };

  template <>
  struct MomHelper<SpatialOps::ZVolField>{

    typedef typename FaceTypes<SpatialOps::ZVolField>::ZFace XFaceT; 
    typedef typename FaceTypes<SpatialOps::ZVolField>::XFace YFaceT; 
    typedef typename FaceTypes<SpatialOps::ZVolField>::YFace ZFaceT; 

    typedef typename SpatialOps::XVolField VT; 
    typedef typename SpatialOps::YVolField WT; 

  };

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
      std::string _u_name; 
      std::string _v_name; 
      std::string _w_name; 
      int _matl_index; 

  };

protected: 

    void register_initialize( std::vector<VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ); 

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
    typedef SpatialOps::SpatFldPtr<typename MomHelper<UT>::VT> VFieldTP;
    typedef SpatialOps::SpatFldPtr<typename MomHelper<UT>::WT> WFieldTP;
    typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

    //operators
    typedef typename SpatialOps::BasicOpTypes<UT>::GradX UGradX; 
    typedef typename SpatialOps::BasicOpTypes<UT>::GradY UGradY; 
    typedef typename SpatialOps::BasicOpTypes<UT>::GradZ UGradZ; 

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomHelper<UT>::XFaceT, UT>::type UDivX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomHelper<UT>::YFaceT, UT>::type UDivY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Divergence, typename MomHelper<UT>::ZFaceT, UT>::type UDivZ;

    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomHelper<UT>::XFaceT>::type IUFX;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomHelper<UT>::YFaceT >::type IUFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename MomHelper<UT>::VT, typename MomHelper<UT>::YFaceT >::type IVFY;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, UT, typename MomHelper<UT>::ZFaceT >::type IUFZ;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename MomHelper<UT>::WT, typename MomHelper<UT>::ZFaceT >::type IWFZ;

    std::string _rhou_name; 
    std::string _Fconv_name;
    const std::string _u_name; 
    const std::string _v_name; 
    const std::string _w_name; 

    std::string _rhs_name; 
    std::string _conv_scheme; 

    bool _do_conv; 

    VAR_TYPE _U_type; 
    VAR_TYPE _V_type; 
    VAR_TYPE _W_type; 
    Wasatch::ConvInterpMethods _limiter_type;

  }; //class URHS


  //Functions:
  template <typename UT>
  void URHS<UT>::create_local_labels(){ 

    register_new_variable( _rhs_name   , _U_type );
    register_new_variable( _u_name     , _U_type );
    register_new_variable( _rhou_name  , _U_type );
    register_new_variable( _Fconv_name , _U_type );
    
  }

  //-------constructor-------------------
  template <typename UT>
  URHS<UT>::URHS( std::string task_name, int matl_index, 
                          const std::string u_name, const std::string v_name, const std::string w_name ) : 
    _u_name(u_name), _v_name(v_name), _w_name(w_name), TaskInterface( task_name, matl_index ){ 

    _rhs_name = task_name + "_RHS";
    _rhou_name = task_name; 
    _Fconv_name = task_name +"_Fconv";

    //set the type: 
    VarTypeHelper<UT> uhelper; 
    _U_type = uhelper.get_vartype(); 
    VarTypeHelper<typename MomHelper<UT>::VT > vhelper; 
    _V_type = vhelper.get_vartype(); 
    VarTypeHelper<typename MomHelper<UT>::WT > whelper; 
    _W_type = whelper.get_vartype(); 

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
  void URHS<UT>::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

    register_variable( _rhs_name, _U_type, COMPUTES, 0, NEWDW, variable_registry ); 
    register_variable( _u_name, _U_type, COMPUTES, 0, NEWDW, variable_registry ); 
    register_variable( _rhou_name, _U_type, COMPUTES, 0, NEWDW, variable_registry ); 

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
  void URHS<UT>::register_timestep_init( std::vector<VariableInformation>& variable_registry){

    register_variable( _u_name, _U_type, COMPUTES, 0, NEWDW, variable_registry );
    register_variable( _u_name, _U_type, REQUIRES, 0, OLDDW, variable_registry );

  }

  template <typename UT> 
  void URHS<UT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                                    SpatialOps::OperatorDatabase& opr ){ 

    
    using namespace SpatialOps;
    using SpatialOps::operator *; 

    UFieldTP u_new = tsk_info->get_so_field<UT>( _u_name );
    UFieldTP u_old = tsk_info->get_const_so_field<UT>( _u_name );

    *u_new <<= *u_old; 

  }

  //-------timestep work------------------------
  template <typename UT>
  void URHS<UT>::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

    register_variable( _rhou_name, _U_type, COMPUTES, 0, NEWDW, variable_registry );
    register_variable( _Fconv_name, _U_type, COMPUTES, 0, NEWDW, variable_registry );
    register_variable( _u_name, _U_type, REQUIRES, 1, NEWDW, variable_registry );
    register_variable( _v_name, _V_type, REQUIRES, 1, NEWDW, variable_registry );
    register_variable( _w_name, _W_type, REQUIRES, 1, NEWDW, variable_registry );
    register_variable( _rhs_name, _U_type, COMPUTES, 0, NEWDW, variable_registry );
    register_variable( "density", CC_DOUBLE, REQUIRES, 1, LATEST, variable_registry );

  }

  template <typename UT>
  void URHS<UT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                           SpatialOps::OperatorDatabase& opr ){

    using namespace SpatialOps;
    using SpatialOps::operator *; 

    typedef typename MomHelper<UT>::VT VT; 
    typedef typename MomHelper<UT>::WT WT; 

    MomHelper<UT> mom_helper; 

    const double dt = tsk_info->get_dt();  

    //fields
    UFieldTP rhou  = tsk_info->get_so_field<UT>( _rhou_name );
    UFieldTP rhs   = tsk_info->get_so_field<UT>( _rhs_name ); 
    UFieldTP Fconv = tsk_info->get_so_field<UT>( _Fconv_name );
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

    //Convection 
    //Central:
    *Fconv <<= (*divx)( (*uinterpx)(*rhou) * (*uinterpx)(*u) ) +
               (*divy)( (*uinterpy)(*rhou) * (*vinterpy)(*v) ); 
               (*divz)( (*uinterpz)(*rhou) * (*winterpz)(*w) ); 


    //RHS update
    *rhs <<= *rhou + dt * ( -*Fconv ); 

  }

  //------BCs------------------------------------
  template <typename UT>
  void URHS<UT>::register_compute_bcs( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 
  }
  
  template <typename UT>
  void URHS<UT>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                                  SpatialOps::OperatorDatabase& opr ){ 
  }


}


#endif
