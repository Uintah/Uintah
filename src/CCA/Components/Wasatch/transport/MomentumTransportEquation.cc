#include "MomentumTransportEquation.h"

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/Stress.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>

using std::string;


namespace Wasatch{

  //==================================================================

  // note that the ordering of Vel1T and Vel2T are very important, and
  // must be consistent with the order of the velocity tags passed
  // into the stress constructor.
  template< typename FaceT > struct StressHelper;

  template<> struct StressHelper<SpatialOps::structured::XSurfXField>
  {
    typedef XVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::XSurfYField>
  {
    typedef XVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::XSurfZField>
  {
    typedef XVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StressHelper<SpatialOps::structured::YSurfXField>
  {
    typedef YVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::YSurfYField>
  {
    typedef YVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::YSurfZField>
  {
    typedef YVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StressHelper<SpatialOps::structured::ZSurfXField>
  {
    typedef ZVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::ZSurfYField>
  {
    typedef ZVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::ZSurfZField>
  {
    typedef ZVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  //==================================================================

  template< typename FieldT >
  string get_mom_dir_name()
  {
    switch( FieldT::Location::StagLoc::value ){
    case SpatialOps::XDIR::value:  return "x";
    case SpatialOps::YDIR::value:  return "y";
    case SpatialOps::ZDIR::value:  return "z";
    }
    return "-INVALID-";
  }

  //==================================================================

  /**
   *  \brief Register the stress expression for the given face field
   */
  template< typename FaceFieldT >
  void setup_stress( const Expr::Tag& stressTag,
                     const Expr::Tag& viscTag,
                     const Expr::Tag& vel1Tag,
                     const Expr::Tag& vel2Tag,
                     const Expr::Tag& dilTag,
                     Expr::ExpressionFactory& factory )
  {
    typedef typename StressHelper<FaceFieldT>::Vel1T Vel1T;  // type of velocity component 1
    typedef typename StressHelper<FaceFieldT>::Vel2T Vel2T;  // type of velocity component 2
    typedef SVolField                                ViscT;  // type of velocity component 3

    typedef typename Stress< FaceFieldT, Vel1T, Vel2T, ViscT >::Builder StressT;

    factory.register_expression( stressTag,
                                 new StressT( viscTag, vel1Tag, vel2Tag, dilTag ) );
  }

  //==================================================================

  template< typename FluxT, typename AdvelT >
  void
  setup_convective_flux( const Expr::Tag& fluxTag,
                         const Expr::Tag& momTag,
                         const Expr::Tag& advelTag, Expr::ExpressionFactory& factory )
  {
    typedef typename SpatialOps::structured::VolType<FluxT>::VolField  MomT;
    typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, MomT,   FluxT >::type  MomInterpOp;
    typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, AdvelT, FluxT >::type  AdvelInterpOp;

    typedef typename ConvectiveFlux< MomInterpOp, AdvelInterpOp >::Builder ConvFlux;
    factory.register_expression( fluxTag, new ConvFlux( momTag, advelTag ) );
  }

  //==================================================================

  void set_vel_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& velTags )
  {
    string u, v, w;
    params->require( "X-Velocity", u );
    const Uintah::ProblemSpecP pv = params->get( "Y-Velocity", v );
    const Uintah::ProblemSpecP pw = params->get( "Z-Velocity", w );
    velTags.push_back( Expr::Tag(u,Expr::STATE_NONE) );
    if( pv ) velTags.push_back( Expr::Tag(v,Expr::STATE_NONE) );
    if( pw ) velTags.push_back( Expr::Tag(w,Expr::STATE_NONE) );

  }

  //==================================================================

  void set_mom_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& tags )
  {
    string u, v, w;
    params->require( "X-Momentum", u );
    const Uintah::ProblemSpecP pv = params->get( "Y-Momentum", v );
    const Uintah::ProblemSpecP pw = params->get( "Z-Momentum", w );
    tags.push_back( Expr::Tag(u,Expr::STATE_N) );
    if( pv ) tags.push_back( Expr::Tag(v,Expr::STATE_N) );
    if( pw ) tags.push_back( Expr::Tag(w,Expr::STATE_N) );
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  get_mom_rhs_id( Expr::ExpressionFactory& factory,
                  const Expr::Tag& thisVelTag,
                  const Expr::Tag& thisMomTag,
                  const Expr::Tag& dilTag,
                  const Uintah::ProblemSpecP params )
  {
    const Expr::Tag viscTag = parse_nametag( params );
    Expr::TagList velTags, momTags;
    set_vel_tags( params, velTags  );
    set_mom_tags( params, momTags );

    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;

    //___________________________________
    // diffusive flux (stress components)

    const Expr::Tag taux( "tau_" + get_mom_dir_name<FieldT>() + "x", Expr::STATE_NONE );
    const Expr::Tag tauy( "tau_" + get_mom_dir_name<FieldT>() + "y", Expr::STATE_NONE );
    const Expr::Tag tauz( "tau_" + get_mom_dir_name<FieldT>() + "z", Expr::STATE_NONE );

    setup_stress< XFace >( taux, viscTag, thisVelTag, velTags[0], dilTag, factory );
    setup_stress< YFace >( tauy, viscTag, thisVelTag, velTags[1], dilTag, factory );
    setup_stress< ZFace >( tauz, viscTag, thisVelTag, velTags[2], dilTag, factory );


    //__________________
    // convective fluxes
    const Expr::Tag cfx( thisMomTag.name() + "_convFlux_x", Expr::STATE_NONE );
    const Expr::Tag cfy( thisMomTag.name() + "_convFlux_y", Expr::STATE_NONE );
    const Expr::Tag cfz( thisMomTag.name() + "_convFlux_z", Expr::STATE_NONE );

    setup_convective_flux< XFace, XVolField >( cfx, thisMomTag, velTags[0], factory );
    setup_convective_flux< YFace, YVolField >( cfy, thisMomTag, velTags[1], factory );
    setup_convective_flux< ZFace, ZVolField >( cfz, thisMomTag, velTags[2], factory );

    /*
      jcs still to do:
        - register dilatation expression (need only be done once, on scalar volume)
        - create expressions to calculate div of fluxes.
        - create expression for RHS of momentum, taking p, body force, and div of fluxes.
        - create pressure expression.
        - create expression for body force
    */
  }

  //==================================================================

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  MomentumTransportEquation( const std::string velName,
                             const std::string momName,
                             Expr::ExpressionFactory& factory,
                             Uintah::ProblemSpecP params )
    : Expr::TransportEquation( momName,
                               get_mom_rhs_id<FieldT>( factory,
                                                       Expr::Tag(velName,Expr::STATE_NONE),
                                                       Expr::Tag(momName,Expr::STATE_N),
                                                       Expr::Tag("dilatation",Expr::STATE_NONE),
                                                       params ) )
  {
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  ~MomentumTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentumTransportEquation<FieldT>::
  setup_boundary_conditions( Expr::ExpressionFactory& factory )
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID
  MomentumTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_registry().get_id( Expr::Tag( this->solution_variable_name(),
                                                       Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class MomentumTransportEquation< XVolField >;
  template class MomentumTransportEquation< YVolField >;
  template class MomentumTransportEquation< ZVolField >;
  //==================================================================

} // namespace Wasatch
