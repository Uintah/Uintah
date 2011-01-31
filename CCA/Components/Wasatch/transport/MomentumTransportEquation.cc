#include "MomentumTransportEquation.h"

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>
//#include <CCA/Ports/SimulationInterface.h>
//#include <CCA/Ports/Scheduler.h>
//#include <CCA/Ports/LoadBalancer.h>
//#include <CCA/Ports/Scheduler.h>
//#include <Core/Grid/AMR.h>
//#include <Core/Grid/Task.h>
//#include <Core/Grid/SimulationState.h>
//#include <Core/Grid/Variables/CellIterator.h>
//#include <Core/Grid/Variables/VarTypes.h>
//#include <Core/Exceptions/ConvergenceFailure.h>
//#include <Core/Exceptions/ProblemSetupException.h> 
//#include <Core/Exceptions/InvalidValue.h>
//#include <Core/Parallel/ProcessorGroup.h>
//#include <Core/Parallel/UintahParallelPort.h>
//#include <Core/Util/DebugStream.h>
//#include <Core/Math/MiscMath.h>
//#include <Core/Exceptions/InternalError.h>
//#include <Core/Parallel/Parallel.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/MomentumPartialRHS.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/Stress.h>
#include <CCA/Components/Wasatch/Expressions/Dilatation.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
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

    factory.register_expression( stressTag, scinew StressT( viscTag, vel1Tag, vel2Tag, dilTag ) );
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
    factory.register_expression( fluxTag, scinew ConvFlux( momTag, advelTag ) );
  }

  //==================================================================
  
  void
  setup_pressure( const Expr::Tag& momTag,
                  Expr::ExpressionFactory& factory )
  {
    
  }
  
  //==================================================================

  void set_vel_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& velTags )
  {
    std::string xvelname, yvelname, zvelname;
    Uintah::ProblemSpecP doxvel,doyvel,dozvel;
    doxvel = params->get( "X-Velocity", xvelname );
    doyvel = params->get( "Y-Velocity", yvelname );
    dozvel = params->get( "Z-Velocity", zvelname );
    if (doxvel) velTags.push_back( Expr::Tag(xvelname, Expr::STATE_N) );
    if (doyvel) velTags.push_back( Expr::Tag(yvelname, Expr::STATE_N) );
    if (dozvel) velTags.push_back( Expr::Tag(zvelname, Expr::STATE_N) );    
  }

  //==================================================================

  void set_mom_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& momTags )
  {
    std::string xmomname, ymomname, zmomname;
    Uintah::ProblemSpecP doxmom,doymom,dozmom;
    doxmom = params->get( "X-Momentum", xmomname );
    doymom = params->get( "Y-Momentum", ymomname );
    dozmom = params->get( "Z-Momentum", zmomname );    
    if (doxmom) momTags.push_back( Expr::Tag(xmomname, Expr::STATE_N) );
    if (doymom) momTags.push_back( Expr::Tag(ymomname, Expr::STATE_N) );
    if (dozmom) momTags.push_back( Expr::Tag(zmomname, Expr::STATE_N) );
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  get_mom_rhs_id( Expr::ExpressionFactory& factory,
                 const std::string velName,
                 const std::string momName,
                  Uintah::ProblemSpecP params )
  {   
    const Expr::Tag thisVelTag( velName, Expr::STATE_NONE );
    const Expr::Tag thisMomTag( momName, Expr::STATE_NONE );
    //
    const Expr::Tag dilTag( "dilatation", Expr::STATE_NONE );    
    const Expr::Tag viscTag = parse_nametag( params );
    //
    Expr::TagList velTags, momTags;    
    set_vel_tags( params, velTags  );
    set_mom_tags( params, momTags );

    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;

    //___________________________________
    // diffusive flux (stress components)

    const Expr::Tag tauxt( "tau_" + get_mom_dir_name<FieldT>() + "x", Expr::STATE_NONE );
    const Expr::Tag tauyt( "tau_" + get_mom_dir_name<FieldT>() + "y", Expr::STATE_NONE );
    const Expr::Tag tauzt( "tau_" + get_mom_dir_name<FieldT>() + "z", Expr::STATE_NONE );

    setup_stress< XFace >( tauxt, viscTag, thisVelTag, velTags[0], dilTag, factory );
    setup_stress< YFace >( tauyt, viscTag, thisVelTag, velTags[1], dilTag, factory );
    setup_stress< ZFace >( tauzt, viscTag, thisVelTag, velTags[2], dilTag, factory );


    //__________________
    // convective fluxes
    const Expr::Tag cfxt( thisMomTag.name() + "_convFlux_x", Expr::STATE_NONE );
    const Expr::Tag cfyt( thisMomTag.name() + "_convFlux_y", Expr::STATE_NONE );
    const Expr::Tag cfzt( thisMomTag.name() + "_convFlux_z", Expr::STATE_NONE );

    setup_convective_flux< XFace, XVolField >( cfxt, thisMomTag, velTags[0], factory );
    setup_convective_flux< YFace, YVolField >( cfyt, thisMomTag, velTags[1], factory );
    setup_convective_flux< ZFace, ZVolField >( cfzt, thisMomTag, velTags[2], factory );
    
    //__________________
    // dilatation
    if (!factory.get_registry().have_entry( dilTag )) {
      // if dilatation expression has not been registered, then register it
      factory.register_expression( dilTag, new typename Dilatation<SVolField,XVolField,YVolField,ZVolField>::Builder(velTags[0],velTags[1],velTags[2]));
    }

    /*
      jcs still to do:
        - DONE. register dilatation expression (need only be done once, on scalar volume)
        - DONE. create pressure expression (need only be done once)
        - create expression for body force
    */
    const Expr::Tag bodyForcet;  // for now, this is empty.

    //__________________
    // pressure
    const Expr::Tag d2rhodt2t; // for now this is empty
    const Expr::Tag pressuret( "pressure", Expr::STATE_NONE); // jcs need to fill in.
    Uintah::SolverInterface* solver;
    const Uintah::ProcessorGroup* myworld;    
    // THIS IS INCORRECT - PLACED HERE FOR CODE TO COMPILE. MUST GET PORT SOME
    // OTHER MANNER
    Uintah::UintahParallelPort* solverPort = Uintah::UintahParallelComponent( myworld ).getPort("solver");
    solver = dynamic_cast<Uintah::SolverInterface*>(solverPort);
    if(!solver) {
      throw Uintah::InternalError("Wasatch: couldn't get solver port", __FILE__, __LINE__);
    }
    Uintah::ProblemSpecP pressureParams = params->findBlock("Pressure");
    Uintah::SolverParameters* sparams = solver->readParameters(pressureParams, "");
    sparams->setSolveOnExtraCells(false);
    
    if (!factory.get_registry().have_entry(pressuret)) {
      // if pressure expression has not be registered, then register it
      std::string xmomname, ymomname, zmomname; // these are needed to construct fx, fy, and fz for pressure RHS
      Uintah::ProblemSpecP doxmom,doymom,dozmom;
      doxmom = params->get( "X-Momentum", xmomname );
      doymom = params->get( "Y-Momentum", ymomname );
      dozmom = params->get( "Z-Momentum", zmomname );
      Expr::Tag fxt, fyt, fzt;
      if (doxmom) fxt = Expr::Tag( xmomname + "_rhs_partial", Expr::STATE_NONE);
      if (doymom) fyt = Expr::Tag( ymomname + "_rhs_partial", Expr::STATE_NONE);
      if (dozmom) fzt = Expr::Tag( zmomname + "_rhs_partial", Expr::STATE_NONE);      
      factory.register_expression( pressuret, new typename Pressure::Builder( fxt, fyt, fzt,
                                                                             d2rhodt2t, *sparams, *solver));
    }
    //_________________________________________________________
    // register expression to calculate the partial RHS (absent
    // pressure gradient) for use in the projection
    const Expr::Tag rhsPart( thisMomTag.name() + "_rhs_partial", Expr::STATE_NONE );
    factory.register_expression( rhsPart, new typename MomRHSPart<FieldT>::Builder( cfyt, cfyt, cfzt,
                                                                                    tauxt, tauyt, cfxt,
                                                                                    bodyForcet) );
    
    //__________________________________________
    // The RHS (including the pressure gradient)
    const Expr::Tag rhsFull( thisMomTag.name() + "_rhs_full", Expr::STATE_NONE );
    const Expr::ExpressionID rhsID = factory.register_expression
      ( rhsFull, new typename MomRHS<FieldT>::Builder( pressuret, rhsPart ) );

    return rhsID;
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
                                                       velName,
                                                       momName,
                                                       params ) )
  {}

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
