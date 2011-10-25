#include "MomentumTransportEquation.h"

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>

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
  // nomenclature: XSurfXField - first letter is volume type: S, X, Y, Z
  // then it is followed by the field type
  template<> struct StressHelper<SpatialOps::structured::XSurfXField>
  {
    // XSurfXField - XVol-XSurf
    // tau_xx
    typedef XVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::XSurfYField>
  {
    // XSurfYField - XVol-YSurf
    // tau_yx (tau on a y face in the x direction)
    typedef XVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::XSurfZField>
  {
    // XSurfZField - XVol-ZSurf
    // tau_zx (tau on a z face in the x direction)
    typedef XVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StressHelper<SpatialOps::structured::YSurfXField>
  {
    // tau_xy
    typedef YVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::YSurfYField>
  {
    // tau_yy
    typedef YVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::YSurfZField>
  {
    // tau_zy
    typedef YVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StressHelper<SpatialOps::structured::ZSurfXField>
  {
    // tau_xz
    typedef ZVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::ZSurfYField>
  {
    // tau_yz
    typedef ZVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StressHelper<SpatialOps::structured::ZSurfZField>
  {
    // tau_zz
    typedef ZVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  //==================================================================

  Expr::Tag mom_tag( const std::string& momName )
  {
    return Expr::Tag( momName, Expr::STATE_N );
  }

  //==================================================================

  Expr::Tag pressure_tag()
  {
    return Expr::Tag( "pressure", Expr::STATE_NONE );
  }

  //==================================================================

  Expr::Tag rhs_part_tag( const Expr::Tag& momTag )
  {
    return Expr::Tag( momTag.name() + "_rhs_partial", Expr::STATE_NONE );
  }

  //==================================================================

  /**
   *  \brief Register the stress expression for the given face field
   */
  template< typename FaceFieldT >
  Expr::ExpressionID
  setup_stress( const Expr::Tag& stressTag,
                const Expr::Tag& viscTag,
                const Expr::Tag& vel1Tag,
                const Expr::Tag& vel2Tag,
                const Expr::Tag& dilTag,
                Expr::ExpressionFactory& factory )
  {
    typedef typename StressHelper<FaceFieldT>::Vel1T Vel1T;  // type of velocity component 1
    typedef typename StressHelper<FaceFieldT>::Vel2T Vel2T;  // type of velocity component 2
    typedef SVolField                                ViscT;  // type of viscosity

    typedef typename Stress< FaceFieldT, Vel1T, Vel2T, ViscT >::Builder StressT;

    return factory.register_expression( stressTag, scinew StressT( viscTag, vel1Tag, vel2Tag, dilTag ) );
  }

  //==================================================================

  template< typename FluxT, typename AdvelT >
  Expr::ExpressionID
  setup_convective_flux( const Expr::Tag& fluxTag,
                         const Expr::Tag& momTag,
                         const Expr::Tag& advelTag, Expr::ExpressionFactory& factory )
  {
    typedef typename SpatialOps::structured::VolType<FluxT>::VolField  MomT;
    typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, MomT,   FluxT >::type  MomInterpOp;
    typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, AdvelT, FluxT >::type  AdvelInterpOp;

    typedef typename ConvectiveFlux< MomInterpOp, AdvelInterpOp >::Builder ConvFlux;
    return factory.register_expression( fluxTag, scinew ConvFlux( momTag, advelTag ) );
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
    if( doxvel ) velTags.push_back( Expr::Tag(xvelname, Expr::STATE_NONE) );
    else         velTags.push_back( Expr::Tag() );
    if( doyvel ) velTags.push_back( Expr::Tag(yvelname, Expr::STATE_NONE) );
    else         velTags.push_back( Expr::Tag() );
    if( dozvel ) velTags.push_back( Expr::Tag(zvelname, Expr::STATE_NONE) );
    else         velTags.push_back( Expr::Tag() );
  }

  //==================================================================

  template< typename FieldT >
  void
  set_tau_tags( Uintah::ProblemSpecP params,
                    Expr::TagList& tauTags,
               const std::string thisMomDirName)
  {
    std::string xmomname, ymomname, zmomname;
    Uintah::ProblemSpecP doxmom,doymom,dozmom;
    Uintah::ProblemSpecP isviscous;
    isviscous = params->findBlock("Viscosity");
    doxmom = params->get( "X-Momentum", xmomname );
    doymom = params->get( "Y-Momentum", ymomname );
    dozmom = params->get( "Z-Momentum", zmomname );
    //
    if( doxmom && isviscous ) tauTags.push_back( Expr::Tag("tau_x" + thisMomDirName , Expr::STATE_NONE) );
    else         tauTags.push_back( Expr::Tag() );
    if( doymom && isviscous) tauTags.push_back( Expr::Tag("tau_y" + thisMomDirName , Expr::STATE_NONE) );
    else         tauTags.push_back( Expr::Tag() );
    if( dozmom && isviscous) tauTags.push_back( Expr::Tag("tau_z" + thisMomDirName , Expr::STATE_NONE) );
    else         tauTags.push_back( Expr::Tag() );
  }

  //==================================================================

  void set_convflux_tags( Uintah::ProblemSpecP params,
               Expr::TagList& cfTags,
               const Expr::Tag thisMomTag )
  {
    std::string xmomname, ymomname, zmomname;
    Uintah::ProblemSpecP doxmom,doymom,dozmom;
    doxmom = params->get( "X-Momentum", xmomname );
    doymom = params->get( "Y-Momentum", ymomname );
    dozmom = params->get( "Z-Momentum", zmomname );
    //
    if( doxmom ) cfTags.push_back( Expr::Tag(thisMomTag.name() + "_convFlux_x", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
    if( doymom ) cfTags.push_back( Expr::Tag(thisMomTag.name() + "_convFlux_y", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
    if( dozmom ) cfTags.push_back( Expr::Tag(thisMomTag.name() + "_convFlux_z", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  get_mom_rhs_id( Expr::ExpressionFactory& factory,
                  const std::string velName,
                  const std::string momName,
                  Uintah::ProblemSpecP params,
                  Uintah::SolverInterface& linSolver )
  {
    const Expr::Tag momTag = mom_tag( momName );
    const Expr::Tag rhsFull( momTag.name() + "_rhs_full", Expr::STATE_NONE );
    return factory.register_expression( rhsFull, new typename MomRHS<FieldT>::Builder( pressure_tag(), rhs_part_tag(momTag) ) );
  }

  //==================================================================

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  MomentumTransportEquation( const std::string velName,
                             const std::string momName,
                             const Expr::Tag densTag,
                             Expr::ExpressionFactory& factory,
                             Uintah::ProblemSpecP params,
                             Uintah::SolverInterface& linSolver)
    : Wasatch::TransportEquation( momName,
                                  get_mom_rhs_id<FieldT>( factory, velName, momName, params, linSolver ),
                                  get_staggered_location<FieldT>() ),
      normalStressID_  ( Expr::ExpressionID::null_id() ),
      normalConvFluxID_( Expr::ExpressionID::null_id() ),
      pressureID_      ( Expr::ExpressionID::null_id() ),
      isviscous_       ( params->findBlock("Viscosity") ? true : false )
  {
    set_vel_tags( params, velTags_ );

    const Expr::Tag thisVelTag( velName, Expr::STATE_NONE );
    const Expr::Tag thisMomTag = mom_tag( momName );

    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;
    //__________________
    // dilatation
    const Expr::Tag dilTag( "dilatation", Expr::STATE_N );    
    if( !factory.get_registry().have_entry( dilTag ) ){
      // if dilatation expression has not been registered, then register it
      const Expr::ExpressionID dilID = factory.register_expression( dilTag, new typename Dilatation<SVolField,XVolField,YVolField,ZVolField>::Builder(velTags_[0],velTags_[1],velTags_[2]));
    }    

    //___________________________________
    // diffusive flux (stress components)
    std::string xmomname, ymomname, zmomname; // these are needed to construct fx, fy, and fz for pressure RHS
    Uintah::ProblemSpecP doxmom,doymom,dozmom;
    doxmom = params->get( "X-Momentum", xmomname );
    doymom = params->get( "Y-Momentum", ymomname );
    dozmom = params->get( "Z-Momentum", zmomname );
    //
    if (stagLoc_ == XDIR) thisMomName_ = xmomname;
    if (stagLoc_ == YDIR) thisMomName_ = ymomname;
    if (stagLoc_ == ZDIR) thisMomName_ = zmomname;
    Expr::TagList tauTags;
    const std::string thisMomDirName = this->dir_name();
    set_tau_tags<FieldT>( params, tauTags, thisMomDirName );
    const Expr::Tag tauxt = tauTags[0];
    const Expr::Tag tauyt = tauTags[1];
    const Expr::Tag tauzt = tauTags[2];

     // check if inviscid or not
    if (isviscous_) {
      const Expr::Tag viscTag = parse_nametag( params->findBlock("Viscosity")->findBlock("NameTag") );
      if( doxmom ){
        const Expr::ExpressionID stressID = setup_stress< XFace >( tauxt, viscTag, thisVelTag, velTags_[0], dilTag, factory );
        if( stagLoc_ == XDIR )  normalStressID_ = stressID;
      }
      if( doymom ){
        const Expr::ExpressionID stressID = setup_stress< YFace >( tauyt, viscTag, thisVelTag, velTags_[1], dilTag, factory );
        if( stagLoc_ == YDIR )  normalStressID_ = stressID;
      }
      if( dozmom ){
        const Expr::ExpressionID stressID = setup_stress< ZFace >( tauzt, viscTag, thisVelTag, velTags_[2], dilTag, factory );
        if( stagLoc_ == ZDIR )  normalStressID_ = stressID;
      }
      factory.cleave_from_children( normalStressID_   );
      factory.cleave_from_parents( normalStressID_   );
    }

    //__________________
    // convective fluxes
    Expr::TagList cfTags;
    set_convflux_tags( params, cfTags, thisMomTag );
    const Expr::Tag cfxt = cfTags[0];
    const Expr::Tag cfyt = cfTags[1];
    const Expr::Tag cfzt = cfTags[2];

    if( doxmom ){
      const Expr::ExpressionID id = setup_convective_flux< XFace, XVolField >( cfxt, thisMomTag, velTags_[0], factory );
      if( stagLoc_ == XDIR )  normalConvFluxID_ = id;
    }
    if( doymom ){
      const Expr::ExpressionID id = setup_convective_flux< YFace, YVolField >( cfyt, thisMomTag, velTags_[1], factory );
      if( stagLoc_ == YDIR )  normalConvFluxID_ = id;
    }
    if( dozmom ){
      const Expr::ExpressionID id = setup_convective_flux< ZFace, ZVolField >( cfzt, thisMomTag, velTags_[2], factory );
      if( stagLoc_ == ZDIR )  normalConvFluxID_ = id;
    }

    /*
      jcs still to do:
      - create expression for body force
    */
    const Expr::Tag bodyForcet;//( "body-force", Expr::STATE_NONE);  // for now, this is empty.
    //_________________________________________________________
    // register expression to calculate the partial RHS (absent
    // pressure gradient) for use in the projection
    factory.register_expression( rhs_part_tag( thisMomTag ),
                                 new typename MomRHSPart<FieldT>::Builder( cfxt, cfyt, cfzt,
                                                                           tauxt, tauyt, tauzt,
                                                                           bodyForcet) );

    //__________________

    // Here we should register an expression to get \nabla.(\rho*v)
    // I.C for \nabla.(\rho*v)???...

    // density time derivative
    const Expr::Tag d2rhodt2t;//( "density-acceleration", Expr::STATE_NONE); // for now this is empty

    factory.register_expression( thisVelTag, new typename PrimVar<FieldT,SVolField>::Builder( thisMomTag, densTag));

    //__________________
    // pressure
    Uintah::ProblemSpecP pressureParams = params->findBlock( "Pressure" );
    Uintah::SolverParameters* sparams = linSolver.readParameters( pressureParams, "" );
    sparams->setSolveOnExtraCells( false );

    if( !factory.get_registry().have_entry( pressure_tag() ) ){
      // if pressure expression has not be registered, then register it
      Expr::Tag fxt, fyt, fzt;
      if( doxmom )  fxt = Expr::Tag( xmomname + "_rhs_partial", Expr::STATE_NONE );
      if( doymom )  fyt = Expr::Tag( ymomname + "_rhs_partial", Expr::STATE_NONE );
      if( dozmom )  fzt = Expr::Tag( zmomname + "_rhs_partial", Expr::STATE_NONE );

      Expr::TagList ptags;
      ptags.push_back( pressure_tag() );
      ptags.push_back( Expr::Tag( pressure_tag().name() + "_rhs", pressure_tag().context() ) );
      pressureID_ = factory.register_expression( ptags,
                                                 new typename Pressure::Builder( fxt, fyt, fzt,
                                                                                 d2rhodt2t, *sparams, linSolver) );
      factory.cleave_from_children( pressureID_   );
      factory.cleave_from_parents( pressureID_       );
    }
    else{
      pressureID_ = factory.get_registry().get_id( pressure_tag() );
    }

    //________________________________________________________________
    // Several expressions require ghost updates after they are calculated
    // jcs note that we need to set BCs on these quantities as well.
    factory.cleave_from_children( normalConvFluxID_   );
    factory.cleave_from_parents( normalConvFluxID_ );
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
  setup_boundary_conditions( const GraphHelper& graphHelper,
                                 const Uintah::PatchSet* const localPatches,
                                 const PatchInfoMap& patchInfoMap,
                                 const Uintah::MaterialSubset* const materials)
  {
//    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
//    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
//    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;
//
//
//    // build bcs for momentum
//    process_boundary_conditions( Expr::Tag( this->solution_variable_name(),
//                         Expr::STATE_N ),
//              this->staggered_location(),
//              graphHelper,
//              localPatches,
//              patchInfoMap,
//              materials );
//
//    // build bcs for velocity - cos we don't have a mechanism now to set them
//    // on interpolated density field
//    Expr::Tag velTag;
//    switch (this->staggered_location()) {
//      case XDIR:
//        velTag = velTags_[0];
//        break;
//      case YDIR:
//        velTag = velTags_[1];
//        break;
//      case ZDIR:
//        velTag = velTags_[2];
//        break;
//      default:
//        break;
//    }
//    process_boundary_conditions( velTag,
//              this->staggered_location(),
//              graphHelper,
//              localPatches,
//              patchInfoMap,
//              materials);
//
//
////    // set bcs for density
////    const Expr::Tag densTag( "density", Expr::STATE_N );
////    const Direction denDir = NODIR;
////    build_bcs( densTag,
////              denDir,
////              graphHelper,
////              localPatches,
////              patchInfoMap,
////              materials);
////
////    // set bcs for viscosity
////    const Expr::Tag viscTag( "viscosity", Expr::STATE_N );
////    const Direction viscDir = NODIR;
////    build_bcs( viscTag,
////              viscDir,
////              graphHelper,
////              localPatches,
////              patchInfoMap,
////              materials);
//
//    // build bcs for normal stresses and normal convective fluxes
//
//    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
//    if(isviscous_) {
//      Expr::Tag normalStressTag = factory.get_registry().get_label(normalStressID_);
//      process_boundary_conditions( normalStressTag,
//                this->staggered_location(),
//                graphHelper,
//                localPatches,
//                patchInfoMap,
//                materials,
//                false);
//    }
//
////      Expr::Tag normalConvFluxTag = factory.get_registry().get_label(normalConvFluxID_);
////      process_boundary_conditions( normalConvFluxTag,
////                                  this->staggered_location(),
////                                  graphHelper,
////                                  localPatches,
////                                  patchInfoMap,
////                                  materials,
////                                  true);
//
////    build_bcs( rhs_part_tag(mom_tag(thisMomName_)),
////            this->staggered_location(),
////            graphHelper,
////            localPatches,
////            patchInfoMap,
////            materials );
////    if( stagLoc_ == XDIR ) {
////	    build_bcs<XFace>( normalStressTag.name(),
////              this->staggered_location(),
////              graphHelper,
////              localPatches,
////              patchInfoMap,
////              materials );
////      build_bcs<XFace>( normalConvFluxTag.name(),
////                this->staggered_location(),
////                graphHelper,
////                localPatches,
////                patchInfoMap,
////                materials );
////    }
////    if (stagLoc_ == YDIR ) {
////      build_bcs<YFace>( normalStressTag.name(),
////                       this->staggered_location(),
////                       graphHelper,
////                       localPatches,
////                       patchInfoMap,
////                       materials );
////      build_bcs<YFace>( normalConvFluxTag.name(),
////                       this->staggered_location(),
////                       graphHelper,
////                       localPatches,
////                       patchInfoMap,
////                       materials );
////
////    }
////    if (stagLoc_ == ZDIR) {
////      build_bcs<ZFace>( normalStressTag.name(),
////                       this->staggered_location(),
////                       graphHelper,
////                       localPatches,
////                       patchInfoMap,
////                       materials );
////      build_bcs<ZFace>( normalConvFluxTag.name(),
////                       this->staggered_location(),
////                       graphHelper,
////                       localPatches,
////                       patchInfoMap,
////                       materials );
////
////    }
//
//
//    // build bcs for pressure
////    build_bcs( pressure_tag(),
////              NODIR,
////              graphHelper,
////              localPatches,
////              patchInfoMap,
////              materials );
  }

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
