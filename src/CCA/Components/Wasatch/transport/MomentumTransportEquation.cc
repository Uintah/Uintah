/*
 * Copyright (c) 2012 The University of Utah
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
#include "MomentumTransportEquation.h"

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/StringNames.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Expressions/MomentumPartialRHS.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/Stress.h>
#include <CCA/Components/Wasatch/Expressions/Dilatation.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>

using std::string;

namespace Wasatch{

  //==================================================================
  
  void register_turbulence_expressions (const TurbulenceParameters& turbParams,
                                        Expr::ExpressionFactory& factory,
                                        const Expr::TagList& velTags,
                                        const Expr::Tag densTag) {

    Expr::Tag strTsrMagTag  = Expr::Tag();
    Expr::Tag waleTsrMagTag  = Expr::Tag();
    Expr::Tag dynSmagConstTag = Expr::Tag();
    Expr::Tag vremanTsrMagTag  = Expr::Tag();    
    const Expr::Tag turbViscTag = turbulent_viscosity_tag();
    
    // we got turbulence turned on. create an expression for the strain tensor magnitude. this is used by all eddy viscosity models
    
    switch (turbParams.turbulenceModelName) {
      case SMAGORINSKY: {
        strTsrMagTag = straintensormagnitude_tag();//( "StrainTensorMagnitude", Expr::STATE_NONE );
        if( !factory.have_entry( strTsrMagTag ) ){
          typedef StrainTensorMagnitude::Builder StrTsrMagT;
          factory.register_expression( scinew StrTsrMagT(strTsrMagTag, velTags[0], velTags[1], velTags[2]) );
        }
      }
        break;
        
      case VREMAN: {
        // if WALE model is turned on, then create an expression for the square velocity gradient tensor
        vremanTsrMagTag = vreman_tensormagnitude_tag();
        if( !factory.have_entry( vremanTsrMagTag ) ){
          typedef VremanTensorMagnitude::Builder VremanTsrMagT;
          factory.register_expression( scinew VremanTsrMagT(vremanTsrMagTag, velTags[0], velTags[1], velTags[2] ) );
        }
      }
        break;
        
      case WALE: {
        
        strTsrMagTag = straintensormagnitude_tag();//( "StrainTensorMagnitude", Expr::STATE_NONE );
        if( !factory.have_entry( strTsrMagTag ) ){
          typedef StrainTensorMagnitude::Builder StrTsrMagT;
          factory.register_expression( scinew StrTsrMagT(strTsrMagTag, velTags[0], velTags[1], velTags[2]) );
        }
        
        // if WALE model is turned on, then create an expression for the square velocity gradient tensor
        waleTsrMagTag = wale_tensormagnitude_tag();
        if( !factory.have_entry( waleTsrMagTag ) ){
          typedef WaleTensorMagnitude::Builder waleStrTsrMagT;
          factory.register_expression( scinew waleStrTsrMagT(waleTsrMagTag, velTags[0], velTags[1], velTags[2] ) );
        }
      }
        break;
      case DYNAMIC: {
        // if DYNAMIC model is turned on, then create an expression for the dynamic smagorinsky expression
        dynSmagConstTag = Expr::Tag("DynamicSmagorinskyConstant", Expr::STATE_NONE);
        if( !factory.have_entry( dynSmagConstTag ) ){
        }
        
      }
        break;
      default:
        break;
    }   

    if( !factory.have_entry( turbViscTag ) ){
      typedef TurbulentViscosity::Builder TurbViscT;
      factory.register_expression( scinew TurbViscT(turbViscTag, densTag, strTsrMagTag, waleTsrMagTag, vremanTsrMagTag, turbParams ) );
    }
  }
  
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

  template< typename FieldT> struct NormalFaceSelector;

  template<> struct NormalFaceSelector<SpatialOps::structured::XVolField>
  {
  private:
    typedef SpatialOps::structured::XVolField FieldT;
  public:
    typedef SpatialOps::structured::FaceTypes<FieldT>::XFace NormalFace;
  };

  template<> struct NormalFaceSelector<SpatialOps::structured::YVolField>
  {
  private:
    typedef SpatialOps::structured::YVolField FieldT;
  public:
    typedef SpatialOps::structured::FaceTypes<FieldT>::YFace NormalFace;
  };

  template<> struct NormalFaceSelector<SpatialOps::structured::ZVolField>
  {
  private:
    typedef SpatialOps::structured::ZVolField FieldT;
  public:
    typedef SpatialOps::structured::FaceTypes<FieldT>::ZFace NormalFace;
  };

  //==================================================================

  Expr::Tag mom_tag( const std::string& momName )
  {
    return Expr::Tag( momName, Expr::STATE_N );
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

    return factory.register_expression( scinew StressT( stressTag, viscTag, vel1Tag, vel2Tag, dilTag ) );
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
    typedef typename ConvectiveFlux<MomInterpOp, AdvelInterpOp >::Builder ConvFlux;
    return factory.register_expression( scinew ConvFlux( fluxTag, momTag, advelTag ) );
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
    else                      tauTags.push_back( Expr::Tag() );
    if( doymom && isviscous ) tauTags.push_back( Expr::Tag("tau_y" + thisMomDirName , Expr::STATE_NONE) );
    else                      tauTags.push_back( Expr::Tag() );
    if( dozmom && isviscous ) tauTags.push_back( Expr::Tag("tau_z" + thisMomDirName , Expr::STATE_NONE) );
    else                      tauTags.push_back( Expr::Tag() );
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
  MomentumTransportEquation<FieldT>::
  get_mom_rhs_id( Expr::ExpressionFactory& factory,
                  const std::string velName,
                  const std::string momName,
                  Uintah::ProblemSpecP params,
                  Uintah::SolverInterface& linSolver )
  {
    const Expr::Tag momTag = mom_tag( momName );
    const Expr::Tag rhsFull( momTag.name() + "_rhs_full", Expr::STATE_NONE );
    return factory.register_expression( new typename MomRHS<FieldT>::Builder( rhsFull, pressure_tag(), rhs_part_tag(momTag) ) );
  }

  //==================================================================

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  MomentumTransportEquation( const std::string velName,
                             const std::string momName,
                             const Expr::Tag densTag,
                             const Expr::Tag bodyForceTag,
                             Expr::ExpressionFactory& factory,
                             Uintah::ProblemSpecP params,
                             TurbulenceParameters turbulenceParams,
                             const Expr::ExpressionID rhsID,
                             Uintah::SolverInterface& linSolver,
                             Uintah::SimulationStateP sharedState)
    : Wasatch::TransportEquation( momName,
                                  rhsID,
                                  get_staggered_location<FieldT>() ),
      isviscous_       ( params->findBlock("Viscosity") ? true : false ),
      isTurbulent_     ( turbulenceParams.turbulenceModelName != NONE ),
      thisVelTag_      ( Expr::Tag(velName, Expr::STATE_NONE) ),
      densityTag_      ( densTag                              ),
      normalStressID_  ( Expr::ExpressionID::null_id()        ),
      normalConvFluxID_( Expr::ExpressionID::null_id()        ),
      pressureID_      ( Expr::ExpressionID::null_id()        )
  {
    solverParams_ = NULL;
    set_vel_tags( params, velTags_ );

    const Expr::Tag thisMomTag = mom_tag( momName );

    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;
    //__________________
    // dilatation
    const Expr::Tag dilTag( "dilatation", Expr::STATE_NONE );
    if( !factory.have_entry( dilTag ) ){
      typedef typename Dilatation<SVolField,XVolField,YVolField,ZVolField>::Builder Dilatation;
      // if dilatation expression has not been registered, then register it
      factory.register_expression( new Dilatation(dilTag, velTags_[0],velTags_[1],velTags_[2]) );
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
    //
    const Expr::Tag viscTag = (isviscous_) ? parse_nametag( params->findBlock("Viscosity")->findBlock("NameTag") ) : Expr::Tag();
    //--------------------------------------
    // TURBULENCE
    // check if we have a turbulence model turned on
    bool enableTurbulenceModel = !(params->findBlock("DisableTurbulenceModel"));
    const Expr::Tag turbViscTag = turbulent_viscosity_tag();
    if ( isTurbulent_ && isviscous_ && enableTurbulenceModel ) {
      register_turbulence_expressions(turbulenceParams, factory, velTags_, densTag);      
      factory.attach_dependency_to_expression(turbViscTag, viscTag);
    }
    // END TURBULENCE
    //--------------------------------------
    // check if inviscid or not
    if ( isviscous_ ) {
      //const Expr::Tag viscosityTag = isTurbulent_ ? turbViscTag : viscTag;
      if( doxmom ){
        const Expr::ExpressionID stressID = setup_stress< XFace >( tauxt, viscTag, thisVelTag_, velTags_[0], dilTag, factory );
        if( stagLoc_ == XDIR )  normalStressID_ = stressID;
      }
      if( doymom ){
        const Expr::ExpressionID stressID = setup_stress< YFace >( tauyt, viscTag, thisVelTag_, velTags_[1], dilTag, factory );
        if( stagLoc_ == YDIR )  normalStressID_ = stressID;
      }
      if( dozmom ){
        const Expr::ExpressionID stressID = setup_stress< ZFace >( tauzt, viscTag, thisVelTag_, velTags_[2], dilTag, factory );
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
    // convective fluxes require ghost updates after they are calculated
    // jcs note that we need to set BCs on these quantities as well.
    factory.cleave_from_children( normalConvFluxID_ );
    factory.cleave_from_parents ( normalConvFluxID_ );    

    //_________________________________________________________
    // partial rhs:
    // register expression to calculate the partial RHS (absent
    // pressure gradient) for use in the projection
    const Expr::ExpressionID momRHSPartID = factory.register_expression(
        new typename MomRHSPart<FieldT>::Builder( rhs_part_tag( thisMomTag ),
                                                  cfxt, cfyt, cfzt,
                                                  tauxt, tauyt, tauzt, densityTag_,
                                                  bodyForceTag ) );
    factory.cleave_from_parents ( momRHSPartID );
    //__________________

    // Here we should register an expression to get \nabla.(\rho*v)
    // I.C for \nabla.(\rho*v)???...

    //__________________
    // density time derivative
    const Expr::Tag d2rhodt2t;//( "density-acceleration", Expr::STATE_NONE); // for now this is empty

    factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( thisVelTag_, thisMomTag, densityTag_ ));

    //__________________
    // pressure
    if( !factory.have_entry( pressure_tag() ) ){
      Uintah::ProblemSpecP pressureParams = params->findBlock( "Pressure" );
      
      bool usePressureRefPoint = false;
      double refPressureValue = 0.0;
      SCIRun::IntVector refPressureLocation(0,0,0);
      if (pressureParams->findBlock("ReferencePressure")) {
        usePressureRefPoint = true;
        Uintah::ProblemSpecP refPressureParams = pressureParams->findBlock("ReferencePressure");
        refPressureParams->getAttribute("value", refPressureValue);
        refPressureParams->get("ReferenceCell", refPressureLocation);
      }
      
      bool use3DLaplacian = true;
      pressureParams->getWithDefault("Use3DLaplacian",use3DLaplacian, true);
      
      solverParams_ = linSolver.readParameters( pressureParams, "",
                                               sharedState );
      solverParams_->setSolveOnExtraCells( false );
      solverParams_->setUseStencil4( true );
      solverParams_->setOutputFileName( "WASATCH" );      
      
      // if pressure expression has not be registered, then register it
      Expr::Tag fxt, fyt, fzt;
      if( doxmom )  fxt = Expr::Tag( xmomname + "_rhs_partial", Expr::STATE_NONE );
      if( doymom )  fyt = Expr::Tag( ymomname + "_rhs_partial", Expr::STATE_NONE );
      if( dozmom )  fzt = Expr::Tag( zmomname + "_rhs_partial", Expr::STATE_NONE );

      const StringNames& sName = StringNames::self();
      const Expr::Tag timestepTag(sName.timestep,Expr::STATE_NONE);

      Expr::TagList ptags;
      ptags.push_back( pressure_tag() );
      ptags.push_back( Expr::Tag( pressure_tag().name() + "_rhs", pressure_tag().context() ) );
      const Expr::ExpressionBuilder* const pbuilder = new typename Pressure::Builder( ptags, fxt, fyt, fzt, dilTag,
                                                                                       d2rhodt2t, timestepTag, usePressureRefPoint, refPressureValue, refPressureLocation, use3DLaplacian,
                                                                                       *solverParams_, linSolver);
      proc0cout << "PRESSURE: " << std::endl
          << pbuilder->get_computed_field_tags() << std::endl;
      pressureID_ = factory.register_expression( pbuilder );
      //factory.cleave_from_children( pressureID_ );
      factory.cleave_from_parents ( pressureID_ );
    }
    else {
      pressureID_ = factory.get_id( pressure_tag() );
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  ~MomentumTransportEquation()
  {
    delete solverParams_;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentumTransportEquation<FieldT>::
  setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     const Uintah::PatchSet* const localPatches,
                                     const PatchInfoMap& patchInfoMap,
                                     const Uintah::MaterialSubset* const materials)
  {
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;

    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;
    typedef typename NormalFaceSelector<FieldT>::NormalFace NormalFace;

    // set initial bcs for momentum
    if (factory.have_entry(mom_tag(thisMomName_))) {
      process_boundary_conditions<FieldT>( Expr::Tag( this->solution_variable_name(),
                                                      Expr::STATE_N ),
                                           this->solution_variable_name(),
                                           this->staggered_location(),
                                           graphHelper,
                                           localPatches,
                                           patchInfoMap,
                                           materials );
    }

    // set bcs for velocity - cos we don't have a mechanism now to set them
    // on interpolated density field
    Expr::Tag velTag;
    switch (this->staggered_location()) {
      case XDIR:  velTag=velTags_[0];  break;
      case YDIR:  velTag=velTags_[1];  break;
      case ZDIR:  velTag=velTags_[2];  break;
      default:                         break;
    }
    if (factory.have_entry(velTag)) {
//      process_boundary_conditions<FieldT>( velTag,
//                                           velTag.name(),
//                                           this->staggered_location(),
//                                           graphHelper,
//                                           localPatches,
//                                           patchInfoMap,
//                                           materials );
    }
    // set bcs for pressure
    // We cannot set pressure BCs here using Wasatch's BC techniques because
    // we need to set the BCs AFTER the pressure solve. We had to create
    // a uintah task for that. See Pressure.cc
    
    // set bcs for partial rhs
    if (factory.have_entry(rhs_part_tag(mom_tag(thisMomName_)))) {
      process_boundary_conditions<FieldT>( rhs_part_tag(mom_tag(thisMomName_)),
                                           rhs_part_tag(mom_tag(thisMomName_)).name(),
                                           this->staggered_location(),
                                           graphHelper,
                                           localPatches,
                                           patchInfoMap,
                                           materials );
    }

  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentumTransportEquation<FieldT>::
  setup_boundary_conditions( const GraphHelper& graphHelper,
                                 const Uintah::PatchSet* const localPatches,
                                 const PatchInfoMap& patchInfoMap,
                                 const Uintah::MaterialSubset* const materials)
  {
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace ZFace;
    typedef typename NormalFaceSelector<FieldT>::NormalFace NormalFace;

    // set bcs for momentum
    process_boundary_conditions<FieldT>( Expr::Tag( this->solution_variable_name(),
                                                    Expr::STATE_N ),
                                         this->solution_variable_name(),
                                         this->staggered_location(),
                                         graphHelper,
                                         localPatches,
                                         patchInfoMap,
                                         materials );

    // set bcs for velocity - cos we don't have a mechanism now to set them
    // on interpolated density field
    Expr::Tag velTag;
    switch (this->staggered_location()) {
      case XDIR:  velTag=velTags_[0];  break;
      case YDIR:  velTag=velTags_[1];  break;
      case ZDIR:  velTag=velTags_[2];  break;
      default:                         break;
    }
    process_boundary_conditions<FieldT>( velTag,
                                         velTag.name(),
                                         this->staggered_location(),
                                         graphHelper,
                                         localPatches,
                                         patchInfoMap,
                                         materials );

    // set bcs for pressure
//    process_boundary_conditions<SVolField>( pressure_tag(),
//                                            "pressure",
//                                            NODIR,
//                                            graphHelper,
//                                            localPatches,
//                                            patchInfoMap,
//                                            materials );
    // set bcs for partial rhs
    process_boundary_conditions<FieldT>( rhs_part_tag(mom_tag(thisMomName_)),
                                         rhs_part_tag(mom_tag(thisMomName_)).name(),
                                         this->staggered_location(),
                                         graphHelper,
                                         localPatches,
                                         patchInfoMap,
                                         materials );
    // set bcs for partial full rhs
    process_boundary_conditions<FieldT>( Expr::Tag(thisMomName_ + "_rhs_full", Expr::STATE_NONE),
                                        thisMomName_ + "_rhs_full",
                                        this->staggered_location(),
                                        graphHelper,
                                        localPatches,
                                        patchInfoMap,
                                        materials );


//    // set bcs for density
//    const Expr::Tag densTag( "density", Expr::STATE_NONE );
//    process_boundary_conditions<SVolField>( densTag,
//                                           "density",
//                                           NODIR,
//                                           graphHelper,
//                                           localPatches,
//                                           patchInfoMap,
//                                           materials );
//    // set bcs for viscosity
//    const Expr::Tag viscTag( "viscosity", Expr::STATE_N );
//    const Direction viscDir = NODIR;
//    build_bcs( viscTag,
//              viscDir,
//              graphHelper,
//              localPatches,
//              patchInfoMap,
//              materials);

    // set bcs for normal stresses
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    if(isviscous_) {
      Expr::Tag normalStressTag = factory.get_label(normalStressID_);
      process_boundary_conditions<NormalFace>( normalStressTag,
                                  normalStressTag.name(),
                NODIR,
                graphHelper,
                localPatches,
                patchInfoMap,
                materials);
    }

    // set bcs for normal convective fluxes
    Expr::Tag normalConvFluxTag = factory.get_label(normalConvFluxID_);
    process_boundary_conditions<NormalFace>( normalConvFluxTag,
                                normalConvFluxTag.name(),
                                NODIR,
                                graphHelper,
                                localPatches,
                                patchInfoMap,
                                materials);

  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID
  MomentumTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {     
    if( icFactory.have_entry( thisVelTag_ ) ) {
      typedef typename InterpolateExpression<SVolField, FieldT>::Builder Builder;
      Expr::Tag interpolatedDensityTag(densityTag_.name() +"_interp_" + this->dir_name(), Expr::STATE_NONE);
      icFactory.register_expression(scinew Builder(interpolatedDensityTag, Expr::Tag(densityTag_.name(),Expr::STATE_NONE)));
      
      // register expression to calculate the momentum initial condition from the initial conditions on
      // velocity and density in the cases that we are initializing velocity in the input file
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      Expr::TagList theTagList;
      theTagList.push_back(thisVelTag_);
      theTagList.push_back(interpolatedDensityTag);
      return icFactory.register_expression( new typename ExprAlgbr::Builder( mom_tag(thisMomName_),
                                                                             theTagList,
                                                                             ExprAlgbr::PRODUCT ) );
    }
    
    return icFactory.get_id( Expr::Tag( this->solution_variable_name(), Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class MomentumTransportEquation< XVolField >;
  template class MomentumTransportEquation< YVolField >;
  template class MomentumTransportEquation< ZVolField >;
  //==================================================================

} // namespace Wasatch
