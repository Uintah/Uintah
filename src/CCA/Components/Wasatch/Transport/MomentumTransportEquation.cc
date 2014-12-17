/*
 * The MIT License
 *
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
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquation.h>

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Solvers/HypreSolver.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/OldVariable.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

#include <CCA/Components/Wasatch/Expressions/Strain.h>
#include <CCA/Components/Wasatch/Expressions/Dilatation.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/TimeDerivative.h>
#include <CCA/Components/Wasatch/Expressions/MomentumPartialRHS.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorBase.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/DynamicSmagorinskyCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OutflowBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OpenBC.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/PressureSource.h>
#include <CCA/Components/Wasatch/Expressions/VelEst.h>
#include <CCA/Components/Wasatch/Expressions/DivmomStar.h>
#include <CCA/Components/Wasatch/Expressions/WeakConvectiveTerm.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/ContinuityResidual.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/KineticEnergy.h>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

using std::string;

namespace Wasatch{

  //==================================================================
  
  void register_turbulence_expressions (const TurbulenceParameters& turbParams,
                                        Expr::ExpressionFactory& factory,
                                        const Expr::TagList& velTags,
                                        const Expr::Tag densTag,
                                        const bool isConstDensity) {

    const TagNames& tagNames = TagNames::self();
    
    Expr::Tag strTsrMagTag      = Expr::Tag();
    Expr::Tag waleTsrMagTag     = Expr::Tag();
    Expr::Tag dynSmagCoefTag    = Expr::Tag();
    Expr::Tag vremanTsrMagTag   = Expr::Tag();
    const Expr::Tag turbViscTag = tagNames.turbulentviscosity;

    // Disallow users from using turbulence models in 1 or 2 dimensions
    if (!( velTags[0]!=Expr::Tag() && velTags[1]!=Expr::Tag() && velTags[2]!=Expr::Tag() )) {
      std::ostringstream msg;
      msg << "ERROR: You cannot use a turbulence model in one or two dimensions. Please revise your input file and make sure that you specify all three velocity/momentum components." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    // we have turbulence turned on. create an expression for the strain tensor magnitude. this is used by all eddy viscosity models
    switch (turbParams.turbModelName) {
        
        // ---------------------------------------------------------------------
      case TurbulenceParameters::SMAGORINSKY: {
        strTsrMagTag = tagNames.straintensormag;//( "StrainTensorMagnitude", Expr::STATE_NONE );
        if( !factory.have_entry( strTsrMagTag ) ){
          typedef StrainTensorSquare::Builder StrTsrMagT;
          factory.register_expression( scinew StrTsrMagT(strTsrMagTag,
                                                         tagNames.tauxx,tagNames.tauyx,tagNames.tauzx,
                                                         tagNames.tauyy,tagNames.tauzy,
                                                         tagNames.tauzz) );
        }
      }
        break;

        // ---------------------------------------------------------------------
      case TurbulenceParameters::VREMAN: {
        vremanTsrMagTag = tagNames.vremantensormag;
        if( !factory.have_entry( vremanTsrMagTag ) ){
          typedef VremanTensorMagnitude::Builder VremanTsrMagT;
          factory.register_expression( scinew VremanTsrMagT(vremanTsrMagTag, velTags ) );
        }
      }
        break;
        
        // ---------------------------------------------------------------------
      case TurbulenceParameters::WALE: {
        strTsrMagTag = tagNames.straintensormag;
        if( !factory.have_entry( strTsrMagTag ) ){
          typedef StrainTensorSquare::Builder StrTsrMagT;
          factory.register_expression( scinew StrTsrMagT(strTsrMagTag,
                                                         tagNames.tauxx,tagNames.tauyx,tagNames.tauzx,
                                                         tagNames.tauyy,tagNames.tauzy,
                                                         tagNames.tauzz) );
        }
        
        // if WALE model is turned on, then create an expression for the square velocity gradient tensor
        waleTsrMagTag = tagNames.waletensormag;
        if( !factory.have_entry( waleTsrMagTag ) ){
          typedef WaleTensorMagnitude::Builder waleStrTsrMagT;
          factory.register_expression( scinew waleStrTsrMagT(waleTsrMagTag, velTags ) );
        }
      }
        break;
        
        // ---------------------------------------------------------------------
      case TurbulenceParameters::DYNAMIC: {
        strTsrMagTag = tagNames.straintensormag;//( "StrainTensorMagnitude", Expr::STATE_NONE );

        Expr::TagList dynamicSmagTagList;
        dynamicSmagTagList.push_back( strTsrMagTag );
        dynamicSmagTagList.push_back( tagNames.dynamicsmagcoef);

        // if the DYNAMIC model is turned on, then create an expression for the dynamic smagorinsky coefficient
        dynSmagCoefTag = tagNames.dynamicsmagcoef;
        
        if( !factory.have_entry( dynSmagCoefTag )&&
            !factory.have_entry( strTsrMagTag )     ){
          typedef DynamicSmagorinskyCoefficient::Builder dynSmagConstT;
          factory.register_expression( scinew dynSmagConstT(dynamicSmagTagList,
                                                            velTags,
                                                            densTag,
                                                            isConstDensity) );
        }
        
      }
        break;

        // ---------------------------------------------------------------------
      default:
        break;
    }

    if( !factory.have_entry( turbViscTag ) ){
      // NOTE: You may need to cleave the turbulent viscosity from its parents
      // in case you run into problems with your simulation. The default behavior
      // of Wasatch is to extrapolate the turbulent viscosity at all patch boundaries.
      // If this extrapolation leads to problems you should consider excluding
      // the extrapolation and communicating the TurbulentViscosity instead.
      // To get rid of extrapolation, go to the TurbulentViscosity.cc expression
      // and comment out the "exOp_->apply_to_field(result)" line at the end
      // of the evaluate method.
      typedef TurbulentViscosity::Builder TurbViscT;
      factory.register_expression( scinew TurbViscT(turbViscTag, densTag, strTsrMagTag, waleTsrMagTag, vremanTsrMagTag, dynSmagCoefTag, turbParams ) );
//      const Expr::ExpressionID turbViscID = factory.register_expression( scinew TurbViscT(turbViscTag, densTag, strTsrMagTag, waleTsrMagTag, vremanTsrMagTag, dynSmagCoefTag, turbParams ) );
//      factory.cleave_from_parents(turbViscID);
    }
  }
  
  //==================================================================

  // note that the ordering of Vel1T and Vel2T are very important, and
  // must be consistent with the order of the velocity tags passed
  // into the Strain constructor.
  template< typename FaceT > struct StrainHelper;
  // nomenclature: XSurfXField - first letter is volume type: S, X, Y, Z
  // then it is followed by the field type
  template<> struct StrainHelper<SpatialOps::XSurfXField>
  {
    // XSurfXField - XVol-XSurf
    // tau_xx
    typedef XVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::XSurfYField>
  {
    // XSurfYField - XVol-YSurf
    // tau_yx (tau on a y face in the x direction)
    typedef XVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::XSurfZField>
  {
    // XSurfZField - XVol-ZSurf
    // tau_zx (tau on a z face in the x direction)
    typedef XVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StrainHelper<SpatialOps::YSurfXField>
  {
    // tau_xy
    typedef YVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::YSurfYField>
  {
    // tau_yy
    typedef YVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::YSurfZField>
  {
    // tau_zy
    typedef YVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  template<> struct StrainHelper<SpatialOps::ZSurfXField>
  {
    // tau_xz
    typedef ZVolField Vel1T;
    typedef XVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::ZSurfYField>
  {
    // tau_yz
    typedef ZVolField Vel1T;
    typedef YVolField Vel2T;
  };
  template<> struct StrainHelper<SpatialOps::ZSurfZField>
  {
    // tau_zz
    typedef ZVolField Vel1T;
    typedef ZVolField Vel2T;
  };

  //==================================================================

  Expr::Tag mom_tag( const std::string& momName, const bool old = false )
  {
    if (old) return Expr::Tag( momName, Expr::STATE_N );
    return Expr::Tag( momName, Expr::STATE_DYNAMIC );
  }

  //==================================================================

  Expr::Tag rhs_part_tag( const std::string& momName )
  {
    return Expr::Tag( momName + "_rhs_partial", Expr::STATE_NONE );
  }
  Expr::Tag rhs_part_tag( const Expr::Tag& momTag )
  {
    return rhs_part_tag( momTag.name() );
  }

  
  //==================================================================
  
  void set_vel_star_tags( Expr::TagList velTags,
                         Expr::TagList& velStarTags )
  {
    const TagNames& tagNames = TagNames::self();
    if( velTags[0] != Expr::Tag() ) velStarTags.push_back( tagNames.make_star(velTags[0]) );
    else         velStarTags.push_back( Expr::Tag() );
    if( velTags[1] != Expr::Tag() ) velStarTags.push_back( tagNames.make_star(velTags[1]) );
    else         velStarTags.push_back( Expr::Tag() );
    if( velTags[2] != Expr::Tag() ) velStarTags.push_back( tagNames.make_star(velTags[2]) );
    else         velStarTags.push_back( Expr::Tag() );
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
  
  void set_mom_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& momTags,
                     const bool old=false)
  {
    std::string xmomname, ymomname, zmomname;
    Uintah::ProblemSpecP doxmom,doymom,dozmom;
    doxmom = params->get( "X-Momentum", xmomname );
    doymom = params->get( "Y-Momentum", ymomname );
    dozmom = params->get( "Z-Momentum", zmomname );
    if( doxmom ) momTags.push_back( mom_tag(xmomname, old) );
    else         momTags.push_back( Expr::Tag() );
    if( doymom ) momTags.push_back( mom_tag(ymomname, old) );
    else         momTags.push_back( Expr::Tag() );
    if( dozmom ) momTags.push_back( mom_tag(zmomname, old) );
    else         momTags.push_back( Expr::Tag() );
  }
  
  //==================================================================
  
  template< typename FieldT >
  void
  set_tau_tags( const bool* doMom,
               const bool isViscous,
               Expr::TagList& tauTags )
  {
    const Direction stagLoc = get_staggered_location<FieldT>();
    std::string thisMomDirName;
    switch (stagLoc) {
      case XDIR:
        thisMomDirName = "x";
        break;
      case YDIR:
        thisMomDirName = "y";
        break;
      case ZDIR:
        thisMomDirName = "z";
        break;
      case NODIR:
      default:
        thisMomDirName = "";
        break;
    }

    if( doMom[0] && isViscous ) tauTags.push_back( Expr::Tag("tau_x" + thisMomDirName , Expr::STATE_NONE) );
    else                        tauTags.push_back( Expr::Tag() );
    if( doMom[1] && isViscous ) tauTags.push_back( Expr::Tag("tau_y" + thisMomDirName , Expr::STATE_NONE) );
    else                        tauTags.push_back( Expr::Tag() );
    if( doMom[2] && isViscous ) tauTags.push_back( Expr::Tag("tau_z" + thisMomDirName , Expr::STATE_NONE) );
    else                        tauTags.push_back( Expr::Tag() );
  }
  
  //==================================================================
  
  void set_convflux_tags( const bool* doMom,
                         Expr::TagList& cfTags,
                         const Expr::Tag thisMomTag )
  {
    const TagNames& tagNames = TagNames::self();
    if( doMom[0] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "x", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
    if( doMom[1] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "y", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
    if( doMom[2] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "z", Expr::STATE_NONE) );
    else         cfTags.push_back( Expr::Tag() );
  }

  //==================================================================

  /**
   *  \brief Register the Strain expression for the given face field
   */
  template< typename FaceFieldT >
  Expr::ExpressionID
  setup_strain( const Expr::Tag& strainTag,
                const Expr::Tag& vel1Tag,
                const Expr::Tag& vel2Tag,
                const Expr::Tag& dilTag,
                Expr::ExpressionFactory& factory )
  {
    typedef typename StrainHelper<FaceFieldT>::Vel1T Vel1T;  // type of velocity component 1
    typedef typename StrainHelper<FaceFieldT>::Vel2T Vel2T;  // type of velocity component 2
    typedef typename Strain< FaceFieldT, Vel1T, Vel2T >::Builder StrainT;
    return factory.register_expression( scinew StrainT( strainTag, vel1Tag, vel2Tag, dilTag ) );
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  register_strain_tensor( const bool* const doMom,
                         const bool isViscous,
                         const Expr::TagList& velTags,
                         Expr::TagList& tauTags,
                         const Expr::Tag& dilTag,
                         Expr::ExpressionFactory& factory )
  {
    const Direction stagLoc = get_staggered_location<FieldT>();

    typedef typename SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::ZFace ZFace;

    set_tau_tags<FieldT>( doMom, isViscous, tauTags );
    const Expr::Tag& tauxt = tauTags[0];
    const Expr::Tag& tauyt = tauTags[1];
    const Expr::Tag& tauzt = tauTags[2];
    
    Expr::ExpressionID normalStrainID;
    
    const int thisVelIdx = (stagLoc == XDIR) ? 0 : ( (stagLoc == YDIR) ? 1 : 2 );    
    const Expr::Tag& thisVelTag = velTags[thisVelIdx];

    // register necessary strain expression when the flow is viscous
    if ( isViscous ) {
      if( doMom[0] ){
        const Expr::ExpressionID strainID = setup_strain< XFace >( tauxt, thisVelTag, velTags[0], dilTag, factory );
        if( stagLoc == XDIR )  normalStrainID = strainID;
      }
      if( doMom[1] ){
        const Expr::ExpressionID strainID = setup_strain< YFace >( tauyt, thisVelTag, velTags[1], dilTag, factory );
        if( stagLoc == YDIR )  normalStrainID = strainID;
      }
      if( doMom[2] ){
        const Expr::ExpressionID strainID = setup_strain< ZFace >( tauzt, thisVelTag, velTags[2], dilTag, factory );
        if( stagLoc == ZDIR )  normalStrainID = strainID;
      }
      factory.cleave_from_children( normalStrainID );
      factory.cleave_from_parents( normalStrainID  );
    }
    return normalStrainID;
  }
  
  //==================================================================

  template< typename FluxT, typename AdvelT >
  Expr::ExpressionID
  setup_convective_flux( const Expr::Tag& fluxTag,
                         const Expr::Tag& momTag,
                         const Expr::Tag& advelTag,
                        ConvInterpMethods convInterpMethod,
                        const Expr::Tag& volFracTag,
                        Expr::ExpressionFactory& factory )
  {
    using namespace SpatialOps;
    if (convInterpMethod == CENTRAL) {
      typedef typename SpatialOps::VolType<FluxT>::VolField  MomT;
      typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, MomT,   FluxT >::type  MomInterpOp;
      typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, AdvelT, FluxT >::type  AdvelInterpOp;
      typedef typename ConvectiveFlux<MomInterpOp, AdvelInterpOp >::Builder ConvFlux;
      return factory.register_expression( scinew ConvFlux( fluxTag, momTag, advelTag ) );
    } else {
      typedef typename SpatialOps::VolType<FluxT>::VolField  MomT;
      typedef typename ConvectiveFluxLimiter<
      FluxLimiterInterpolant< MomT, FluxT >,
      UpwindInterpolant< MomT, FluxT >,
      typename OperatorTypeBuilder<Interpolant,MomT,FluxT>::type, // scalar interp type
      typename OperatorTypeBuilder<Interpolant,AdvelT,FluxT>::type  // velocity interp type
      >::Builder ConvFluxLim;
      return factory.register_expression( scinew ConvFluxLim( fluxTag, momTag, advelTag, convInterpMethod, volFracTag ) );
      
    }
  }

  //==================================================================
  
  template< typename FieldT >
  Expr::ExpressionID
  register_convective_fluxes( const bool* const doMom,
                              const Expr::TagList& velTags,
                              Expr::TagList& cfTags,
                              ConvInterpMethods convInterpMethod,
                              const Expr::Tag& momTag,
                              const Expr::Tag& volFracTag,
                              Expr::ExpressionFactory& factory )
  {
    set_convflux_tags( doMom, cfTags, momTag );
    const Expr::Tag cfxt = cfTags[0];
    const Expr::Tag cfyt = cfTags[1];
    const Expr::Tag cfzt = cfTags[2];

    typedef typename SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::ZFace ZFace;

    Expr::ExpressionID normalConvFluxID;
    Direction stagLoc = get_staggered_location<FieldT>();
    
    if( doMom[0] ){
      const Expr::ExpressionID id = setup_convective_flux< XFace, XVolField >( cfxt, momTag, velTags[0],convInterpMethod, volFracTag, factory );
      if( stagLoc == XDIR )  normalConvFluxID = id;
    }
    if( doMom[1] ){
      const Expr::ExpressionID id = setup_convective_flux< YFace, YVolField >( cfyt, momTag, velTags[1], convInterpMethod, volFracTag, factory );
      if( stagLoc == YDIR )  normalConvFluxID = id;
    }
    if( doMom[2] ){
      const Expr::ExpressionID id = setup_convective_flux< ZFace, ZVolField >( cfzt, momTag, velTags[2], convInterpMethod, volFracTag, factory );
      if( stagLoc == ZDIR )  normalConvFluxID = id;
    }
    // convective fluxes require ghost updates after they are calculated
    // jcs note that we need to set BCs on these quantities as well.
    factory.cleave_from_children( normalConvFluxID );
    factory.cleave_from_parents ( normalConvFluxID );
    return normalConvFluxID;
  }

  //==================================================================

  bool is_normal_to_boundary(const Direction stagLoc,
                             const Uintah::Patch::FaceType face)
  {
    bool isNormal = false;
    switch (stagLoc) {
      case XDIR:
      {
        if (face == Uintah::Patch::xminus || face == Uintah::Patch::xplus) {
          isNormal = true;
        }
      }
        break;
      case YDIR:
        if (face == Uintah::Patch::yminus || face == Uintah::Patch::yplus) {
          isNormal = true;
        }
        break;
      case ZDIR:
        if (face == Uintah::Patch::zminus || face == Uintah::Patch::zplus) {
          isNormal = true;
        }
        break;
      default:
        break;
    }
    return isNormal;
  }

  //==================================================================

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  MomentumTransportEquation( const std::string velName,
                             const std::string momName,
                             const Expr::Tag densTag,
                             const bool isConstDensity,
                             const Expr::Tag bodyForceTag,
                             const Expr::Tag srcTermTag,
                             GraphCategories& gc,
                             Uintah::ProblemSpecP params,
                             TurbulenceParameters turbulenceParams,
                             VarDenParameters varDenParams,
                             Uintah::SolverInterface& linSolver,
                             Uintah::SimulationStateP sharedState)
    : TransportEquation( gc,
                         momName,
                         params,
                         get_staggered_location<FieldT>(),
                         isConstDensity ),
      isViscous_       ( params->findBlock("Viscosity") ? true : false ),
      isTurbulent_     ( turbulenceParams.turbModelName != TurbulenceParameters::NOTURBULENCE ),
      thisVelTag_      ( Expr::Tag(velName, Expr::STATE_NONE) ),
      densityTag_      ( densTag                              ),
      normalStrainID_  ( Expr::ExpressionID::null_id()        ),
      normalConvFluxID_( Expr::ExpressionID::null_id()        ),
      pressureID_      ( Expr::ExpressionID::null_id()        )
  {
    solverParams_ = NULL;
    set_vel_tags( params, velTags_ );

    GraphHelper& graphHelper   = *(gc[ADVANCE_SOLUTION  ]);
    Expr::ExpressionFactory& factory = *(graphHelper.exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    
    std::string xmomname, ymomname, zmomname; // these are needed to construct fx, fy, and fz for pressure RHS
    bool doMom[3];
    doMom[0] = params->get( "X-Momentum", xmomname );
    doMom[1] = params->get( "Y-Momentum", ymomname );
    doMom[2] = params->get( "Z-Momentum", zmomname );

    const bool enablePressureSolve = !(params->findBlock("DisablePressureSolve"));
    
    //_____________
    // volume fractions for embedded boundaries Terms
    const EmbeddedGeometryHelper& embedGeom = EmbeddedGeometryHelper::self();
    thisVolFracTag_ = embedGeom.vol_frac_tag<FieldT>();
    
    //__________________
    // convective fluxes
    Expr::TagList cfTags; // these tags will be filled by register_convective_fluxes
    std::string convInterpMethod = "CENTRAL";
    if (params_->findBlock("ConvectiveInterpMethod")) {
      params_->findBlock("ConvectiveInterpMethod")->getAttribute("method",convInterpMethod);
    }
    
    normalConvFluxID_ = register_convective_fluxes<FieldT>(doMom, velTags_, cfTags, get_conv_interp_method(convInterpMethod), solnVarTag_, thisVolFracTag_, factory );

    //__________________
    // dilatation - needed by pressure source term and strain tensor
    const Expr::Tag dilTag = tagNames.dilatation;
    if( !factory.have_entry( dilTag ) ){
      typedef typename Dilatation<SVolField,XVolField,YVolField,ZVolField>::Builder Dilatation;
      // if dilatation expression has not been registered, then register it
      factory.register_expression( new Dilatation(dilTag, velTags_) );
    }

    //__________________
    // dilatation - needed by pressure source term and strain tensor
    const bool computeContinuityResidual = params->findBlock("ComputeMassResidual");
    if( computeContinuityResidual ){
      GraphHelper& postProcGH   = *(gc[POSTPROCESSING]);
      Expr::ExpressionFactory& postProcFactory = *(postProcGH.exprFactory);

      const Expr::Tag contTag = tagNames.continuityresidual;

      if( !postProcFactory.have_entry( contTag ) ){
        typedef typename ContinuityResidual<SVolField,XVolField,YVolField,ZVolField>::Builder ContResT;
        // if dilatation expression has not been registered, then register it
        Expr::TagList np1MomTags;
        if(doMom[0]) np1MomTags.push_back(Expr::Tag("x-mom",Expr::STATE_NP1));
        else         np1MomTags.push_back(Expr::Tag());
        if(doMom[1]) np1MomTags.push_back(Expr::Tag("y-mom",Expr::STATE_NP1));
        else         np1MomTags.push_back(Expr::Tag());
        if(doMom[2]) np1MomTags.push_back(Expr::Tag("z-mom",Expr::STATE_NP1));
        else         np1MomTags.push_back(Expr::Tag());

        Expr::Tag drhodtTag = Expr::Tag();
        if( !isConstDensity_ ){
          drhodtTag = tagNames.drhodtnp1;
          typedef Expr::PlaceHolder<SVolField>  FieldExpr;
          postProcFactory.register_expression( new typename FieldExpr::Builder(drhodtTag),true );
        }
        Expr::ExpressionID contID = postProcFactory.register_expression( new ContResT(contTag, drhodtTag, np1MomTags) );
        postProcGH.rootIDs.insert(contID);
      }
    }

    //___________________________________
    // diffusive flux (strain components)
    Expr::TagList tauTags;
    normalStrainID_ = register_strain_tensor<FieldT>(doMom, isViscous_, velTags_, tauTags, dilTag, factory);
    
    //--------------------------------------
    // TURBULENCE
    // check if we have a turbulence model turned on
    // check if the flow is viscous
    const Expr::Tag viscTag = (isViscous_) ? parse_nametag( params->findBlock("Viscosity")->findBlock("NameTag") ) : Expr::Tag();
    
    const bool enableTurbulenceModel = !(params->findBlock("DisableTurbulenceModel"));
    const Expr::Tag turbViscTag = tagNames.turbulentviscosity;
    if( isTurbulent_ && isViscous_ && enableTurbulenceModel ){
      register_turbulence_expressions(turbulenceParams, factory, velTags_, densTag, is_constant_density() );
      factory.attach_dependency_to_expression(turbViscTag, viscTag);
    }
    // END TURBULENCE
    //--------------------------------------

    //_________________________________________________________
    // partial rhs:
    // register expression to calculate the partial RHS (absent
    // pressure gradient) for use in the projection
    const Expr::ExpressionID momRHSPartID = factory.register_expression(
        new typename MomRHSPart<FieldT>::Builder( rhs_part_tag( solnVarTag_ ),
                                                  cfTags[0] , cfTags[1] , cfTags[2] , viscTag,
                                                  tauTags[0], tauTags[1], tauTags[2], densityTag_,
                                                  bodyForceTag, srcTermTag,
                                                  thisVolFracTag_) );
    factory.cleave_from_parents ( momRHSPartID );
    
    //__________________
    // Pressure source term
    
    if( !isConstDensity_ ){
      // calculating velocity at the next time step
      const Expr::Tag thisVelStarTag = tagNames.make_star(thisVelTag_);
      const Expr::Tag convTermWeak   = Expr::Tag( thisVelTag_.name() + "_weak_convective_term", Expr::STATE_NONE);
      if( !factory.have_entry( thisVelStarTag ) ){
        OldVariable& oldPressure = OldVariable::self();
        if( enablePressureSolve ) oldPressure.add_variable<SVolField>( ADVANCE_SOLUTION, pressure_tag() );
        const Expr::Tag oldPressureTag = Expr::Tag (pressure_tag().name() + "_old", Expr::STATE_NONE);
        convTermWeakID_ = factory.register_expression( new typename WeakConvectiveTerm<FieldT>::Builder( convTermWeak, thisVelTag_, velTags_));
        factory.cleave_from_parents ( convTermWeakID_ );
        factory.register_expression( new typename VelEst<FieldT>::Builder( thisVelStarTag, thisVelTag_, convTermWeak, tauTags, densTag, viscTag, oldPressureTag, tagNames.dt ));
      }
    }
    
    if( !factory.have_entry( tagNames.pressuresrc ) ){
      const Expr::Tag densStarTag  = tagNames.make_star(densTag, Expr::CARRY_FORWARD);
      const Expr::Tag dens2StarTag = tagNames.make_double_star(densTag, Expr::CARRY_FORWARD);
      Expr::TagList velStarTags = Expr::TagList();
      
      set_vel_star_tags( velTags_, velStarTags );
      set_mom_tags( params, momTags_ );
      set_mom_tags( params, oldMomTags_, true );
        // register the expression for pressure source term
        Expr::TagList psrcTagList;
        psrcTagList.push_back(tagNames.pressuresrc);
        if( !isConstDensity ){
          psrcTagList.push_back(tagNames.drhodt     );
          psrcTagList.push_back(tagNames.vardenalpha);
          psrcTagList.push_back(tagNames.vardenbeta );
//          psrcTagList.push_back(tagNames.divmomstar );
          psrcTagList.push_back(tagNames.drhodtstar );
        }
      factory.register_expression( new typename DivmomStar::Builder( tagNames.divmomstar, velStarTags, densStarTag ) );
      factory.register_expression( new typename PressureSource::Builder( psrcTagList, momTags_, oldMomTags_, velTags_, velStarTags, isConstDensity, densTag, densStarTag, dens2StarTag, varDenParams, tagNames.divmomstar ) );
    }
    
    //__________________
    // calculate velocity at the current time step    
    factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( thisVelTag_, solnVarTag_, densityTag_, thisVolFracTag_ ) );
    
    //__________________
    // pressure
    if( enablePressureSolve ){
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
        
        // ALAS, we cannot throw an error here because setupFrequency is parsed using getWithDefault
        // which means it will be specified in the input.xml file that is generated by uintah...
        if (pressureParams->findBlock("Parameters")->findBlock("setupFrequency")) {
          std::ostringstream msg;
          msg << "WARNING: Wasatch does NOT allow specification of setupFrequency for the pressure solver. "
          << "The setupFrequency will be determined by Wasatch."
          << std::endl;
          std::cout << msg;
          //throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        solverParams_ = linSolver.readParameters( pressureParams, "", sharedState );
        solverParams_->setSolveOnExtraCells( false );
        solverParams_->setUseStencil4( true );
        solverParams_->setOutputFileName( "WASATCH" );
        
        // matrix update in hypre: If we have a moving geometry, then update every timestep.
        // Otherwise, no update is needed since the coefficient matrix is constant        
        solverParams_->setSetupFrequency(embedGeom.has_moving_geometry() ? 1 : 0);
        
        // if pressure expression has not be registered, then register it
        Expr::Tag fxt, fyt, fzt;
        if( doMom[0] )  fxt = rhs_part_tag( xmomname );
        if( doMom[1] )  fyt = rhs_part_tag( ymomname );
        if( doMom[2] )  fzt = rhs_part_tag( zmomname );

        Expr::TagList ptags;
        ptags.push_back( pressure_tag() );
        ptags.push_back( Expr::Tag( pressure_tag().name() + "_rhs", pressure_tag().context() ) );
        const Expr::ExpressionBuilder* const pbuilder = new typename Pressure::Builder( ptags, fxt, fyt, fzt,
                                                                                        tagNames.pressuresrc, tagNames.dt, embedGeom.vol_frac_tag<SVolField>(),
                                                                                        embedGeom.has_moving_geometry(), usePressureRefPoint, refPressureValue,
                                                                                        refPressureLocation, use3DLaplacian,
                                                                                        *solverParams_, linSolver);
        pressureID_ = factory.register_expression( pbuilder );
        factory.cleave_from_children( pressureID_ );
        factory.cleave_from_parents ( pressureID_ );
      }
      else {
        pressureID_ = factory.get_id( pressure_tag() );
      }
    }
    
    // Kinetic energy calculation, if necessary
    if ( params->findBlock("ComputeKE") ) {
      Uintah::ProblemSpecP keSpec = params->findBlock("ComputeKE");
      bool isTotalKE = true;
      keSpec->getAttribute("total", isTotalKE);
      if (isTotalKE) { // calculate total kinetic energy. then follow that with a reduction variable
        if (!factory.have_entry( TagNames::self().totalKineticEnergy )) {
          bool outputKE = true;
          keSpec->getAttribute("output", outputKE);
          
          // we need to create two expressions
          const Expr::Tag tkeTempTag("TotalKE_temp", Expr::STATE_NONE);
          factory.register_expression(scinew typename TotalKineticEnergy<XVolField,YVolField,ZVolField>::Builder( tkeTempTag,
                                                                                                                 velTags_[0],velTags_[1],velTags_[2] ),true);
          
          ReductionHelper::self().add_variable<SpatialOps::SingleValueField, ReductionSumOpT>(ADVANCE_SOLUTION, TagNames::self().totalKineticEnergy, tkeTempTag, outputKE, false);
        }
      } else if (!factory.have_entry( TagNames::self().kineticEnergy )) { // calculate local, pointwise kinetic energy
        const Expr::ExpressionID keID = factory.register_expression(
            scinew typename KineticEnergy<SVolField,XVolField,YVolField,ZVolField>::Builder(
                TagNames::self().kineticEnergy, velTags_[0],velTags_[1],velTags_[2] ), true);
        graphHelper.rootIDs.insert( keID );
      }
    }

    setup();
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  MomentumTransportEquation<FieldT>::
  ~MomentumTransportEquation()
  {
    delete solverParams_;
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID  MomentumTransportEquation<FieldT>::
  setup_rhs( FieldTagInfo&,
             const Expr::TagList& srcTags )
  {
    const bool enablePressureSolve = !(params_->findBlock("DisablePressureSolve"));

    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    Expr::Tag volFracTag = vNames.vol_frac_tag<FieldT>();

    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    typedef typename MomRHS<FieldT>::Builder RHS;
    return factory.register_expression( scinew RHS( rhsTag_,
                                                    (enablePressureSolve ? pressure_tag() : Expr::Tag()),
                                                    rhs_part_tag(solnVarTag_),
                                                    volFracTag ) );
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void MomentumTransportEquation<FieldT>::
  setup_boundary_conditions( BCHelper& bcHelper, GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory = *(graphCat[INITIALIZATION]->exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    //
    // Add dummy modifiers on all patches. This is used to inject new dpendencies across all patches.
    // Those new dependencies, result for example from complicated boundary conditions added in this
    // function. NOTE: whenever you want to add a new complex boundary condition, please use this
    // functionality to inject new dependencies across patches.
    //
    if( !isConstDensity_ ){
      const Expr::Tag rhoTagInit(densityTag_.name(), Expr::STATE_NONE);
      const Expr::Tag rhoStarTag = tagNames.make_star(densityTag_); // get the tagname of rho*
      bcHelper.create_dummy_dependency<SVolField>(rhoStarTag, tag_list(rhoTagInit), INITIALIZATION);
      const Expr::Tag rhoTagAdv(densityTag_.name(), Expr::CARRY_FORWARD);
      bcHelper.create_dummy_dependency<SVolField>(rhoStarTag, tag_list(rhoTagAdv), ADVANCE_SOLUTION);
    }

    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() )
    {
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;

      const bool isNormal = is_normal_to_boundary(this->staggered_location(), myBndSpec.face);
      
      // variable density: add bcopiers on all boundaries
      if( !isConstDensity_ ){
        // if we are solving a variable density problem, then set bcs on density estimate rho*
        const Expr::Tag rhoStarTag = tagNames.make_star(densityTag_); // get the tagname of rho*
        // check if this boundary applies a bc on the density
        if( myBndSpec.has_field(densityTag_.name()) ){
          // create a bc copier for the density estimate
          const Expr::Tag rhoStarBCTag( rhoStarTag.name() + "_" + bndName +"_bccopier", Expr::STATE_NONE);
          BndCondSpec rhoStarBCSpec = {rhoStarTag.name(), rhoStarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
          if( !initFactory.have_entry(rhoStarBCTag) ){
            const Expr::Tag rhoTag(densityTag_.name(), Expr::STATE_NONE);
            initFactory.register_expression ( new typename BCCopier<SVolField>::Builder(rhoStarBCTag, rhoTag) );
            bcHelper.add_boundary_condition(bndName, rhoStarBCSpec);
          }
          if( !advSlnFactory.have_entry(rhoStarBCTag) ){
            const Expr::Tag rhoTag(densityTag_.name(), Expr::CARRY_FORWARD);
            advSlnFactory.register_expression ( new typename BCCopier<SVolField>::Builder(rhoStarBCTag, rhoTag) );
            bcHelper.add_boundary_condition(bndName, rhoStarBCSpec);
          }
        }
      }
      
      switch (myBndSpec.type) {
        case WALL:
        {
          // first check if the user specified momentum boundary conditions at the wall
          if( myBndSpec.has_field(thisVelTag_.name()) || myBndSpec.has_field(solnVarName_) ||
              myBndSpec.has_field(rhs_name()) || myBndSpec.has_field(solnVarName_ + "_rhs_part") ){
            std::ostringstream msg;
            msg << "ERROR: You cannot specify any momentum-related boundary conditions at a stationary wall. "
            << "This error occured while trying to analyze boundary " << bndName
            << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          }

          BndCondSpec momBCSpec = {solution_variable_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          
          BndCondSpec velBCSpec = {thisVelTag_.name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, velBCSpec);          

          if( isNormal ){
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(mom_tag(solnVarName_))).name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
            BndCondSpec rhsFullBCSpec = {rhs_name(), "none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          
          // Variable Density:
          // apply 0 dirichlet on velocity estimates (u*) @ walls
          if( !isConstDensity_ ){
            const Expr::Tag thisVelStarTag = tagNames.make_star(thisVelTag_);
            // first check if the user specified momentum boundary conditions at the wall
            if( myBndSpec.has_field(thisVelStarTag.name()) ){
              std::ostringstream msg;
              msg << "ERROR: You cannot specify any momentum-related boundary conditions at a stationary wall. "
              << "This error occured while trying to analyze boundary " << bndName
              << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
            BndCondSpec velStarBCSpec = {thisVelStarTag.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, velStarBCSpec);
          }

          break;
        }
        case VELOCITY:
        {
          // tsaad: please keep the commented code below. This should process velocity BCs and infer momentum bcs from those
//          if (myBndSpec.find(thisVelTag_.name()) ) {
//            const BndCondSpec* velBCSpec = myBndSpec.find(thisVelTag_.name());
//            BndCondSpec momBCSpec = *velBCSpec;
//            momBCSpec.varName = solution_variable_name();
//            bcHelper.add_boundary_condition(bndName, momBCSpec);
//          }
          
          // tsaad: If this VELOCITY boundary does NOT have this velocity AND this momentum specified
          // then assume that they are zero and create boundary conditions for them accordingly
          if( !myBndSpec.has_field(thisVelTag_.name()) && !myBndSpec.has_field(solnVarName_) ) {
            BndCondSpec velBCSPec = {thisVelTag_.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, velBCSPec);
            BndCondSpec momBCSPec = {solnVarName_, "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, momBCSPec);
          }

          if( isNormal ){
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(mom_tag(solnVarName_))).name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
            BndCondSpec rhsFullBCSpec = {rhs_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          
          BndCondSpec pressureBCSpec = {pressure_tag().name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          
          // Variable Density:
          // For variable density flows, copy the velocity estimate values from the velocity
          // use a BCCopier to do this. This must be done AT EVERY boundary where there is a velocity
          // specification
          if( !isConstDensity_ ){
            // if the velocity specification is a constant - then use a simple constant value on the velocity estimates
            const Expr::Tag thisVelStarTag = tagNames.make_star(thisVelTag_);
            
            // first check if the user specified velocity estimate conditions
            if( myBndSpec.has_field(thisVelStarTag.name()) ){
              std::ostringstream msg;
              msg << "ERROR: You cannot specify velocity estimate boundary conditions at a Velocity boundary. Those are automatically inferred from the velocity specification. "
                  << "This error occurred while trying to analyze boundary " << bndName
                  << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }

            // check if this boundary has velocity specification on it. it better have!
            if( myBndSpec.has_field(thisVelTag_.name()) ){
              // grab the bc specification of the velocity on this boundary. Note that here we
              // should guarantee that the spec is found!
              const BndCondSpec* velBCSpec = myBndSpec.find(thisVelTag_.name());
              assert(velBCSpec);
              if( !velBCSpec->is_functor() ){
                // if the boundary condition is not a functor (i.e. a constant value), then simply
                // copy that value into a new BCSpec for the velocity estimate
                BndCondSpec velStarBCSpec = *velBCSpec; // copy the spec from the velocity
                velStarBCSpec.varName = thisVelStarTag.name(); // change the name to the starred velocity
                bcHelper.add_boundary_condition(bndName, velStarBCSpec);
              } else {
                // if it is a functor type, then create a BCCopier
                // create tagname for the bc copier
                const Expr::Tag velStarBCTag(thisVelStarTag.name() + "_" + bndName + "_copier", Expr::STATE_NONE);
                // create and register the BCCopier
                typedef typename BCCopier<FieldT>::Builder Copier;
                advSlnFactory.register_expression(scinew Copier(velStarBCTag,thisVelTag_));
                // specify the bc on velstart using the bc copier functor
                BndCondSpec velStarBCSpec = {thisVelStarTag.name(), velStarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
                // add it to the boundary conditions!
                bcHelper.add_boundary_condition(bndName, velStarBCSpec);
              }
            }
          }
          break;
        }
        case OUTFLOW:
        {
          if( isNormal ){
            // register outflow functor for this boundary. we'll register one functor per boundary
            const Expr::Tag outBCTag(bndName + "_outflow_bc", Expr::STATE_NONE);
            typedef typename OutflowBC<FieldT>::Builder Builder;
            //bcHelper.register_functor_expression( scinew Builder( outBCTag, thisVelTag_ ), ADVANCE_SOLUTION );
            advSlnFactory.register_expression( scinew Builder( outBCTag, solution_variable_tag() ) );
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(solution_variable_tag())).name(),outBCTag.name(), 0.0, DIRICHLET,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
          } else {
            BndCondSpec rhsFullBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
            
            // Variable Density:
            // For the tangential velocity estimates, apply simple Neumann conditions at the outflow
            if( !isConstDensity_ ){
              const Expr::Tag velStarTag = tagNames.make_star(thisVelTag_);
              BndCondSpec velStarBCSpec = {velStarTag.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
              bcHelper.add_boundary_condition(bndName, velStarBCSpec);
            }
          }
          
          // variable density:
          // for variable density outflows, force the value of divmomStar to zero in the interior cells of the outflow boundary
          if( !isConstDensity_ ){
            const Expr::Tag divmomstarBCTag( tagNames.divmomstar.name() + "_" + bndName + "_outflow_bc",Expr::STATE_NONE);
            if (!advSlnFactory.have_entry(divmomstarBCTag)) {
              advSlnFactory.register_expression ( new typename OneSidedDirichletBC<SVolField>::Builder(divmomstarBCTag, 0.0) );
              BndCondSpec divmomstarBCSpec = {tagNames.divmomstar.name(), divmomstarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
              bcHelper.add_boundary_condition(bndName, divmomstarBCSpec);
            }
          }

          // after the correction has been made, update the momentum and velocities in the extra cells using simple Neumann conditions
          BndCondSpec momBCSpec = {solnVarName_, "none", 0.0, NEUMANN, DOUBLE_TYPE};
          BndCondSpec velBCSpec = {thisVelTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          bcHelper.add_boundary_condition(bndName, velBCSpec);

          // Set the pressure to Dirichlet 0 (atmospheric conditions)
          BndCondSpec pressureBCSpec = {pressure_tag().name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          break;
        }
        case OPEN:
        {
          if( isNormal ){
            // register pressurebc functor for this boundary. we'll register one functor per boundary
            const Expr::Tag openBCTag(bndName + "_open_bc", Expr::STATE_NONE);
            typedef typename OpenBC<FieldT>::Builder Builder;
            advSlnFactory.register_expression( scinew Builder( openBCTag, solution_variable_tag() ) );
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(solution_variable_tag())).name(),openBCTag.name(), 0.0, DIRICHLET,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
            // variable density:
            // for variable density outflows, change the velocity estimate at the outflow boundary.
            // instead of using an outflow boundary condition similar to that applied on momentum (see above)
            // simply use the old velocity value at the outflow boundary, i.e. u*_at_outflow = un_at_outflow (old velocity)
            if( !isConstDensity_ ){
              const Expr::Tag velStarTag = tagNames.make_star(thisVelTag_);
              const Expr::Tag velStarBCTag( velStarTag.name() + bndName + "_open_bc",Expr::STATE_NONE);
              advSlnFactory.register_expression ( new typename BCCopier<FieldT>::Builder(velStarBCTag, thisVelTag_) );
              BndCondSpec velStarBCSpec = {velStarTag.name(), velStarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
              bcHelper.add_boundary_condition(bndName, velStarBCSpec);
            }
            
          } else {
            BndCondSpec rhsFullBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
            
            // Variable Density:
            // For the tangential velocity estimates, apply simple Neumann conditions at the outflow
            if( !isConstDensity_ ){
              const Expr::Tag velStarTag = tagNames.make_star(thisVelTag_);
              BndCondSpec velStarBCSpec = {velStarTag.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
              bcHelper.add_boundary_condition(bndName, velStarBCSpec);
            }            
          }

          // after the correction has been made, update the momentum and velocities in the extra cells using simple Neumann conditions
          BndCondSpec momBCSpec = {solnVarName_, "none", 0.0, NEUMANN, DOUBLE_TYPE};
          BndCondSpec velBCSpec = {thisVelTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          bcHelper.add_boundary_condition(bndName, velBCSpec);

          // Set the pressure to Dirichlet 0 (atmospheric conditions)
          BndCondSpec pressureBCSpec = {pressure_tag().name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          break;
        }
        case USER:
        {
          // pass through the list of user specified BCs that are relevant to this transport equation
          break;
        }
          
        default:
          break;
      } // SWITCH BOUNDARY TYPE
    } // BOUNDARY LOOP
  }
  
  //==================================================================
  
  template< typename FieldT >
  void MomentumTransportEquation<FieldT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     BCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
  
    // apply velocity boundary condition, if specified
    bcHelper.apply_boundary_condition<FieldT>(thisVelTag_, taskCat);
    // apply momentum boundary condition, if specified
    bcHelper.apply_boundary_condition<FieldT>( initial_condition_tag(), taskCat );

    if( !isConstDensity_ ){
      const TagNames& tagNames = TagNames::self();
      
      // set bcs for density
      const Expr::Tag densTag( densityTag_.name(), Expr::STATE_NONE );
      bcHelper.apply_boundary_condition<SVolField>(densTag, taskCat);
      
      // set bcs for density_*
      const Expr::Tag densStarTag = tagNames.make_star(densityTag_, Expr::STATE_NONE);
      bcHelper.apply_boundary_condition<SVolField>(densStarTag, taskCat);
    }
  }

  //==================================================================
  
  template< typename FieldT >
  void MomentumTransportEquation<FieldT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             BCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
      
    // set bcs for momentum
    bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
    // set bcs for velocity
    bcHelper.apply_boundary_condition<FieldT>( thisVelTag_, taskCat );
    // set bcs for partial rhs
    bcHelper.apply_boundary_condition<FieldT>( rhs_part_tag(mom_tag(solnVarName_)), taskCat, true);
    // set bcs for partial full rhs
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true);

    if( !isConstDensity_ ){
      const TagNames& tagNames = TagNames::self();
      // set bcs for starred velocities. Those are now set through setup_boundary_conditions
      const Expr::Tag velStarTag = tagNames.make_star(thisVelTag_);
      bcHelper.apply_boundary_condition<FieldT>(velStarTag, taskCat);

      // set bcs for density
      const Expr::Tag densTag( densityTag_.name(), Expr::CARRY_FORWARD );
      bcHelper.apply_boundary_condition<SVolField>(densTag, taskCat);
      
      // set bcs for density_*
      bcHelper.apply_boundary_condition<SVolField>( tagNames.make_star(densityTag_,Expr::CARRY_FORWARD), taskCat );

      // set bcs for divmom*
      bcHelper.apply_boundary_condition<SVolField>(tagNames.divmomstar, taskCat);
    }
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  MomentumTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    // register an initial condition for da pressure
    if( !icFactory.have_entry( pressure_tag() ) ) {
      icFactory.register_expression( new typename Expr::ConstantExpr<SVolField>::Builder(pressure_tag(), 0.0 ) );
    }
    
    if( icFactory.have_entry( thisVelTag_ ) ) {
      typedef typename InterpolateExpression<SVolField, FieldT>::Builder Builder;
      Expr::Tag interpolatedDensityTag(densityTag_.name() +"_interp_" + this->dir_name(), Expr::STATE_NONE);
      icFactory.register_expression(scinew Builder(interpolatedDensityTag, Expr::Tag(densityTag_.name(),Expr::STATE_NONE)));
      
      // register expression to calculate the momentum initial condition from the initial conditions on
      // velocity and density in the cases that we are initializing velocity in the input file
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList( tag_list( thisVelTag_, interpolatedDensityTag ) );
      icFactory.register_expression( new typename ExprAlgbr::Builder( initial_condition_tag(),
                                                                      theTagList,
                                                                      ExprAlgbr::PRODUCT ) );
    }

    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& geomHelper = EmbeddedGeometryHelper::self();
    if( geomHelper.has_embedded_geometry() ){
      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList( tag_list( thisVolFracTag_ ) );
      Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE );
      icFactory.register_expression( new typename ExprAlgbr::Builder( modifierTag,
                                                                      theTagList,
                                                                      ExprAlgbr::PRODUCT,
                                                                      true ) );
      icFactory.attach_modifier_expression( modifierTag, initial_condition_tag() );
    }
    return icFactory.get_id( initial_condition_tag() );
  }

  //==================================================================
  
  //==================================================================  
  // Explicit template instantiation
  template class MomentumTransportEquation< XVolField >;
  template class MomentumTransportEquation< YVolField >;
  template class MomentumTransportEquation< ZVolField >;
  //==================================================================

} // namespace Wasatch
