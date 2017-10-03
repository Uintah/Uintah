/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Solvers/HypreSolver.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
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
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/ContinuityResidual.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/KineticEnergy.h>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

using std::string;

namespace WasatchCore{

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
        strTsrMagTag = tagNames.straintensormag;
        if( !factory.have_entry( strTsrMagTag ) ){
          typedef StrainTensorSquare::Builder StrTsrMagT;
          factory.register_expression( scinew StrTsrMagT(strTsrMagTag,
                                                         tagNames.strainxx,tagNames.strainyx,tagNames.strainzx,
                                                         tagNames.strainyy,tagNames.strainzy,
                                                         tagNames.strainzz) );
        }
      }
        break;

        // ---------------------------------------------------------------------
      case TurbulenceParameters::VREMAN: {
        vremanTsrMagTag = tagNames.vremantensormag;
        if( !factory.have_entry( vremanTsrMagTag ) ){
          typedef VremanTensorMagnitude<SVolField,XVolField,YVolField,ZVolField>::Builder VremanTsrMagT;
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
                                                         tagNames.strainxx,tagNames.strainyx,tagNames.strainzx,
                                                         tagNames.strainyy,tagNames.strainzy,
                                                         tagNames.strainzz) );
        }
        
        // if WALE model is turned on, then create an expression for the square velocity gradient tensor
        waleTsrMagTag = tagNames.waletensormag;
        if( !factory.have_entry( waleTsrMagTag ) ){
          typedef WaleTensorMagnitude<SVolField,XVolField,YVolField,ZVolField>::Builder waleStrTsrMagT;
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
          typedef DynamicSmagorinskyCoefficient<SVolField,XVolField,YVolField,ZVolField>::Builder dynSmagConstT;
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

  Expr::Tag mom_tag( const std::string& momName, const bool old)
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
                     const bool old)
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
  set_strain_tags( const Direction momComponent,
                   const bool* doMom,
                   const bool isViscous,
                   Expr::TagList& strainTags )
  {
    const TagNames& tagNames = TagNames::self();
    strainTags.clear();
    Expr::Tag xTag, yTag, zTag;
    switch( momComponent ){
      case XDIR : xTag=tagNames.strainxx; yTag=tagNames.strainyx; zTag=tagNames.strainzx; break;
      case YDIR : xTag=tagNames.strainxy; yTag=tagNames.strainyy; zTag=tagNames.strainzy; break;
      case ZDIR : xTag=tagNames.strainxz; yTag=tagNames.strainyz; zTag=tagNames.strainzz; break;
      case NODIR:
      default   : break;
    }
    Expr::Tag empty;
    if( doMom[0] && isViscous ) strainTags.push_back( xTag  );
    else                        strainTags.push_back( empty );
    if( doMom[1] && isViscous ) strainTags.push_back( yTag  );
    else                        strainTags.push_back( empty );
    if( doMom[2] && isViscous ) strainTags.push_back( zTag  );
    else                        strainTags.push_back( empty );
  }
  
  //==================================================================
  
  void set_convflux_tags( const bool* doMom,
                          Expr::TagList& cfTags,
                          const Expr::Tag thisMomTag )
  {
    const TagNames& tagNames = TagNames::self();
    if( doMom[0] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "x", Expr::STATE_NONE) );
    else           cfTags.push_back( Expr::Tag() );
    if( doMom[1] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "y", Expr::STATE_NONE) );
    else           cfTags.push_back( Expr::Tag() );
    if( doMom[2] ) cfTags.push_back( Expr::Tag(thisMomTag.name() + tagNames.convectiveflux + "z", Expr::STATE_NONE) );
    else           cfTags.push_back( Expr::Tag() );
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
                Expr::ExpressionFactory& factory )
  {
    typedef typename StrainHelper<FaceFieldT>::Vel1T Vel1T;  // type of velocity component 1
    typedef typename StrainHelper<FaceFieldT>::Vel2T Vel2T;  // type of velocity component 2
    typedef typename Strain< FaceFieldT, Vel1T, Vel2T >::Builder StrainT;
    return factory.register_expression( scinew StrainT( strainTag, vel1Tag, vel2Tag ) );
  }
  
  template< typename FaceFieldT, typename DirT >
  Expr::ExpressionID
  setup_collocated_strain( const Expr::Tag& strainTag,
               const Expr::Tag& vel1Tag,
               const Expr::Tag& vel2Tag,
               Expr::ExpressionFactory& factory )
  {
    typedef typename CollocatedStrain< FaceFieldT, DirT >::Builder StrainT;
    return factory.register_expression( scinew StrainT( strainTag, vel1Tag, vel2Tag ) );
  }


  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  register_strain_tensor( const Direction momComponent,
                          const bool* const doMom,
                          const bool isViscous,
                          const Expr::TagList& velTags,
                          Expr::TagList& strainTags,
                          const Expr::Tag& dilTag,
                          Expr::ExpressionFactory& factory,
                          Expr::Tag& normalStrainTag)
  {
    typedef typename SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::ZFace ZFace;

    set_strain_tags<FieldT>( momComponent, doMom, isViscous, strainTags );
    const Expr::Tag& strainXt = strainTags[0];
    const Expr::Tag& strainYt = strainTags[1];
    const Expr::Tag& strainZt = strainTags[2];
    
    Expr::ExpressionID normalStrainID;
    
    const int thisVelIdx = (momComponent == XDIR) ? 0 : ( (momComponent == YDIR) ? 1 : 2 );
    const Expr::Tag& thisVelTag = velTags[thisVelIdx];

    // register necessary strain expression when the flow is viscous
    if( isViscous ) {
      if( doMom[0] ){
        const Expr::ExpressionID strainID = setup_strain< XFace >( strainXt, thisVelTag, velTags[0], factory );
        if( momComponent == XDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainXt;
        }
      }
      if( doMom[1] ){
        const Expr::ExpressionID strainID = setup_strain< YFace >( strainYt, thisVelTag, velTags[1], factory );
        if( momComponent == YDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainYt;
        }
      }
      if( doMom[2] ){
        const Expr::ExpressionID strainID = setup_strain< ZFace >( strainZt, thisVelTag, velTags[2], factory );
        if( momComponent == ZDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainZt;
        }
      }
      factory.cleave_from_children( normalStrainID );
      factory.cleave_from_parents ( normalStrainID );
    }
    return normalStrainID;
  }

  //==================================================================
  
  template<>
  Expr::ExpressionID
  register_strain_tensor<SVolField>(const Direction momComponent,
                                    const bool* const doMom,
                         const bool isViscous,
                         const Expr::TagList& velTags,
                         Expr::TagList& strainTags,
                         const Expr::Tag& dilTag,
                         Expr::ExpressionFactory& factory,
                         Expr::Tag& normalStrainTag)
  {
    typedef SVolField FieldT;
    typedef SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef SpatialOps::FaceTypes<FieldT>::ZFace ZFace;
    
    set_strain_tags<FieldT>(momComponent, doMom, isViscous, strainTags );
    const Expr::Tag& strainXt = strainTags[0];
    const Expr::Tag& strainYt = strainTags[1];
    const Expr::Tag& strainZt = strainTags[2];
    
    Expr::ExpressionID normalStrainID;
    
    const int thisVelIdx = (momComponent == XDIR) ? 0 : ( (momComponent == YDIR) ? 1 : 2 );
    const Expr::Tag& thisVelTag = velTags[thisVelIdx];
    
    // register necessary strain expression when the flow is viscous
    if( isViscous ) {
      if( doMom[0] ){
        const Expr::ExpressionID strainID = setup_collocated_strain< XFace, SpatialOps::XDIR >( strainXt, thisVelTag, velTags[0], factory );
        if( momComponent == XDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainXt;
        }
      }
      if( doMom[1] ){
        const Expr::ExpressionID strainID = setup_collocated_strain< YFace, SpatialOps::YDIR >( strainYt, thisVelTag, velTags[1], factory );
        if( momComponent == YDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainYt;
        }
      }
      if( doMom[2] ){
        const Expr::ExpressionID strainID = setup_collocated_strain< ZFace, SpatialOps::ZDIR >( strainZt, thisVelTag, velTags[2], factory );
        if( momComponent == ZDIR ) {
          normalStrainID = strainID;
          normalStrainTag = strainZt;
        }
      }
      factory.cleave_from_children( normalStrainID );
      factory.cleave_from_parents ( normalStrainID );
    }
    return normalStrainID;

  }
  
  //==================================================================

  template< typename FluxT, typename AdvelT >
  Expr::ExpressionID
  setup_momentum_convective_flux( const Expr::Tag& fluxTag,
                         const Expr::Tag& momTag,
                         const Expr::Tag& advelTag,
                         ConvInterpMethods convInterpMethod,
                         const Expr::Tag& volFracTag,
                         Expr::ExpressionFactory& factory )
  {
    using namespace SpatialOps;
    if( convInterpMethod == CENTRAL ){
      typedef typename SpatialOps::VolType<FluxT>::VolField  MomT;
      typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, MomT,   FluxT >::type  MomInterpOp;
      typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, AdvelT, FluxT >::type  AdvelInterpOp;
      typedef typename ConvectiveFlux<MomInterpOp, AdvelInterpOp >::Builder ConvFlux;
      return factory.register_expression( scinew ConvFlux( fluxTag, momTag, advelTag ) );
    }
    else{
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
  register_momentum_convective_fluxes(const Direction momComponent,
                                      const bool* const doMom,
                                      const Expr::TagList& velTags,
                                      Expr::TagList& cfTags,
                                      ConvInterpMethods convInterpMethod,
                                      const Expr::Tag& momTag,
                                      const Expr::Tag& volFracTag,
                                      Expr::ExpressionFactory& factory,
                                      Expr::Tag& normalConvFluxTag )
  {
    set_convflux_tags( doMom, cfTags, momTag );
    const Expr::Tag& cfxt = cfTags[0];
    const Expr::Tag& cfyt = cfTags[1];
    const Expr::Tag& cfzt = cfTags[2];

    typedef typename SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<FieldT>::ZFace ZFace;

    Expr::ExpressionID normalConvFluxID;
    Direction stagLoc = get_staggered_location<FieldT>();
    
    if( doMom[0] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< XFace, XVolField >( cfxt, momTag, velTags[0],convInterpMethod, volFracTag, factory );
      if( stagLoc == XDIR )  {
        normalConvFluxID = id;
        normalConvFluxTag = cfxt;
      }
    }
    if( doMom[1] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< YFace, YVolField >( cfyt, momTag, velTags[1], convInterpMethod, volFracTag, factory );
      if( stagLoc == YDIR ){
        normalConvFluxID = id;
        normalConvFluxTag = cfyt;
      }
    }
    if( doMom[2] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< ZFace, ZVolField >( cfzt, momTag, velTags[2], convInterpMethod, volFracTag, factory );
      if( stagLoc == ZDIR ) {
        normalConvFluxID = id;
        normalConvFluxTag = cfzt;
      }
    }
    // convective fluxes require ghost updates after they are calculated
    // jcs note that we need to set BCs on these quantities as well.
    factory.cleave_from_children( normalConvFluxID );
    factory.cleave_from_parents ( normalConvFluxID );
    return normalConvFluxID;
  }

  //==================================================================
  
  template<>
  Expr::ExpressionID
  register_momentum_convective_fluxes<SVolField>( const Direction momComponent,
                                                  const bool* const doMom,
                                                  const Expr::TagList& velTags,
                                                  Expr::TagList& cfTags,
                                                  ConvInterpMethods convInterpMethod,
                                                  const Expr::Tag& momTag,
                                                  const Expr::Tag& volFracTag,
                                                  Expr::ExpressionFactory& factory,
                                                  Expr::Tag& normalConvFluxTag )
  {
    set_convflux_tags( doMom, cfTags, momTag );
    const Expr::Tag& cfxt = cfTags[0];
    const Expr::Tag& cfyt = cfTags[1];
    const Expr::Tag& cfzt = cfTags[2];
    
    typedef SVolField FieldT;
    typedef SpatialOps::FaceTypes<FieldT>::XFace XFace;
    typedef SpatialOps::FaceTypes<FieldT>::YFace YFace;
    typedef SpatialOps::FaceTypes<FieldT>::ZFace ZFace;
    
    Expr::ExpressionID normalConvFluxID;
    
    if( doMom[0] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< XFace, SVolField >( cfxt, momTag, velTags[0],convInterpMethod, volFracTag, factory );
      if( momComponent == XDIR )  {
        normalConvFluxID = id;
        normalConvFluxTag = cfxt;
      }
    }
    if( doMom[1] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< YFace, SVolField >( cfyt, momTag, velTags[1], convInterpMethod, volFracTag, factory );
      if( momComponent == YDIR ) {
        normalConvFluxID = id;
        normalConvFluxTag = cfyt;
      }
    }
    if( doMom[2] ){
      const Expr::ExpressionID id = setup_momentum_convective_flux< ZFace, SVolField >( cfzt, momTag, velTags[2], convInterpMethod, volFracTag, factory );
      if( momComponent == ZDIR ) {
        normalConvFluxID = id;
        normalConvFluxTag = cfzt;
      }
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
        if( face == Uintah::Patch::xminus || face == Uintah::Patch::xplus ){
          isNormal = true;
        }
        break;
      case YDIR:
        if( face == Uintah::Patch::yminus || face == Uintah::Patch::yplus ){
          isNormal = true;
        }
        break;
      case ZDIR:
        if( face == Uintah::Patch::zminus || face == Uintah::Patch::zplus ){
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
  MomentumTransportEquationBase<FieldT>::
  MomentumTransportEquationBase( const Direction momComponent,
                                 const std::string velName,
                                 const std::string momName,
                                 const Expr::Tag densTag,
                                 const bool isConstDensity,
                                 const Expr::Tag bodyForceTag,
                                 const Expr::Tag srcTermTag,
                                 GraphCategories& gc,
                                 Uintah::ProblemSpecP params,
                                 TurbulenceParameters turbulenceParams )
    : TransportEquation( gc,
                         momName,
                         get_staggered_location<FieldT>(),
                         isConstDensity ),
      momComponent_    ( momComponent),
      params_          ( params ),
      isViscous_       ( params->findBlock("Viscosity") ? true : false ),
      isTurbulent_     ( turbulenceParams.turbModelName != TurbulenceParameters::NOTURBULENCE ),
      thisVelTag_      ( Expr::Tag(velName, Expr::STATE_NONE) ),
      densityTag_      ( densTag                              ),
      pressureTag_     ( TagNames::self().pressure            ),
      normalStrainID_  ( Expr::ExpressionID::null_id()        ),
      normalConvFluxID_( Expr::ExpressionID::null_id()        ),
      pressureID_      ( Expr::ExpressionID::null_id()        )
  {
    set_vel_tags( params, this->velTags_ );
    
    GraphHelper& graphHelper   = *(gc[ADVANCE_SOLUTION  ]);
    Expr::ExpressionFactory& factory = *(graphHelper.exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    
    std::string xmomname, ymomname, zmomname; // these are needed to construct fx, fy, and fz for pressure RHS
    bool doMom[3];
    doMom[0] = params->get( "X-Momentum", xmomname );
    doMom[1] = params->get( "Y-Momentum", ymomname );
    doMom[2] = params->get( "Z-Momentum", zmomname );
    
    set_mom_tags( params, this->momTags_ );
    set_mom_tags( params, this->oldMomTags_, true );

    //_____________
    // volume fractions for embedded boundaries Terms
    const EmbeddedGeometryHelper& embedGeom = EmbeddedGeometryHelper::self();
    this->thisVolFracTag_ = embedGeom.vol_frac_tag<FieldT>();
    
    //__________________
    // convective fluxes
    Expr::TagList cfTags; // these tags will be filled by register_convective_fluxes
    std::string convInterpMethod = "CENTRAL";
    if( this->params_->findBlock("ConvectiveInterpMethod") ){
      this->params_->findBlock("ConvectiveInterpMethod")->getAttribute("method",convInterpMethod);
    }
    
    this->normalConvFluxID_ = register_momentum_convective_fluxes<FieldT>(momComponent, doMom, this->velTags_, cfTags, get_conv_interp_method(convInterpMethod), this->solnVarTag_, this->thisVolFracTag_, factory, this->normalConvFluxTag_ );
    
    //__________________
    // dilatation - needed by pressure source term and strain tensor
    const Expr::Tag dilTag = tagNames.dilatation;
    // if dilatation expression has not been registered, then register it
    if( !factory.have_entry( dilTag ) ){
      if (get_staggered_location<FieldT>() == NODIR ) { // collocated
        typedef typename Dilatation<SVolField,SVolField,SVolField,SVolField>::Builder Dilatation;
        factory.register_expression( new Dilatation(dilTag, this->velTags_) );
      } else { // staggered, const density & low-Mach projection
        typedef typename Dilatation<SVolField,XVolField,YVolField,ZVolField>::Builder Dilatation;
        factory.register_expression( new Dilatation(dilTag, this->velTags_) );
        const Expr::Tag divRhoUTag = tagNames.divrhou;
        factory.register_expression( new Dilatation(divRhoUTag, this->momTags_) );
      }
    }
    
    //___________________________________
    // diffusive flux (strain components)
    Expr::TagList strainTags;
    this->normalStrainID_ = register_strain_tensor<FieldT>(momComponent, doMom, this->isViscous_, this->velTags_, strainTags, dilTag, factory, this->normalStrainTag_);
    
    //--------------------------------------
    // TURBULENCE
    // check if we have a turbulence model turned on
    // check if the flow is viscous
    const Expr::Tag viscTag = (this->isViscous_) ? parse_nametag( params->findBlock("Viscosity")->findBlock("NameTag") ) : Expr::Tag();
    
    const bool enableTurbulenceModel = !(params->findBlock("DisableTurbulenceModel"));
    const Expr::Tag turbViscTag = tagNames.turbulentviscosity;
    if( this->isTurbulent_ && this->isViscous_ && enableTurbulenceModel ){
      register_turbulence_expressions(turbulenceParams, factory, this->velTags_, densTag, this->is_constant_density() );
      factory.attach_dependency_to_expression(turbViscTag, viscTag);
    }
    // END TURBULENCE
    //--------------------------------------
    
    //_________________________________________________________
    // partial rhs:
    // register expression to calculate the partial RHS (absent
    // pressure gradient) for use in the projection
    const Expr::ExpressionID momRHSPartID = factory.register_expression(
                                                                        new typename MomRHSPart<FieldT>::Builder( rhs_part_tag( this->solnVarTag_ ),
                                                                                                                 cfTags[0] , cfTags[1] , cfTags[2] ,
                                                                                                                 viscTag,
                                                                                                                 strainTags[0], strainTags[1], strainTags[2],
                                                                                                                 dilTag,
                                                                                                                 this->densityTag_, bodyForceTag, srcTermTag,
                                                                                                                 this->thisVolFracTag_) );
    factory.cleave_from_parents ( momRHSPartID );
    
    //__________________
    // continuity residual
    const bool computeContinuityResidual = params->findBlock("ComputeMassResidual");
    if( computeContinuityResidual ) {
      const Expr::Tag contTag = tagNames.continuityresidual;
      
      if( !factory.have_entry( contTag ) ){
        typedef typename ContinuityResidual<SVolField,XVolField,YVolField,ZVolField>::Builder ContResT;
        
        Expr::Tag drhodtTag = Expr::Tag();
        if( !this->is_constant_density() ){
          drhodtTag = tagNames.drhodtstar;
        }
        Expr::ExpressionID contID = factory.register_expression( new ContResT(contTag, drhodtTag, this->momTags_) );
        graphHelper.rootIDs.insert(contID);
      }
    }
    
    
    //__________________
    // calculate velocity at the current time step
    factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( this->thisVelTag_, this->solnVarTag_, this->densityTag_, this->thisVolFracTag_ ) );
    
    // Kinetic energy calculation, if necessary
    if( params->findBlock("ComputeKE") ){
      Uintah::ProblemSpecP keSpec = params->findBlock("ComputeKE");
      bool isTotalKE = true;
      keSpec->getAttribute("total", isTotalKE);
      if( isTotalKE ){ // calculate total kinetic energy. then follow that with a reduction variable
        if( !factory.have_entry( TagNames::self().totalKineticEnergy )){
          bool outputKE = true;
          keSpec->getAttribute("output", outputKE);
          
          // we need to create two expressions
          const Expr::Tag tkeTempTag("TotalKE_temp", Expr::STATE_NONE);
          factory.register_expression(scinew typename TotalKineticEnergy<XVolField,YVolField,ZVolField>::Builder( tkeTempTag,
                                                                                                                 this->velTags_[0],this->velTags_[1],this->velTags_[2] ),true);
          
          ReductionHelper::self().add_variable<SpatialOps::SingleValueField, ReductionSumOpT>(ADVANCE_SOLUTION, TagNames::self().totalKineticEnergy, tkeTempTag, outputKE, false);
        }
      }
      else if( !factory.have_entry( TagNames::self().kineticEnergy ) ){ // calculate local, pointwise kinetic energy
        const Expr::ExpressionID keID = factory.register_expression(
                                                                    scinew typename KineticEnergy<SVolField,XVolField,YVolField,ZVolField>::Builder(
                                                                                                                                                    TagNames::self().kineticEnergy, this->velTags_[0],this->velTags_[1],this->velTags_[2] ), true);
        graphHelper.rootIDs.insert( keID );
      }
    }
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  MomentumTransportEquationBase<FieldT>::
  ~MomentumTransportEquationBase()
  {}
  
  //==================================================================
  
  //==================================================================  
  // Explicit template instantiation
  template class MomentumTransportEquationBase< SVolField >;
  template class MomentumTransportEquationBase< XVolField >;
  template class MomentumTransportEquationBase< YVolField >;
  template class MomentumTransportEquationBase< ZVolField >;
  
#define REGISTER_STRAIN(VOL)                                    \
  template Expr::ExpressionID                                   \
  register_strain_tensor<VOL>( const Direction momComponent,    \
                               const bool* const doMom,         \
                               const bool isViscous,            \
                               const Expr::TagList& velTags,    \
                               Expr::TagList& strainTags,       \
                               const Expr::Tag& dilTag,         \
                               Expr::ExpressionFactory& factory, Expr::Tag& normalStrainTag );
  
  REGISTER_STRAIN( XVolField )
  REGISTER_STRAIN( YVolField )
  REGISTER_STRAIN( ZVolField )

#define REGISTER_CONVECTIVE_FLUXES(VOL)                                         \
  template Expr::ExpressionID                                                   \
  register_momentum_convective_fluxes<VOL> ( const Direction momComponent,      \
                                             const bool* const doMom,           \
                                             const Expr::TagList& velTags,      \
                                             Expr::TagList& cfTags,             \
                                             ConvInterpMethods convInterpMethod,\
                                             const Expr::Tag& momTag,           \
                                             const Expr::Tag& volFracTag,       \
                                             Expr::ExpressionFactory& factory, Expr::Tag& normalConvFluxTag );
  
  REGISTER_CONVECTIVE_FLUXES( XVolField )
  REGISTER_CONVECTIVE_FLUXES( YVolField )
  REGISTER_CONVECTIVE_FLUXES( ZVolField )

  //==================================================================

} // namespace WasatchCore
