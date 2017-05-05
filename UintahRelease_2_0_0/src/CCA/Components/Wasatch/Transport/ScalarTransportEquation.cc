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

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/structured/IndexTriplet.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Transport/ScalarTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSCoupling.h>
//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace WasatchCore{

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalarTransportEquation<FieldT>::
  ScalarTransportEquation( const std::string solnVarName,
                           Uintah::ProblemSpecP params,
                           GraphCategories& gc,
                           const Expr::Tag densityTag,
                           const bool isConstDensity,
                           const TurbulenceParameters& turbulenceParams,
                           const bool callSetup )
    : WasatchCore::TransportEquation( gc,
                                  solnVarName,
                                  get_staggered_location<FieldT>(),
                                  isConstDensity ),
      params_( params ),
      hasConvection_( params_->findBlock("ConvectiveFlux") ),
      densityTag_( densityTag ),
      enableTurbulence_( !params->findBlock("DisableTurbulenceModel") && (turbulenceParams.turbModelName != TurbulenceParameters::NOTURBULENCE) )
  {
    //_____________
    // Turbulence
    if( enableTurbulence_ ){
      Expr::Tag turbViscTag = TagNames::self().turbulentviscosity;
      turbDiffTag_ = turbulent_diffusivity_tag();

      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      if( !factory.have_entry( turbDiffTag_ ) ){
        typedef typename TurbulentDiffusivity::Builder TurbDiffT;
        factory.register_expression( scinew TurbDiffT( turbDiffTag_, densityTag_, turbulenceParams.turbSchmidt, turbViscTag ) );
      }
    }

    // define the primitive variable and solution variable tags and trap errors
    std::string form = "strong"; // default to strong form
    // get attribute for form. if none provided, then use default    
    if( params->findAttribute("form") ) params->getAttribute("form",form);

    isStrong_ = (form == "strong") ? true : false;
    const bool existPrimVar = params->findBlock("PrimitiveVariable");

    if( isConstDensity_ ){
      primVarTag_ = solution_variable_tag();
      if( existPrimVar ){
        std::ostringstream msg;
        msg << "ERROR: For constant density cases the primitive variable will be the same as the solution variable. So, you don't need to specify it. Please remove the \"PrimitiveVariable\" block from the \"TransportEquation\" block in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    else{
      if( isStrong_ && !existPrimVar ){
        std::ostringstream msg;
        msg << "ERROR: When you are solving a transport equation with variable density in its strong form, you need to specify your primitive and solution variables separately. Please include the \"PrimitiveVariable\" block in your input file in the \"TransportEquation\" block." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      else if( isStrong_ && existPrimVar ){
        const std::string primVarName = get_primvar_name( params );
        primVarTag_ = Expr::Tag( primVarName, Expr::STATE_NONE );
      }
      else if( !isStrong_ && existPrimVar ){
        std::ostringstream msg;
        msg << "ERROR: When solving a transport equation in weak form, the primitive variable will be the same as the solution variable. So, you don't need to specify it. Please remove the \"PrimitiveVariable\" block from the \"TransportEquation\" block in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      } else {
        primVarTag_ = solution_variable_tag();
      }
    }
    assert( primVarTag_ != Expr::Tag() );

    if( callSetup ) setup();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalarTransportEquation<FieldT>::~ScalarTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  ScalarTransportEquation<FieldT>::setup_diffusive_flux( FieldTagInfo& info )
  {
    // these expressions all get registered on the advance solution graph
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

    if( !isConstDensity_ ){
      for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
           diffFluxParams != nullptr;
           diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") ) {
        Expr::Tag densityTag, primVarTag;
        if (is_strong_form()) {
          densityTag = Expr::Tag(densityTag_.name(), Expr::STATE_NONE);
          primVarTag = Expr::Tag(primVarTag_.name(), Expr::STATE_NONE);
        } else {
          densityTag = densityTag_;
          primVarTag = primVarTag_;
        }
        
        setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                 densityTag,
                                                 primVarTag,
                                                 turbDiffTag_,
                                                 factory,
                                                 info );

        // if doing convection, we will likely have a pressure solve that requires
        // predicted scalar values to approximate the density time derivatives
        if( params_->findBlock("ConvectiveFlux") ){
          
          const Expr::Tag densityCorrectedTag = Expr::Tag(densityTag.name() + TagNames::self().star, Expr::STATE_NONE);
          const Expr::Tag primVarCorrectedTag = Expr::Tag(primVarTag.name() + TagNames::self().star, Expr::STATE_NONE);
          
          setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                   densityCorrectedTag,
                                                   primVarCorrectedTag,
                                                   turbDiffTag_,
                                                   factory,
                                                   infoStar_ );
        }
      } // loop over each flux specification
    }
    else{ // constant density
      for( Uintah::ProblemSpecP diffVelParams=params_->findBlock("DiffusiveFlux");
           diffVelParams != nullptr;
           diffVelParams=diffVelParams->findNextBlock("DiffusiveFlux") ) {
        setup_diffusive_velocity_expression<FieldT>( diffVelParams,
                                                     primVarTag_,
                                                     turbDiffTag_,
                                                     factory,
                                                     info );
      } // loop over each flux specification
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  ScalarTransportEquation<FieldT>::setup_convective_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    const Expr::Tag solnVarTag( solnVarName_, Expr::STATE_DYNAMIC );
    for( Uintah::ProblemSpecP convFluxParams=params_->findBlock("ConvectiveFlux");
         convFluxParams != nullptr;
         convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") ) {
      setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag, factory, info );
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  ScalarTransportEquation<FieldT>::setup_source_terms( FieldTagInfo& info,
                                                       Expr::TagList& srcTags )
  {
    for( Uintah::ProblemSpecP sourceTermParams=params_->findBlock("SourceTermExpression");
         sourceTermParams != nullptr;
         sourceTermParams=sourceTermParams->findNextBlock("SourceTermExpression") ) {
      srcTags.push_back( parse_nametag( sourceTermParams->findBlock("NameTag") ) );
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID
  ScalarTransportEquation<FieldT>::setup_rhs( FieldTagInfo& info,
                                              const Expr::TagList& srcTags )
  {

    typedef typename ScalarRHS<FieldT>::Builder RHSBuilder;
    typedef typename ScalarEOSCoupling<FieldT>::Builder ScalarEOSBuilder;
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    const TagNames& tagNames = TagNames::self();

    info[PRIMITIVE_VARIABLE] = primVarTag_;

    // for variable density flows:
    if( !isConstDensity_ || !isStrong_ ){
      factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarTag_, solnVarTag_, densityTag_) );

      if( hasConvection_ ){
        const Expr::Tag rhsStarTag     = tagNames.make_star_rhs(solnVarName_);
        const Expr::Tag densityStarTag = tagNames.make_star(densityTag_, Expr::STATE_NONE);
        const Expr::Tag primVarStarTag = tagNames.make_star(primVarTag_);
        const Expr::Tag solnVarStarTag = tagNames.make_star(solnVarName_);
        infoStar_[PRIMITIVE_VARIABLE]  = primVarStarTag;
        
        EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
        if( vNames.has_embedded_geometry() ){
          infoStar_[VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
          infoStar_[AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
          infoStar_[AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
          infoStar_[AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();
        }

        factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarStarTag, this->solnvar_np1_tag(), densityStarTag ) );

        const Expr::Tag scalEOSTag (primVarStarTag.name() + "_EOS_Coupling", Expr::STATE_NONE);
        const Expr::Tag dRhoDfStarTag("drhod" + primVarStarTag.name(), Expr::STATE_NONE);
        factory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoStar_, srcTags, densityStarTag, dRhoDfStarTag, isStrong_) );
        
        // register an expression for divu. divu is just a constant expression to which we add the
        // necessary couplings from the scalars that represent the equation of state.
        if( !factory.have_entry( tagNames.divu ) ) { // if divu has not been registered yet, then register it!
          typedef typename Expr::ConstantExpr<SVolField>::Builder divuBuilder;
          factory.register_expression( new divuBuilder(tagNames.divu, 0.0)); // set the value to zero so that we can later add sources to it
        }

        factory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
      }
    }

    return factory.register_expression( scinew RHSBuilder( rhsTag_, info, srcTags, densityTag_, isConstDensity_, isStrong_, tagNames.divrhou ) );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
   
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    //const Expr::Tag phiTag( this->solution_variable_name(), Expr::STATE_N );
    
    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ) {

      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList = tag_list( vNames.vol_frac_tag<SVolField>() );
      const Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE);
      factory.register_expression( new typename ExprAlgbr::Builder(modifierTag,
                                                                   theTagList,
                                                                   ExprAlgbr::PRODUCT,
                                                                   true) );
      
      factory.attach_modifier_expression( modifierTag, initial_condition_tag() );
    }
    
    if( factory.have_entry(initial_condition_tag()) ){
      bcHelper.apply_boundary_condition<FieldT>( initial_condition_tag(), taskCat );
    }
    
    if( !isConstDensity_ ){
      bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat );
    }
  }

  //------------------------------------------------------------------
  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;
      
      if (!isConstDensity_) {
        // for variable density problems, we must ALWAYS guarantee proper boundary conditions for
        // rhof_{n+1}. Since we apply bcs on rhof at the bottom of the graph, we can't apply
        // the same bcs on rhof (time advanced). Hence, we set rhof_rhs to zero always :)
        if( !myBndSpec.has_field(rhs_name()) ){ // if nothing has been specified for the RHS
          const BndCondSpec rhsBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
          bcHelper.add_boundary_condition(bndName, rhsBCSpec);
        }
      }
      
      switch ( myBndSpec.type ){
        case WALL:
        case VELOCITY:
        case OUTFLOW:
        case OPEN:{
          // for constant density problems, on all types of boundary conditions, set the scalar rhs
          // to zero. The variable density case requires setting the scalar rhs to zero ALL the time
          // and is handled in the code above.
          if( isConstDensity_ ){
            if( myBndSpec.has_field(rhs_name()) ){
              std::ostringstream msg;
              msg << "ERROR: You cannot specify scalar rhs boundary conditions unless you specify USER "
              << "as the type for the boundary condition. Please revise your input file. "
              << "This error occured while trying to analyze boundary " << bndName
              << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
            const BndCondSpec rhsBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
            bcHelper.add_boundary_condition(bndName, rhsBCSpec);
          }
          
          break;
        }
        case USER:{
          // parse through the list of user specified BCs that are relevant to this transport equation
          break;
        }
          
        default:
          break;
      }
    }
    
  }

  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell
  
    if( !isConstDensity_ && hasConvection_ ){
      // set bcs for solnVar_*
      const TagNames& tagNames = TagNames::self();
      const Expr::Tag solnVarStarTag = tagNames.make_star(this->solution_variable_name());
      //bcHelper.apply_boundary_condition<FieldT>( solnVarStarTag, taskCat );
      //bcHelper.apply_boundary_condition<FieldT>( Expr::Tag(rhs_tag().name() + tagNames.star, Expr::STATE_NONE), taskCat, true );
      bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalarTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    if( isStrong_ && !is_constant_density() ){
      // register expression to calculate the initial condition of the solution variable from the initial
      // conditions on primitive variable and density in the cases that we are solving for e.g. rho*phi
      typedef ExprAlgebra<SVolField> ExprAlgbr;
      return icFactory.register_expression( new typename ExprAlgbr::Builder( initial_condition_tag(),
                                                                             tag_list( primVarTag_, Expr::Tag(densityTag_.name(),Expr::STATE_NONE) ),
                                                                             ExprAlgbr::PRODUCT ) );
    }
    return icFactory.get_id( initial_condition_tag() );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  std::string
  ScalarTransportEquation<FieldT>::get_solnvar_name( Uintah::ProblemSpecP params )
  {
    std::string solnVarName;
    params->get("SolutionVariable",solnVarName);
    return solnVarName;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  std::string
  ScalarTransportEquation<FieldT>::get_primvar_name( Uintah::ProblemSpecP params )
  {
    std::string primVarName;
    params->get("PrimitiveVariable",primVarName);
    return primVarName;
  }

  //------------------------------------------------------------------

  //==================================================================
  // explicit template instantiation
  template class ScalarTransportEquation< SVolField >;

} // namespace WasatchCore
