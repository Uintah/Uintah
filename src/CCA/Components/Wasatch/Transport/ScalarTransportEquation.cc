/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
                           const TurbulenceParameters& turbulenceParams,
                           std::set<std::string>& persistentFields,
                           const bool callSetup )
    : WasatchCore::TransportEquation( gc,
                                  solnVarName,
                                  get_staggered_location<FieldT>() ),
      params_( params ),
      hasConvection_   ( params_->findBlock("ConvectiveFlux") ),
      densityInitTag_  ( densityTag.name(), Expr::STATE_NONE ),
      densityTag_      ( densityTag ),
      densityNP1Tag_   ( densityTag.name(), Expr::STATE_NP1 ),
      enableTurbulence_( !params->findBlock("DisableTurbulenceModel") && (turbulenceParams.turbModelName != TurbulenceParameters::NOTURBULENCE) ),
      persistentFields_( persistentFields )
  {
    //_____________
    // Turbulence
    const TagNames& tagNames = TagNames::self();
    if( enableTurbulence_ ){
      Expr::Tag turbViscTag = tagNames.turbulentviscosity;

      turbDiffTag_ = tagNames.turbulentdiffusivity;

      Expr::ExpressionFactory& factory   = *gc_[ADVANCE_SOLUTION]->exprFactory;

      typedef typename TurbulentDiffusivity::Builder TurbDiffT;
      if (!factory.have_entry(turbDiffTag_))
        factory.register_expression( scinew TurbDiffT( turbDiffTag_, densityTag_, turbulenceParams.turbSchmidt, turbViscTag ) );
    } // if(enableTurbulence_)

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
        primVarTag_ = Expr::Tag( primVarName, flowTreatment_==LOWMACH ? Expr::STATE_N : Expr::STATE_NONE );
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

    primVarInitTag_ = Expr::Tag( primVarTag_.name(), Expr::STATE_NONE);
    primVarNP1Tag_  = Expr::Tag( primVarTag_.name(), Expr::STATE_NP1);

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
    Expr::ExpressionFactory& factory  = *gc_[ADVANCE_SOLUTION]->exprFactory;
    Expr::ExpressionFactory& iFactory = *gc_[INITIALIZATION  ]->exprFactory;

    switch(flowTreatment_){
      case INCOMPRESSIBLE: {
        for( Uintah::ProblemSpecP diffVelParams=params_->findBlock("DiffusiveFlux");
             diffVelParams != nullptr;
             diffVelParams=diffVelParams->findNextBlock("DiffusiveFlux") ) {
          setup_diffusive_velocity_expression<FieldT>( diffVelParams,
                                                       primVarTag_,
                                                       turbDiffTag_,
                                                       factory,
                                                       info );
        } // loop over each flux specification
      } // case INCOMPRESSIBLE
      break;

      case LOWMACH: {
       /* If scalars are being transported in strong form using the low-Mach algorithm, we need
        * density and primitive scalar at the current state (STATE_N) to calculate scalar RHS terms
        * at the new state (STATE_NP1), and subsequently, div(u) at STATE_NP1. As a result,
        * expressions for diffusive fluxes at STATE_N are registered as placeholders.
        */

        for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
          diffFluxParams != nullptr;
          diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") ) {

             // if doing convection, we will likely have a pressure solve that requires
             // predicted scalar values to approximate the density time derivatives
             if( params_->findBlock("ConvectiveFlux") ){

              /* if the low-Mach algorithm is used with a turbulence model, we need to compute
               * the diffusive flux at STATE_N and an estimate for diffusive flux at STATE_NP1.
               * We cannot calculate the STATE_NP1 value exactly attempting to do so would
               * induce a circular dependency like so:
               * diffusive-flux_NP1 --> (...) --> velocity_NP1 --> (...) --> diffusive-flux_NP1
               */
               if(enableTurbulence_){
                 setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                          densityTag_,
                                                          primVarTag_,
                                                          turbDiffTag_,
                                                          factory,
                                                          info );

                 setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                          densityTag_,
                                                          primVarTag_,
                                                          turbDiffTag_,
                                                          factory,
                                                          infoNP1_,
                                                          Expr::STATE_NONE,
                                                          "-NP1-estimate");
               } // if(enableTurbulence_)

               else{
                 assert(primVarNP1Tag_.context() == Expr::STATE_NP1);
                 setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                          densityNP1Tag_,
                                                          primVarNP1Tag_,
                                                          turbDiffTag_,
                                                          factory,
                                                          infoNP1_,
                                                          Expr::STATE_NP1 );

                 // fieldTagInfo used for diffusive fluxes calculated at initialization
                 FieldTagInfo infoInit;

                 assert(primVarInitTag_.context() == Expr::STATE_NONE);
                 setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                          densityInitTag_,
                                                          primVarInitTag_,
                                                          turbDiffTag_,
                                                          iFactory,
                                                          infoInit,
                                                          primVarInitTag_.context() );

                 // set info for diffusive flux based on infoNP1_, add tag names to set of persistent fields
                 const std::vector<FieldSelector> fsVec = {DIFFUSIVE_FLUX_X, DIFFUSIVE_FLUX_Y, DIFFUSIVE_FLUX_Z};
                 for( FieldSelector fs : fsVec ){
                  if( infoNP1_.find(fs) != infoNP1_.end() ){
                    const std::string diffFluxName = infoNP1_[fs].name();
                    info[fs] = Expr::Tag(diffFluxName, Expr::STATE_N);
                    persistentFields_.insert(diffFluxName);

                    // Force diffusive flux expression on initialization graph.
                    const Expr::ExpressionID id = iFactory.get_id( infoInit[fs] );
                    gc_[INITIALIZATION]->rootIDs.insert(id);
                  }
                 }

                 // Register placeholders for diffusive flux parameters at STATE_N
                 register_diffusive_flux_placeholders<FieldT>( factory, info );
               } // else -- (enableTurbulence_ == false)
             } //if( params_->findBlock("ConvectiveFlux") )

             else{
               setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                        densityTag_,
                                                        primVarTag_,
                                                        turbDiffTag_,
                                                        factory,
                                                        info );
             }
        } // loop over each flux specification
      } // case LOWMACH
      break;

      case COMPRESSIBLE: {
        Expr::Tag densityTag, primVarTag;
        if (is_strong_form()) {
          densityTag = Expr::Tag(densityTag_.name(), Expr::STATE_NONE);
          primVarTag = Expr::Tag(primVarTag_.name(), Expr::STATE_NONE);
        } else {
          densityTag = densityTag_;
          primVarTag = primVarTag_;
        }

        for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
          diffFluxParams != nullptr;
          diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") ) {

             setup_diffusive_flux_expression<FieldT>( diffFluxParams,
                                                      densityTag,
                                                      primVarTag,
                                                      turbDiffTag_,
                                                      factory,
                                                      info );
        } // loop over each flux specification
      } // case COMPRESSIBLE
      break;

      default:
        std::ostringstream msg;
        msg << "ERROR: unhandled flow treatment." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
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
    /* todo: Deal with source terms from input file properly when flowTreatment == LOWMACH
     * and scalars are transported in strong form. This may be a bit tricky since we need
     * source terms at both STATE_N and STATE_NP1 for this case.
     */
    for( Uintah::ProblemSpecP sourceTermParams=params_->findBlock("SourceTermExpression");
         sourceTermParams != nullptr;
         sourceTermParams=sourceTermParams->findNextBlock("SourceTermExpression") ) {
      srcTags.push_back( parse_nametag( sourceTermParams->findBlock("NameTag") ) );
    }
    
    for( Uintah::ProblemSpecP sourceTermParams=params_->findBlock("TargetValueSource");
        sourceTermParams != nullptr;
        sourceTermParams=sourceTermParams->findNextBlock("TargetValueSource") ) {
      
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      
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

    if( flowTreatment_ == COMPRESSIBLE ){
      factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarTag_, solnVarTag_, densityTag_) );
    }

    // for variable density flows:
    if( flowTreatment_ == LOWMACH ){
      const Expr::Tag rhsNP1Tag     = Expr::Tag(rhsTag_.name(), Expr::STATE_NP1);
      infoNP1_[PRIMITIVE_VARIABLE]  = primVarNP1Tag_;

      EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
      if( vNames.has_embedded_geometry() ){
        infoNP1_
        [VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
        infoNP1_[AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
        infoNP1_[AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
        infoNP1_[AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();
      }

      if(isStrong_){
        const Expr::ExpressionID primVarNP1ID = 
        factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarNP1Tag_, this->solnvar_np1_tag(), densityNP1Tag_ ) );
        factory.register_expression( new typename Expr::PlaceHolder<FieldT>::Builder(primVarTag_) );

        gc_[ADVANCE_SOLUTION]->rootIDs.insert(primVarNP1ID);
      }

      const Expr::Tag scalEOSTag (primVarTag_.name() + "_EOS_Coupling", Expr::STATE_NONE);
      const Expr::Tag dRhoDfTag = tagNames.derivative_tag(densityTag_,primVarTag_);
      factory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoNP1_, srcTags, densityNP1Tag_, dRhoDfTag, isStrong_) );

      // register an expression for divu. divu is just a constant expression to which we add the
      // necessary couplings from the scalars that represent the equation of state.
      if( !factory.have_entry( tagNames.divu ) ) { // if divu has not been registered yet, then register it!
        typedef typename Expr::ConstantExpr<SVolField>::Builder divuBuilder;
        factory.register_expression( new divuBuilder(tagNames.divu, 0.0)); // set the value to zero so that we can later add sources to it
      }

      factory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
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
      bcHelper.apply_boundary_condition<FieldT>( primVarInitTag_, taskCat );
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
    // Should we always apply BCs to STATE_NP1 or STATE_N? Does it even matter?
    const Category taskCat = ADVANCE_SOLUTION;
    const Expr::Tag solnVarTag = flowTreatment_==LOWMACH ? solnvar_np1_tag() : solution_variable_tag();
    bcHelper.apply_boundary_condition<FieldT>( solnVarTag, taskCat );
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell

    const std::string normalDiffVelName_nodir = primVarTag_.name() + "_diffVelocity_";
    bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag(normalDiffVelName_nodir + 'X', Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag(normalDiffVelName_nodir + 'Y', Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag(normalDiffVelName_nodir + 'Z', Expr::STATE_NONE), taskCat);

    if( !isConstDensity_ && hasConvection_ ){
      const Expr::Tag primVarTag = flowTreatment_==LOWMACH ? primVarNP1Tag_ : primVarTag_;
      bcHelper.apply_boundary_condition<FieldT>( primVarTag, taskCat );
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
      const Expr::Tag icPrimVarTag = Expr::Tag( primVarTag_.name(), Expr::STATE_NONE );
      const Expr::Tag icDensityTag = Expr::Tag( densityTag_.name(), Expr::STATE_NONE );

      return icFactory.register_expression( new typename ExprAlgbr::Builder( initial_condition_tag(),
                                                                             tag_list( icPrimVarTag, icDensityTag ),
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
