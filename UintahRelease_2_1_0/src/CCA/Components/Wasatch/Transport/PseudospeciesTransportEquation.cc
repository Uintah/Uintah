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
#include <CCA/Components/Wasatch/Transport/PseudospeciesTransportEquation.h>
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
  PseudospeciesTransportEquation<FieldT>::
  PseudospeciesTransportEquation( const std::string pseudospeciesName,
                                  Uintah::ProblemSpecP params,
                                  GraphCategories& gc,
                                  const Expr::Tag densityTag,
                                  const TurbulenceParameters& turbulenceParams,
                                  const bool callSetup )
    : WasatchCore::TransportEquation( gc,
                                      "rho_" + pseudospeciesName,       // this will need changed if solving under constant density conditions is allowed
                                      get_staggered_location<FieldT>(),
                                      false ),
      params_        ( params                                   ),
      psParams_      ( params->findBlock("TarAndSootEquations") ),
      solnVarName_   ( "rho_" + pseudospeciesName               ),
      pseudoSpecName_( pseudospeciesName                        ),
      densityTag_    ( densityTag                               ),
      primVarTag_    ( pseudospeciesName, Expr::STATE_NONE      ),
      enableTurbulence_(  turbulenceParams.turbModelName != TurbulenceParameters::NOTURBULENCE ),
      isStrong_      ( true   ),
      isConstDensity_( false  )
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
    if( callSetup ) setup();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  PseudospeciesTransportEquation<FieldT>::~PseudospeciesTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  PseudospeciesTransportEquation<FieldT>::
  setup_convective_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *(gc_[ADVANCE_SOLUTION]->exprFactory);
    const Expr::Tag solnVarTag( solnVarName_, Expr::STATE_DYNAMIC );

    if( psParams_ && psParams_->findBlock("ConvectiveFlux") ){
      for( Uintah::ProblemSpecP convFluxParams=psParams_->findBlock("ConvectiveFlux");
          convFluxParams != nullptr;
          convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") )
      {
        setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag, factory, info );
      }

    }
    else{
      // jtm: should this be the default flux limiter?
      const ConvInterpMethods convMethod = CENTRAL;

      proc0cout << std::endl
                << "WARNING:\n"
                << "Default convective interpolation method ("
                << get_conv_interp_method(convMethod)
                <<") used for " << solution_variable_name()
                << std::endl;

      std::string dir;
      Uintah::ProblemSpecP momentumSpec = params_->findBlock("MomentumEquations");
      std::string xVelName, yVelName, zVelName;
      Expr::Tag advVelocityTag;
      const bool doX = momentumSpec->get( "X-Velocity", xVelName );
      const bool doY = momentumSpec->get( "Y-Velocity", yVelName );
      const bool doZ = momentumSpec->get( "Z-Velocity", zVelName );

      if( doX){
        dir = "X";
        advVelocityTag = Expr::Tag(xVelName, Expr::STATE_NONE);

        setup_convective_flux_expression<FieldT>( dir, solnVarTag,
                                                  Expr::Tag(), convMethod,
                                                  advVelocityTag, factory,
                                                  info );
      }
      if( doY){
        dir = "Y";
        advVelocityTag = Expr::Tag(yVelName, Expr::STATE_NONE);

        setup_convective_flux_expression<FieldT>( dir, solnVarTag,
                                                  Expr::Tag(), convMethod,
                                                  advVelocityTag, factory,
                                                  info );
      }
      if( doZ){
        dir = "Z";
        advVelocityTag = Expr::Tag(zVelName, Expr::STATE_NONE);

        setup_convective_flux_expression<FieldT>( dir, solnVarTag,
                                                  Expr::Tag(), convMethod,
                                                  advVelocityTag, factory,
                                                  info );
      }

    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID
  PseudospeciesTransportEquation<FieldT>::
  setup_rhs( FieldTagInfo& info,
             const Expr::TagList& srcTags )
  {
    const TagNames& tagNames = TagNames::self();

    info[PRIMITIVE_VARIABLE] = primVarTag_;

    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

    factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarTag_, solnVarTag_, densityTag_) );

    typedef typename ScalarRHS<FieldT>::Builder RHS;
    return factory.register_expression( scinew RHS( rhsTag_, info, srcTags, densityTag_, isConstDensity_, isStrong_, tagNames.drhodt ) );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void PseudospeciesTransportEquation<FieldT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;

    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;

    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ) {

      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList = tag_list( vNames.vol_frac_tag<SVolField>() );
      const Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE);
      factory.register_expression( new typename ExprAlgbr::Builder( modifierTag,
                                                                    theTagList,
                                                                    ExprAlgbr::PRODUCT,
                                                                    true ) );

      factory.attach_modifier_expression( modifierTag, initial_condition_tag() );
    }

    if( factory.have_entry(initial_condition_tag()) ){
      bcHelper.apply_boundary_condition<FieldT>( initial_condition_tag(), taskCat );
    }

  }

  //------------------------------------------------------------------
  template< typename FieldT >
  void PseudospeciesTransportEquation<FieldT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    assert( !isConstDensity_ );

    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;

      // for variable density problems, we must ALWAYS guarantee proper boundary conditions for
      // (rho y_i)_{n+1}. Since we apply bcs on (rho y_i) at the bottom of the graph, we can't apply
      // the same bcs on (rho yi) (time advanced). Hence, we set the rhs to zero always :)
      if( !myBndSpec.has_field(rhs_name()) ){
        const BndCondSpec rhsBCSpec = { rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
        bcHelper.add_boundary_condition(bndName, rhsBCSpec);
      }

      switch ( myBndSpec.type ){
        case WALL:
        {
          const TagNames& tagNames = TagNames::self();
          const Uintah::Patch::FaceType& face = myBndSpec.face;
          const std::string dir = (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus) ? "X"
                                : (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus) ? "Y"
                                : (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus) ? "Z" : "INVALID";

          const std::string diffFluxName( primVarTag_.name() + "_" + tagNames.diffusiveflux + "_"  + dir );
          const std::string convFluxName( solnVarName_       + "_" + tagNames.convectiveflux + "_" + dir );

          // set zero convective and diffusive fluxes
          const BndCondSpec diffFluxBCSpec = { diffFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
          const BndCondSpec convFluxBCSpec = { convFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
          bcHelper.add_boundary_condition( bndName, diffFluxBCSpec );
          bcHelper.add_boundary_condition( bndName, convFluxBCSpec );
        }
        case VELOCITY:
        case OUTFLOW:
        case OPEN:
        case USER:
        default:
        {
          std::ostringstream msg;
          msg << "ERROR: VELOCITY, OPEN, and OUTFLOW boundary conditions are not currently supported for compressible flows in Wasatch. " << bndName
          << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          break;
        }
      }
    }
}

  //------------------------------------------------------------------

  template< typename FieldT >
  void PseudospeciesTransportEquation<FieldT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  PseudospeciesTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    if( is_constant_density() ){
      return icFactory.get_id( initial_condition_tag() );
    }
    typedef typename ExprAlgebra<FieldT>::Builder Algebra;
    return icFactory.register_expression( scinew Algebra( initial_condition_tag(),
                                                          tag_list( this->primVarTag_, Expr::Tag(densityTag_.name(),Expr::STATE_NONE) ),
                                                          ExprAlgebra<FieldT>::PRODUCT) );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::Tag
  PseudospeciesTransportEquation<FieldT>::
  get_species_rhs_tag( std::string name)
  {
    return Expr::Tag("rho_" + name + "_rhs", Expr::STATE_NONE);
  }

  //------------------------------------------------------------------

  //==================================================================
  // explicit template instantiation
  template class PseudospeciesTransportEquation< SVolField >;

} // namespace WasatchCore
