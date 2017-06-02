/*
 * The MIT License
 *
 * Copyright (c) 2016-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Transport/SpeciesTransportEquation.h>

#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>

#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

// PoKiTT expressions that we require here
#include <pokitt/CanteraObjects.h>
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/SpeciesN.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for species transport

std::vector<EqnTimestepAdaptorBase*>
setup_species_equations( Uintah::ProblemSpecP specEqnParams,
                         const TurbulenceParameters& turbParams,
                         const Expr::Tag densityTag,
                         const Expr::Tag temperatureTag,
                         GraphCategories& gc )
{
  std::vector<EqnTimestepAdaptorBase*> specEqns;

  const TagNames& tagNames = TagNames::self();

  std::string canteraInputFileName, canteraGroupName;
  specEqnParams->get("CanteraInputFile",canteraInputFileName);
  specEqnParams->get("CanteraGroup",    canteraGroupName    );

  CanteraObjects::Setup canteraSetup( "Mix", canteraInputFileName, canteraGroupName );
  CanteraObjects::setup_cantera( canteraSetup );

  const int nspec = CanteraObjects::number_species();

  Expr::ExpressionFactory&   factory = *gc[ADVANCE_SOLUTION]->exprFactory;
  Expr::ExpressionFactory& icFactory = *gc[INITIALIZATION  ]->exprFactory;

  Expr::TagList yiTags;
  for( int i=0; i<nspec; ++i ){
    const std::string specName = CanteraObjects::species_name(i);
    const Expr::Tag specTag( specName, Expr::STATE_NONE );
    yiTags.push_back( specTag );

    if( i == nspec-1 ) continue; // don't build the nth species equation

    proc0cout << "Setting up transport equation for species " << specName << std::endl;
    SpeciesTransportEquation* specEqn = scinew SpeciesTransportEquation( specEqnParams, turbParams, i, gc, densityTag, temperatureTag, tagNames.mixMW );
    specEqns.push_back( new EqnTimestepAdaptor<FieldT>(specEqn) );

    // default to zero mass fraction for species unless specified otherwise
    if( !icFactory.have_entry( specTag ) ){
      icFactory.register_expression( scinew Expr::ConstantExpr<FieldT>::Builder(specTag,0.0) );
    }
    //_____________________________________________________
    // set up initial conditions on this equation
    try{
      GraphHelper* const icGraphHelper = gc[INITIALIZATION];
      icGraphHelper->rootIDs.insert( specEqn->initial_condition( *icGraphHelper->exprFactory ) );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
          << std::endl
          << "ERORR while setting initial conditions on species: '" << specName << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
  }

  // mixture molecular weight
  factory  .register_expression( scinew pokitt::MixtureMolWeight<FieldT>::Builder( tagNames.mixMW, yiTags, pokitt::MASS ) );
  icFactory.register_expression( scinew pokitt::MixtureMolWeight<FieldT>::Builder( tagNames.mixMW, yiTags, pokitt::MASS ) );

  // the last species mass fraction
  typedef pokitt::SpeciesN<FieldT>::Builder SpecN;
  factory  .register_expression( scinew SpecN( yiTags[nspec-1], yiTags, pokitt::ERRORSPECN, 0 /* no ghost cells for RHS calculation */ ) );
  icFactory.register_expression( scinew SpecN( yiTags[nspec-1], yiTags, pokitt::ERRORSPECN, 0 /* no ghost cells for RHS calculation */ ) );

  return specEqns;
}

//==============================================================================

SpeciesTransportEquation::
SpeciesTransportEquation( Uintah::ProblemSpecP params,
                          const TurbulenceParameters& turbParams,
                          const int specNum,
                          GraphCategories& gc,
                          const Expr::Tag densityTag,
                          const Expr::Tag temperatureTag,
                          const Expr::Tag mmwTag )
: TransportEquation( gc, "rho_"+CanteraObjects::species_name(specNum), NODIR, false ),  // jcs should we allow constant density?  possibly...
  params_( params ),
  turbParams_( turbParams ),
  specNum_( specNum ),
  primVarTag_( CanteraObjects::species_name(specNum), Expr::STATE_NONE ),
  densityTag_    ( densityTag     ),
  temperatureTag_( temperatureTag ),
  mmwTag_        ( mmwTag         ),
  nspec_( CanteraObjects::number_species() )
{
  for( int i=0; i<nspec_; ++i ){
    yiTags_.push_back( Expr::Tag( CanteraObjects::species_name(i), Expr::STATE_NONE ) );
  }

  setup();
}

//------------------------------------------------------------------------------

SpeciesTransportEquation::~SpeciesTransportEquation(){}

//------------------------------------------------------------------------------

Expr::ExpressionID
SpeciesTransportEquation::initial_condition( Expr::ExpressionFactory& icFactory )
{
  if( is_constant_density() ){
    return icFactory.get_id( initial_condition_tag() );
  }
  typedef ExprAlgebra<FieldT>::Builder Algebra;
  return icFactory.register_expression( scinew Algebra( initial_condition_tag(),
                                                        tag_list( this->primVarTag_, Expr::Tag(densityTag_.name(),Expr::STATE_NONE) ),
                                                        ExprAlgebra<FieldT>::PRODUCT) );
}

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::
setup_boundary_conditions( WasatchBCHelper& bcHelper,
                           GraphCategories& graphCat )
{
  // jcs at some point we will need to include support for NSCBC here.

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

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::apply_initial_boundary_conditions( const GraphHelper& graphHelper,
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

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::apply_boundary_conditions( const GraphHelper& graphHelper,
                                                     WasatchBCHelper& bcHelper )
{
  const Category taskCat = ADVANCE_SOLUTION;
  bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
  bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell
}

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::setup_convective_flux( FieldTagInfo& info )
{
  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  const Expr::Tag solnVarTag( solnVarName_, Expr::STATE_DYNAMIC );
  for( Uintah::ProblemSpecP convFluxParams=params_->findBlock("ConvectiveFlux");
      convFluxParams != nullptr;
      convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") )
  {
    setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag, factory, info );
  }
}

//------------------------------------------------------------------------------

Expr::ExpressionID
SpeciesTransportEquation::setup_rhs( FieldTagInfo& info,
                                     const Expr::TagList& srcTags )
{
  const TagNames& tagNames = TagNames::self();

  assert( srcTags.empty() );

  info[PRIMITIVE_VARIABLE] = primVarTag_;

  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

  factory.register_expression( new PrimVar<FieldT,SVolField>::Builder( primVarTag_, solnVarTag_, densityTag_) );

  const bool isConstDensity = false;
  const bool isStrongForm   = true;
  typedef ScalarRHS<FieldT>::Builder RHS;
  return factory.register_expression( scinew RHS( rhsTag_, info, srcTags, densityTag_, isConstDensity, isStrongForm, tagNames.drhodt ) );
}

//------------------------------------------------------------------------------

} // namespace WasatchCore
