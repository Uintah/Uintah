/*
 * The MIT License
 *
 * Copyright (c) 2016-2023 The University of Utah
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

#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSCoupling.h>

#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>

#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionsOneSided.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <boost/algorithm/string.hpp>

#ifdef HAVE_POKITT
// PoKiTT expressions that we require here
#include <pokitt/CanteraObjects.h>
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/SpeciesN.h>
#endif

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for species transport


template< typename FaceT, typename GradT >
struct SpeciesBoundaryTyper
{
  typedef SpatialOps::SVolField CellT;
  typedef SpatialOps::Divergence DivT;

  typedef typename SpatialOps::OperatorTypeBuilder<GradT, CellT, CellT>::type CellNeumannT;
  typedef typename SpatialOps::OperatorTypeBuilder<DivT, FaceT, CellT>::type FaceNeumannT;

  typedef typename SpatialOps::NeboBoundaryConditionBuilder<CellNeumannT> CellNeumannBCOpT;
  typedef typename SpatialOps::NeboBoundaryConditionBuilder<FaceNeumannT> FaceNeumannBCOpT;

  typedef typename ConstantBCNew<CellT,CellNeumannBCOpT>::Builder ConstantCellNeumannBC;
  typedef typename ConstantBCNew<FaceT,FaceNeumannBCOpT>::Builder ConstantFaceNeumannBC;
};

std::vector<EqnTimestepAdaptorBase*>
setup_species_equations( Uintah::ProblemSpecP specEqnParams,
                         Uintah::ProblemSpecP wasatchSpec,
                         const TurbulenceParameters& turbParams,
                         const Expr::Tag densityTag,
                         const Expr::TagList velTags,
                         const Expr::Tag temperatureTag,
                         GraphCategories& gc,
                         std::set<std::string>& persistentFields,
                         WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                         bool computeKineticsJacobian )
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

  const bool isLowMach =  Wasatch::flow_treatment() == LOWMACH;
  Expr::Context context = isLowMach ? Expr::STATE_NP1 : Expr::STATE_NONE;

  Expr::TagList yiTags, yiInitTags;
  for( int i=0; i<nspec; ++i ){
    const std::string specName = CanteraObjects::species_name(i);
    yiTags    .push_back( Expr::Tag( specName, context          ) );
    yiInitTags.push_back( Expr::Tag( specName, Expr::STATE_NONE ) );

    if( i == nspec-1 ) continue; // don't build the nth species equation

    proc0cout << "Setting up transport equation for species " << specName << std::endl;
    SpeciesTransportEquation* specEqn = scinew SpeciesTransportEquation( specEqnParams,
                                                                         wasatchSpec,
                                                                         turbParams,
                                                                         i,
                                                                         gc,
                                                                         persistentFields,
                                                                         densityTag,
                                                                         velTags,
                                                                         temperatureTag,
                                                                         tagNames.mixMW,
                                                                         dualTimeMatrixInfo,
                                                                         computeKineticsJacobian );
    specEqns.push_back( new EqnTimestepAdaptor<FieldT>(specEqn) );

    // default to zero mass fraction for species unless specified otherwise
    const Expr::Tag& specTag = yiInitTags[i];
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
          << "ERROR while setting initial conditions on species: '" << specName << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
  }

  // mixture molecular weight
  factory  .register_expression( scinew pokitt::MixtureMolWeight<FieldT>::Builder( tagNames.mixMW, yiTags    , pokitt::MASS ) );
  icFactory.register_expression( scinew pokitt::MixtureMolWeight<FieldT>::Builder( tagNames.mixMW, yiInitTags, pokitt::MASS ) );

  // the last species mass fraction
  typedef pokitt::SpeciesN<FieldT>::Builder SpecN;
  factory  .register_expression( scinew SpecN( yiTags    [nspec-1], yiTags    , pokitt::CLIPSPECN,  DEFAULT_NUMBER_OF_GHOSTS ) );
  icFactory.register_expression( scinew SpecN( yiInitTags[nspec-1], yiInitTags, pokitt::ERRORSPECN, DEFAULT_NUMBER_OF_GHOSTS ) );

  if( isLowMach ){
    const Expr::Tag ynTag = Expr::Tag( yiTags[nspec-1].name(), Expr::STATE_N );
    factory.register_expression( scinew Expr::PlaceHolder<FieldT>::Builder( ynTag ) );
    persistentFields.insert(ynTag.name());
  }

  // register stuff for dual time
  Expr::TagList rhoYiTags;
  for( int i=0; i<nspec; ++i ){
    rhoYiTags.push_back( Expr::Tag( "rho_" + CanteraObjects::species_name(i), Expr::STATE_DYNAMIC ) );
  }

  dualTimeMatrixInfo.doSpecies = true;
  dualTimeMatrixInfo.set_mass_fraction_tags( yiTags );
  dualTimeMatrixInfo.set_species_density_tags( rhoYiTags );
  dualTimeMatrixInfo.set_molecular_weights( CanteraObjects::molecular_weights() );
  dualTimeMatrixInfo.mmw = tagNames.mixMW;

  return specEqns;
}

//==============================================================================

SpeciesTransportEquation::
SpeciesTransportEquation( Uintah::ProblemSpecP params,
                          Uintah::ProblemSpecP wasatchSpec,
                          const TurbulenceParameters& turbParams,
                          const int specNum,
                          GraphCategories& gc,
                          std::set<std::string>& persistentFields,
                          const Expr::Tag densityTag,
                          const Expr::TagList velTags,
                          const Expr::Tag temperatureTag,
                          const Expr::Tag mmwTag,
                          WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                          const bool computeKineticsJacobian )
: TransportEquation( gc, "rho_"+CanteraObjects::species_name(specNum), NODIR ),  // jcs should we allow constant density?  possibly...
  params_     ( params      ),
  wasatchSpec_( wasatchSpec ),
  turbParams_ ( turbParams  ),
  specNum_    ( specNum     ),
  primVarTag_( CanteraObjects::species_name(specNum), flowTreatment_ == LOWMACH ? Expr::STATE_N : Expr::STATE_NONE ),
  primVarNP1Tag_ ( primVarTag_.name(), Expr::STATE_NP1  ),
  primVarInitTag_( primVarTag_.name(), Expr::STATE_NONE ),
  densityTag_    ( densityTag                           ),
  densityNP1Tag_ ( densityTag.name(), Expr::STATE_NP1   ),
  densityInitTag_( densityTag.name(), Expr::STATE_NONE  ),
  temperatureTag_( temperatureTag                       ),
  mmwTag_        ( mmwTag                               ),
  velTags_       ( velTags                              ),
  nspec_         ( CanteraObjects::number_species()     ),
  isStrong_      ( true                                 ),
  persistentFields_  ( persistentFields   ),
  dualTimeMatrixInfo_( dualTimeMatrixInfo )
{
  Expr::Context contextN = flowTreatment_== LOWMACH ? Expr::STATE_N : Expr::STATE_NONE;
  for( int i=0; i<nspec_; ++i ){
    const std::string specName = CanteraObjects::species_name(i);
    yiTags_    .push_back( Expr::Tag( specName, contextN         ) );
    yiNP1Tags_ .push_back( Expr::Tag( specName, Expr::STATE_NP1  ) );
    yiInitTags_.push_back( Expr::Tag( specName, Expr::STATE_NONE ) );
  }

  // set the Jacobian
  if( computeKineticsJacobian ){
    jacobian_ = boost::make_shared<pokitt::ChemicalSourceJacobian>();
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
                                                        tag_list( primVarInitTag_, densityInitTag_ ),
                                                        ExprAlgebra<FieldT>::PRODUCT) );
}

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::
setup_boundary_conditions( WasatchBCHelper& bcHelper,
                           GraphCategories& graphCat )
{
  Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
  Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);

  assert( !isConstDensity_ );

  // set up the extra fields for setting BCs on primitives
  const Expr::Tag temporaryYTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
  const Expr::Tag temporaryRhoTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
  const Expr::Tag temporaryRhoYTag( "temporary_rho" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );

  if( !( advSlnFactory.have_entry( temporaryYTag ) ) ){
    advSlnFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryYTag, 0.0 ) );
  }
  if( !( initFactory.have_entry( temporaryYTag ) ) ){
    initFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryYTag, 0.0 ) );
  }
  if( !( advSlnFactory.have_entry( temporaryRhoYTag ) ) ){
    typedef ExprAlgebra<SVolField>::Builder RhoY;
    advSlnFactory.register_expression( new RhoY( temporaryRhoYTag, Expr::tag_list( temporaryRhoTag, temporaryYTag ), ExprAlgebra<SVolField>::PRODUCT ) );
  }
  if( !( initFactory.have_entry( temporaryRhoYTag ) ) ){
    typedef ExprAlgebra<SVolField>::Builder RhoY;
    initFactory.register_expression( new RhoY( temporaryRhoYTag, Expr::tag_list( temporaryRhoTag, temporaryYTag ), ExprAlgebra<SVolField>::PRODUCT ) );
  }

  BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
    const std::string& bndName = bndPair.first;
    const BndSpec& myBndSpec = bndPair.second;

    // a lambda to make decorated tags for boundary condition expressions for a this boundary spec
    //
    // param: exprName: a string, the name of the field on which we will impose the boundary condition
    // param: description: a string describing the boundary condition, such as "neumann-zero-for-outflow" or "dirichlet-for-inflow"
    // param: direction: a string for the direction of the boundary face, such as "X", "Y", or "Z"
    auto get_decorated_tag = [&myBndSpec]( const std::string exprName, const std::string description, const std::string direction )
    {
      return Expr::Tag( exprName + "_STATE_NONE_" + description + "_bc_" + myBndSpec.name + "_" + direction + "dir", Expr::STATE_NONE );
    };

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
        const Uintah::Patch::FaceType& face = myBndSpec.face;
        const std::string dir = (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus) ? "X"
                              : (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus) ? "Y"
                              : (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus) ? "Z" : "INVALID";

        const std::string diffFluxName( primVarTag_.name() + "_diffFlux_" + dir );
        const std::string convFluxName( this->solnVarName_ + "_convFlux_" + dir );

        // set zero convective and diffusive fluxes
        const BndCondSpec diffFluxBCSpec = { diffFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
        const BndCondSpec convFluxBCSpec = { convFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
        bcHelper.add_boundary_condition( bndName, diffFluxBCSpec );
        bcHelper.add_boundary_condition( bndName, convFluxBCSpec );
      }
      break;


      case OUTFLOW:
      case OPEN:{

        const std::string normalConvFluxNameBase = this->solnVarName_ + "_convFlux_"; // normal convective flux
        const std::string normalDiffFluxNameBase = primVarTag_.name() + "_diffFlux_"; // normal diffusive flux

        Expr::Tag neumannZeroConvectiveFluxTag,
                  neumannZeroDiffFluxTag,
                  neumannZeroRhoYTag;
        std::string normalConvFluxName,
                    normalDiffFluxName;

        // build boundary conditions for x, y, and z faces
        switch( myBndSpec.face ) {
          case Uintah::Patch::xplus:
          case Uintah::Patch::xminus:
          {
            std::string dir = "X";
            typedef SpeciesBoundaryTyper<SpatialOps::SSurfXField, SpatialOps::GradientX> BCTypes;
            BCTypes bcTypes;

            normalConvFluxName = normalConvFluxNameBase + dir;
            normalDiffFluxName = normalDiffFluxNameBase + dir;

            neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "nuemann-zero", dir );
            neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "nuemann-zero", dir );
            neumannZeroRhoYTag           = get_decorated_tag( this->solnVarName_, "nuemann-zero", dir );

            if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroRhoYTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
            if( !initFactory  .have_entry( neumannZeroRhoYTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
          }
          break;
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
          {
            std::string dir = "Y";
            typedef SpeciesBoundaryTyper<SpatialOps::SSurfYField, SpatialOps::GradientY> BCTypes;
            BCTypes bcTypes;

            normalConvFluxName = normalConvFluxNameBase + dir;
            normalDiffFluxName = normalDiffFluxNameBase + dir;

            neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "nuemann-zero", dir );
            neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "nuemann-zero", dir );
            neumannZeroRhoYTag           = get_decorated_tag( this->solnVarName_, "nuemann-zero", dir );

            if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroRhoYTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
            if( !initFactory  .have_entry( neumannZeroRhoYTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
          }
          break;
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
          {
            std::string dir = "Z";
            typedef SpeciesBoundaryTyper<SpatialOps::SSurfZField, SpatialOps::GradientZ> BCTypes;
            BCTypes bcTypes;

            normalConvFluxName = normalConvFluxNameBase + dir;
            normalDiffFluxName = normalDiffFluxNameBase + dir;

            neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "nuemann-zero", dir );
            neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "nuemann-zero", dir );
            neumannZeroRhoYTag           = get_decorated_tag( this->solnVarName_, "nuemann-zero", dir );

            if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
            if( !advSlnFactory.have_entry( neumannZeroRhoYTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
            if( !initFactory  .have_entry( neumannZeroRhoYTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoYTag          , 0.0 ) );
          }
          break;
          default:
            break;
        }

        // make boundary condition specifications, connecting the name of the field (first) and the name of the modifier expression (second)
        BndCondSpec diffFluxSpec = {normalDiffFluxName, neumannZeroDiffFluxTag.name()      , 0.0, NEUMANN, FUNCTOR_TYPE};
        BndCondSpec convFluxSpec = {normalConvFluxName, neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
        BndCondSpec rhoYSpec     = {this->solnVarName_, neumannZeroRhoYTag.name()          , 0.0, NEUMANN, FUNCTOR_TYPE};

        // add boundary condition specifications to this boundary
        bcHelper.add_boundary_condition( bndName, diffFluxSpec );
        bcHelper.add_boundary_condition( bndName, convFluxSpec );
        bcHelper.add_boundary_condition( bndName, rhoYSpec     );
      }
      break;

      case VELOCITY:{

        Expr::Tag bcCopiedRhoYTag;

        switch( myBndSpec.face ) {
          case Uintah::Patch::xplus:
          case Uintah::Patch::xminus:
          {
            std::string dir = "X";
            typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
            bcCopiedRhoYTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
            if( !initFactory  .have_entry( bcCopiedRhoYTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
            if( !advSlnFactory.have_entry( bcCopiedRhoYTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
          }
          break;
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
          {
            std::string dir = "Y";
            typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
            bcCopiedRhoYTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
            if( !initFactory  .have_entry( bcCopiedRhoYTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
            if( !advSlnFactory.have_entry( bcCopiedRhoYTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
          }
          break;
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
          {
            std::string dir = "Z";
            typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
            bcCopiedRhoYTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
            if( !initFactory  .have_entry( bcCopiedRhoYTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
            if( !advSlnFactory.have_entry( bcCopiedRhoYTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoYTag, temporaryRhoYTag ) );
          }
          break;
          default:
            break;
        }

        BndCondSpec rhoYDirichletBC = {this->solnVarTag_.name(), bcCopiedRhoYTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
        bcHelper.add_boundary_condition( bndName, rhoYDirichletBC );
      }
      break;

      case USER:
      default:
        // do nothing
        break;
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
  if( factory.have_entry(primVarTag_) ){
    bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat, true );
  }
  // bcs for hard inflow - set primitive and conserved variables
  const Expr::Tag temporaryYTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
  bcHelper.apply_boundary_condition<FieldT>( temporaryYTag, taskCat, true );
}

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::apply_boundary_conditions( const GraphHelper& graphHelper,
                                                     WasatchBCHelper& bcHelper )
{
  const bool isLowMach = flowTreatment_ == LOWMACH;
  const bool setOnExtraOnly = isLowMach;
  const Category taskCat = ADVANCE_SOLUTION;
  bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
  bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell

  // bcs for hard inflow - set primitive and conserved variables
  bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat );
  const Expr::Tag temporaryYTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
  bcHelper.apply_boundary_condition<FieldT>( temporaryYTag, taskCat, true );

  const std::string normalConvFluxName_nodir = this->solnVarName_ + "_convFlux_";
  bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag(normalConvFluxName_nodir + 'X', Expr::STATE_NONE), taskCat, setOnExtraOnly);
  bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag(normalConvFluxName_nodir + 'Y', Expr::STATE_NONE), taskCat, setOnExtraOnly);
  bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag(normalConvFluxName_nodir + 'Z', Expr::STATE_NONE), taskCat, setOnExtraOnly);
  bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBC::SPECIES, taskCat, this->specNum_);

  // if the flow treatment is low-Mach, diffusive fluxes for STATE_NP1 are used to estimate div(u) so we must
  // set boundary conditions at STATE_NP1 as well as STATE_NONE when initial conditions are set
  const Expr::Context diffFluxContext = isLowMach ? Expr::STATE_NP1 : Expr::STATE_NONE;
  const std::string normalDiffFluxName_nodir = primVarTag_.name() + "_diffFlux_";
  bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag(normalDiffFluxName_nodir + 'X', diffFluxContext), taskCat, setOnExtraOnly);
  bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag(normalDiffFluxName_nodir + 'Y', diffFluxContext), taskCat, setOnExtraOnly);
  bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag(normalDiffFluxName_nodir + 'Z', diffFluxContext), taskCat, setOnExtraOnly);

  if(isLowMach){
//    const Category initCat = INITIALIZATION;
//    const Expr::Context initContext = Expr::STATE_NONE;
//    bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag(normalDiffFluxName_nodir + 'X', initContext), initCat, setOnExtraOnly);
//    bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag(normalDiffFluxName_nodir + 'Y', initContext), initCat, setOnExtraOnly);
//    bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag(normalDiffFluxName_nodir + 'Z', initContext), initCat, setOnExtraOnly);
  }
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

  Expr::ExpressionFactory& solnFactory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  Expr::ExpressionFactory& initFactory = *gc_[INITIALIZATION  ]->exprFactory;

  if( flowTreatment_ == COMPRESSIBLE ){
    solnFactory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarTag_, solnVarTag_, densityTag_) );
  }

  // for variable density flows:
  if( flowTreatment_ == LOWMACH ){
    infoNP1_ [PRIMITIVE_VARIABLE]  = primVarNP1Tag_;
    infoInit_[PRIMITIVE_VARIABLE]  = primVarInitTag_;

    EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ){
      infoNP1_ [VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
      infoNP1_ [AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
      infoNP1_ [AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
      infoNP1_ [AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();

      infoInit_[VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
      infoInit_[AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
      infoInit_[AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
      infoInit_[AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();
    }

    // put source term tag in infoInit_ if that tag is found in infoNP1_
    const FieldTagInfo::const_iterator ifld = infoNP1_.find( SOURCE_TERM );
    if( ifld != infoNP1_.end() )
    {
      infoInit_[SOURCE_TERM] =  Expr::Tag( ifld->second.name(), Expr::STATE_NONE);
    }

    if(isStrong_){
      solnFactory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarNP1Tag_, this->solnvar_np1_tag(), densityNP1Tag_ ) );
      solnFactory.register_expression( new typename Expr::PlaceHolder<FieldT>::Builder(primVarTag_) );
      persistentFields_.insert( primVarTag_.name() );
    }

    const Expr::Tag scalEOSTag(primVarTag_.name() + "_EOS_Coupling", Expr::STATE_NONE);
    const Expr::Tag dRhoDYiTag = tagNames.derivative_tag( densityTag_, primVarTag_ );

    typedef typename ScalarEOSCoupling<FieldT>::Builder ScalarEOSBuilder;
    // todo: check whether or not 'srcTags' should be for ScalarEOSBuilder at initialization
    solnFactory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoNP1_ , srcTags, densityNP1Tag_ , dRhoDYiTag, isStrong_) );
    initFactory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoInit_, srcTags, densityInitTag_, dRhoDYiTag, isStrong_) );

    // register an expression for divu. divu is just a constant expression to which we add the
    // necessary couplings from the scalars that represent the equation of state.
    typedef typename Expr::ConstantExpr<SVolField>::Builder ConstBuilder;

    if( !solnFactory.have_entry( tagNames.divu ) ) { // if divu has not been registered yet, then register it!
      solnFactory.register_expression( new ConstBuilder(tagNames.divu, 0.0)); // set the value to zero so that we can later add sources to it
    }
    if( !initFactory.have_entry( tagNames.divu ) ) {
      initFactory.register_expression( new ConstBuilder(tagNames.divu, 0.0));
    }

    solnFactory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
    initFactory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
  }

  typedef ScalarRHS<FieldT>::Builder RHS;
  return solnFactory.register_expression( scinew RHS( rhsTag_, info, srcTags, densityTag_, isConstDensity_, isStrong_, tagNames.drhodt ) );
}

//------------------------------------------------------------------------------

} // namespace WasatchCore
