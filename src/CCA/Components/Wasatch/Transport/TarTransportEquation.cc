/*
 * The MIT License
 *
 * Copyright (c) 2016-2018 The University of Utah
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

#include <CCA/Components/Wasatch/Transport/TarTransportEquation.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>

// includes for tar source terms
#include <CCA/Components/Wasatch/Expressions/TarAndSoot/TarOxidationRate.h>
#include <CCA/Components/Wasatch/Expressions/TarAndSoot/SootFormationRate.h>
#include <CCA/Components/Wasatch/Transport/TarAndSootInfo.h>

#include <Core/Exceptions/ProblemSetupException.h>

// PoKiTT expressions that we require here
#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;

EqnTimestepAdaptorBase*
setup_tar_equation( Uintah::ProblemSpecP params,
                    const TurbulenceParameters& turbParams,
                    const Expr::Tag densityTag,
                    GraphCategories& gc )
{
  const TagNames& tagNames = TagNames::self();
  Expr::ExpressionFactory& icFactory = *gc[INITIALIZATION  ]->exprFactory;

  proc0cout << "Setting up transport equation for tar" << std::endl;
  TarTransportEquation* tarEqn = scinew TarTransportEquation( params, gc, densityTag, turbParams);

    // default to zero mass fraction unless specified otherwise
    const Expr::Tag tarTag = tagNames.tar;
    if( !icFactory.have_entry( tarTag ) ){
      icFactory.register_expression( scinew Expr::ConstantExpr<FieldT>::Builder(tarTag ,0.0) );
    }

    //_____________________________________________________
    // set up initial conditions on this equation
    try{
      GraphHelper* const icGraphHelper = gc[INITIALIZATION];
      icGraphHelper->rootIDs.insert( tarEqn->initial_condition( *icGraphHelper->exprFactory ) );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
          << std::endl
          << "ERROR while setting initial conditions on tar mass fraction" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

  return new EqnTimestepAdaptor<FieldT>(tarEqn);
}

//===========================================================================

TarTransportEquation::
TarTransportEquation( Uintah::ProblemSpecP params,
                      GraphCategories& gc,
                      const Expr::Tag densityTag,
                      const TurbulenceParameters& turbulenceParams )
: PseudospeciesTransportEquation<SVolField>( TagNames::self().tar.name(),
                                             params, gc,
                                             densityTag,
                                             turbulenceParams ),
  densityTag_     ( densityTag                         ),
  tarOxRateTag_   ( TagNames::self().tarOxidationRate  ),
  sootFormRateTag_( TagNames::self().sootFormationRate )
{
 setup();
}

//---------------------------------------------------------------------------

TarTransportEquation::~TarTransportEquation()
{}

//---------------------------------------------------------------------------

void
TarTransportEquation::
setup_diffusive_flux( FieldTagInfo& info )
{
  typedef DiffusiveVelocity< SpatialOps::SSurfXField >::Builder XFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfYField >::Builder YFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfZField >::Builder ZFlux;

  TarAndSootInfo tsInfo = TarAndSootInfo::self();

  Uintah::ProblemSpecP momentumSpec = params_->findBlock("MomentumEquations");
  std::string dummyName;
  const bool doX = momentumSpec->get( "X-Velocity", dummyName );
  const bool doY = momentumSpec->get( "Y-Velocity", dummyName );
  const bool doZ = momentumSpec->get( "Z-Velocity", dummyName );


  const double diffCoeff = tsInfo.tarDiffusivity;

  const TagNames& tagNames = TagNames::self();
  const Expr::Tag xDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "x", Expr::STATE_NONE );
  const Expr::Tag yDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "y", Expr::STATE_NONE );
  const Expr::Tag zDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "z", Expr::STATE_NONE );

  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  if( doX ){ info[DIFFUSIVE_FLUX_X]=xDiffFluxTag; factory.register_expression( scinew XFlux( xDiffFluxTag, primVarTag_, diffCoeff, turbDiffTag_ ) ); }
  if( doY ){ info[DIFFUSIVE_FLUX_Y]=yDiffFluxTag; factory.register_expression( scinew YFlux( yDiffFluxTag, primVarTag_, diffCoeff, turbDiffTag_ ) ); }
  if( doZ ){ info[DIFFUSIVE_FLUX_Z]=zDiffFluxTag; factory.register_expression( scinew ZFlux( zDiffFluxTag, primVarTag_, diffCoeff, turbDiffTag_ ) ); }

  // if doing convection, we will likely have a pressure solve that requires
  // predicted scalar values to approximate the density time derivatives
  if( params_->findBlock("MomentumEquations") ){
    const std::string& suffix = tagNames.star;
    const Expr::Tag xFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "x", Expr::STATE_NONE    );
    const Expr::Tag yFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "y", Expr::STATE_NONE    );
    const Expr::Tag zFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "z", Expr::STATE_NONE    );
    const Expr::Tag primVarTagNew( primVarTag_.name()  + suffix, Expr::STATE_NONE    );
    if( doX ){ infoStar_[DIFFUSIVE_FLUX_X]=xFluxTagNew; factory.register_expression( scinew XFlux( xFluxTagNew, primVarTagNew, diffCoeff, turbDiffTag_ ) ); }
    if( doY ){ infoStar_[DIFFUSIVE_FLUX_Y]=yFluxTagNew; factory.register_expression( scinew YFlux( yFluxTagNew, primVarTagNew, diffCoeff, turbDiffTag_ ) ); }
    if( doZ ){ infoStar_[DIFFUSIVE_FLUX_Z]=zFluxTagNew; factory.register_expression( scinew ZFlux( zFluxTagNew, primVarTagNew, diffCoeff, turbDiffTag_ ) ); }
  }
}

//---------------------------------------------------------------------------

void
TarTransportEquation::
setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
{
  assert( srcTags.empty() );

  TagNames tagNames = TagNames::self();
  TarAndSootInfo tsInfo = TarAndSootInfo::self();

  // get solution variable name for energy equation
  std::string e0Name;
  Uintah::ProblemSpecP e0Params = params_->findBlock("EnergyEquation");
  params_->findBlock("EnergyEquation")->get( "SolutionVariable", e0Name );

  if(!e0Params){
    std::ostringstream msg;
    msg << std::endl
        << "ERROR: An energy transport equation is required for tar transport" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  // define tags we'll need here
  const std::string suffix = "_src_tar_oxidation";
  const Expr::Tag o2Tag       ( "O2"           , Expr::STATE_NONE );
  const Expr::Tag coTag       ( "CO"           , Expr::STATE_NONE );
  const Expr::Tag h2oTag      ( "H2O"          , Expr::STATE_NONE );
  const Expr::Tag o2SrcTag    ( "O2"   + suffix, Expr::STATE_NONE );
  const Expr::Tag coSrcTag    ( "CO"   + suffix, Expr::STATE_NONE );
  const Expr::Tag h2oSrcTag   ( "H2O"  + suffix, Expr::STATE_NONE );
  const Expr::Tag energySrcTag( e0Name + suffix, Expr::STATE_NONE );
  const Expr::Tag energyRHSTag( e0Name + "_rhs", Expr::STATE_NONE );
  const Expr::Tag tempTag  = tagNames.temperature;

  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  if(!factory.have_entry(o2Tag) || !factory.have_entry(coTag) || !factory.have_entry(h2oTag)){
    std::ostringstream msg;
    msg << std::endl
        << "ERROR: transport equations for O2, CO, and H2O are required for tar transport" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }



  // calculate rate of tar formation and subtract it from tar RHS, add to soot RHS
  typedef SootFormationRate<FieldT>::Builder SootFormRate;
  factory.register_expression( scinew SootFormRate(sootFormRateTag_, primVarTag_, densityTag_, tempTag) );

  typedef TarOxidationRate<FieldT>::Builder TarOxRate;
  factory.register_expression( scinew TarOxRate(tarOxRateTag_, densityTag_, o2Tag, primVarTag_, tempTag) );


  // calculate the rate at which O$_{2}$ and CO source terms due to tar oxidation under the following assumptions:
  // - tar is composed completely of carbon and hydrogen ($C_{x}H_{y}$)
  // - oxidation reaction: $C_{x}H_{y} + (x/2 + y/4)*O2 \rightarrow (x)CO + (y/2)H_{2}O$

  // get stochiometric coefficients for elements in tar
  const double x = tsInfo.tarCarbon;
  const double y = tsInfo.tarHydrogen;

  // get molecular weights of CO, O2, and H2O
  Cantera::IdealGasMix* gas = CanteraObjects::get_gasmix();
  const std::vector<double>& mwVec = gas->molecularWeights();

  const double mwO2  = mwVec[gas->speciesIndex("O2" )];
  const double mwCo  = mwVec[gas->speciesIndex("CO" )];
  const double mwH2o = mwVec[gas->speciesIndex("H2O")];
  const double mwTar = tsInfo.tarMW;

  // calculate the mass of species produced per mass of tar reacted
  const double rO2  = -(x/2. + y/4.)*mwO2/mwTar; // O2 is consumed, so rO2 is negative
  const double rCo  =  x*mwCo/mwTar;
  const double rH2o =  (y/2.)*mwH2o/mwTar;

  // calculate production rates of species participating in tar oxidation and
  typedef Expr::LinearFunction<FieldT>::Builder LinFun;

  // O2 production rate:
  factory.register_expression( scinew LinFun( o2SrcTag , tarOxRateTag_, rO2 , 0) );
  factory.attach_dependency_to_expression( o2SrcTag,
                                           get_species_rhs_tag("O2"),
                                           Expr::ADD_SOURCE_EXPRESSION );

  // CO production rate:
  factory.register_expression( scinew LinFun( coSrcTag , tarOxRateTag_, rCo , 0) );
  factory.attach_dependency_to_expression( coSrcTag,
                                           get_species_rhs_tag("CO"),
                                           Expr::ADD_SOURCE_EXPRESSION );

  // H2O production rate:
  factory.register_expression( scinew LinFun( h2oSrcTag, tarOxRateTag_, rH2o, 0) );
  factory.attach_dependency_to_expression( h2oSrcTag,
                                           get_species_rhs_tag("H2O"),
                                           Expr::ADD_SOURCE_EXPRESSION );

  factory.register_expression( scinew LinFun( energySrcTag, tarOxRateTag_, tsInfo.tarHeatOfOxidation, 0) );
  factory.attach_dependency_to_expression( energySrcTag,
                                           energyRHSTag,
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );

  CanteraObjects::restore_gasmix(gas);
}

//---------------------------------------------------------------------------

Expr::ExpressionID
TarTransportEquation::
setup_rhs( FieldTagInfo& info, Expr::TagList& srcTags )
{
  Expr::ExpressionID rhsID = PseudospeciesTransportEquation<FieldT>::setup_rhs( info, srcTags);

  // soot formation and tar oxidation are both sink terms for tar so they are not added to srcTags
  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  factory.attach_dependency_to_expression( sootFormRateTag_,
                                           rhs_tag(),
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );

  factory.attach_dependency_to_expression( tarOxRateTag_,
                                           rhs_tag(),
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );

  return rhsID;
}

//---------------------------------------------------------------------------

} /* namespace WasatchCore */
