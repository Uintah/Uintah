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

#include <CCA/Components/Wasatch/Transport/SootTransportEquation.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>

// includes for soot source terms
#include <CCA/Components/Wasatch/Expressions/TarAndSoot/SootOxidationRate.h>
#include <CCA/Components/Wasatch/Transport/TarAndSootInfo.h>

#include <Core/Exceptions/ProblemSetupException.h>

// PoKiTT expressions that we require here
#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;

EqnTimestepAdaptorBase*
setup_soot_equation( Uintah::ProblemSpecP params,
                    const TurbulenceParameters& turbParams,
                    const Expr::Tag densityTag,
                    GraphCategories& gc )
{
  const TagNames& tagNames = TagNames::self();
  Expr::ExpressionFactory& icFactory = *gc[INITIALIZATION  ]->exprFactory;

  proc0cout << "Setting up transport equation for soot" << std::endl;
  SootTransportEquation* sootEqn = scinew SootTransportEquation( params, gc, densityTag, turbParams);

  // default to zero mass fraction unless specified otherwise
  const Expr::Tag sootTag = tagNames.soot;
  if( !icFactory.have_entry( sootTag ) ){
    icFactory.register_expression( scinew Expr::ConstantExpr<FieldT>::Builder(sootTag ,0.0) );
  }

  //_____________________________________________________
  // set up initial conditions on this equation
  try{
    GraphHelper* const icGraphHelper = gc[INITIALIZATION];
    icGraphHelper->rootIDs.insert( sootEqn->initial_condition( *icGraphHelper->exprFactory ) );
  }
  catch( std::runtime_error& e ){
    std::ostringstream msg;
    msg << e.what()
        << std::endl
        << "ERROR while setting initial conditions on soot mass fraction" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  return new EqnTimestepAdaptor<FieldT>(sootEqn);
}

//===========================================================================

SootTransportEquation::
SootTransportEquation( Uintah::ProblemSpecP params,
                      GraphCategories& gc,
                      const Expr::Tag densityTag,
                      const TurbulenceParameters& turbulenceParams )
: PseudospeciesTransportEquation<SVolField>( TagNames::self().soot.name(),
                                             params, gc,
                                             densityTag,
                                             turbulenceParams, false ),
  densityTag_       ( densityTag                                 ),
  sootOxRateTag_    ( TagNames::self().sootOxidationRate         ),
  sootFormRateTag_  ( TagNames::self().sootFormationRate         ),
  sootNumDensityTag_( TagNames::self().sootParticleNumberDensity )
{
  setup();
}

//---------------------------------------------------------------------------

SootTransportEquation::~SootTransportEquation()
{}

//---------------------------------------------------------------------------

void
SootTransportEquation::
setup_diffusive_flux( FieldTagInfo& info )
{
  typedef DiffusiveVelocity< SpatialOps::SSurfXField >::Builder XFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfYField >::Builder YFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfZField >::Builder ZFlux;

  Uintah::ProblemSpecP momentumSpec = params_->findBlock("MomentumEquations");
  std::string dummyName;
  const bool doX = momentumSpec->get( "X-Velocity", dummyName );
  const bool doY = momentumSpec->get( "Y-Velocity", dummyName );
  const bool doZ = momentumSpec->get( "Z-Velocity", dummyName );

  const double diffCoeff = 0.0; // soot  molecular diffusivity is effectively zero.

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
SootTransportEquation::
setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
{
  assert( srcTags.empty() );
  srcTags.push_back(sootFormRateTag_);

  TagNames tagNames = TagNames::self();
  TarAndSootInfo tsInfo = TarAndSootInfo::self();

  // get solution variable name for energy equation
  std::string e0Name;
  Uintah::ProblemSpecP e0Params = params_->findBlock("EnergyEquation");
  params_->findBlock("EnergyEquation")->get( "SolutionVariable", e0Name );

  if(!e0Params){
    std::ostringstream msg;
    msg << std::endl
        << "ERROR: An energy transport equation is required for soot transport" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  // define tags we'll need here
  const std::string suffix = "_src_soot_oxidation";
  const Expr::Tag o2Tag       ( "O2"           , Expr::STATE_NONE );
  const Expr::Tag coTag       ( "CO"           , Expr::STATE_NONE );
  const Expr::Tag o2SrcTag    ( "O2"   + suffix, Expr::STATE_NONE );
  const Expr::Tag coSrcTag    ( "CO"   + suffix, Expr::STATE_NONE );
  const Expr::Tag energySrcTag( e0Name + suffix, Expr::STATE_NONE );
  const Expr::Tag energyRHSTag( e0Name + "_rhs", Expr::STATE_NONE );
  const Expr::Tag mmwTag   = tagNames.mixMW;
  const Expr::Tag tempTag  = tagNames.temperature;
  const Expr::Tag pressTag = tagNames.pressure;

  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  if(!factory.have_entry(o2Tag) || !factory.have_entry(coTag)){
    std::ostringstream msg;
    msg << std::endl
        << "ERROR: transport equations for O2 and CO are required for soot transport" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  typedef SootOxidationRate<FieldT>::Builder SootOxRate;
  factory.register_expression( scinew SootOxRate( sootOxRateTag_, o2Tag, primVarTag_, sootNumDensityTag_,
                                                  mmwTag, pressTag, densityTag_, tempTag, tsInfo.sootDensity ) );


  // Calculate the O2 and CO source terms due to soot oxidation under the following assumptions:
  // - soot is 100% carbon
  // - oxidation reaction: C + O2 --> CO

  // get molecular weights of CO and O2
  Cantera::IdealGasMix* gas = CanteraObjects::get_gasmix();
  const std::vector<double>& mwVec = gas->molecularWeights();

  const double mwO2     = mwVec[gas->speciesIndex("O2")];
  const double mwCo     = mwVec[gas->speciesIndex("CO")];
  const double mwCarbon = 12.01;

  // calculate the mass of species produced per mass of soot reacted
  const double rO2  = -0.5*mwO2/mwCarbon; // O2 is consumed, so rO2 is negative
  const double rCo  =  mwCo/mwCarbon;

  // calculate production rates of species participating in soot oxidation and
  typedef Expr::LinearFunction<FieldT>::Builder LinFun;

  // O2 production rate:
  factory.register_expression( scinew LinFun( o2SrcTag , sootOxRateTag_, rO2 , 0) );
  factory.attach_dependency_to_expression( o2SrcTag,
                                           get_species_rhs_tag("O2"),
                                           Expr::ADD_SOURCE_EXPRESSION );

  // CO production rate:
  factory.register_expression( scinew LinFun( coSrcTag , sootOxRateTag_, rCo , 0) );
  factory.attach_dependency_to_expression( coSrcTag,
                                           get_species_rhs_tag("CO"),
                                           Expr::ADD_SOURCE_EXPRESSION );

  factory.register_expression( scinew LinFun( energySrcTag, sootOxRateTag_, tsInfo.sootHeatOfOxidation, 0) );
  factory.attach_dependency_to_expression( energySrcTag,
                                           energyRHSTag,
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );

}

//---------------------------------------------------------------------------

Expr::ExpressionID
SootTransportEquation::
setup_rhs( FieldTagInfo& info, Expr::TagList& srcTags )
{
  Expr::ExpressionID rhsID = PseudospeciesTransportEquation<FieldT>::setup_rhs( info, srcTags);

  // soot oxidation is a sink term for soot so it is not added to srcTags
  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  factory.attach_dependency_to_expression( sootOxRateTag_,
                                           rhs_tag(),
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );
  return rhsID;
}

//---------------------------------------------------------------------------

} /* namespace WasatchCore */
