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

#include <CCA/Components/Wasatch/Transport/SootParticleTransportEquation.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>

// includes for soot source terms
#include <CCA/Components/Wasatch/Expressions/TarAndSoot/SootAgglomerationRate.h>
#include <CCA/Components/Wasatch/Transport/TarAndSootInfo.h>

#include <Core/Exceptions/ProblemSetupException.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;

EqnTimestepAdaptorBase*
setup_soot_particle_equation( Uintah::ProblemSpecP params,
                              const TurbulenceParameters& turbParams,
                              const Expr::Tag densityTag,
                              GraphCategories& gc )
{
  const TagNames& tagNames = TagNames::self();
  Expr::ExpressionFactory& icFactory = *gc[INITIALIZATION  ]->exprFactory;

  proc0cout << "Setting up transport equation for soot particle number density" << std::endl;
  SootParticleTransportEquation* sootEqn = scinew SootParticleTransportEquation( params, gc, densityTag, turbParams);

  // default to zero mass fraction unless specified otherwise
  const Expr::Tag numberDensityTag = tagNames.sootParticleNumberDensity;
  if( !icFactory.have_entry( numberDensityTag ) ){
    icFactory.register_expression( scinew Expr::ConstantExpr<FieldT>::Builder(numberDensityTag ,0.0) );
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
        << "ERROR while setting initial conditions on soot particle number density" << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  return new EqnTimestepAdaptor<FieldT>(sootEqn);
}

//===========================================================================

SootParticleTransportEquation::
SootParticleTransportEquation( Uintah::ProblemSpecP params,
                      GraphCategories& gc,
                      const Expr::Tag densityTag,
                      const TurbulenceParameters& turbulenceParams )
: PseudospeciesTransportEquation<SVolField>( TagNames::self().sootParticleNumberDensity.name(),
                                             params, gc,
                                             densityTag,
                                             turbulenceParams, false ),
  densityTag_       ( densityTag                             ),
  sootAgglomRateTag_( TagNames::self().sootAgglomerationRate )
{
 setup();
}

//---------------------------------------------------------------------------

SootParticleTransportEquation::~SootParticleTransportEquation()
{}

//---------------------------------------------------------------------------

void
SootParticleTransportEquation::
setup_diffusive_flux( FieldTagInfo& info )
{

  typedef DiffusiveVelocity< SpatialOps::SSurfXField >::Builder XFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfYField >::Builder YFlux;
  typedef DiffusiveVelocity< SpatialOps::SSurfZField >::Builder ZFlux;

  Uintah::ProblemSpecP momentumSpec = params_->findBlock("MomentumEquations");
  std::string dummyName;
  Expr::Tag advVelocityTag;
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
SootParticleTransportEquation::
setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
{
  assert( srcTags.empty() );

  TagNames tagNames = TagNames::self();
  TarAndSootInfo tsInfo = TarAndSootInfo::self();

  // define tags we'll need here
  const std::string suffix = "_src_formation";
  const Expr::Tag particleSrcTag( primVarTag_.name() + suffix, Expr::STATE_NONE );
  const Expr::Tag tempTag         = tagNames.temperature;
  const Expr::Tag sootTag         = tagNames.soot;
  const Expr::Tag sootFormRateTag = tagNames.sootFormationRate;

  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

  // soot particle agglomeration rate (particles/second):
  typedef SootAgglomerationRate<FieldT>::Builder SootAgglomRate;
  factory.register_expression( scinew SootAgglomRate( sootAgglomRateTag_, densityTag_, tempTag, sootTag, primVarTag_, tsInfo.sootDensity ) );

  const double mwCarbon = 12.01e-3;  // kg/mol
  const double avagadro = 6.022e23;  // 1/mol

  const double incipientMass = tsInfo.cMin*mwCarbon/avagadro; // mass per incipient soot particle


  // calculate production rates of species participating in soot oxidation and
  typedef Expr::LinearFunction<FieldT>::Builder LinFun;

  // soot particle formation rate (particles/second):
  factory.register_expression( scinew LinFun( particleSrcTag , sootFormRateTag, 1./incipientMass, 0) );

  srcTags.push_back(particleSrcTag);
}

//---------------------------------------------------------------------------

Expr::ExpressionID
SootParticleTransportEquation::
setup_rhs( FieldTagInfo& info, Expr::TagList& srcTags )
{
  Expr::ExpressionID rhsID = PseudospeciesTransportEquation<FieldT>::setup_rhs( info, srcTags);

  // agglomeration is a sink term for the soot particle number density so it is not added to srcTags
  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  factory.attach_dependency_to_expression( sootAgglomRateTag_,
                                           rhs_tag(),
                                           Expr::SUBTRACT_SOURCE_EXPRESSION );
  return rhsID;
}

//---------------------------------------------------------------------------

} /* namespace WasatchCore */
