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

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/TestLowMachSpeciesDensitySolver.h>

//--- Local (Wasatch) includes ---//
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>

#include <expression/Functions.h>

#include <pokitt/CanteraObjects.h>
#include <pokitt/SpeciesN.h>
#include <pokitt/thermo/Density.h>
#include <pokitt/thermo/Temperature.h>
#include <pokitt/thermo/Enthalpy.h>
#include <pokitt/MixtureMolWeight.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromSpeciesAndEnthalpy.h>
namespace WasatchCore{

void test_low_mach_species_density_solver( Uintah::ProblemSpecP&  params,
                                           GraphCategories&       gc,
                                           std::set<std::string>& persistentFields )
{
  Expr::ExpressionID id;

  std::string canteraInputFileName, canteraGroupName;

  double rTol;           // residual tolerance
  int    maxIter;        // maximum number of iterations

  params->get("CanteraInputFile"  , canteraInputFileName);
  params->get("CanteraGroup"      , canteraGroupName    );
  params->get("tolerance"         , rTol                );
  params->get("maxIterations"     , maxIter             );

  CanteraObjects::Setup canteraSetup( "Mix", canteraInputFileName, canteraGroupName );
  CanteraObjects::setup_cantera( canteraSetup );


  GraphHelper& gh  = *gc[ADVANCE_SOLUTION];

  const int nspec = CanteraObjects::number_species();
  const TagNames& tagNames = TagNames::self();

  typedef Expr::ConstantExpr<SVolField>::Builder ConstBuilder;

  Expr::TagList yiExactTags;
  for( int i=0; i<nspec; ++i ){
    const std::string specName = CanteraObjects::species_name(i);
    const Expr::Tag specTag( specName, Expr::STATE_NONE );
    yiExactTags.push_back( specTag );

    if( i == nspec-1 ) continue; // don't build the nth species equation

    // default to zero mass fraction for species unless specified otherwise
    if( !gh.exprFactory->have_entry( specTag ) ){
      gh.exprFactory->register_expression( scinew ConstBuilder(specTag, 0.) );
    }
  }

  // nth species
  const Expr::Tag nthSpecTag( CanteraObjects::species_name(nspec-1), Expr::STATE_NONE );
  gh.exprFactory->
  register_expression( scinew pokitt::SpeciesN<SVolField>::Builder( nthSpecTag,
                                                                    yiExactTags,
                                                                    pokitt::CLIPSPECN ));


  const Expr::Tag temperatureExactTag = tagNames.temperature;
  const Expr::Tag pressureTag         = tagNames.thermodynamicPressure;

  // mixture molecular weight
  const Expr::Tag mmwTag = tagNames.mixMW;
  gh.exprFactory->
  register_expression( scinew pokitt::MixtureMolWeight<SVolField>::Builder( mmwTag,
                                                                            yiExactTags,
                                                                            pokitt::MASS ));

  // true density
  const Expr::Tag densityExactTag("rho", Expr::STATE_NONE);
  gh.exprFactory->
  register_expression( scinew pokitt::Density<SVolField>::Builder( densityExactTag,
                                                                   temperatureExactTag,
                                                                   pressureTag,
                                                                   mmwTag ));

  // true enthalpy
  const Expr::Tag enthalpyExactTag = tagNames.enthalpy;
  gh.exprFactory->
  register_expression( scinew pokitt::Enthalpy<SVolField>::Builder( enthalpyExactTag,
                                                                    temperatureExactTag,
                                                                    yiExactTags ));

  // density-weighted species mass fractions
  typedef ExprAlgebra<SVolField> Algebra;
  Expr::TagList rhoYiTags, yiGuessTags, yiCalcTags;
  for( int i=0; i<nspec; ++i ){
    const Expr::Tag& yiTrueTag = yiExactTags[i];

    yiGuessTags.push_back(Expr::Tag(yiTrueTag.name() + "_guess", Expr::STATE_NONE) );
    yiCalcTags .push_back(Expr::Tag(yiTrueTag.name() + "_calc" , Expr::STATE_NONE) );

    if(i == nspec-1) continue;

    const Expr::Tag rhoYiTag("rho_" + yiTrueTag.name(), Expr::STATE_NONE);
    rhoYiTags.push_back(rhoYiTag);
    persistentFields.insert(rhoYiTag.name());
    gh.exprFactory->
    register_expression( scinew Algebra::Builder( rhoYiTag,
                                                  Expr::tag_list(densityExactTag, yiTrueTag),
                                                  Algebra::PRODUCT ));

    if( !gh.exprFactory->have_entry( yiGuessTags[i] ) ){
      gh.exprFactory->register_expression( scinew ConstBuilder(yiGuessTags[i], 0.) );
    }
  }

  gh.exprFactory->
  register_expression( scinew pokitt::SpeciesN<SVolField>::Builder( yiGuessTags[nspec-1],
                                                                    yiGuessTags ));

  // perturbed temperature
  const Expr::Tag temperatureGuessTag = Expr::Tag( temperatureExactTag.name() + "_guess", Expr::STATE_NONE);

  // perturbed molecular_weight
  const Expr::Tag mmwGuessTag = Expr::Tag( mmwTag.name() + "_guess", Expr::STATE_NONE);
  gh.exprFactory->
  register_expression( scinew pokitt::MixtureMolWeight<SVolField>::Builder( mmwGuessTag,
                                                                            yiGuessTags,
                                                                            pokitt::MASS ));
  // perturbed enthalpy
  const Expr::Tag enthalpyGuessTag = Expr::Tag( enthalpyExactTag.name() + "_guess", Expr::STATE_NONE);
  gh.exprFactory->
  register_expression( scinew pokitt::Enthalpy<SVolField>::Builder( enthalpyGuessTag,
                                                                    temperatureGuessTag,
                                                                    yiGuessTags ));                                                                 

  // perturbed density
  const Expr::Tag densityGuessTag("rho_guess", Expr::STATE_NONE);
  gh.exprFactory->
  register_expression( scinew pokitt::Density<SVolField>::Builder( densityGuessTag,
                                                                   temperatureGuessTag,
                                                                   pressureTag,
                                                                   mmwGuessTag ));                                                                      


  // density-weighted enthalpy
  const Expr::Tag rhoEnthalpyTag("rho_" + enthalpyExactTag.name(), Expr::STATE_NONE);
  persistentFields.insert(rhoEnthalpyTag.name());
  gh.exprFactory->
  register_expression( scinew Algebra::Builder( rhoEnthalpyTag,
                                                Expr::tag_list(densityExactTag, enthalpyExactTag),
                                                Algebra::PRODUCT ));

  Expr::TagList dRhodYiTags;
  for( int i=0; i<nspec-1; ++i ){
    dRhodYiTags.push_back( tagNames.derivative_tag(densityExactTag.name(), yiExactTags[i].name()) );
  }
  Expr::TagList dRhodPhiTags = dRhodYiTags;
  
  const Expr::Tag dRhodHTag = tagNames.derivative_tag(densityExactTag.name(), tagNames.enthalpy.name());
  dRhodPhiTags.push_back( dRhodHTag );

  for(const Expr::Tag& tag : dRhodPhiTags) persistentFields.insert( tag.name() );

  // calculate rho, temperature
  const Expr::Tag densityCalcTag     = Expr::Tag(densityExactTag    .name() + "_calc", Expr::STATE_NONE);
  const Expr::Tag temperatureCalcTag = Expr::Tag(temperatureExactTag.name() + "_calc", Expr::STATE_NONE);

  typedef DensityFromSpeciesAndEnthalpy<SVolField>::Builder DensBuilder;
  gh.exprFactory->
  register_expression(new DensBuilder( densityCalcTag,
                                       temperatureCalcTag,
                                       dRhodYiTags,
                                       dRhodHTag,
                                       densityGuessTag,
                                       rhoYiTags,
                                       rhoEnthalpyTag,
                                       yiGuessTags,
                                       enthalpyGuessTag,
                                       temperatureGuessTag,
                                       mmwGuessTag,
                                       pressureTag,
                                       rTol,
                                       maxIter ) );

  typedef typename PrimVar<SVolField,SVolField>::Builder PrimVar;
  for( int i=0; i<nspec-1; ++i ){
    gh.exprFactory->
      register_expression(scinew PrimVar(yiCalcTags[i], rhoYiTags[i], densityCalcTag) );

  }
  gh.exprFactory->
  register_expression( scinew pokitt::SpeciesN<SVolField>::Builder( yiCalcTags[nspec-1],
                                                                    yiCalcTags ));


  typedef typename Expr::LinearFunction<SVolField>::Builder LinFun;
  const Expr::Tag densityErrorTag = Expr::Tag(densityExactTag.name() + "_error", Expr::STATE_NONE);

  id =
  gh.exprFactory->
  register_expression( scinew LinFun(densityErrorTag, densityCalcTag, 1., 0.) );
  gh.rootIDs.insert(id);

  gh.exprFactory->attach_dependency_to_expression(densityExactTag, densityErrorTag, Expr::SUBTRACT_SOURCE_EXPRESSION);

  const Expr::Tag temperatureErrorTag = Expr::Tag(temperatureExactTag.name() + "_error", Expr::STATE_NONE);

  id =
  gh.exprFactory->
  register_expression( scinew LinFun(temperatureErrorTag, temperatureCalcTag, 1., 0.) );
  gh.rootIDs.insert(id);

  gh.exprFactory->attach_dependency_to_expression(temperatureExactTag, temperatureErrorTag, Expr::SUBTRACT_SOURCE_EXPRESSION);

  for( int i=0; i<nspec; ++i ){
    const Expr::Tag yiErrorTag = Expr::Tag(yiExactTags[i].name() + "_error", Expr::STATE_NONE);

    id =
    gh.exprFactory->
    register_expression( scinew LinFun(yiErrorTag, yiCalcTags[i], 1., 0.) );
    gh.rootIDs.insert(id);

    gh.exprFactory->attach_dependency_to_expression(yiExactTags[i], yiErrorTag, Expr::SUBTRACT_SOURCE_EXPRESSION);
  }

}

//---------------------------------------------------------------------------------------------------------

} // namespace WasatchCore
