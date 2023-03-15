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
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

// PoKiTT expressions that we require here
#include <pokitt/CanteraObjects.h>
#include <pokitt/kinetics/ReactionRates.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for species transport

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::setup_source_terms( FieldTagInfo& info, Expr::TagList& tags )
{
  if( !params_->findBlock("DetailedKinetics") ) return;

  if( turbParams_.turbModelName != TurbulenceParameters::NOTURBULENCE ){
    throw Uintah::ProblemSetupException( "Closure for reaction source terms is not supported\n", __FILE__, __LINE__ );
  }

  if( specNum_ == 0 ){
    Expr::TagList srcTags, srcNP1Tags, srcInitTags;
    for( int i=0; i<nspec_; ++i ){
      const std::string srcName = "rr_"+CanteraObjects::species_name(i);
      srcTags    .push_back( Expr::Tag( srcName, primVarTag_    .context() ) );
      srcNP1Tags .push_back( Expr::Tag( srcName, primVarNP1Tag_ .context() ) );
      srcInitTags.push_back( Expr::Tag( srcName, primVarInitTag_.context() ) );
    }
    Expr::ExpressionFactory& factory   = *gc_[ADVANCE_SOLUTION]->exprFactory;
    Expr::ExpressionFactory& icFactory = *gc_[INITIALIZATION  ]->exprFactory;
    typedef pokitt::ReactionRates<FieldT>::Builder RxnRates;
    if( flowTreatment_ == LOWMACH ){
      const Expr::ExpressionID id =
      icFactory.register_expression( scinew RxnRates( srcInitTags, temperatureTag_, densityInitTag_, yiInitTags_, mmwTag_, jacobian_ ) );
      factory  .register_expression( scinew RxnRates( srcNP1Tags , temperatureTag_, densityNP1Tag_ , yiNP1Tags_ , mmwTag_, jacobian_ ) );

      gc_[INITIALIZATION]->rootIDs.insert(id);

      typedef Expr::PlaceHolder<FieldT>::Builder PlaceHolder;
      for( const Expr::Tag& srcTag : srcTags ){
        factory.register_expression( scinew PlaceHolder( srcTag ) );
        persistentFields_.insert( srcTag.name() );
      }
    }

    else{
      factory.register_expression( scinew RxnRates( srcTags, temperatureTag_, densityTag_, yiTags_, mmwTag_, jacobian_ ) );
    }

    // todo: consider what needs to be done if both the low-Mach model and dual timestepping are enabled
    dualTimeMatrixInfo_.set_production_rates( srcTags );
  }
  const std::string srcName = "rr_"+primVarTag_.name();
  info[SOURCE_TERM] = Expr::Tag( srcName, primVarTag_.context() );

  if( flowTreatment_ == LOWMACH ){
    infoNP1_[SOURCE_TERM] = Expr::Tag( srcName, primVarNP1Tag_.context() );
  }
}

//------------------------------------------------------------------------------

} // namespace WasatchCore
