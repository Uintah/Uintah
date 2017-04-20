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

#include <sci_defs/wasatch_defs.h>

#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Transport/ParticleTemperatureEquation.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleConvectiveHeatTransferCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleTemperatureSrc_convection.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <expression/ExprLib.h>


namespace WasatchCore{

  //============================================================================

  ParticleTemperatureEquation::
  ParticleTemperatureEquation( const std::string&   solnVarName,
                               const Expr::TagList& particlePositionTags,
                               const Expr::Tag&     particleSizeTag,
                               const Expr::Tag&     particleMassTag,
                               const Expr::Tag&     gasViscosityTag,
                               Uintah::ProblemSpecP WasatchSpec,
                               Uintah::ProblemSpecP particleTempSpec,
                               GraphCategories&     gc )
  : ParticleEquationBase( solnVarName, NODIR, particlePositionTags, particleSizeTag, gc ),
    implementCoalModels_( WasatchSpec->findBlock("Coal") ),
    pSrcTag_( parse_nametag(particleTempSpec->findBlock("SourceTerm")) ),
    pMassTag_( particleMassTag ),
    pSizeTag_( particleSizeTag ),
    gViscTag_( gasViscosityTag ),
    pPosTags_( particlePositionTags )
  {
#   ifndef HAVE_POKITT
    if( implementCoalModels_ ){
      std::ostringstream msg;
      msg << "ERROR: Coal models require PoKiTT\n";
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
#   endif
    setup();
  }

  //------------------------------------------------------------------

  void ParticleTemperatureEquation::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }

  //------------------------------------------------------------------

  Expr::ExpressionID ParticleTemperatureEquation::setup_rhs()
  {
    if( !implementCoalModels_ ){
      std::ostringstream msg;
      msg << "Temperature solve without coal model implementation not supported:"
          << std::endl;
      throw std::runtime_error( msg.str() );
    }

    const TagNames& tagNames         = TagNames::self();
    const Expr::Tag pTempSrcConvTag(rhs_name() + "_convection", Expr::STATE_NONE);

    // register necessary expressions
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    typedef Expr::ConstantExpr<ParticleField>::Builder RHSBuilder;

    const Expr::ExpressionID
    rhsID = factory.register_expression( scinew RHSBuilder(rhsTag_, 0.0 ) );

    typedef ParticleConvectiveHeatTransferCoefficient<SVolField>::Builder pHTCBuilder;
    factory.register_expression( scinew pHTCBuilder( tagNames.pHeatTransCoef,
                                                    pPosTags_,
                                                    tagNames.preynolds,
                                                    pSizeTag_,
                                                    tagNames.heatCapacity,
                                                    gViscTag_,
                                                    tagNames.thermalConductivity) );

    typedef ParticleTemperatureSrc_convection<SVolField>::Builder pTSrcConvBuilder;
    factory.register_expression( scinew pTSrcConvBuilder( pTempSrcConvTag,
                                                          tagNames.pHeatTransCoef,
                                                          pMassTag_,
                                                          solution_variable_tag(),
                                                          pSizeTag_,
                                                          tagNames.pHeatCapacity,
                                                          tagNames.temperature ) );

    factory.attach_dependency_to_expression( pTempSrcConvTag,
                                             rhs_tag(),
                                             Expr::ADD_SOURCE_EXPRESSION);

    return rhsID;
  }

  //------------------------------------------------------------------

  ParticleTemperatureEquation::~ParticleTemperatureEquation()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ParticleTemperatureEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    return exprFactory.get_id( initial_condition_tag() );
  }

  //==================================================================

} // namespace WasatchCore
