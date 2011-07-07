#ifndef Wasatch_BasicExprBuilder_h
#define Wasatch_BasicExprBuilder_h

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>

/**
 *  \file BasicExprBuilder.h
 *  \brief parser support for creating some basic expressions.
 */


namespace Expr{
  class ExpressionBuilder;
}



namespace Wasatch{


  /**
   *  \ingroup WasatchParser
   *  \brief Creates expressions from the ones explicitly defined in the input file
   *  \param parser the Uintah::ProblemSpec block that contains <BasicExpression> tags
   *  \param gc the GraphCategories object that this expression should be associated with.
   */
  void
  create_expressions_from_input( Uintah::ProblemSpecP parser,
                                 GraphCategories& gc );

} // namespace Wasatch


#endif // Wasatch_BasicExprBuilder_h
