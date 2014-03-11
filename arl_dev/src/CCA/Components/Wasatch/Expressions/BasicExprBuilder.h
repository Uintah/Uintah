/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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
   *  \addtogroup WasatchParser
   *  \addtogroup Expressions
   *
   *  \brief Creates expressions from the ones explicitly defined in the input file
   *
   *  \param parser - the Uintah::ProblemSpec block that contains \verbatim <BasicExpression> \endverbatim tags
   *  \param gc - the GraphCategories object that this expression should be associated with.
   */
  void
  create_expressions_from_input( Uintah::ProblemSpecP parser,
                                 GraphCategories& gc );

} // namespace Wasatch


#endif // Wasatch_BasicExprBuilder_h
