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

#include <CCA/Components/Wasatch/Expressions/TestNestedExpression.h>

//--- Local (Wasatch) includes ---//
#include <CCA/Components/Wasatch/Expressions/NestedExpression.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/ParseTools.h>

#include <expression/Functions.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace WasatchCore{

void test_nested_expression( Uintah::ProblemSpecP&  params,
                             GraphCategories&       gc,
                             std::set<std::string>& persistentFields )
{
  GraphHelper& gh  = *gc[ADVANCE_SOLUTION];

  const Expr::Tag resultTag = parse_nametag( params->findBlock("NameTag") );

  unsigned numIter = 100;
  params->get("numIterations", numIter);

  Uintah::ProblemSpecP  inputParams = params->findBlock("dependentVariables");

  Expr::TagList inputTags;
  for( Uintah::ProblemSpecP tagParams = inputParams->findBlock("NameTag");
      tagParams != nullptr;
      tagParams = tagParams->findNextBlock("NameTag") ) {
    inputTags.push_back( parse_nametag( tagParams ) );
  }

  typedef NestedExpression<SpatialOps::SVolField>::Builder NestedBuilder;

  Expr::ExpressionID id =
  gh.exprFactory->
  register_expression( scinew NestedBuilder(resultTag, inputTags, numIter) );

  gh.rootIDs.insert(id);
  persistentFields.insert(resultTag.name());

}

} // namespace WasatchCore
