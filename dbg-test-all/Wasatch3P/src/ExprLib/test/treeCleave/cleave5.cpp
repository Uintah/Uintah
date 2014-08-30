/**
 *  \file   cleave5.cpp
 *  \date   Aug 19, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */


#include <iostream>
#include <sstream>

#include <expression/ExprLib.h>
#include <expression/Functions.h>

#include "expressions.h"
#include <test/TestHelper.h>

using namespace Expr;
using namespace std;
using namespace SpatialOps;

typedef Expr::ExprPatch Patch;

int main()
{
  try{
    TestHelper status(true);

    Patch patch( 1 );

    ExpressionFactory exprFactory;

    const Tag tagA("A",STATE_NONE);
    const Tag tagB("B",STATE_NONE);
    const Tag tagC("C",STATE_NONE);
    const Tag tagD("D",STATE_NONE);
    const Tag tagE("E",STATE_NONE);
    const Tag tagF("F",STATE_NONE);

    typedef ConstantExpr<SingleValueField>::Builder ExprT;

    const ExpressionID id_A = exprFactory.register_expression( new ExprT(tagA,1.0) );
    const ExpressionID id_B = exprFactory.register_expression( new ExprT(tagB,2.0) );
    const ExpressionID id_C = exprFactory.register_expression( new ExprT(tagC,3.0) );
    const ExpressionID id_D = exprFactory.register_expression( new ExprT(tagD,4.0) );
    const ExpressionID id_E = exprFactory.register_expression( new ExprT(tagE,5.0) );
    const ExpressionID id_F = exprFactory.register_expression( new ExprT(tagF,6.0) );

    exprFactory.attach_dependency_to_expression( tagC, tagA, ADD_SOURCE_EXPRESSION );
    exprFactory.attach_dependency_to_expression( tagC, tagB, ADD_SOURCE_EXPRESSION );
    exprFactory.attach_dependency_to_expression( tagD, tagC, ADD_SOURCE_EXPRESSION );
    exprFactory.attach_dependency_to_expression( tagD, tagA, ADD_SOURCE_EXPRESSION );
    exprFactory.attach_dependency_to_expression( tagE, tagD, ADD_SOURCE_EXPRESSION );
    exprFactory.attach_dependency_to_expression( tagF, tagE, ADD_SOURCE_EXPRESSION );

    exprFactory.cleave_from_parents( id_C );

    ExpressionTree tree( id_A, exprFactory, patch.id(), "cleave5" );
    tree.insert_tree( id_B );

//tree.dump_vertex_properties(cout);

    {
      ofstream fout("cleave5_precleave.dot");
      tree.write_tree(fout);
    }
std::cout << "\n\n==================================================\n\n";
    // cleave the tree.
    typedef ExpressionTree::TreeList TreeList;
    const TreeList treeList = tree.split_tree();

    FieldManagerList fml;
    const FieldAllocInfo fieldInfo = patch.field_info();

    BOOST_FOREACH( const ExpressionTree::TreePtr tchild, treeList ){
      tchild->register_fields(fml);
      std::ostringstream nam;
      nam << tchild->name() << ".dot";
      std::ofstream fout( nam.str().c_str() );
      tchild->write_tree( fout );
    }

    status( treeList[0]->is_persistent( tagC ), "C persistence" );
    status( treeList[0]->is_persistent( tagD ), "D persistence" );
    status(!treeList[0]->is_persistent( tagE ), "E persistence" );
    status( treeList[0]->is_persistent( tagF ), "F persistence" );

    status( treeList[1]->is_persistent( tagA ), "A persistence" );
    status( treeList[1]->is_persistent( tagB ), "B persistence" );

    if( status.ok() ){
      cout << "PASS\n";
      return 0;
    }
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl;
  }
  std::cout << "\n\nFAIL!\n";
  return -1;
}

