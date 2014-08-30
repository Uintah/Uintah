/**
 *  \file   cleave4.cpp
 *  \date   Mar 29, 2013
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

#include <test/TestHelper.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Expr;
using namespace std;

typedef Expr::ExprPatch Patch;

int main()
{
  try{
    TestHelper status;
    Patch patch( 1 );

    ExpressionFactory exprFactory;

    typedef SpatialOps::SVolField FieldT;
    typedef ConstantExpr  <FieldT>::Builder Constant;
    typedef LinearFunction<FieldT>::Builder Linear;

    const Tag phi1Tag ( "phi1",  Expr::STATE_NONE );
    const Tag phi2Tag ( "phi2",  Expr::STATE_NONE );
    const Tag alphaTag( "alpha", Expr::STATE_NONE );
    const Tag betaTag ( "beta",  Expr::STATE_NONE );

    const ExpressionID phi1ID  = exprFactory.register_expression( new Constant(phi1Tag, 1.0) );
    const ExpressionID phi2ID  = exprFactory.register_expression( new Constant(phi2Tag, 2.0) );
    const ExpressionID alphaID = exprFactory.register_expression( new Linear(alphaTag,betaTag,1.0,3.0) );
    const ExpressionID betaID  = exprFactory.register_expression( new Constant(betaTag, 4.0) );

    exprFactory.attach_dependency_to_expression( alphaTag, phi1Tag );
    exprFactory.attach_dependency_to_expression( alphaTag, phi2Tag );

    exprFactory.cleave_from_children( phi1ID );

    ExpressionTree::RootIDList ids;  ids.insert( phi1ID ); ids.insert( phi2ID );
    ExpressionTree tree( ids, exprFactory, patch.id() );

    {
      ofstream fout("tree4.dot");
      tree.write_tree(fout);
    }
    {
      ofstream fout("tree4T.dot");
      tree.write_tree(fout,true);
    }

    FieldManagerList fml;
    const FieldAllocInfo fieldInfo = patch.field_info();
    tree.register_fields( fml );
    fml.allocate_fields( fieldInfo );
    tree.bind_fields( fml );
    tree.execute_tree();

    {
      FieldT& phi1 = fml.field_ref<FieldT>(phi1Tag);
      FieldT& phi2 = fml.field_ref<FieldT>(phi2Tag);

#     ifdef ENABLE_CUDA
      phi1.add_device(CPU_INDEX);
      phi1.set_device_as_active(CPU_INDEX);
      phi2.add_device(CPU_INDEX);
      phi2.set_device_as_active(CPU_INDEX);
#     endif

      status( phi1[0] == 8, "pre-cleave field value 1" );
      status( phi2[0] == 9, "pre-cleave field value 2" );

#     ifdef ENABLE_CUDA
      phi1.set_device_as_active(GPU_INDEX);
      phi2.set_device_as_active(GPU_INDEX);
#     endif
    }

    ExpressionTree::TreeList trees = tree.split_tree();
    unsigned inum = 1;
    BOOST_FOREACH( ExpressionTree::TreePtr treePtr, trees ){
      ostringstream fnam;
      fnam << "tree4_" << inum++ << ".dot";
      ofstream fout( fnam.str().c_str() );
      treePtr->write_tree(fout);
      treePtr->register_fields( fml );
      fml.allocate_fields( fieldInfo );
      treePtr->bind_fields( fml );
      treePtr->execute_tree();
    }

    {
      FieldT& phi1 = fml.field_ref<FieldT>(phi1Tag);
      FieldT& phi2 = fml.field_ref<FieldT>(phi2Tag);
#     ifdef ENABLE_CUDA
      // move fields to CPU and activate them there so we can check values below.
      phi1.validate_device(CPU_INDEX);
      phi2.validate_device(CPU_INDEX);
      phi1.set_device_as_active(CPU_INDEX);
      phi2.set_device_as_active(CPU_INDEX);
#     endif

      status( phi1[0] == 8, "post-cleave field value 1" );
      status( phi2[0] == 9, "post-cleave field value 2" );
    }

    if( status.ok() ){
      cout << "PASS" << endl;
      return 0;
    }
  }
  catch( exception& err ){
    cout << err.what() << endl;
  }
  cout << "FAIL" << endl;
  return -1;
}
