/**
 *  \file   PollerTest.cpp
 *  \date   Dec 5, 2013
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

#include <expression/ExprLib.h>
#include <expression/Poller.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FieldComparisons.h>

#include <test/TestHelper.h>

#include <fstream>

typedef SpatialOps::SingleValueField FieldT;


class DummyPollWorker : public Expr::PollWorker
{
  const Expr::Tag tag_;
  int count_;
protected:
  bool operator()( Expr::FieldManagerList* fml )
  {
    using namespace SpatialOps;
    ++count_;
//    std::cout << "In DummyPollWorker : " << count_ << std::endl;
    if( count_ > 30 || isBlocked_ ){
//      std::cout << "DummyPollWorker -- DONE!!" << std::endl;
      count_ = 0;
      FieldT& f = fml->field_ref<FieldT>(tag_);
      f <<= 1.234;
      return true;
    }
    return false;
  }
public:
  DummyPollWorker( const Expr::Tag& tag ) : tag_(tag)
  {
    std::cout << "Building DummyPollWorker\n";
    count_=0;
  }
};



int main()
{
  using namespace SpatialOps;
  try{
    const Expr::Tag aTag("a",Expr::STATE_NONE);
    const Expr::Tag bTag("b",Expr::STATE_NONE);
    Expr::ExprPatch patch(1,1,1);
    Expr::ExpressionFactory factory;
    const Expr::ExpressionID aID = factory.register_expression( new Expr::ConstantExpr<FieldT>::Builder(aTag,1.0) );
    const Expr::ExpressionID bID = factory.register_expression( new Expr::ConstantExpr<FieldT>::Builder(bTag,2.0) );
    factory.attach_dependency_to_expression(bTag,aTag,Expr::ADD_SOURCE_EXPRESSION);
    factory.get_poller(bTag)->add_new( Expr::PollWorkerPtr( new DummyPollWorker(bTag) ) );
    Expr::ExpressionTree tree( aID, factory, 0 );

    Expr::FieldManagerList& fml = patch.field_manager_list();
    tree.register_fields( fml );
    fml.allocate_fields( patch.field_info() );

    std::cout << "Performing iterations of the graph...\n";
    for( size_t i=0; i<1000; ++i ){
        tree.execute_tree();
    }
    std::cout << "DONE.\nPerforming comparisons...\n";

    {
      std::ofstream of("tree.dot");
      tree.write_tree(of);
    }

    const FieldT& a = fml.field_ref<FieldT>(aTag);
    const FieldT& b = fml.field_ref<FieldT>(bTag);
    SpatFldPtr<FieldT> f2 = SpatialFieldStore::get<FieldT>(a);
    SpatFldPtr<FieldT> f3 = SpatialFieldStore::get<FieldT>(a);
    *f2 <<= 1.234; // expected value if the poller does its job.
    *f3 <<= 2.234;

    TestHelper status(true);
    status( field_equal_abs(b,*f2) && field_equal_abs(a,*f3), "approach 1" );

    // an alternative way to calculate the error...

    *f2 <<= abs( *f2 - b );
    *f3 <<= abs( *f3 - a );

    const double erra = field_max( *f2 );
    const double errb = field_max( *f3 );

    status( erra < 1e-13, "a" );
    status( errb < 1e-13, "b" );

    if( status.ok() ){
      std::cout << "PASS\n";
      return 0;
    }
  }
  catch( std::exception& err ){
    std::cout << err.what();
  }
  std::cout << "\nFAIL\n";
  return -1;
}
