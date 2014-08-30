/**g
 *  \file   agmultipleFieldModifier.cpp
 *  \date   Apr 19, 2014
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2014 The University of Utah
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
#include <expression/Functions.h>

#include "Modifier.h"
#include <test/TestHelper.h>

using namespace std;
using namespace Expr;

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>

typedef SpatialOps::SVolField FieldT;


class MultipleField
 : public Expression<FieldT>
{
  const vector<double> values_;

  MultipleField( const vector<double>& values )
    : Expression<FieldT>(),
      values_( values )
  {}

public:
  class Builder : public ExpressionBuilder
  {
  public:
    Builder( const TagList& resultTags )
      : ExpressionBuilder( resultTags )
    {
      double val = 0.0;
      BOOST_FOREACH( const Tag& t, resultTags ){
        values_.push_back( val += 1.1 );
      }
    }

    ExpressionBase* build() const{ return new MultipleField(values_); }

  private:
     vector<double> values_;
  };

  ~MultipleField(){}
  void advertise_dependents( ExprDeps& exprDeps ){}
  void bind_fields( const FieldManagerList& fml ){}

  void evaluate()
  {
    using SpatialOps::operator<<=;
    ValVec& results = this->get_value_vec();
    vector<double>::const_iterator ival = values_.begin();
    BOOST_FOREACH( FieldT* f, results ){
      *f <<= *ival;
      ++ival;
    }
  }
};

//==============================================================================

int main()
{
  ExpressionFactory factory(false);

  const TagList tags = tag_list(
      Tag("A",STATE_NONE),
      Tag("B",STATE_NONE),
      Tag("C",STATE_NONE)
      );

  const Tag tmpTag1 ("tmp1",  STATE_NONE);
  const Tag tmpTag2 ("tmp2",  STATE_NONE);
  const Tag bmodTag1("B-mod1",STATE_NONE);
  const Tag bmodTag2("B-mod2",STATE_NONE);

  const ExpressionID id = factory.register_expression( new MultipleField::Builder(tags) );
  factory.register_expression( new ConstantExpr<FieldT>::Builder(tmpTag1,10.0) );
  factory.register_expression( new ConstantExpr<FieldT>::Builder(tmpTag2,20.0) );
  factory.register_expression( new Modifier<FieldT>::Builder(bmodTag1,tmpTag1) );
  factory.register_expression( new Modifier<FieldT>::Builder(bmodTag2,tmpTag2) );

  try{
    // this results in a depending on b even though it doesn't above (both are constants)
    // because the Modifier implies a dependency.
    factory.attach_modifier_expression( bmodTag1, tags[1], 0 );
    factory.attach_modifier_expression( bmodTag2, tags[1], 0 );

    ExpressionTree tree( id, factory, 0, "graphMultiple" );

    FieldManagerList fml;
    tree.register_fields(fml);
    SpatialOps::IntVec npts(2,1,1);
    fml.allocate_fields( FieldAllocInfo( npts, 0, 0, false, false, false ) );
    tree.bind_fields(fml);
    {
      ofstream fout("graphMultiple.dot");
      tree.write_tree(fout);
    }
    tree.execute_tree();

    const FieldT& a = fml.field_manager<FieldT>().field_ref(tags[0]);
    const FieldT& b = fml.field_manager<FieldT>().field_ref(tags[1]);
    const FieldT& c = fml.field_manager<FieldT>().field_ref(tags[2]);
#   ifdef ENABLE_CUDA
    const_cast<FieldT&>(a).add_device( CPU_INDEX );
    const_cast<FieldT&>(b).add_device( CPU_INDEX );
    const_cast<FieldT&>(c).add_device( CPU_INDEX );
#   endif
    TestHelper status(true);
    status( abs( a[0] - 1.1  ) < 1e-15, "a field value" );
    status( abs( b[0] - 32.2 ) < 1e-15, "b field value" );
    status( abs( c[0] - 3.3  ) < 1e-15, "c field value" );
    if( status.ok() ){
      cout << "PASS" << endl;
      return 0;
    }
    else{
      std::cout << a[0] << ", " << b[0] << ", " << c[0] << std::endl;
    }
  }
  catch( exception& err ){
    cout << err.what() << endl;
  }
  cout << "FAIL!" << endl;
  return -1;
}
