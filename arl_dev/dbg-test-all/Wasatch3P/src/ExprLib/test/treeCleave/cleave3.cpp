#include <iostream>
#include <sstream>

#include <expression/ExprLib.h>

#include "DummyExpr.h"
#include <test/TestHelper.h>

using namespace Expr;
using namespace std;
namespace so=SpatialOps;

int main()
{
  try{
    TestHelper status(true);

    ExpressionFactory factory;

    const Tag a1( "a1", STATE_NONE ),
              a2( "a2", STATE_NONE ),
              b ( "b",  STATE_NONE ),
              c ( "c",  STATE_NONE ),
              d ( "d",  STATE_NONE );

    TagList bd; bd.push_back(b); bd.push_back(d);

    const ExpressionID ida1 = factory.register_expression( new Dummy::Builder(a1,c) );
    const ExpressionID ida2 = factory.register_expression( new Dummy::Builder(a2,c) );
    const ExpressionID idb  = factory.register_expression( new Dummy::Builder(b,d ) );
    const ExpressionID idc  = factory.register_expression( new Dummy::Builder(c,bd) );
    const ExpressionID idd  = factory.register_expression( new PlaceHolder<so::SingleValueField>::Builder(d) );

    factory.cleave_from_children( ida1 );
    factory.cleave_from_children( ida2 );

    ExpressionTree::RootIDList roots;
    roots.insert(ida1); roots.insert(ida2); roots.insert(idb);
    ExpressionTree tree( roots, factory, 0, "base" );
    {
      ofstream fout("orig3.dot");
      tree.write_tree(fout);
    }
    ExpressionTree::TreeList trees = tree.split_tree();

    int inum=0;
    for( ExpressionTree::TreeList::const_iterator i=trees.begin();
        i!=trees.end(); ++i, ++inum )
    {
      ostringstream fnam;
      fnam << "tree3_" << inum << ".dot";
      ofstream fout(fnam.str().c_str());
      (*i)->write_tree(fout);
    }

    std::cout << "\nPASS\n";
    return 0;
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }
  std::cout << "\nFAIL\n";
  return -1;
}
