#include <expression/ExprLib.h>
#include <expression/Functions.h>

#include "Modifier.h"

using namespace std;
using namespace Expr;

#include <spatialops/structured/FVStaggeredFieldTypes.h>

typedef SpatialOps::SVolField FieldT;

int main()
{
  ExpressionFactory factory(true);

  Tag atag    ("A",     STATE_NONE);
  Tag btag    ("B",     STATE_NONE);
  Tag ctag    ("C",     STATE_NONE);
  Tag amodTag1("A-mod1",STATE_NONE);
  Tag amodTag2("A-mod2",STATE_NONE);

  const ExpressionID aid  = factory.register_expression( new ConstantExpr<FieldT>::Builder(atag,1.0) );
  const ExpressionID bid  = factory.register_expression( new ConstantExpr<FieldT>::Builder(btag,2.0) );
  const ExpressionID cid  = factory.register_expression( new ConstantExpr<FieldT>::Builder(ctag,3.0) );
  const ExpressionID amid1= factory.register_expression( new Modifier<FieldT>::Builder(amodTag1,btag) );
  const ExpressionID amid2= factory.register_expression( new Modifier<FieldT>::Builder(amodTag2,ctag) );

  try{
    // this results in a depending on b even though it doesn't above (both are constants)
    // because the Modifier implies a dependency.
    factory.attach_modifier_expression( amodTag1, atag, 0 );
    factory.attach_modifier_expression( amodTag2, atag, 0 );

    factory.dump_expressions(cout);

    ExpressionTree tree( aid, factory, 0, "graph" );

    FieldManagerList fml;
    tree.register_fields(fml);
    SpatialOps::IntVec npts(2,1,1);
    fml.allocate_fields( Expr::FieldAllocInfo( npts, 0, 0, false, false, false ) );
    tree.bind_fields(fml);
    {
      std::ofstream fout("graph.dot");
      tree.write_tree(fout);
    }
    tree.execute_tree();

    const FieldT& a = fml.field_manager<FieldT>().field_ref(atag);
#   ifdef ENABLE_CUDA
    const_cast<FieldT&>(a).add_device( CPU_INDEX );
#   endif
    if( *a.begin() == 6 ){
      cout << "PASS" << endl;
      return 0;
    }
    else{
      cout << *a.begin() << endl;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }
  cout << "FAIL!" << endl;
  return -1;
}
