#include <iostream>

#include <expression/ExprLib.h>

#include "expressions.h"

typedef Expr::ExprPatch Patch;

using namespace std;
using namespace Expr;
namespace so = SpatialOps;

int main()
{
  Patch patch( 1 );

  ExpressionFactory factory;

  const Tag Atag("A",STATE_NONE);
  const Tag Btag("B",STATE_NONE);
  const Tag Ctag("C",STATE_NONE);
  const Tag Dtag("D",STATE_NONE);

  const ExpressionID id_A = factory.register_expression( new TestExpr::Builder(Atag,1.1) );
  const ExpressionID id_B = factory.register_expression( new TestExpr::Builder(Btag,2.2) );
  const ExpressionID id_C = factory.register_expression( new TestExpr::Builder(Ctag,3.3) );
  const ExpressionID id_D = factory.register_expression( new TestExpr::Builder(Dtag,4.4) );

  factory.attach_dependency_to_expression( Btag, Atag );
  factory.attach_dependency_to_expression( Ctag, Btag );
  factory.attach_dependency_to_expression( Dtag, Atag );

  Expr::ExpressionTree tree(id_A,factory,patch.id());
  FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<so::SingleValueField>::type& fmx = fml.field_manager<so::SingleValueField>();
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() /*FieldInfo( patch.dim(),
                                  patch.get_n_particles(),
                                  patch.has_physical_bc_xplus(),
                                  patch.has_physical_bc_yplus(),
                                  patch.has_physical_bc_zplus() ) */);
  fmx.lock_field( Atag );
  fmx.lock_field( Btag );
  fmx.lock_field( Ctag );
  fmx.lock_field( Dtag );

  tree.bind_fields( fml );
  tree.write_tree(cout);
  tree.execute_tree();

  const Expr::FieldMgrSelector<so::SingleValueField>::type& fmgr = fml.field_manager<so::SingleValueField>();
  cout << endl
       << "A = " << fmgr.field_ref( Atag )[0] << endl
       << "B = " << fmgr.field_ref( Btag )[0] << endl
       << "C = " << fmgr.field_ref( Ctag )[0] << endl
       << "D = " << fmgr.field_ref( Dtag )[0] << endl
       << endl;

  if( fmgr.field_ref( Atag )[0] == 1.1+2.2+3.3+4.4 ){
    cout << "PASS" << endl;
    return 0;
  }
  else{
    cout << "FAIL - expected A=" << 1.1+2.2+3.3+4.4 << endl;
  }

  return 1;
}
