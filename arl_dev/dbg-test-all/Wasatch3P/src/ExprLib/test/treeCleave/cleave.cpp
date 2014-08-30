#include <iostream>
#include <sstream>

#include <expression/ExprLib.h>

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

    const Tag tagA("A",Expr::STATE_NONE);
    const Tag tagB("B",Expr::STATE_NONE);
    const Tag tagC("C",Expr::STATE_NONE);
    const Tag tagD("D",Expr::STATE_NONE);
    const Tag tagE("E",Expr::STATE_NONE);
    const Tag tagF("F",Expr::STATE_NONE);
    const Tag tagG("G",Expr::STATE_NONE);
    const Tag tagH("H",Expr::STATE_NONE);
    const Tag tagI("I",Expr::STATE_NONE);

    const ExpressionID id_A = exprFactory.register_expression( new A::Builder(tagA,tagB,tagC,tagD) );
    const ExpressionID id_B = exprFactory.register_expression( new B::Builder(tagB,tagE,tagG) );
    const ExpressionID id_C = exprFactory.register_expression( new C::Builder(tagC,tagF,tagG) );
    const ExpressionID id_D = exprFactory.register_expression( new D::Builder(tagD,tagF) );
    const ExpressionID id_E = exprFactory.register_expression( new E::Builder(tagE) );
    const ExpressionID id_F = exprFactory.register_expression( new F::Builder(tagF,tagG,tagH) );
    const ExpressionID id_G = exprFactory.register_expression( new G::Builder(tagG) );
    const ExpressionID id_H = exprFactory.register_expression( new H::Builder(tagH) );
    const ExpressionID id_I = exprFactory.register_expression( new I::Builder(tagI,tagD) );

    // tag B, C and F as targets for cleaving.
    exprFactory.cleave_from_parents( id_C );
    exprFactory.cleave_from_parents( id_B );
    exprFactory.cleave_from_parents( id_F );

    ExpressionTree tree( id_A, exprFactory, patch.id(), "orig" );
    FieldManagerList& fml = patch.field_manager_list();
    tree.insert_tree( id_I );

    tree.register_fields( fml );
    fml.allocate_fields( patch.field_info() );

    {
      ofstream fout("orig_precleave.dot");
      tree.write_tree(fout);
    }

    tree.set_expr_is_persistent( tagA, fml );
    tree.set_expr_is_persistent( tagD, fml );
    tree.set_expr_is_persistent( tagE, fml );

    tree.bind_fields( fml );
    tree.execute_tree();

    // cleave the tree.
    typedef ExpressionTree::TreeList TreeList;
    const TreeList treeList = tree.split_tree();

    status( treeList[0]->has_expression(id_F) == true, "0 has F" );
    status( treeList[0]->has_expression(id_G) == true, "0 has G" );
    status( treeList[0]->has_expression(id_H) == true, "0 has H" );
    status( treeList[0]->computes_field(tagF) == true, "0 computes F" );
    status( treeList[0]->computes_field(tagG) == true, "0 computes G" );
    status( treeList[0]->computes_field(tagH) == true, "0 computes H" );

    status( treeList[1]->has_expression(id_F) == true, "1 has F expr" );
    status( treeList[1]->computes_field(tagF) == false, "1 doesn't compute F" );
    status( treeList[1]->has_expression(id_B) == true, "1 has B" );
    status( treeList[1]->has_expression(id_C) == true, "1 has C" );
    status( treeList[1]->has_expression(id_E) == true, "1 has E" );
    status( treeList[1]->is_persistent(tagE)  == true, "E is persistent" );
    status( treeList[1]->has_field(tagF)      == true, "1 has F field" );
    status( treeList[1]->has_field(tagG)      == true, "1 has G field" );
    status( treeList[1]->computes_field(tagB) == true, "1 computes B" );
    status( treeList[1]->computes_field(tagC) == true, "1 computes C" );
    status( treeList[1]->computes_field(tagE) == true, "1 computes E" );
    status( treeList[1]->computes_field(tagG) == false,"1 doesn't compute G" );

    status( treeList[2]->has_expression(id_A) == true, "2 has A" );
    status( treeList[2]->is_persistent( tagA) == true, "A is persistent" );
    status( treeList[2]->has_expression(id_I) == true, "2 has I" );
    status( treeList[2]->has_expression(id_D) == true, "2 has D" );
    status( treeList[2]->is_persistent( tagD) == true, "D is persistent" );
    status( treeList[2]->has_expression(id_B) == true, "2 has B" );
    status( treeList[2]->has_expression(id_C) == true, "2 has C" );
    status( treeList[2]->computes_field(tagA) == true, "2 computes A" );
    status( treeList[2]->computes_field(tagI) == true, "2 computes I" );
    status( treeList[2]->computes_field(tagD) == true, "2 computes D" );
    status( treeList[2]->computes_field(tagB) == false, "2 doesn't compute B" );
    status( treeList[2]->computes_field(tagC) == false, "2 doesn't compute C" );
    status( treeList[2]->computes_field(tagE) == false, "2 doesn't compute E" );
    status( treeList[2]->computes_field(tagF) == false, "2 doesn't compute F" );
    status( treeList[2]->computes_field(tagG) == false, "2 doesn't compute G" );
    status( treeList[2]->computes_field(tagH) == false, "2 doesn't compute H" );
    status( treeList[2]->has_field(tagB)      == true, "2 has B field" );
    status( treeList[2]->has_field(tagC)      == true, "2 has C field" );

    const FieldAllocInfo fieldInfo = patch.field_info();
    for( TreeList::const_iterator i=treeList.begin(); i!=treeList.end(); ++i ){
      Expr::ExpressionTree::TreePtr tt( *i );
      tt->register_fields(fml);
      fml.allocate_fields( fieldInfo );
      tt->bind_fields( fml );
      tt->execute_tree();
      {
        std::ostringstream nam;
        nam << tt->name() << ".dot";
        ofstream fout( nam.str().c_str() );
        tt->write_tree( fout );
      }
    }

    Expr::FieldMgrSelector<SingleValueField>::type& fmgr = fml.field_manager<SingleValueField>();

    const double e = 5.5;
    const double g = 7.7;
    const double h = 8.8;

    const double f = 6.6 + g + h;
    const double d = 4.4 + f;
    const double c = 3.3 + f + g;
    const double b = 2.2 + e + g;
    const double a = 1.1 + b + c + d;

    const double i = 9.9 * d;

    SingleValueField afld = fmgr.field_ref(tagA);
    SingleValueField ifld = fmgr.field_ref(tagI);

#   ifdef ENABLE_CUDA
    // move fields to CPU to allow us to dereference them and get values for comparison
    afld.add_device(CPU_INDEX);
    ifld.add_device(CPU_INDEX);
    afld.set_device_as_active(CPU_INDEX);
    ifld.set_device_as_active(CPU_INDEX);
#   endif

    double& aVal = afld[0];
    double& iVal = ifld[0];

    /* Fix some floating-point roundoff error */
    aVal = ((int) (aVal * 1000)) /1000.0;
    iVal = ((int) (iVal * 1000)) /1000.0;

    status( a == aVal, "a value" );
    status( i == iVal, "i value" );

    if( status.ok() ){
      cout << endl << "PASS" << endl << endl;
      return 0;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }
  cout << endl << "***FAIL***" << endl << endl;
  return -1;
}

