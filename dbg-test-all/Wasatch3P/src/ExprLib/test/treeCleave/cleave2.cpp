#include <iostream>
#include <sstream>

#include <expression/ExprLib.h>

#include "expressions.h"
#include <test/TestHelper.h>

using namespace Expr;
using namespace std;
namespace so=SpatialOps;

typedef Expr::ExprPatch Patch;

int main()
{
  try{
    TestHelper status(true);

    Patch patch( 1 );

    const FieldAllocInfo fieldInfo = patch.field_info();

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

    // tag B, and D as targets for cleaving.
    exprFactory.cleave_from_parents( id_B );
    exprFactory.cleave_from_children( id_C );
    exprFactory.cleave_from_parents( id_G );

    ExpressionTree tree( id_A, exprFactory, patch.id() );
    FieldManagerList& fml = patch.field_manager_list();
    tree.insert_tree( id_I );

    tree.register_fields( fml );
    fml.allocate_fields( fieldInfo );

    {
      ofstream fout("orig2.dot");
      tree.write_tree(fout);
    }

    tree.bind_fields( fml );
    tree.execute_tree();

    tree.set_expr_is_persistent( tagE, fml );
    tree.set_expr_is_persistent( tagD, fml );


    // cleave the tree.
    typedef ExpressionTree::TreeList TreeList;
    const TreeList treeList = tree.split_tree();

    BOOST_FOREACH( const ExpressionTree::TreePtr tchild, treeList ){
      std::ostringstream nam;
      nam << tchild->name() << ".dot";
      std::ofstream fout( nam.str().c_str() );
      tchild->write_tree( fout );
    }
    fml.dump_fields(std::cout);

    status( treeList[0]->has_expression(id_G) == true, "0 has G" );

    status( treeList[1]->has_expression(id_B) == true, "1 has B" );
    status( treeList[1]->has_expression(id_E) == true, "1 has E" );
    status( treeList[1]->is_persistent(tagE) == true, "E is persistent");
    status( treeList[1]->has_expression(id_F) == true, "1 has F" );
    status( treeList[1]->has_expression(id_H) == true, "1 has H" );
    status( treeList[1]->has_field(tagG)      == true, "1 has G field" );
    status( treeList[1]->computes_field(tagB) == true, "1 computes E" );
    status( treeList[1]->computes_field(tagE) == true, "1 computes E" );
    status( treeList[1]->computes_field(tagF) == true, "1 computes F" );
    status( treeList[1]->computes_field(tagH) == true, "1 computes H" );
    status( treeList[1]->computes_field(tagG) == false,"1 doesn't compute G" );

    status( treeList[2]->has_expression(id_A) == true, "2 has A" );
    status( treeList[2]->has_expression(id_C) == true, "2 has C" );
    status( treeList[2]->has_expression(id_D) == true, "2 has D" );
    status( treeList[2]->is_persistent(tagD)  == true, "D is persistent");
    status( treeList[2]->has_expression(id_I) == true, "2 has I" );
    status( treeList[2]->computes_field(tagA) == true, "2 computes A" );
    status( treeList[2]->computes_field(tagC) == true, "2 computes C" );
    status( treeList[2]->computes_field(tagD) == true, "2 computes D" );
    status( treeList[2]->computes_field(tagI) == true, "2 computes I" );
    status( treeList[2]->computes_field(tagE) == false, "2 doesn't compute E" );
    status( treeList[2]->computes_field(tagF) == false, "2 doesn't compute F" );
    status( treeList[2]->computes_field(tagG) == false, "2 doesn't compute G" );
    status( treeList[2]->computes_field(tagH) == false, "2 doesn't compute H" );
    status( treeList[2]->has_field(tagB)      == true, "2 has B field" );
    status( treeList[2]->has_field(tagG)      == true, "2 has G field" );
    status( treeList[2]->has_field(tagF)      == true, "2 has F field" );

    BOOST_FOREACH( const ExpressionTree::TreePtr tt, treeList ){
      tt->register_fields(fml);
      fml.allocate_fields( fieldInfo );
      tt->bind_fields( fml );
      tt->execute_tree();
    }

    const Expr::FieldMgrSelector<so::SingleValueField>::type& fmgr = fml.field_manager<so::SingleValueField>();

    const double e = 5.5;
    const double g = 7.7;
    const double h = 8.8;

    const double f = 6.6 + g + h;
    const double d = 4.4 + f;
    const double c = 3.3 + f + g;
    const double b = 2.2 + e + g;
    const double a = 1.1 + b + c + d;

    const double i = 9.9 * d;

    so::SingleValueField tempA = fmgr.field_ref(tagA);
    so::SingleValueField tempI = fmgr.field_ref(tagI);

#   ifdef ENABLE_CUDA
    // fields are added to CPU to enable indexing operator
    tempA.add_device(CPU_INDEX);
    tempI.add_device(CPU_INDEX);
    tempA.set_device_as_active(CPU_INDEX);
    tempI.set_device_as_active(CPU_INDEX);
#   endif

    double afld = tempA[0];
    double ifld = tempI[0];

    /* Fix some floating-point roundoff error */
    afld = ((int) (afld * 1000)) /1000.0;
    ifld = ((int) (ifld * 1000)) /1000.0;

    status( a == afld, "a value" );
    status( i == ifld, "i value" );

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

