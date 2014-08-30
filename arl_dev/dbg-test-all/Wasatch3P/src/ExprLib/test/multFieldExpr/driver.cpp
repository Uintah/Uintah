#include <fstream>

#include "A.h"
#include "BandC.h"

#include <expression/ExpressionTree.h>

int main()
{
  try{
    Expr::ExpressionFactory factory;

    Expr::Tag a("A",Expr::STATE_NONE), b("B",Expr::STATE_NONE), c("C",Expr::STATE_NONE), d("D",Expr::STATE_NONE);

    {
      Expr::TagList bc;
      bc.push_back(b);
      bc.push_back(c);
      factory.register_expression( new BandC::Builder(bc) );
    }

    Expr::ExpressionID aid = factory.register_expression( new A::Builder(a,b,c) );
    Expr::ExpressionID did = factory.register_expression( new A::Builder(d,a,c) );
    Expr::ExpressionTree tree( did, factory, 0 );

    Expr::ExprPatch patch(1);
    Expr::FieldManagerList fml;
    tree.register_fields( fml );
    fml.allocate_fields( patch.field_info() );
    tree.bind_fields( fml );

    {
      std::ofstream fout("tree.dot");
      tree.write_tree(fout);
    }

    tree.execute_tree();

    std::cout << "PASS" << std::endl;
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl
        << "FAIL" << std::endl;
    return -1;
  }
  return 0;
}
