#include <iostream>

#include <expression/ExprLib.h>

int main()
{
  Expr::FieldManagerList fml( "test" );

  typedef Expr::FieldMgrSelector<double>::type DoubleFM;
  typedef Expr::FieldMgrSelector<int   >::type IntFM;
  typedef Expr::FieldMgrSelector<char  >::type CharFM;

  DoubleFM& doubleFM = fml.field_manager<double>();
  IntFM&    intFM    = fml.field_manager<int   >();
  CharFM&   charFM   = fml.field_manager<char  >();

  // register a few fields.
  doubleFM.register_field( "myDoubleField",Expr::STATE_NONE );
  intFM   .register_field( "myIntField",   Expr::STATE_NONE );
  charFM  .register_field( "myCharField",  Expr::STATE_NP1  );

  // set some modes
  doubleFM.set_field_mode( "myDoubleField",
                           Expr::STATE_NONE,
                           Expr::COMPUTES );

  fml.dump_fields( std::cout );
}
