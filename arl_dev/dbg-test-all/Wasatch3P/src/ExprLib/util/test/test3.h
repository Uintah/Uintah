#ifndef test3_Expr_h
#define test3_Expr_h

#include <expression/Expression.h>

// DEFINE THE TYPE OF FIELD FOR THIS EXPRESSION HERE
typedef /* insert field type here */ FieldT;

/**
 *  \class test1
 */
class test1
 : public Expr::Expression<FieldT>
{
  /* declare tags that need to be saved here:
  Expr::Tag myVar;
  */

  /* declare private variables here */

  /* declare operators associated with this expression here */

    test1( /* class-specific arguments (typically Expr::Tag objects) */ ) );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a test1 expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag /* add additional arguments here */ );

    Expr::ExpressionBase* build() const;

  private:
     /* add additional arguments here */
  };

  ~test1();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



test1::
test1( /* class-specific arguments (typically Expr::Tag objects) */ ) )
  : Expr::Expression<FieldT>()
{}

//--------------------------------------------------------------------

test1::
~test1()
{}

//--------------------------------------------------------------------

void
test1::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  /* add dependencies as follows (TAG represents the Expr::Tag for the depenency): */
  // exprDeps.requires_expression( TAG );
}

//--------------------------------------------------------------------

void
test1::
bind_fields( const Expr::FieldManagerList& fml )
{
  /* add additional code here to bind any fields required by this expression */
  // const Expr::FieldManager<FieldT>& fm = fml.field_manager<FieldT>();

}

//--------------------------------------------------------------------

void
test1::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();
}

//--------------------------------------------------------------------

void
test1::
evaluate()
{
  FieldT& result = this->value();

  /* evaluation code goes here - be sure to assign the appropriate value to 'result' */
}

//--------------------------------------------------------------------

test1::
Builder::Builder( const Expr::Tag& resultTag /* add arguments here */ )
  : ExpressionBuilder( resultTag )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
test1::
Builder::build() const
{
  return new test1( /* insert additional arguments here */ );
}


#endif // test3_Expr_h
