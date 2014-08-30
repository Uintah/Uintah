#ifndef test2_Expr_h
#define test2_Expr_h

#include <expression/Expression.h>

/**
 *  \class test1
 */
template< typename FluxT,
          typename DivT >
class test1
 : public Expr::Expression<FluxT>
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



template< typename FluxT, typename DivT >
test1<FluxT,DivT>::
test1( /* class-specific arguments (typically Expr::Tag objects) */ ) )
  : Expr::Expression<FluxT>()
{}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
test1<FluxT,DivT>::
~test1()
{}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
void
test1<FluxT,DivT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  /* add dependencies as follows (TAG represents the Expr::Tag for the depenency): */
  // exprDeps.requires_expression( TAG );
}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
void
test1<FluxT,DivT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  /* add additional code here to bind any fields required by this expression */
  // const Expr::FieldManager<FluxT>& fm = fml.template field_manager<FluxT>();

}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
void
test1<FluxT,DivT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();
}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
void
test1<FluxT,DivT>::
evaluate()
{
  FluxT& result = this->value();

  /* evaluation code goes here - be sure to assign the appropriate value to 'result' */
}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
test1<FluxT,DivT>::
Builder::Builder( const Expr::Tag& resultTag /* add arguments here */ )
  : ExpressionBuilder( resultTag )
{}

//--------------------------------------------------------------------

template< typename FluxT, typename DivT >
Expr::ExpressionBase*
test1<FluxT,DivT>::
Builder::build() const
{
  return new test1<FluxT,DivT>( /* insert additional arguments here */ );
}


#endif // test2_Expr_h
