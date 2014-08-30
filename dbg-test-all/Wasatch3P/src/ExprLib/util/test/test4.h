#ifndef test4_Expr_h
#define test4_Expr_h

#include <expression/Expression.h>

/**
 *  \class test1
 */
template< typename PatchT,
          typename FieldT,
          typename DivT >
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



template< typename PatchT, typename FieldT, typename DivT >
test1<PatchT,FieldT,DivT>::
test1( /* class-specific arguments (typically Expr::Tag objects) */ ) )
  : Expr::Expression<FieldT>()
{}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
test1<PatchT,FieldT,DivT>::
~test1()
{}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
void
test1<PatchT,FieldT,DivT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  /* add dependencies as follows (TAG represents the Expr::Tag for the depenency): */
  // exprDeps.requires_expression( TAG );
}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
void
test1<PatchT,FieldT,DivT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  /* add additional code here to bind any fields required by this expression */
  // const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();

}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
void
test1<PatchT,FieldT,DivT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();
}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
void
test1<PatchT,FieldT,DivT>::
evaluate()
{
  FieldT& result = this->value();

  /* evaluation code goes here - be sure to assign the appropriate value to 'result' */
}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
test1<PatchT,FieldT,DivT>::
Builder::Builder( const Expr::Tag& resultTag /* add arguments here */ )
  : ExpressionBuilder( resultTag )
{}

//--------------------------------------------------------------------

template< typename PatchT, typename FieldT, typename DivT >
Expr::ExpressionBase*
test1<PatchT,FieldT,DivT>::
Builder::build() const
{
  return new test1<PatchT,FieldT,DivT>( /* insert additional arguments here */ );
}


#endif // test4_Expr_h
