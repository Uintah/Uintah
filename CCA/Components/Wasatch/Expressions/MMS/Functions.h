#ifndef Wasatch_MMS_Functions
#define Wasatch_MMS_Functions

#include <expression/Expr_Expression.h>

/**
 *  \class SineTime
 *  \author Tony Saad
 *  \date September, 2011
 *  \brief Implements a sin(t) function. This is useful for testing time integrators
           with ODEs. Note that we can't pass time as a argument to the functions
					 provided by ExprLib at the moment.
 */
template< typename ValT >
class SineTime : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds a Sin(t) expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag tTag);
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                const Expr::ExpressionRegistry& reg ) const;
  private:
    const Expr::Tag tt_;
  };

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  SineTime( const Expr::Tag& tTag,
           const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg);
  const Expr::Tag tTag_;
  const double* t_;
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
SineTime<ValT>::
SineTime( const Expr::Tag& ttag,
         const Expr::ExpressionID& id,
         const Expr::ExpressionRegistry& reg )
: Expr::Expression<ValT>( id, reg ),
tTag_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<double>& timeFM = fml.template field_manager<double>();
  t_ = &timeFM.field_ref( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  //std::cout << "Time in source term = " << *t_ << std::endl;
  phi <<= sin( *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
SineTime<ValT>::Builder::
Builder(         const Expr::Tag ttag)
: tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
SineTime<ValT>::Builder::
build( const Expr::ExpressionID& id,
      const Expr::ExpressionRegistry& reg ) const
{
  return new SineTime<ValT>( tt_, id, reg );
}

//--------------------------------------------------------------------

#endif // Wasatch_MMS_Functions
