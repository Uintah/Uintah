#ifndef Wasatch_MMS_Functions
#define Wasatch_MMS_Functions

#include <expression/Expression.h>

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
    Builder( const Expr::Tag& result,
             const Expr::Tag& tTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag tt_;
  };

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  SineTime( const Expr::Tag& tTag );
  const Expr::Tag tTag_;
  const double* t_;
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
SineTime<ValT>::
SineTime( const Expr::Tag& ttag )
: Expr::Expression<ValT>(),
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
  phi <<= sin( *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
SineTime<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& ttag )
: ExpressionBuilder(result),
  tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
SineTime<ValT>::Builder::build() const
{
  return new SineTime<ValT>( tt_ );
}

//--------------------------------------------------------------------

/**
 *  \class ExprAlgebra
 *  \author Tony Saad
 *  \date September, 2011
 *  \brief Implements simple algebraic operations between expressions. This useful
           for initializing data for debugging without the need to implement 
           expressions such as x + y etc... this was required by the visit team
           and the easiest way to implement this was to create this expression.
           Furthermore, when initializing with embedded boundaries, we must
           multiply the initialized field by the volume fraction to get zero
           values inside solid volumes.
 */
template< typename FieldT >
class ExprAlgebra : public Expr::Expression<FieldT>
{
public:
  
  /**
   *  \brief Builds a Taylor Vortex velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder(const Expr::Tag& result, 
            const Expr::Tag& tag1,
            const Expr::Tag& tag2,
            const std::string& algebraicOperation);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag tag1_, tag2_;
    const std::string algebraicoperation_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  ExprAlgebra( const Expr::Tag& tag1,
            const Expr::Tag& tag2,
              const std::string& algebraicOperation);
  const Expr::Tag tag1_, tag2_;
  const std::string algebraicOperation_;
  const FieldT* field1_;
  const FieldT* field2_;
};

//--------------------------------------------------------------------

template<typename FieldT>
ExprAlgebra<FieldT>::
ExprAlgebra( const Expr::Tag& tag1,
             const Expr::Tag& tag2,
             const std::string& algebraicOperation)
: Expr::Expression<FieldT>(),
  tag1_(tag1), tag2_(tag2), algebraicOperation_( algebraicOperation )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tag1_ );
  exprDeps.requires_expression( tag2_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  field1_ = &fm.field_ref( tag1_ );
  field2_ = &fm.field_ref( tag2_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  if (algebraicOperation_ == "SUM") phi <<= *field1_ + *field2_;
  if (algebraicOperation_ == "DIFFERENCE") phi <<= *field1_ - *field2_;  
  if (algebraicOperation_ == "PRODUCT") phi <<= *field1_ * *field2_;    
  
}

//--------------------------------------------------------------------

template< typename FieldT >
ExprAlgebra<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& tag1,
        const Expr::Tag& tag2,
        const std::string& algebraicOperation)
: ExpressionBuilder(result),
tag1_(tag1),
tag2_(tag2),
algebraicoperation_(algebraicOperation)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ExprAlgebra<FieldT>::Builder::
build() const
{
  return new ExprAlgebra<FieldT>( tag1_, tag2_, algebraicoperation_ );
}

//--------------------------------------------------------------------

//====================================================================

// Explicit template instantiation for supported versions of this expression
#include <CCA/Components/Wasatch/FieldTypes.h>
using namespace Wasatch;

#define INSTANTIATE_EXPR_ALGEBRA( VOL ) 	\
template class ExprAlgebra< VOL >;

INSTANTIATE_EXPR_ALGEBRA( SVolField );
INSTANTIATE_EXPR_ALGEBRA( XVolField );
INSTANTIATE_EXPR_ALGEBRA( YVolField );
INSTANTIATE_EXPR_ALGEBRA( ZVolField );
//==========================================================================


#endif // Wasatch_MMS_Functions
