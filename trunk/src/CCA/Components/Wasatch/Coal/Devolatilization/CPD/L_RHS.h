#ifndef L_RHS_Expr_h
#define L_RHS_Expr_h

#include <expression/Expression.h>

namespace CPD{

/**
 *  \ingroup CPD
 *  \class L_RHS
 *
 *  In this Expression the RHS of the differential equation in below being evaluated :
 *  \f[ \frac{dl}{dt}=-k_{b}\ell \f]
 *  where
 *  - \f$k_{b}\f$ : reaction constant of labile bridge
 *  - \f$\ell\f$ : labile bridge
 */
template<typename FieldT>
class L_RHS
  : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, kb_, l_ )

  L_RHS( const Expr::Tag& kbTag,
         const Expr::Tag& lTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& rhsTag,
             const Expr::Tag& kbTag,
             const Expr::Tag& lTag );

    Expr::ExpressionBase* build() const;
    ~Builder(){}
  private:
    const Expr::Tag kbt_, lt_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
L_RHS<FieldT>::L_RHS( const Expr::Tag& kbTag,
                      const Expr::Tag& lTag )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  kb_ = this->template create_field_request<FieldT>( kbTag );
  l_  = this->template create_field_request<FieldT>( lTag  );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
L_RHS<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& kb = kb_->field_ref();
  const FieldT& l  = l_ ->field_ref();
  result <<= -1.0 * ( kb * l );
}

//--------------------------------------------------------------------

template< typename FieldT >
L_RHS<FieldT>::Builder::Builder( const Expr::Tag& rhsTag,
                                 const Expr::Tag& kbTag,
                                 const Expr::Tag& lTag)
: ExpressionBuilder(rhsTag),
  kbt_( kbTag ),
  lt_ ( lTag  )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
L_RHS<FieldT>::Builder::build() const
{
  return new L_RHS<FieldT>( kbt_, lt_ );
}

//--------------------------------------------------------------------

#endif // L_RHS_Expr_h

} // namespace CPD
