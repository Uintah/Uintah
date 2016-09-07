#ifndef C_RHS_Expr_h
#define C_RHS_Expr_h

#include <expression/Expression.h>

namespace CPD{

/**
 *  \ingroup CPD
 *  \class C_RHS
 *
 *  In this Expression the RHS of the differential equation
 *  \f[ \frac{dC}{dt}=\frac{k_{b}\ell}{\rho+1} \f]
 *  is being evaluated, where
 *  - \f$ C \f$ : Char
 *  - \f$ k_{b} \f$ : reaction constant of labile bridge
 *  - \f$\ell\f$ : labile bridge
 *  - \f$ \rho = 0.9 \f$
 */
template<typename FieldT>
class C_RHS
  : public Expr::Expression<FieldT>
{
  const CPDInformation& cpd_;

  DECLARE_FIELDS( FieldT, kb_, l_ )

  C_RHS( const Expr::Tag& kbTag,
         const Expr::Tag& lTag,
         const CPDInformation& cpd );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& crhsTag,
             const Expr::Tag& kbTag,
             const Expr::Tag& lTag,
             const CPDInformation& cpd );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag lt_, kbt_;
    const CPDInformation& cpd_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
C_RHS<FieldT>::C_RHS( const Expr::Tag& kbTag,
                      const Expr::Tag& lTag,
                      const CPDInformation& cpd )
  : Expr::Expression<FieldT>(),
    cpd_( cpd )
{
  this->set_gpu_runnable(true);

  kb_ = this->template create_field_request<FieldT>( kbTag );
  l_  = this->template create_field_request<FieldT>( lTag  );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
C_RHS<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& kb  = kb_->field_ref();
  const FieldT& ell = l_ ->field_ref();
  const double  mwl0= cpd_.l0_molecular_weight();
  const double rho = 0.9;
  result <<= ( kb * ell ) / (rho + 1.0) * (12.0 / mwl0);  // "ell" is in kg

}

//--------------------------------------------------------------------

template< typename FieldT >
C_RHS<FieldT>::Builder::Builder( const Expr::Tag& crhsTag,
                                 const Expr::Tag& kbTag,
                                 const Expr::Tag& lTag,
                                 const CPDInformation& cpd )
  : ExpressionBuilder(crhsTag),
    lt_ ( lTag  ),
    kbt_( kbTag ),
    cpd_( cpd   )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
C_RHS<FieldT>::Builder::build() const
{
  return new C_RHS<FieldT>( kbt_, lt_, cpd_ );
}

} // namespace CPD

#endif // C_RHS_Expr_h
