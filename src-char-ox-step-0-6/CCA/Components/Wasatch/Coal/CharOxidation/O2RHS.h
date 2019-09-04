#ifndef O2RHS_Expr_h
#define O2RHS_Expr_h

#include <expression/Expression.h>

/**
 *  \class O2RHS
 */
template< typename FieldT >
class O2RHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, oxidation_, co2coratio_ )

  O2RHS( const Expr::Tag& oxidationtag,
         const Expr::Tag& co2coratiotag )
    : Expr::Expression<FieldT>()
  {
    this->set_gpu_runnable(true);
    oxidation_  = this->template create_field_request<FieldT>( oxidationtag  );
    co2coratio_ = this->template create_field_request<FieldT>( co2coratiotag );
  }

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& o2rhsTag,
             const Expr::Tag& oxidationtag,
             const Expr::Tag& co2coratiotag )
      : ExpressionBuilder( o2rhsTag ),
        oxidationtag_ ( oxidationtag  ),
        co2coratiotag_( co2coratiotag )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new O2RHS<FieldT>( oxidationtag_, co2coratiotag_ );
    }
  private:
    const Expr::Tag oxidationtag_, co2coratiotag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();
    const FieldT& oxidation  = oxidation_ ->field_ref();
    const FieldT& co2coratio = co2coratio_->field_ref();
    result <<= oxidation/12.0 * ( 0.5/ (1.0 + co2coratio) + co2coratio /(1.0+ co2coratio)) * -32.0;
  }

};

#endif // O2RHS_Expr_h
