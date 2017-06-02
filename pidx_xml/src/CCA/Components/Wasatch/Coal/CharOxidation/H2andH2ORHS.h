#ifndef H2andH2ORHS_CHAR_h
#define H2andH2ORHS_CHAR_h

#include <expression/Expression.h>

namespace CHAR{

/**
 *  \class H2andH2ORHS
 *  \tparam the type of field to build this expression for.
 */
template< typename FieldT >
class H2andH2ORHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELD( FieldT, hetroh2o_ )

  H2andH2ORHS( const Expr::Tag& hetroh2otag )
    : Expr::Expression<FieldT>()
  {
    this->set_gpu_runnable(true);
    hetroh2o_ = this->template create_field_request<FieldT>( hetroh2otag );
  }


public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& h2h20rhsTag,
             const Expr::Tag& hetroh2otag )
      : ExpressionBuilder( h2h20rhsTag ),
        hetroh2otag_ ( hetroh2otag )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new H2andH2ORHS<FieldT>( hetroh2otag_ );
    }


  private:
    const Expr::Tag hetroh2otag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();
    FieldT& H2rhs = *results[0];
    FieldT& H2Orhs = *results[1];
    const FieldT& hetroh2o = hetroh2o_->field_ref();

    H2rhs  <<= hetroh2o / 12.0 * 2;
    H2Orhs <<= hetroh2o / 12.0 * -18.0;
  }
};

} //namespace CHAR


#endif // H2andH2ORHS_CHAR_h
