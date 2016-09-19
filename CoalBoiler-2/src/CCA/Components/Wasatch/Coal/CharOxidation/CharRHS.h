#ifndef CharRHS_CHAR_h
#define CharRHS_CHAR_h

#include <expression/Expression.h>

/**
 *  \class CharRHS
 *  \brief Calculate Char production rhs due to all the heterogeneous reactions.
 */
namespace CHAR {

template< typename FieldT >
class CharRHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, oxidation_, hetroco2_, hetroh2o_ )

  CharRHS( const Expr::Tag& oxidationtag,
           const Expr::Tag& hetroco2tag,
           const Expr::Tag& hetroh2otag )
    : Expr::Expression<FieldT>()
  {
    this->set_gpu_runnable(true);
    oxidation_ = this->template create_field_request<FieldT>(oxidationtag);
    hetroco2_  = this->template create_field_request<FieldT>(hetroco2tag );
    hetroh2o_  = this->template create_field_request<FieldT>(hetroh2otag );
  }


public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& charRHSTag,
             const Expr::Tag& oxidationtag,
             const Expr::Tag& hetroco2tag,
             const Expr::Tag& hetroh2otag )
    : ExpressionBuilder(charRHSTag),
      oxidationtag_( oxidationtag ),
      hetroco2tag_ ( hetroco2tag  ),
      hetroh2otag_ ( hetroh2otag  )
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new CharRHS<FieldT>( oxidationtag_, hetroco2tag_, hetroh2otag_ );
    }

  private:
    const Expr::Tag oxidationtag_, hetroco2tag_, hetroh2otag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();
    const FieldT& oxidation = oxidation_->field_ref();
    const FieldT& hetroco2  = hetroco2_ ->field_ref();
    const FieldT& hetroh2o  = hetroh2o_ ->field_ref();
    result <<= oxidation + hetroco2 + hetroh2o;
  }
};

} // namespace Char
#endif // CharRHS_CHAR_h
