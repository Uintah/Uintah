#ifndef MvRHS_Expr_h
#define MvRHS_Expr_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/CPDData.h>


namespace CPD{


/**
 *  \class MvRHS
 *  \todo document this.
 */
template< typename FieldT >
class MvRHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELD( FieldT, cRHS_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, dyiRHS_ )

  MvRHS( const Expr::Tag     &cRHStag,
         const Expr::TagList &dyirhstag  );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag     &mvrhsTag,
             const Expr::Tag     &cRHStag,
             const Expr::TagList &dyirhstag );
    Expr::ExpressionBase* build() const;
    ~Builder(){}
  private:
    const Expr::Tag cRHStag_;
    const Expr::TagList dyirhstag_;
  };

  ~MvRHS(){}

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
MvRHS<FieldT>::
MvRHS( const Expr::Tag     &cRHStag,
       const Expr::TagList &dyirhstag )
 : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  cRHS_ = this->template create_field_request<FieldT>( cRHStag );
  this->template create_field_vector_request<FieldT>( dyirhstag, dyiRHS_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MvRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  result <<= - cRHS_->field_ref();
  for( size_t i=0; i<dyiRHS_.size(); ++i ){
    result <<= result + dyiRHS_[i]->field_ref();
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
MvRHS<FieldT>::
Builder::Builder( const Expr::Tag     &mvrhsTag,
                  const Expr::Tag     &cRHStag,
                  const Expr::TagList &dyirhstag )
 : ExpressionBuilder(mvrhsTag),
   cRHStag_  ( cRHStag   ),
   dyirhstag_( dyirhstag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MvRHS<FieldT>::
Builder::build() const
{
  return new MvRHS<FieldT>( cRHStag_, dyirhstag_ );
}

} // namespace CPD

#endif // MvRHS_Expr_h
