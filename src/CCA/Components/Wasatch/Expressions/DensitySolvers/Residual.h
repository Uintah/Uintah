/*
 * The MIT License
 *
 * Copyright (c) 2015 The University of Utah
 *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Residual_Expr_h
#define Residual_Expr_h

#include <expression/Expression.h>

/**
 *  \class Residual
 */
template< typename FieldT >
class Residual
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS( FieldT, rhoPhi_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, phi_    )
  DECLARE_FIELD( FieldT, rho_ )
  
    Residual( const Expr::TagList& rhoPhiTags,
              const Expr::TagList& phiTags,
              const Expr::Tag&     rhoTag );

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a Residual expression
     *  @param residualTags tags for residuals
     */
    Builder( const Expr::TagList residualTags,
             const Expr::TagList rhoPhiTags,
             const Expr::TagList phiTags,
             const Expr::Tag     rhoTag )
      : ExpressionBuilder( residualTags ),
        rhoPhiTags_( rhoPhiTags ),
        phiTags_   ( phiTags    ),
        rhoTag_    ( rhoTag     )
    {
      assert( residualTags.size()==rhoPhiTags.size() );
      assert( residualTags.size()==phiTags.size()    );
    }

    Expr::ExpressionBase* build() const{
      return new Residual<FieldT>( rhoPhiTags_, phiTags_, rhoTag_ );
    }
    const Expr::TagList rhoPhiTags_, phiTags_;
    const Expr::Tag     rhoTag_;
  };

  void evaluate();
  ~Residual(){};
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
Residual<FieldT>::
Residual( const Expr::TagList& rhoPhiTags,
          const Expr::TagList& phiTags,
          const Expr::Tag&     rhoTag )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  this->template create_field_vector_request<FieldT>( rhoPhiTags, rhoPhi_ );
  this->template create_field_vector_request<FieldT>( phiTags   , phi_    );

  rho_ = this->template create_field_request<FieldT>( rhoTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Residual<FieldT>::
evaluate()
{
  typename Expr::Expression<FieldT>::ValVec&  result = this->get_value_vec();
  const FieldT& rho = rho_->field_ref();

  const int n = result.size();
  for(int i=0; i<n; ++i)
  {
    const FieldT& rhoPhi = rhoPhi_[i]->field_ref();
    const FieldT& phi    = phi_   [i]->field_ref();
    *result[i] <<= rho*phi - rhoPhi;
  }
}
//--------------------------------------------------------------------
#endif // Residual_Expr_h
