/*
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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

#ifndef Dissolution_Expr_h
#define Dissolution_Expr_h

#define KAYDISSOLUTION 30.0 * 0.25 * (1.0 - erf(8.0 * log(a / rMin_))) * (1.0 - erf(5.0 * (-2.5-log10(w)))) * (w) * pow((a),momentOrder_)
#include <expression/Expression.h>

/**
 *  \class  Dissolution
 *  \author Alex Abboud
 *  \date   October, 2013
 *  \brief death by dissolution source term
 */
template< typename FieldT >
class Dissolution
: public Expr::Expression<FieldT>
{
  
//  const Expr::TagList weightsTagList_;   // these are the tags of all the known weights
//  const Expr::TagList abscissaeTagList_; // these are the tags of all the known absicissae
//  const Expr::Tag sBarTag_;              // SBar form teh ostwald ripening term
//  const Expr::Tag superSatTag_;          // Supersaturation in aqueous phase
  DECLARE_VECTOR_OF_FIELDS(FieldT, weights_)
  DECLARE_VECTOR_OF_FIELDS(FieldT, abscissae_)
  DECLARE_FIELDS(FieldT, sBar_, superSat_)
  
  const double rMin_;           //smallest radius possible before deaht occurs
  const double momentOrder_;    // order of the current moment
  const double deathCoef_; 
  
  Dissolution( const Expr::TagList& weightsTagList,
               const Expr::TagList& abscissaeTagList,
               const Expr::Tag& sBarTag,
               const Expr::Tag& superSatTag,
               const double rMin,
               const double momentOrder,
               const double deathCoef );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& sBarTag,
             const Expr::Tag& superSatTag,
             const double rMin,
             const double momentOrder,
             const double deathCoef )
    : ExpressionBuilder(result),
      weightstaglist_(weightsTagList),
      abscissaetaglist_(abscissaeTagList),
      sbart_(sBarTag),
      supersatt_(superSatTag),
      rmin_(rMin),
      momentorder_(momentOrder),
      deathcoef_(deathCoef)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new Dissolution<FieldT>( weightstaglist_, abscissaetaglist_, sbart_, supersatt_, rmin_, momentorder_, deathcoef_ );
    }
    
  private:
    const Expr::TagList weightstaglist_; 
    const Expr::TagList abscissaetaglist_; 
    const Expr::Tag sbart_;
    const Expr::Tag supersatt_;
    const double rmin_;
    const double momentorder_;
    const double deathcoef_;
  };
  
  ~Dissolution(){}
  
  void evaluate();
  
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
Dissolution<FieldT>::
Dissolution( const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& sBarTag,
             const Expr::Tag& superSatTag,
             const double rMin,
             const double momentOrder,
             const double deathCoef )
: Expr::Expression<FieldT>(),
  rMin_(rMin),
  momentOrder_(momentOrder),
  deathCoef_(deathCoef)
{
  this->template create_field_vector_request<FieldT>(weightsTagList, weights_);
  this->template create_field_vector_request<FieldT>(abscissaeTagList, abscissae_);
   sBar_ = this->template create_field_request<FieldT>(sBarTag);
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Dissolution<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  
  const FieldT& sBar = sBar_->field_ref();
  const FieldT& S = superSat_->field_ref();
  for (size_t i = 0; i<weights_.size(); ++i) {
    const FieldT& w = weights_[i]->field_ref();
    const FieldT& a = abscissae_[i]->field_ref();
	  result <<= result - cond( sBar > S && a < rMin_ && a > 0.0 && w > 1e-3, deathCoef_ * KAYDISSOLUTION)
                            (  0.0 );
  }
}

#endif // Dissolution_Expr_h
