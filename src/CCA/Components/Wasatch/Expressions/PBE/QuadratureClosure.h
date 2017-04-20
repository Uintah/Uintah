/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef MomentClosure_Expr_h
#define MomentClosure_Expr_h

#include <expression/Expression.h>

/**
 *  \class QuadratureClosure
 *  \author Tony Saad
 *  \todo add documentation
 */
template< typename FieldT >
class QuadratureClosure
 : public Expr::Expression<FieldT>
{

  DECLARE_VECTOR_OF_FIELDS(FieldT, weights_)
  DECLARE_VECTOR_OF_FIELDS(FieldT, abscissae_)
//  const Expr::TagList weightsTagList_; // these are the tags of all the known moments
//  const Expr::TagList abscissaeTagList_; // these are the tags of all the known moments
  const double momentOrder_; // order of this unclosed moment. this will be used int he quadrature

//  typedef std::vector<const FieldT*> FieldVec;
//  FieldVec weights_;
//  FieldVec abscissae_;

  QuadratureClosure( const Expr::TagList& weightsTagList_,
                     const Expr::TagList& abscissaeTagList_,
                     const double momentOrder );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const double momentOrder )
      : ExpressionBuilder(result),
        weightstaglist_(weightsTagList),
        abscissaetaglist_(abscissaeTagList),
        momentorder_(momentOrder)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new QuadratureClosure<FieldT>( weightstaglist_,abscissaetaglist_, momentorder_ );
    }

  private:
    const Expr::TagList weightstaglist_; // these are the tags of all the known moments
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const double momentorder_;
  };

  ~QuadratureClosure();

  void evaluate();

};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
QuadratureClosure<FieldT>::
QuadratureClosure( const Expr::TagList& weightsTagList,
                   const Expr::TagList& abscissaeTagList,
                   const double momentOrder )
  : Expr::Expression<FieldT>(),
    momentOrder_(momentOrder)
{
  this->set_gpu_runnable( true );
  this->template create_field_vector_request<FieldT>(weightsTagList, weights_);
  this->template create_field_vector_request<FieldT>(abscissaeTagList, abscissae_);
}

//--------------------------------------------------------------------

template< typename FieldT >
QuadratureClosure<FieldT>::
~QuadratureClosure()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuadratureClosure<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  assert(abscissae_.size() == weights_.size());  
  for (size_t i=0; i<abscissae_.size(); ++i) {
    const FieldT& a = abscissae_[i]->field_ref();
    const FieldT& w = weights_[i]->field_ref();
    result <<= result + w * pow(a, momentOrder_);
  }
}

#endif // MomentClosure_Expr_h
