/*
 * The MIT License
 *
 * Copyright (c) 2013-2015 The University of Utah
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

#define KAYDISSOLUTION 30.0 * 0.25 * (1.0 - erf(8.0 * log(**abscissaeIterator / rMin_))) * (1.0 - erf(5.0 * (-2.5-log10(**weightsIterator)))) * (**weightsIterator) * pow((**abscissaeIterator),momentOrder_)
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
  
  const Expr::TagList weightsTagList_;   // these are the tags of all the known weights
  const Expr::TagList abscissaeTagList_; // these are the tags of all the known absicissae
  const Expr::Tag sBarTag_;              // SBar form teh ostwald ripening term
  const Expr::Tag superSatTag_;          // Supersaturation in aqueous phase
  const FieldT* sBar_;
  const FieldT* superSat_;
  
  const double rMin_;           //smallest radius possible before deaht occurs
  const double momentOrder_;    // order of the current moment
  const double deathCoef_; 
  
  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  
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
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
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
  weightsTagList_(weightsTagList),
  abscissaeTagList_(abscissaeTagList),
  sBarTag_(sBarTag),
  superSatTag_(superSatTag),
  rMin_(rMin),
  momentOrder_(momentOrder),
  deathCoef_(deathCoef)
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Dissolution<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( weightsTagList_ );
  exprDeps.requires_expression( abscissaeTagList_ );
  exprDeps.requires_expression( sBarTag_ );
  exprDeps.requires_expression( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Dissolution<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& volfm = fml.template field_manager<FieldT>();
  weights_.clear();
  abscissae_.clear();
  for (Expr::TagList::const_iterator iweight=weightsTagList_.begin(); iweight!=weightsTagList_.end(); iweight++) {
    weights_.push_back(&volfm.field_ref(*iweight));
  }
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }
  sBar_     = &volfm.field_ref( sBarTag_ );
  superSat_ = &volfm.field_ref( superSatTag_ );
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
  
  typename FieldVec::const_iterator abscissaeIterator = abscissae_.begin();
  for( typename FieldVec::const_iterator weightsIterator=weights_.begin();
      weightsIterator!=weights_.end();
      ++weightsIterator, ++abscissaeIterator) {
    const FieldT& w = **weightsIterator;
    const FieldT& a = **abscissaeIterator;
	  result <<= result - cond( *sBar_ > *superSat_ && a < rMin_ && a > 0.0 && w > 1e-3, deathCoef_ * KAYDISSOLUTION)
                            (  0.0 );
    
  }
}

#endif // Dissolution_Expr_h
