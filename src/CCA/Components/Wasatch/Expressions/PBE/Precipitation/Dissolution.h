/*
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
  
  const double momentOrder_;    // order of the current moment
  const double rMin_;           //smallest radius possible before deaht occurs
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

  typename FieldT::interior_iterator resultsIter = result.interior_begin();
  int nEnv = weights_.size();
  const FieldT* sampleField = weights_[0];
  typename FieldT::const_interior_iterator sampleIterator = sampleField->interior_begin();
  
  std::vector<typename FieldT::const_interior_iterator> weightIterators;
  std::vector<typename FieldT::const_interior_iterator> abscissaeIterators;
  for( size_t i=0; i<nEnv; ++i ){
    typename FieldT::const_interior_iterator thisIterator = weights_[i]->interior_begin();
    weightIterators.push_back(thisIterator);
    
    typename FieldT::const_interior_iterator otherIterator = abscissae_[i]->interior_begin();
    abscissaeIterators.push_back(otherIterator);
  }
  typename FieldT::const_interior_iterator sBarIterator = sBar_->interior_begin();
  typename FieldT::const_interior_iterator superSatIterator = superSat_->interior_begin();
  
  while( sampleIterator != sampleField->interior_end() ){
    double sum = 0.0;
    
    for( size_t i = 0; i<nEnv; ++i ){
      if (*sBarIterator > *superSatIterator && *abscissaeIterators[i] < rMin_ &&
          *weightIterators[i] > 1.0e-3 && *abscissaeIterators[i] > 0.0) {
        const double k = 0.25 * (1.0 - erf(8.0*log(*abscissaeIterators[i]/rMin_))) *
                                (1.0 - erf(5.0*(-2.5-log10(*weightIterators[i]))) );
        sum = sum  -30.0 * k * *weightIterators[i] * pow(*abscissaeIterators[i], momentOrder_);
      }
    }
    *resultsIter = sum * deathCoef_;
    
    ++superSatIterator;
    ++sBarIterator;
    ++sampleIterator;
    ++resultsIter;
    for( size_t i = 0; i<nEnv; i++ ){
      ++weightIterators[i];
      ++abscissaeIterators[i];
    }
  }
}

#endif // Dissolution_Expr_h
