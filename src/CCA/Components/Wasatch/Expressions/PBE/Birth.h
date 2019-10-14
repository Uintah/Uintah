/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef Birth_Expr_h
#define Birth_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class Birth
 *  \authors Alex Abboud, Tony Saad
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.

 *  \brief Implements any type of birth term
 *  takes the form of
 *  \f$ B  = J  B_0(S)  IB(r) \f$
 *  J is a predetermined coefficeint
 *  \f$ B_0(S) \f$ is an optional functionality for the coefficient
 *  \f$ IB \f$ is the integral of the birth source term
 *  \f$ IB(r) = \int_0^\infty B r^k dr \f$, (k = moment order)
 *  For Point and Uniform birth sources, an analytical
 *  solution is possible and used here
 *  For Normal distribution birth, a 10-point trapezoid method is used,
 *  with limits set as +/- 3 times std dev of \f$ r^* \f$
 */

//birth(birthTag, birthCoefTag, RStarTag, constcoef, momentOrder, birthModel, ConstRStar, stdDev)

template< typename FieldT >
class Birth
: public Expr::Expression<FieldT>
{
public:
  enum BirthModel { POINT, UNIFORM, NORMAL };
    
private:    
//  const Expr::Tag birthCoefTag_, rStarTag_;  //this will correspond to proper tags for constant calc & momnet dependency
  const double constCoef_;   //"pre" coefficient
  const double momentOrder_; // this is the order of the moment equation in which the Birth rate is used
  const BirthModel birthType_;  //enum for birth model
  const double constRStar_;
  const double stdDev_;
  const bool doBirth_, doRStar_;
//  const FieldT* birthCoef_; // this will correspond to the coefficient in the Birth rate term
//  const FieldT* rStar_; // this will correspond to m(k + x) x depends on which Birth model
  
  DECLARE_FIELDS(FieldT, birthCoef_, rStar_)

  Birth( const Expr::Tag& birthCoefTag,
         const Expr::Tag& rStarTag,
         const double constCoef,
         const double momentOrder,
         const BirthModel birthType,
         const double constRStar,
         const double stdDev);
  
  double integrate_birth_kernel( const double rStar );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& birthCoefTag,
             const Expr::Tag& rStarTag,
             const double constCoef,
             const double momentOrder,
             const BirthModel birthType,
             const double constRStar,
             const double stdDev)
    : ExpressionBuilder(result),
    birthcoeft_ (birthCoefTag),
    rstart_     (rStarTag),
    constcoef_  (constCoef),
    momentorder_(momentOrder),
    birthtype_  (birthType),
    constrstar_ (constRStar),
    stddev_     (stdDev)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new Birth<FieldT>( birthcoeft_, rstart_, constcoef_, momentorder_, birthtype_, constrstar_, stddev_ );
    }

  private:
    const Expr::Tag birthcoeft_, rstart_;
    const double constcoef_;
    const double momentorder_;
    const BirthModel birthtype_;
    const double constrstar_;
    const double stddev_;
  };

  ~Birth();

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
Birth<FieldT>::
Birth( const Expr::Tag& birthCoefTag,
       const Expr::Tag& rStarTag,
       const double constCoef,
       const double momentOrder,
       const BirthModel birthType,
       const double constRStar,
       const double stdDev)
: Expr::Expression<FieldT>(),
  constCoef_   (constCoef),
  momentOrder_ (momentOrder),
  birthType_   (birthType),
  constRStar_  (constRStar),
  stdDev_      (stdDev),
  doBirth_     (birthCoefTag != Expr::Tag()),
  doRStar_     (birthCoefTag != Expr::Tag())
{
  if (doBirth_)  birthCoef_ = this->template create_field_request<FieldT>(birthCoefTag);
  if (doRStar_)  rStar_ = this->template create_field_request<FieldT>(rStarTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
Birth<FieldT>::
~Birth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
inline double
Birth<FieldT>::integrate_birth_kernel( const double rStar ) {
  // trapezoidal integration
  const int npts = 10;
  double x[npts];
  const double dx = (6.0*stdDev_)/npts;
  
  x[0] = rStar - 3.0*stdDev_;
  for( int i=1; i<npts; ++i )
    x[i] = x[i-1] + dx;
  
  double intVal = 0.0;
  for(int i =0; i < npts-1; ++i ){
    const double xa = x[i]   - rStar;
    const double xb = x[i+1] - rStar;
    //.399 ~ 1/sqrt(2pi)
    intVal += dx/2.0/0.399*(  pow(x[i]  ,momentOrder_) * exp(-stdDev_/2.0 * xa*xa)
                            + pow(x[i+1],momentOrder_) * exp(-stdDev_/2.0 * xb*xb) );
  }
  return intVal;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  if ( doBirth_ ) {
    const FieldT& birthCoef = birthCoef_->field_ref();
    if ( doRStar_ ) {
      const FieldT& rStar = rStar_->field_ref();
      switch (birthType_) {
        case POINT:
          result <<= constCoef_ * birthCoef * pow(rStar, momentOrder_);
          break;
        case UNIFORM:
          result <<= constCoef_ * birthCoef * ( pow(rStar + stdDev_, momentOrder_ + 1) -
                                                  pow(rStar - stdDev_, momentOrder_ + 1)) / (momentOrder_ + 1);
          break;
        case NORMAL: {
          typename FieldT::const_iterator rStarIter = rStar.interior_begin();
          typename FieldT::const_iterator birthCoefIter = birthCoef.interior_begin();
          typename FieldT::iterator resultsIter = result.interior_begin();
          while (rStarIter!=rStar.interior_end() ) {
            const double intVal = integrate_birth_kernel(*rStarIter);
            *resultsIter = constCoef_ * *birthCoefIter * intVal;
            ++resultsIter;
            ++rStarIter;
            ++birthCoefIter;
          }
          break;
        }
        default:
          break;
      } // SWITCH
      
    } else {// constant r*
      
      switch (birthType_) {
        case POINT:
          result <<= constCoef_ * birthCoef * pow(constRStar_, momentOrder_);
          break;
        case UNIFORM:
          result <<= constCoef_ * birthCoef * ( pow(constRStar_ + stdDev_, momentOrder_ + 1) -
                                                 pow(constRStar_ - stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
          break;
        case NORMAL: {
          const double intVal = integrate_birth_kernel(constRStar_);
          result <<= constCoef_ * birthCoef * intVal;
          break;
        }
        default:
          break;
      } // SWITCH
      
    }

  } else { //const birth coefficient
    if ( doRStar_ ) {
      const FieldT& rStar = rStar_->field_ref();
      switch (birthType_) {
        case POINT:
          result <<= constCoef_ * pow(rStar, momentOrder_);
          break;
        case UNIFORM:
          result <<= constCoef_  * ( pow(rStar + stdDev_, momentOrder_ + 1) -
                                     pow(rStar - stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
          break;
        case NORMAL: {
          typename FieldT::const_iterator rStarIter = rStar.interior_begin();
          typename FieldT::iterator resultsIter = result.interior_begin();
          while (rStarIter!=rStar.interior_end() ) {
            const double intVal = integrate_birth_kernel(*rStarIter);
            *resultsIter = constCoef_ * intVal;
            ++resultsIter;
            ++rStarIter;
          }
          break;
        }
        default:
          break;
      } // SWITCH

    } else { // constant r* & const coef
      
      switch (birthType_) {
        case POINT:
          result <<= constCoef_ * pow(constRStar_, momentOrder_);
          break;
        case UNIFORM:
          result <<= constCoef_ * ( pow(constRStar_ + stdDev_, momentOrder_ + 1) -
                                    pow(constRStar_ - stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
          break;
        case NORMAL: {
          const double intVal = integrate_birth_kernel(constRStar_);
          result <<= constCoef_ * intVal;
          break;
        }
        default:
          break;
      } // SWITCH
      
    }
  }
}

#endif
