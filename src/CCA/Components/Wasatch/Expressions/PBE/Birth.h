
#ifndef Birth_Expr_h
#define Birth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class Birth
 *  \author Alex Abboud	
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 
 *  \brief Implements any type of birth term
 *  takes the form of
 *  \f$ B  = J \times B_0(S) \times IB(r) \f$
 *  J is a predetermined coefficeint
 *  \f$ B_0(S) \f$ is an optional functionality for the coefficient
 *  \f$ IB \f$ is the integral of the birth source term
 *  \f$ IB(r) = \int_0^\inf B r^k dr \f$, (k = moment order)
 *  For Point and Uniform birth sources, an analytical 
 *  solution is possible and used here
 *  For Normal distribution birth, a 10-point trapezoid method is used,
 *  with limits set as +/- 3 \times std dev of \f$ r^* \f$
 */

//birth(birthTag, birthCoefTag, RStarTag, constcoef, momentOrder, birthModel, ConstRStar, stdDev)

template< typename FieldT >
class Birth
: public Expr::Expression<FieldT>
{
  const Expr::Tag birthCoefTag_, rStarTag_;  //this will correspond to proper tags for constant calc & momnet dependency
  const double constCoef_;   //"pre" coefficient
  const double momentOrder_; // this is the order of the moment equation in which the Birth rate is used
  const std::string birthModel_;
  const double constRStar_;
  const double stdDev_;
  const FieldT* birthCoef_; // this will correspond to the coefficient in the Birth rate term
  const FieldT* rStar_; // this will correspond to m(k + x) x depends on which Birth model
  
  Birth( const Expr::Tag& birthCoefTag,
         const Expr::Tag& rStarTag,
         const double constCoef,
         const double momentOrder, 
         const std::string birthModel,
         const double constRStar,
         const double stdDev);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result, 
             const Expr::Tag& birthCoefTag,
             const Expr::Tag& rStarTag,
             const double constCoef,
             const double momentOrder, 
             const std::string birthModel,
             const double constRStar,
             const double stdDev)
    : ExpressionBuilder(result),
    birthcoeft_ (birthCoefTag),
    rstart_     (rStarTag),
    constcoef_  (constCoef),
    momentorder_(momentOrder),
    birthmodel_ (birthModel),
    constrstar_ (constRStar),
    stddev_     (stdDev)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new Birth<FieldT>( birthcoeft_, rstart_, constcoef_, momentorder_, birthmodel_, constrstar_, stddev_ );
    }
    
  private:
    const Expr::Tag birthcoeft_, rstart_;
    const double constcoef_;
    const double momentorder_;
    const std::string birthmodel_;
    const double constrstar_;
    const double stddev_;
  };
  
  ~Birth();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
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
       const std::string birthModel,
       const double constRStar,
       const double stdDev)
: Expr::Expression<FieldT>(),
  birthCoefTag_(birthCoefTag),
  rStarTag_    (rStarTag),
  constCoef_   (constCoef),
  momentOrder_ (momentOrder),
  birthModel_  (birthModel),
  constRStar_  (constRStar),
  stdDev_      (stdDev)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Birth<FieldT>::
~Birth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if ( birthCoefTag_ != Expr::Tag () )
    exprDeps.requires_expression( birthCoefTag_ );
  if (rStarTag_ != Expr::Tag () ) 
    exprDeps.requires_expression( rStarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  if ( birthCoefTag_ != Expr::Tag () )
    birthCoef_ = &fm.field_ref( birthCoefTag_ );
  if (rStarTag_ != Expr::Tag () )
    rStar_ = &fm.field_ref( rStarTag_ ); 
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  if ( birthCoefTag_ != Expr::Tag () ) {
    if ( rStarTag_ != Expr::Tag () ) {
      if (birthModel_ == "POINT" ) {
        result <<= constCoef_ * *birthCoef_ * pow(*rStar_, momentOrder_); 
      } else if (birthModel_ == "UNIFORM" ) {
        result <<= constCoef_ * *birthCoef_ * ( pow(*rStar_ + stdDev_, momentOrder_ + 1) -
                                                pow(*rStar_ + stdDev_, momentOrder_ + 1)) / (momentOrder_ + 1);
      } else { //if (birthModel_ == "NORMAL" {
        typename FieldT::const_interior_iterator rStarIter = rStar_->interior_begin();
        typename FieldT::const_interior_iterator birthCoefIter = birthCoef_->interior_begin();
        typename FieldT::interior_iterator resultsIter = result.interior_begin();
        double IntVal;
        double dx;
        int npts = 10;
        while (rStarIter!=rStar_->interior_end() ) {
          std::vector <double> x;
          x = std::vector<double>(npts);
          dx = (6*stdDev_)/npts;
          x[0] = *rStarIter - 3*stdDev_;
          for(int i =1; i<npts; i++) {
            x[i] = x[i-1] + dx;
          }
          IntVal = 0.0;
          for(int i =0; i < npts-1; i++) { //trap integration, use external package in future?
            //.399 ~ 1/sqrt(2pi)
            IntVal = IntVal + dx/2/.399*( pow(x[i],momentOrder_)* exp(-stdDev_/2 * (x[i] - *rStarIter) * (x[i] - *rStarIter)) +
                                         pow(x[i+1],momentOrder_) * exp(-stdDev_/2 * (x[i+1] - *rStarIter) * (x[i+1] - *rStarIter)) );
          }
          *resultsIter = constCoef_ * *birthCoefIter * IntVal;
          ++resultsIter; 
          ++rStarIter;
          ++birthCoefIter;
        }
      }
      
    } else {// constant r*
      if (birthModel_ == "POINT" ) {
        result <<= constCoef_ * *birthCoef_ * pow(constRStar_, momentOrder_); 
      } else if (birthModel_ == "UNIFORM" ) {
        result <<= constCoef_ * *birthCoef_ * ( pow(constRStar_ + stdDev_, momentOrder_ + 1) -
                                                pow(constRStar_ + stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
      } else { //if (birthModel_ == "NORMAL" {
        std::vector <double> x;
        double dx;
        int npts = 10;
        x = std::vector<double>(npts);
        dx = (6*stdDev_)/npts;
        x[0] = constRStar_ - 3*stdDev_;
        for(int i =1; i < npts; i++) {
          x[i] = x[i-1] + dx;
        }
        double IntVal = 0.0;
        for(int i =0; i < npts-1; i++) { //trap integration, use external package in future?
          //.399 ~ 1/sqrt(2pi)
          IntVal = IntVal + dx/2/.399*( pow(x[i],momentOrder_)* exp(-stdDev_/2 * (x[i] - constRStar_) * (x[i] - constRStar_) ) +
                                        pow(x[i+1],momentOrder_) * exp(-stdDev_/2 * (x[i+1] - constRStar_) * (x[i+1] - constRStar_) ) );
        }
        result <<= constCoef_ * *birthCoef_ * IntVal;
      }
    }   
    
  } else { //const coeff
    if ( rStarTag_ != Expr::Tag () ) {
      if (birthModel_ == "POINT" ) {
        result <<= constCoef_ * pow(*rStar_, momentOrder_); 
      } else if (birthModel_ == "UNIFORM" ) {
        result <<= constCoef_  * ( pow(*rStar_ + stdDev_, momentOrder_ + 1) -
                                   pow(*rStar_ + stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
      } else { //if (birthModel_ == "NORMAL" {
        typename FieldT::const_interior_iterator rStarIter = rStar_->interior_begin();
        typename FieldT::interior_iterator resultsIter = result.interior_begin();
        double IntVal;
        double dx;
        int npts = 10;
        while (rStarIter!=rStar_->interior_end() ) {
          std::vector <double> x;
          x = std::vector<double>(npts);
          dx = (6*stdDev_)/npts;
          x[0] = *rStarIter - 3*stdDev_;
          for(int i =1; i<npts; i++) {
            x[i] = x[i-1] + dx;
          }
          IntVal = 0.0;
          for(int i =0; i < npts-1; i++) { //trap integration, use external package in future?
            //.399 ~ 1/sqrt(2pi)
            IntVal = IntVal + dx/2/.399*( pow(x[i],momentOrder_)* exp(-stdDev_/2 * (x[i] - *rStarIter) * (x[i] - *rStarIter)) +
                                         pow(x[i+1],momentOrder_) * exp(-stdDev_/2 * (x[i+1] - *rStarIter) * (x[i+1] - *rStarIter)) );
          }
          *resultsIter = constCoef_ * IntVal;
          ++resultsIter; 
          ++rStarIter;
        }
      }
      
    } else { // constant r* & const coef
      if (birthModel_ == "POINT" ) {
        result <<= constCoef_ * pow(constRStar_, momentOrder_); 
      } else if (birthModel_ == "UNIFORM" ) {
        result <<= constCoef_ * ( pow(constRStar_ + stdDev_, momentOrder_ + 1) -
                                  pow(constRStar_ + stdDev_, momentOrder_ + 1) ) / (momentOrder_ + 1);
      } else { //if (birthModel_ == "NORMAL" {
        std::vector <double> x;
        double dx;
        int npts = 10;
        x = std::vector<double>(npts);
        dx = (6*stdDev_)/npts;
        x[0] = constRStar_ - 3*stdDev_;
        for(int i =1; i<npts; i++) {
          x[i] = x[i-1] + dx;
        }
        double IntVal = 0.0;
        for(int i =0; i < npts-1; i++) { //trap integration use external package in future?
          //.399 ~ 1/sqrt(2pi)
          IntVal = IntVal + dx/2/.399*( pow(x[i],momentOrder_)* exp(-stdDev_/2 * (x[i] - constRStar_) * (x[i] - constRStar_)) +
                                        pow(x[i+1],momentOrder_) * exp(-stdDev_/2 * (x[i+1] - constRStar_) * (x[i+1] - constRStar_)) );
        }
        result <<= constCoef_ * IntVal;
      }
    }
  }
}

#endif
