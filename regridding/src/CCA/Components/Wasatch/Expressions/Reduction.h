#ifndef Reduction_Expr_h
#define Reduction_Expr_h

#include "ReductionBase.h"

/**
 *  \class Reduction
 *  \author Tony Saad
 *  \date   June, 2013
 *  \brief  Provides an expression-based approach to defining reduction variables.
 */
template< typename SrcFieldT,     // source field type: SVol, XVol...
          typename ReductionOpT > // the type of reduction operation: Reductions::Sum<double> etc...
class Reduction
 : public ReductionBase
{
  DECLARE_FIELD(SrcFieldT, src_)

  Reduction( const Expr::Tag& resultTag,
             const Expr::Tag& srcTag,
             const bool printVar=false );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a Reduction expression
     *  @param resultTag The tag for the value that this expression computes.
     *  @param srcTag The tag of the source field on which we want to apply reduction.
     *  @param printVar A boolean that specifies whether you want the reduced variable output to the cout stream.
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& srcTag,
             const bool printVar=false );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag resultTag_, srcTag_;
    bool printVar_;
  };

  ~Reduction();

  //--------------------------------------------------------------------
  void evaluate();
};


#endif // Reduction_Expr_h
