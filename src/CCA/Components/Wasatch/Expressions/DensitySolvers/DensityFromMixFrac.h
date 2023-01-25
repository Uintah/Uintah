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

#ifndef Wasatch_DensityFromMixFrac_h
#define Wasatch_DensityFromMixFrac_h

#include <tabprops/TabProps.h>

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityCalculatorBase.h>

namespace WasatchCore{

  /**
   * \class DensityFromMixFrac
   *  \author James C. Sutherland, Josh McConnell
   *  \date January 2020
   *
   * Given \f$G_\rho(f)\f$ and \f$\rho f\f$, find \f$f\f$ and \f$\rho\f$.  This is
   * done by defining the residual equation
   *  \f[ r(f) = f G_\rho - (\rho f) \f]
   * with
   *  \f[ r^\prime(f) = \frac{\partial r}{\partial f} = G_\rho + f\frac{\partial G_\rho}{\partial f} \f]
   * so that the newton update is
   *  \f[ f^{new}=f - \frac{r(f)}{r^\prime(f)} \f].
   * 
   * See <CCA/Components/Wasatch/Expressions/DensitySolvers/Residual.h> for the residual expression.
   */
  template< typename FieldT >
  class DensityFromMixFrac : protected DensityCalculatorBase<FieldT>
  {
    const InterpT& rhoEval_;
    const Expr::Tag& fOldTag_;
    const Expr::Tag& fNewTag_;
    const Expr::Tag& dRhodFTag_;
    const Expr::Tag& rhoFTag_;
    const std::pair<double,double> fBounds_;
    DECLARE_FIELDS(FieldT, rhoOld_, rhoF_, fOld_)
    
    DensityFromMixFrac( const InterpT& rhoEval,
                        const Expr::Tag& rhoOldTag,
                        const Expr::Tag& rhoFTag,
                        const Expr::Tag& fOldTag,
                        const double rtol,
                        const unsigned maxIter );

    inline double get_normalization_factor( const unsigned i ) const{
      return 0.5; // nominal value for mixture fraction
    }

    inline const std::pair<double,double>& get_bounds( const unsigned i ) const{
      return fBounds_;
    }


  public:
    /**
     *  @class Builder
     *  @brief Build a DensFromMixfrac expression
     */
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @param rhoNewTag density computed by this expression
       *  @param dRhodFTag derivative of density w.r.t. mixture fraction computed by this expression
       *  @param rhoEval   reference to a density evaluation table
       *  @param rhoOldTag the density from the previous timestep (used as a guess)
       *  @param fOldTag the density from the previous timestep (used as a guess)
       *  @param rhoFTag the density weighted mixture fraction
       *  @param rTol the relative solver tolerance
       *  @param maxIter maximum number of solver iterations allowed
       */
      Builder( const Expr::Tag rhoNewTag,
               const Expr::Tag dRhodFTag,
               const InterpT&  rhoEval,
               const Expr::Tag rhoOldTag,
               const Expr::Tag rhoFTag,
               const Expr::Tag fOldTag,
               const double rtol,
               const unsigned maxIter );
      
      ~Builder(){ delete rhoEval_; }
      Expr::ExpressionBase* build() const{
        return new DensityFromMixFrac<FieldT>( *rhoEval_, rhoOldTag_, rhoFTag_, fOldTag_, rtol_, maxIter_ );
      }

    private:
      const InterpT* const rhoEval_;
      const Expr::Tag rhoOldTag_, rhoFTag_, fOldTag_;
      const double rtol_;    ///< relative error tolerance
      const unsigned maxIter_; ///< maximum number of iterations    
    };

    // void bind_operators( const SpatialOps::OperatorDatabase& opDB );

    ~DensityFromMixFrac();
    void set_initial_guesses();
    Expr::IDSet register_local_expressions();
    void evaluate();
  };

}


#endif // Wasatch_DensityFromMixFrac_h
