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

#ifndef Wasatch_DensityFromMixFracAndHeatLossAndHeatLoss_h
#define Wasatch_DensityFromMixFracAndHeatLossAndHeatLoss_h

#include <tabprops/TabProps.h>

// #include <expression/Expression.h>

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityCalculatorBase.h>

namespace WasatchCore{


/**
 *  \class DensityFromMixFracAndHeatLoss
 *  \author James C. Sutherland, Josh McConnell
 *  \date January 2020
 *
 *  \brief When transporting \f$\rho f\f$ and \f$\rho h\f$, this expression will
 *  calculate \f$\rho\f$ and \f$\gamma\f$.  Note that \f$f\f$ and \f$h\f$ are
 *  calculated by other expressions once \f$\rho\f$ is known.
 *
 *  Given the density-weighted mixture fraction, \f$\rho f\f$ and density-weighted
 *  enthalpy, \f$\rho h\f$, this expression finds the density \f$\rho\f$ and
 *  heat loss \f$\gamma\f$.  It is assumed that we have two functions:
 *  \f[
 *    \rho = \mathcal{G}_\rho (f,\gamma) \\
 *    h = \mathcal{G}_h (f,\gamma)
 *  \f]
 *  that calculate \f$\rho\f$ and \f$h\f$ from \f$f\f$ and \f$\gamma\f$.
 *  In residual form, the equations to be solved are
 *  \f[
 *   r_1 = f \mathcal{G}_\rho \rho f \\
 *   r_2 = \mathcal{G}_\rho \mathcal{G}_h -\rho h
 *  \f]
 *  and the Jacobian matrix is
 *  \f[
 *  J=\left[
 *    \begin{array}{cc}
 *        \mathcal{G}_{\rho}+f \frac{\partial \mathcal{G}_{\rho}}{\partial f}
 *      & f \frac{\partial \mathcal{G}_{\rho}}{\partial \gamma} \\
 *        \mathcal{G}_{\rho}\frac{\partial \mathcal{G}_{h}}{\partial f}     +h\frac{\partial \mathcal{G}_{\rho}}{\partial f}
 *      & \mathcal{G}_{\rho}\frac{\partial \mathcal{G}_{h}}{\partial \gamma}+\mathcal{G}_{h}\frac{\partial \mathcal{G}_{\rho}}{\partial \gamma}
 *      \end{array}
 *    \right]
 *  \f]
 */
  template< typename FieldT >
  class DensityFromMixFracAndHeatLoss : protected DensityCalculatorBase<FieldT>
  {
    const InterpT& rhoEval_;
    const InterpT& enthEval_;
    const Expr::Tag& fOldTag_;
    const Expr::Tag& hOldTag_;
    const Expr::Tag& gammaOldTag_;
    const Expr::Tag& fNewTag_;
    const Expr::Tag& hNewTag_;
    const Expr::Tag& gammaNewTag_;
    const Expr::Tag& dRhodFTag_;
    const Expr::Tag& dRhodHTag_;
    const Expr::Tag& rhoFTag_;
    const Expr::Tag& rhoHTag_;
    const std::pair<double,double> fBounds_, gammaBounds_;

    Expr::TagList jacobianTags_;

    DECLARE_FIELDS(FieldT, rhoOld_, rhoF_, rhoH_, fOld_, hOld_, gammaOld_)
    
    DensityFromMixFracAndHeatLoss( const InterpT& rhoEval,
                                   const InterpT& enthEval,
                                   const Expr::Tag& rhoOldTag,
                                   const Expr::Tag& rhoFTag,
                                   const Expr::Tag& rhoHTag,
                                   const Expr::Tag& fOldTag,
                                   const Expr::Tag& hOldTag,
                                   const Expr::Tag& gammaOldTag,
                                   const double rtol,
                                   const unsigned maxIter );

    inline double get_normalization_factor( const unsigned i ) const{
      return 0.5; // nominal value for mixture fraction and heat loss
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
       *  @param gammaNewTag heat loss computed by this expression
       *  @param dRhodFTag derivative of density w.r.t. mixture fraction computed by this expression
       *  @param dRhodFTag derivative of density w.r.t.enthalpy computed by this expression
       *  @param rhoEval   reference to a density evaluation table
       *  @param rhoEval   reference to an enthalpy evaluation table
       *  @param rhoOldTag the density from the previous timestep (used as a guess)
       *  @param fOldTag the density from the previous timestep (used as a guess)
       *  @param hOldTag the enthalpy from the previous timestep (used as a guess)
       *  @param gammaOldTag the heat loss from the previous timestep (used as a guess)
       *  @param rhoFTag the density weighted mixture fraction
       *  @param rhoHTag the density weighted enthalpy
       *  @param rTol the relative solver tolerance
       *  @param maxIter maximum number of solver iterations allowed
       */
      Builder( const Expr::Tag rhoNewTag,
               const Expr::Tag gammaNewTag,
               const Expr::Tag dRhodFTag,
               const Expr::Tag dRhodHTag,
               const InterpT&  rhoEval,
               const InterpT&  enthEval,
               const Expr::Tag rhoOldTag,
               const Expr::Tag rhoFTag,
               const Expr::Tag rhoHTag,
               const Expr::Tag fOldTag,
               const Expr::Tag hOldTag,
               const Expr::Tag gammaOldTag,
               const double rTol,
               const unsigned maxIter );
      
      ~Builder(){ delete rhoEval_;  delete enthEval_;  }
      Expr::ExpressionBase* build() const{
        return new DensityFromMixFracAndHeatLoss<FieldT>( *rhoEval_, *enthEval_, rhoOldTag_, rhoFTag_, rhoHTag_, 
                                                          fOldTag_, hOldTag_, gammaOldTag_, rtol_, maxIter_ );
      }

    private:
      const InterpT* const rhoEval_;
      const InterpT* const enthEval_;
      const Expr::Tag rhoOldTag_, rhoFTag_, rhoHTag_, fOldTag_, hOldTag_, gammaOldTag_;
      const double rtol_;    ///< relative error tolerance
      const unsigned maxIter_; ///< maximum number of iterations    
    };

    // void bind_operators( const SpatialOps::OperatorDatabase& opDB );

    ~DensityFromMixFracAndHeatLoss();
    void set_initial_guesses();
    Expr::IDSet register_local_expressions();
    void evaluate();
  };

}


#endif // Wasatch_DensityFromMixFracAndHeatLossAndHeatLoss_h
