/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef DensityCalculator_Expr_h
#define DensityCalculator_Expr_h

#include <tabprops/TabProps.h>

#include <expression/Expression.h>

class DensityCalculatorBase
{
public:
  typedef std::vector<double> DoubleVec;
protected:

  DensityCalculatorBase( const int neq,
                         const double rtol,
                         const size_t maxIter=20 );

  virtual ~DensityCalculatorBase();

  /**
   * Given the current guess for the solution, calculate the jacobian and the residual
   * @param soln the current guess for the solution
   * @param jac the jacobian matrix
   * @param res the RHS function
   */
  virtual void calc_jacobian_and_res( const DoubleVec& passThrough,
                                      const DoubleVec& soln,
                                      DoubleVec& jac,
                                      DoubleVec& res ) = 0;

  /**
   * @param soln the initial guess.  Returns as the solution
   * @return true if converged, false otherwise
   */
  bool solve( const DoubleVec& passThroughVals, DoubleVec& soln );

  /**
   * @return the value for each variable that should be used in normalizing the error for that equation.
   */
  virtual double get_normalization_factor( const unsigned i ) const =0;

  virtual std::pair<double,double> get_bounds( const unsigned i ) const =0;

private:
  const int neq_;        ///< number of equations to solve
  const double rtol_;    ///< relative error tolerance
  const size_t maxIter_; ///< maximum number of iterations
  DoubleVec jac_, res_;
  std::vector<int> ipiv_;  ///< integer work array for linear solver
};


/**
 * \class DensFromMixfrac
 *
 * Given \f$G_\rho(f)\f$ and \f$\rho f\f$, find \f$f\f$ and \f$\rho\f$.  This is
 * done by defining the residual equation
 *  \f[ r(f) = f G_\rho - (\rho f)\f]
 * with
 *  \f[ r^\prime(f) = \frac{\partial r}{\partial f} = G_\rho + f\frac{\partial G_\rho}{\partial f} \f]
 * so that the newton update is
 *  \f[ f^{new}=f - \frac{r(f)}{r^\prime(f)} \f].
 */
template< typename FieldT >
class DensFromMixfrac : public Expr::Expression<FieldT>, protected DensityCalculatorBase
{
  const Expr::Tag mixFracTag_, rhoFTag_;
  const FieldT* mixFrac_;
  const FieldT* rhoF_;
  const InterpT& rhoEval_;
  const std::pair<double,double> bounds_;

  DensFromMixfrac( const InterpT& rhoEval,
                   const Expr::Tag& rhoFTag );

  void calc_jacobian_and_res( const DoubleVec& passThrough,
                              const DoubleVec& soln,
                              DoubleVec& jac,
                              DoubleVec& res );
  double get_normalization_factor( const unsigned i ) const;
  std::pair<double,double> get_bounds( const unsigned i ) const;

public:
  /**
   *  @class Builder
   *  @brief Build a DensFromMixfrac expression
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @param rhoEval calculates density given mixture fraction
     *  @param resultTag the tag for the value that this expression computes
     *  @param rhoFTag the density weighted mixture fraction
     */
    Builder( const InterpT& rhoEval,
             const Expr::Tag& resultTag,
             const Expr::Tag& rhoFTag );
    ~Builder(){ delete rhoEval_; }
    Expr::ExpressionBase* build() const;

  private:
    const InterpT* const rhoEval_;
    const Expr::Tag rhoFTag_;
  };

  ~DensFromMixfrac();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};


/**
 *  \class DensHeatLossMixfrac
 *  \author James C. Sutherland
 *  \date November, 2013
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
 *   r_1 = f \mathcal{G}_\rho - \rho f \\
 *   r_2 = \mathcal{G}_\rho \mathcal{G}_h - \rho h
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
class DensHeatLossMixfrac
 : public Expr::Expression<FieldT>, protected DensityCalculatorBase
{
  const Expr::Tag hTag_, rhofTag_, rhohTag_;
  const FieldT *rhoh_, *rhof_;
  const InterpT &densEval_, &enthEval_;
  const std::vector< std::pair<double,double> > bounds_;

  DensHeatLossMixfrac( const Expr::Tag& hTag,
                       const Expr::Tag& rhofTag,
                       const Expr::Tag& rhohTag,
                       const InterpT& densEvaluator,
                       const InterpT& enthEvaluator );

  void calc_jacobian_and_res( const DoubleVec& passThrough,
                              const DoubleVec& soln,
                              DoubleVec& jac,
                              DoubleVec& res );
  double get_normalization_factor( const unsigned i ) const;
  std::pair<double,double> get_bounds( const unsigned i ) const;

public:
    /**
     * @class Builder
     * @brief Build a DensHeatLossMixfrac expression
     */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a DensHeatLossMixfrac expression
     *  @param rhoTag the density (calculated here) - should have CARRY_FORWARD context.
     *  @param gammaTag the heat loss parameter (calculated here)
     *  @param hTag the enthalpy - should have CARRY_FORWARD context.
     *  @param rhofTag the density-weighted mixture fraction
     *  @param rhohTag the density-weighted enthalpy
     */
    Builder( const Expr::Tag& rhoTag,
             const Expr::Tag& gammaTag,
             const Expr::Tag& hTag,
             const Expr::Tag& rhofTag,
             const Expr::Tag& rhohTag,
             const InterpT& densEvaluator,
             const InterpT& enthEvaluator );
    ~Builder(){ delete densEval_; delete enthEval_; }
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag hTag_, rhofTag_, rhohTag_;
    const InterpT * const densEval_, * const enthEval_;
  };

  ~DensHeatLossMixfrac();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};


#endif // DensityCalculator_Expr_h
