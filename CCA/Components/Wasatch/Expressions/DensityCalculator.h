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

/**
 *  \ingroup WasatchExpressions
 *  \class  DensityCalculator
 *  \author Amir Biglari
 *  \date   Feb, 2011
 *
 *  \brief Evaluates density (\f$\rho\f$) and \f$\eta\f$ if we are given
 *         \f$\rho \cdot \eta\f$, using TabProps.
 *
 *  In general, we have \f$\rho = \mathcal{G}(\eta_1,\eta_2,\ldots\eta_{n_\eta})\f$
 *  and we are given some mixture of \f$\eta_i\f$ and $\rho \eta_j\f$. Given this,
 *  we need to find the consistent values of \f$\rho$\f and $\eta_j\f$.  This is
 *  done by solving a nonlinear system of equations using newton's method.
 */
template< typename FieldT >
class DensityCalculator
: public Expr::Expression<FieldT>
{
  typedef std::vector<      FieldT*>  DepVarVec;
  typedef std::vector<const FieldT*>  IndepVarVec;

  typedef std::vector< typename FieldT::      iterator > VarIter;
  typedef std::vector< typename FieldT::const_iterator > ConstIter;

  typedef std::vector<double> PointValues;

  /* RhoetaIncEta and ReIEta mean "Rhoeta Included Eta" and are using to show a
   subset of the eta's which can be weighted by density */
  /* RhoetaExEta and ReEEta mean "Rhoeta Excluded Eta" and are using to show  a
   subset of the eta's which can NOT be weighted by density */
  const Expr::TagList rhoEtaTags_, rhoEtaIncEtaNames_, orderedEtaTags_;
  Expr::TagList rhoEtaExEtaNames_;

  const InterpT* const evaluator_; ///< calculates \f$\rho=\mathcal{G}(\eta_1,\eta_2,\ldots,\eta_{n_\eta})\f$.

  IndepVarVec rhoEta_;
  IndepVarVec rhoEtaExEta_;
  VarIter     rhoEtaIncEtaIters_;
  ConstIter   rhoEtaExEtaIters_;
  ConstIter   rhoEtaIters_;

  PointValues rhoEtaIncEtaPoint_;
  PointValues rhoEtaExEtaPoint_;
  PointValues rhoEtaPoint_;

  std::vector<int> reIindex_;

  // Linear solver function variables
  std::vector<double> jac_;         ///< A vector for the jacobian matrix
  std::vector<double> g_;           ///< A vector for the rhs functions in non-linear solver
  std::vector<double> orderedEta_;  ///< A vector to store all eta values in the same order as the table


  DensityCalculator( const InterpT* const evaluator,            ///< evaluate rho given etas
                     const Expr::TagList& rhoEtaTags,           ///< rho*eta tag
                     const Expr::TagList& rhoetaIncEtaNames,    ///< Tag for ReIEta
                     const Expr::TagList& orderedEtaTags );     ///< Tag for all of the eta's in the correct order

  bool nonlinear_solver( std::vector<double>& reIeta,
                         const std::vector<double>& reEeta,
                         const std::vector<double>& rhoEta,
                         const std::vector<int>& reIindex,
                         double& rho,
                         const InterpT&,
                         const double rtol );

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList rhoEtaTs_, rhoEtaIncEtaNs_, rhoEtaExEtaNs_, orderedEtaTs_;
    const InterpT* const spline_;
  public:
    /**
     * @param result the density
     * @param densEvaluator the interpolant that provides density as a function of the independent variables, \f$\eta_j\f$
     * @param rhoEtaTags Tag for each of the density-weighted independent variables, \f$\rho\eta_j\f$
     * @param rhoEtaIncEtaNames Tags corresponding to the primitive variables, \f$\eta_j\f$
     * @param orderedEtaTags Tags corresponding to the full set of independent variables, \f$\eta_i\f$.
     *   Note that this may be a superset of the rhoEtaIncEtaNames variable set since there may be
     *   some variables that are not density weighted that we include in the analysis.
     */
    Builder( const Expr::Tag& result,
             const InterpT* const densEvaluator,
             const Expr::TagList& rhoEtaTags,
             const Expr::TagList& rhoEtaIncEtaNames,
             const Expr::TagList& orderedEtaTags );

    Expr::ExpressionBase* build() const;
  };

  ~DensityCalculator();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

#endif // DensityCalculator_Expr_h
