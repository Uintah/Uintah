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
 *  \brief Evaluates density (\f$\rho\f$) and \f$\vec{\eta}\f$ if we are given
 *         \f$\rho \cdot \vec{\eta}\f$, \f$\vec{\theta}\f$ and
 *         \f$\rho=G(\vec{\eta},\vec{\theta})\f$.
 *
 *  In general, we have a system of nonlinear equations
 *  \f[
 *     \rho = G(\vec{\eta},\vec{\theta})
 *  \f]
 *  with
 *  \f[
 *     \eta_i = \frac{\rho\eta_i}{G(\vec{\eta},\vec{\theta})}
 *  \f]
 *  where we are given \f$\rho\vec{\eta}\f$ and \f$\theta\f$ but want to find
 *  \f$\rho\f$ and \f$\vec{\eta}\f$.  This is done by solving a nonlinear system
 *  of equations using newton's method.
 *
 *  Note that, although we could populate \f$\vec{\eta}\f$ as part of this expression,
 *  we currently only calculate the density.  This is because \f$\vec{\eta}\f$ is
 *  calculated elsewhere currently.
 *
 *  \tparam FieldT the type of field to build this density evaluator for.
 *    Currently, this class is instantiated only for SVolField.
 */
template< typename FieldT >
class DensityCalculator
: public Expr::Expression<FieldT>
{
  typedef std::vector<      FieldT*>  DepVarVec;
  typedef std::vector<const FieldT*>  IndepVarVec;

  typedef std::vector< typename FieldT::      iterator > VarIter;
  typedef std::vector< typename FieldT::const_iterator > ConstIter;

  typedef std::vector<double> DoubleVec;

  /*
   * rhoEta and reIEta mean "rhoeta included eta" and are using to show a
   * subset of the eta's which can be weighted by density.
   *
   * rhoetaExEta and reEEta mean "rhoeta excluded eta" and are using to show a
   * subset of the eta's which can NOT be weighted by density
   */
  const Expr::TagList rhoEtaTags_, etaTags_, orderedIvarTags_;
  Expr::TagList thetaTags_;

  const InterpT& evaluator_; ///< calculates \f$\rho=\mathcal{G}(\eta_1,\eta_2,\ldots,\eta_{n_\eta})\f$.

  IndepVarVec rhoEta_;      ///< density-weighted independent variables
  IndepVarVec theta_;       ///< non-density-weighted independent variables
  VarIter     rhoEtaIncEtaIters_;
  ConstIter   thetaIters_;
  ConstIter   rhoEtaIters_;

  DoubleVec etaPoint_;    ///< values of \f$\vec{\eta}\f$     at a given point on the mesh
  DoubleVec thetaPoint_;  ///< values of \f$\vec{\theta}\f$   at a given point on the mesh
  DoubleVec rhoEtaPoint_; ///< values of \f$\rho\vec{\eta}\f$ at a given point on the mesh

  std::vector<size_t> etaIndex_;  ///< locations of the eta variables in the full set of indep. vars. for the function.

  // Linear solver function variables - used in nonlinear_solver
  DoubleVec jac_;          ///< A vector for the jacobian matrix
  DoubleVec g_;            ///< A vector for the rhs functions in non-linear solver
  DoubleVec orderedIvars_; ///< A vector to store all independent variables in the same order as the table
  DoubleVec delta_, etaTmp_;
  std::vector<int> ipiv_;  ///< integer work array for linear solver

  DensityCalculator( const InterpT& evaluator,                ///< evaluate rho given etas & thetas
                     const Expr::TagList& rhoEtaTags,         ///< rho*eta tag
                     const Expr::TagList& etaTags,            ///< Tags for eta
                     const Expr::TagList& orderedIvarTags );  ///< Tag for all of the etas & thetas in the correct order

  bool nonlinear_solver( double& rho,                         ///< the density solution
                         std::vector<double>& eta,            ///< solution for \f$\eta\f$ corresponding to the density-weighted independent variables
                         const std::vector<double>& theta,    ///< non-density weighted independent variables
                         const std::vector<double>& rhoEta,   ///< density weighted independent variables
                         const std::vector<size_t>& etaIndex, ///< indices where the density-weighted independent variables should be included in the full set of independent variables.
                         const InterpT&,                      ///< interpolant for density
                         const double rtol );

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList rhoEtaTs_, etaTs_, thetaTs_, orderedIvarTs_;
    const InterpT* const interp_;
  public:
    /**
     * @param result the density
     * @param densEvaluator the interpolant that provides density as a function of the independent variables, \f$\eta_j\f$
     * @param rhoEtaTags Tag for each of the density-weighted independent variables, \f$\rho\eta_j\f$
     * @param etaTags Tags corresponding to the primitive variables, \f$\eta_j\f$
     * @param orderedIvarTags Tags corresponding to the full set of independent variables, \f$\vec{\eta}\f$ and \f$\vec{\theta}\f$.
     *   Note that this may be a superset of etaTags  set since there may be some
     *   variables that are not density weighted that we include in the analysis.
     */
    Builder( const Expr::Tag& result,
             const InterpT& densEvaluator,
             const Expr::TagList& rhoEtaTags,
             const Expr::TagList& etaTags,
             const Expr::TagList& orderedIvarTags );
    ~Builder(){ delete interp_; }
    Expr::ExpressionBase* build() const;
  };

  ~DensityCalculator();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

#endif // DensityCalculator_Expr_h
