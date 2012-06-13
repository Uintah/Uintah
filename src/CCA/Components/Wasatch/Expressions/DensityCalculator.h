/*
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

#include <tabprops/StateTable.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  DensityCalculator
 *  \author Amir Biglari
 *  \date   Feb, 2011
 *
 *  \brief Evaluates density and eta if we are given rho*eta, using TabProps,
 *         an external library for tabular property evaluation.
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

  const InterpT* const evaluator_;

  IndepVarVec rhoEta_;
  IndepVarVec rhoEtaExEta_;
  VarIter     rhoEtaIncEtaIters_;
  ConstIter   rhoEtaExEtaIters_;
  ConstIter   rhoEtaIters_;

  PointValues rhoEtaIncEtaPoint_;
  PointValues rhoEtaExEtaPoint_;
  PointValues rhoEtaPoint_;

  std::vector<int> ReIindex;

  // Linear solver function variables
  std::vector<double> J;           ///< A vector for the jacobian matrix
  std::vector<double> g;           ///< A vector for the functions
  std::vector<double> orderedEta;  ///< A vector to store all eta values in the same order as the table


  DensityCalculator( const InterpT* const spline,
                     const Expr::TagList& RhoEtaTags,           ///< rho*eta tag
                     const Expr::TagList& RhoetaIncEtaNames,    ///< Tag for ReIEta
                     const Expr::TagList& OrderedEtaTags );     ///< Tag for all of the eta's in the corect order

  bool nonlinear_solver( std::vector<double>& ReIeta,
                         const std::vector<double>& ReEeta,
                         const std::vector<double>& RhoEta,
                         std::vector<int>& ReIindex,
                         double& rho,
                         const InterpT&,
                         const double rtol );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList rhoEtaTs_, rhoEtaIncEtaNs_, rhoEtaExEtaNs_, orderedEtaTs_;
    const InterpT* const spline_;
  public:
    Builder( const Expr::Tag& result,
             const InterpT* const spline,
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



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
DensityCalculator<FieldT>::
DensityCalculator( const InterpT* const spline,
                   const Expr::TagList& rhoEtaTags,
                   const Expr::TagList& rhoetaIncEtaNames,
                   const Expr::TagList& orderedEtaTags )
: Expr::Expression<FieldT>(),
  rhoEtaTags_       ( rhoEtaTags        ),
  rhoEtaIncEtaNames_( rhoetaIncEtaNames ),
  orderedEtaTags_   ( orderedEtaTags    ),
  evaluator_        ( spline            )
{
  ReIindex.clear();
  int counter=0;
  bool match;
  for( typename Expr::TagList::const_iterator i=orderedEtaTags_.begin();
      i!=orderedEtaTags_.end();
      ++i, ++counter )
  {
    match=0;
    for( typename Expr::TagList::const_iterator j=rhoEtaIncEtaNames_.begin();
         j!=rhoEtaIncEtaNames_.end();
         ++j )
    {
      if (*i==*j) {
        ReIindex.push_back( counter );
        match=1;
        break;
      }
    }

    if (!match)  {
      rhoEtaExEtaNames_.push_back( *i );
    }
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
~DensityCalculator()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  for( Expr::TagList::const_iterator inam=rhoEtaTags_.begin(); inam!=rhoEtaTags_.end(); ++inam ){
    exprDeps.requires_expression( *inam );
  }
  for( Expr::TagList::const_iterator inam=rhoEtaExEtaNames_.begin(); inam!=rhoEtaExEtaNames_.end(); ++inam ){
    exprDeps.requires_expression( *inam );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();

  rhoEta_.clear();
  for( Expr::TagList::const_iterator inam=rhoEtaTags_.begin(); inam!=rhoEtaTags_.end(); ++inam ){
    rhoEta_.push_back( &fm.field_ref( *inam ) );
  }
  rhoEtaExEta_.clear();
  for( Expr::TagList::const_iterator inam=rhoEtaExEtaNames_.begin(); inam!=rhoEtaExEtaNames_.end(); ++inam ){
    rhoEtaExEta_.push_back( &fm.field_ref( *inam ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
evaluate()
{
  DepVarVec& results = this->get_value_vec();

  DepVarVec rhoEtaIncEta;
  typename DepVarVec::iterator iReIEta = results.begin();
  FieldT& rho = **iReIEta;
  ++iReIEta;
  for( ; iReIEta!=results.end(); ++iReIEta ){
    rhoEtaIncEta.push_back( *iReIEta );
  }

  rhoEtaIncEtaIters_.clear();
  rhoEtaExEtaIters_.clear();
  rhoEtaIters_.clear();

  for( typename DepVarVec::iterator i=rhoEtaIncEta.begin(); i!=rhoEtaIncEta.end(); ++i ){
    rhoEtaIncEtaIters_.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::iterator i=rhoEtaExEta_.begin(); i!=rhoEtaExEta_.end(); ++i ){
    rhoEtaExEtaIters_.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::const_iterator i=rhoEta_.begin(); i!=rhoEta_.end(); ++i ){
    rhoEtaIters_.push_back( (*i)->begin() );
  }

  unsigned int convergeFailed=0;

  // loop over grid points.
  for( typename FieldT::iterator irho= rho.begin(); irho!=rho.end(); ++irho ){

    // extract indep vars at this grid point
    rhoEtaIncEtaPoint_.clear();
    rhoEtaPoint_.clear();
    for( typename ConstIter::const_iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i ){
      rhoEtaPoint_.push_back( **i );
      rhoEtaIncEtaPoint_.push_back( **i / (*irho + 1e-11) );
    }
    rhoEtaExEtaPoint_.clear();
    for( typename ConstIter::const_iterator i=rhoEtaExEtaIters_.begin(); i!=rhoEtaExEtaIters_.end(); ++i ){
      rhoEtaExEtaPoint_.push_back( **i );
    }

    // calculate the result
    const bool converged = nonlinear_solver( rhoEtaIncEtaPoint_, rhoEtaExEtaPoint_, rhoEtaPoint_, ReIindex, *irho , *evaluator_, 1e-9 );
    if( !converged )  ++convergeFailed;

    // increment all iterators to the next grid point
    typename PointValues::const_iterator iReIEta=rhoEtaIncEtaPoint_.begin();
    for( typename VarIter::iterator i=rhoEtaIncEtaIters_.begin(); i!=rhoEtaIncEtaIters_.end(); ++i, ++iReIEta ){
      **i = *iReIEta;
      ++(*i);
    }
    for( typename ConstIter::iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i )  ++(*i);
    for( typename ConstIter::iterator i=rhoEtaExEtaIters_.begin(); i!=rhoEtaExEtaIters_.end(); ++i )  ++(*i);

  } // grid loop

  if( convergeFailed>0 )
    std::cout << convergeFailed << " of " << rho.window_with_ghost().local_npts() << " points failed to converge on density solver" << std::endl;
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const InterpT* const spline,
                  const Expr::TagList& rhoEtaTags,
                  const Expr::TagList& rhoEtaIncEtaNames,
                  const Expr::TagList& orderedEtaTags )
: ExpressionBuilder(result),
  rhoEtaTs_      ( rhoEtaTags        ),
  rhoEtaIncEtaNs_( rhoEtaIncEtaNames ),
  orderedEtaTs_  ( orderedEtaTags    ),
  spline_        ( spline            )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensityCalculator<FieldT>::
Builder::build() const
{
  return new DensityCalculator<FieldT>( spline_, rhoEtaTs_, rhoEtaIncEtaNs_, orderedEtaTs_);
}

//====================================================================

extern "C"{
  void dgetrf_( const int* m,
                const int* n,
                double A[],
                const int* lda,
                int* ipiv,
                int* info );

  void dgetrs_( const char* trans,
                const int* n,
                const int* nrhs,
                const double A[],
                const int* lda,
                const int* ipiv,
                double b[],
                const int* ldb,
                int* info );
}

template< typename FieldT >
bool
DensityCalculator<FieldT>::nonlinear_solver( std::vector<double>& ReIeta,
                                             const std::vector<double>& ReEeta,
                                             const std::vector<double>& rhoEta,
                                             std::vector<int>& ReIindex,
                                             double& rho,
                                             const InterpT& eval,
                                             const double rtol )
{
  const size_t neq = rhoEta.size();
  unsigned int itCounter = 0;
  const unsigned int maxIter = 100;
//  double rlxFac = 1.0;      // jcs note that if you apply relaxation you need to update your guess for rho...

  if( neq==0 ){
    orderedEta.clear();
    orderedEta=ReEeta;
    rho = eval.value(orderedEta);
  }
  else{

    J.resize(neq*neq);                // A vector for the jacobian matrix
    g.resize(neq);                     // A vector for the functions
    std::vector<int> ipiv(neq);        // work array for linear solver

    PointValues ReIetaTmp(ReIeta);
    double relErr=rtol+1;

    do{
      ++itCounter;
      // Create the ordered eta vector
      orderedEta.clear();
      orderedEta=ReEeta;
      for( size_t i=0; i<ReIindex.size(); i++ ) {
        orderedEta.insert(orderedEta.begin()+ReIindex[i],ReIeta[i]);
      }

      const double rhotmp = eval.value( orderedEta );
      const double shiftFactor = 0.001;  // multiplier for finite difference approximation in jacobian

      // Loop over different etas
      for( size_t k=0; k<neq; ++k ){
        for( size_t i=0; i<neq; ++i ) {
          if( k==i )
            ReIetaTmp[i]=ReIeta[k] + (std::abs(ReIeta[k])+rtol) * shiftFactor;
          else
            ReIetaTmp[i] = ReIeta[i];
        }

        // Recreate the ordered eta vector
        orderedEta.clear();
        orderedEta=ReEeta;
        for( size_t i=0; i<ReIindex.size(); i++ ) {
          orderedEta.insert( orderedEta.begin()+ReIindex[i], ReIetaTmp[i] );
        }

        const double rhoplus = eval.value(&orderedEta[0]);

        // Calculating the function members
        g[k] = -( ReIeta[k] - (rhoEta[k] / rhotmp));

        for( size_t i=0; i<neq; ++i ) {
          J[i + k*neq] = (( ReIetaTmp[i] - rhoEta[i]/rhoplus ) - ( ReIeta[i] - rhoEta[i]/rhotmp )) / ((std::abs(ReIeta[k])+ rtol) * shiftFactor );
        }
      }

      // Solve the linear system
      const char mode = 'n';
      int one=1, info;

      // jcs why not use dgesv instead?
      // Factor general matrix J using Gaussian elimination with partial pivoting
      const int numEqns = neq;
      dgetrf_(&numEqns, &numEqns, &J[0], &numEqns, &ipiv[0], &info);
      assert( info==0 );
      // Solving J * delta = g
      // note that on entry, g is the rhs and on exit, g is the solution (delta).
      dgetrs_(&mode, &numEqns, &one, &J[0], &numEqns, &ipiv[0], &g[0], &numEqns, &info);
      assert(info==0);

      relErr = 0.0;
      for( size_t i=0; i<neq; ++i ){
        // jcs note that if you apply relaxation you need to update your guess for rho...
        ReIeta[i] += g[i]; //*rlxFac;
        relErr += std::abs( g[i]/(std::abs(ReIeta[i])+rtol) );
      }

      rho = rhotmp;  // Update rho

    } while ( relErr>rtol && itCounter<maxIter );

    if( itCounter >= maxIter ){
//      std::cout << itCounter << " problems!  " << rho << " , " << relErr << ", " << g[0] << ", " << ReIeta[0] << std::endl;
      return false;
    }
//    std::cout << "converged in " << itCounter << " iterations.  eta=" << ReIeta[0] << ", rhoeta=" << rho*ReIeta[0] << ", rho=" << rho << std::endl;
  }
  return true;
}

//====================================================================

#endif // DensityCalculator_Expr_h
