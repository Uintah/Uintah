#ifndef DensityFromZChi_h
#define DensityFromZChi_h


#include <expression/Expression.h>
#include <spatialops/OperatorDatabase.h>
#include <boost/math/special_functions/erf.hpp>
#include <expression/matrix-assembly/Compounds.h>
#include <expression/matrix-assembly/MatrixExpression.h>
#include <expression/matrix-assembly/MapUtilities.h>
#include <expression/matrix-assembly/DenseSubMatrix.h>

#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

#ifndef ZTOL
#  define ZTOL 1.e-10
#endif

typedef SpatialOps::SVolField ScalarT;

template< typename ScalarT >
class DensityFromZChi : public Expr::Expression<ScalarT>
{
  typedef typename SpatialOps::FaceTypes<ScalarT>::XFace FluxX;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,    ScalarT, FluxX>::type sfGradT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, FluxX,   ScalarT>::type fsInterpT;

public:

  class Builder : public Expr::ExpressionBuilder
  {
  private:
    const Expr::Tag fOldTag_, chimaxOldTag_, rhoFTag_;
    const InterpT* const rhoEval_;
    const InterpT* const DEval_;

    const double rTol_;
    const double s_;
    const int maxIter_;
    const bool logscale_;

  public:

    Builder( const Expr::Tag rhoNewTag,
             const Expr::Tag chimaxNewTag,
             const Expr::Tag fOldTag,
             const Expr::Tag chimaxOldTag,
             const Expr::Tag rhoFTag,
             const InterpT&   rhoEval,
             const InterpT&   DEval,
             const double rTol,
             const int maxIter,
             const bool logscale,
             const double sNewton,
             const SpatialOps::GhostData nghost = DEFAULT_NUMBER_OF_GHOSTS )
    : ExpressionBuilder( tag_list(rhoNewTag, chimaxNewTag), nghost ),
      fOldTag_ ( fOldTag ),
      chimaxOldTag_ ( chimaxOldTag ),
      rhoFTag_ ( rhoFTag ),
      rhoEval_ ( rhoEval.clone() ),
      DEval_ ( DEval.clone() ),
      rTol_(rTol),
      maxIter_(maxIter),
      logscale_(logscale),
      s_(sNewton)
    {}

    ~Builder(){delete rhoEval_;  delete DEval_; }

    Expr::ExpressionBase* build() const{
      return new DensityFromZChi<ScalarT>( fOldTag_, chimaxOldTag_, rhoFTag_, *rhoEval_, *DEval_, rTol_, maxIter_, logscale_, s_ );
    }
  };

  void evaluate();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );

  protected:

  const int  numVariables_=2;

  DensityFromZChi( 
                  const Expr::Tag& fOldTag,
                  const Expr::Tag& chimaxOldTag,
                  const Expr::Tag& rhoFTag,
                  const InterpT&   rhoEval,
                  const InterpT&   DEval,
                  const double rTol,
                  const int maxIter,
                  const bool logscale,
                  const double sNewton
                );

  ~DensityFromZChi(){}
  
  const sfGradT*   gradOp_;
  const fsInterpT* interpOp_;

  const InterpT& rhoEval_;
  const InterpT& DEval_;

  const double fMin_;
  const double fMax_;
  const double xMin_;
  const double xMax_;
  const double  rTol_;
  const int  maxIter_;
  const bool logscale_;
  const double s_;
  
  DECLARE_FIELDS( ScalarT, fOld_, chimaxOld_, rhoF_ )

  class AbsVal{
  public:
    AbsVal(){}
    double operator()( const double z ) const{
      return std::abs(z);
    }
  };

  class Presumed{
    bool logscale_;
  public:
    Presumed(const bool logscale) : logscale_(logscale){}
    double operator()( const double z, const double chimax ) const{
      if(z<ZTOL || z>1. - ZTOL){return 1.e-16;}
      else{
        double erfinv_z = boost::math::erf_inv(2. * z - 1.);
        const double coeff = (logscale_) ? std::pow(10., chimax) : chimax;
        return coeff * std::exp(-2. * erfinv_z * erfinv_z);
      }
    }
  };

  class PresumedDchiref{
    bool logscale_;
  public:
    PresumedDchiref(const bool logscale) : logscale_(logscale){}
    double operator()( const double z, const double chimax ) const{
      if(z<ZTOL || z>1. - ZTOL){return 1.e-16;}
      else{
        double erfinv_z = boost::math::erf_inv(2. * z - 1.);
        const double coeff = (logscale_) ? std::log(10.) * std::pow(10., chimax) : 1.;
        return coeff * std::exp(-2. * erfinv_z * erfinv_z);
      }
    }
  };

  class PresumedDf{
    bool logscale_;
  public:
    PresumedDf(const bool logscale) : logscale_(logscale){}
    double operator()( const double z, const double chimax ) const{
      if(z<ZTOL || z>1. - ZTOL){return 1.e-16;}
      else{
        double erfinv_z = boost::math::erf_inv(2. * z - 1.);
        const double coeff = (logscale_) ? std::pow(10., chimax) : chimax;
        return coeff * -4. * std::sqrt(PI) * erfinv_z * std::exp(-erfinv_z * erfinv_z);
      }
    }
  };

  class InterpEval{
    const InterpT& eval_;
  public:
    InterpEval(const InterpT& interp) : eval_(interp){}
    double operator()( const double x1, const double x2 ) const{
      double vals[2] = {x1,x2};
      return eval_.value(vals);
    }
  };

  class InterpDerEval{
    const InterpT& eval_;
    const int dim_;
  public:
    InterpDerEval(const InterpT& interp, const int dim) : eval_(interp), dim_(dim){}
    double operator()( const double x1, const double x2 ) const{
      double vals[2] = {x1,x2};
      return eval_.derivative( vals, dim_ );
    }
  };

};


// ###################################################################
//
//                         Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template<typename ScalarT>
DensityFromZChi<ScalarT>::
DensityFromZChi(  const Expr::Tag& fOldTag,
                  const Expr::Tag& chimaxOldTag,
                  const Expr::Tag& rhoFTag,
                  const InterpT&   rhoEval,
                  const InterpT&   DEval,
                  const double rTol,
                  const int maxIter,
                  const bool logscale,
                  const double sNewton
                )
  : Expr::Expression<ScalarT>(),
    rhoEval_(rhoEval),
    DEval_(DEval),
    fMin_(rhoEval_.get_bounds()[0].first),
    fMax_(rhoEval_.get_bounds()[0].second),
    xMin_(rhoEval_.get_bounds()[1].first),
    xMax_(rhoEval_.get_bounds()[1].second),
    rTol_(rTol),
    maxIter_(maxIter),
    logscale_(logscale),
    s_(sNewton)
{
//   this->set_gpu_runnable( true );

  fOld_ = this->template create_field_request<ScalarT>( fOldTag );
  chimaxOld_ = this->template create_field_request<ScalarT>( chimaxOldTag );
  rhoF_ = this->template create_field_request<ScalarT>( rhoFTag );
}

//--------------------------------------------------------------------

template<typename ScalarT>
void
DensityFromZChi<ScalarT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<sfGradT  >();
  interpOp_ = opDB.retrieve_operator<fsInterpT>();
}

//--------------------------------------------------------------------

template<typename ScalarT>
void
DensityFromZChi<ScalarT>::evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<ScalarT>::ValVec&  results = this->get_value_vec();

  const ScalarT& rhoF = rhoF_ ->field_ref();
  const ScalarT& fOld = fOld_ ->field_ref();
  const ScalarT& chimaxOld = chimaxOld_ ->field_ref();

  SpatialOps::SpatFldPtr<ScalarT> f = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> chimax = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> gradf2 = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> rho = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> D = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> dRhodF = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> dRhodChimax = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> dDdF = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> dDdChimax = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> atBounds = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );
  SpatialOps::SpatFldPtr<ScalarT> chiP = SpatialOps::SpatialFieldStore::get<ScalarT>( fOld );

  // initial guess
  *f <<= fOld;
  *chimax <<= chimaxOld;

  // obtain memory for the jacobian and residual
  std::vector<SpatialOps::SpatFldPtr<ScalarT> > resPtrs, jacPtrs;
  for( int i=0; i<numVariables_; ++i ){
    resPtrs.push_back( SpatialOps::SpatialFieldStore::get<ScalarT>( fOld ) );

    for( int j=0; j<numVariables_; ++j ){
      jacPtrs.push_back( SpatialOps::SpatialFieldStore::get<ScalarT>( fOld ) );
    }
  }
  FieldVector<ScalarT> residualVec( resPtrs );
  FieldMatrix<ScalarT> jacobian   ( jacPtrs );

  unsigned numIter = 0;
  bool converged = false;
  double maxError = 0.;

  while(numIter<maxIter_ && !converged){
    ++numIter;

    *atBounds <<= cond( abs(*f-fMin_) < ZTOL, 1 )
                      ( abs(*f-fMax_) < ZTOL, 1 )
                      ( 0 );

    *gradf2 <<= (*interpOp_)((*gradOp_) ( *f )) * (*interpOp_)((*gradOp_) ( *f ));

    *rho <<= apply_pointwise(factory<InterpEval>(rhoEval_), *f, *chimax);
    *dRhodF <<= apply_pointwise(factory<InterpDerEval>(rhoEval_, 0), *f, *chimax);
    *dRhodChimax <<= apply_pointwise(factory<InterpDerEval>(rhoEval_, 1), *f, *chimax);

    *D <<= apply_pointwise(factory<InterpEval>(DEval_), *f, *chimax);
    *dDdF <<= apply_pointwise(factory<InterpDerEval>(DEval_, 0), *f, *chimax);
    *dDdChimax <<= apply_pointwise(factory<InterpDerEval>(DEval_, 1), *f, *chimax);

    *chiP <<=apply_pointwise(factory<Presumed>(logscale_), *f, *chimax);

    jacobian(0,0) <<= *rho + *f * *dRhodF;

    jacobian(0,1) <<=  cond( *atBounds > 0, 0 )
                          ( *f * *dRhodChimax ); 

    jacobian(1,0) <<= cond( *atBounds > 0, 0 )
                          ( apply_pointwise(factory<PresumedDf>(logscale_), *f, *chimax) - 2. * *gradf2 * *dDdF );

    jacobian(1,1) <<= cond( *atBounds > 0, 1 )
                          ( apply_pointwise(factory<PresumedDchiref>(logscale_), *f, *chimax) - 2. * *gradf2 * *dDdChimax );

    residualVec(0) <<= *rho * *f - rhoF;
    residualVec(1) <<= cond( *atBounds > 0, 0 )
                           ( *chiP - 2. * *D * *gradf2  );
    // solve for updates
    residualVec = jacobian.solve( residualVec );

    *f <<= *f - s_ * residualVec(0);
    *chimax <<= *chimax - s_ * residualVec(1);

    *f <<= max( min( *f, fMax_ ), fMin_ );
    *chimax <<= max( min( *chimax, xMax_ ), xMin_ );

    *gradf2 <<= (*interpOp_)((*gradOp_) ( *f )) * (*interpOp_)((*gradOp_) ( *f ));

    *rho <<= apply_pointwise(factory<InterpEval>(rhoEval_), *f, *chimax);
    *D <<= apply_pointwise(factory<InterpEval>(DEval_), *f, *chimax);
    *chiP <<=apply_pointwise(factory<Presumed>(logscale_), *f, *chimax);

    residualVec(0) <<= *rho * *f - rhoF;
    residualVec(1) <<= cond( *atBounds > 0, 0 )
                          ( *chiP - 2. * *D * *gradf2  );

    const double ff = nebo_max_interior(apply_pointwise(factory<AbsVal>(),residualVec(0)) / (apply_pointwise(factory<AbsVal>(),rhoF) + 1.e-4));
    const double fchimax = nebo_max_interior(apply_pointwise(factory<AbsVal>(),residualVec(1)) / (apply_pointwise(factory<AbsVal>(),*chiP) + 1.e-4));

    maxError = ff;
    if(fchimax>maxError){
      maxError = fchimax;
    }
    converged = (maxError <= rTol_);
  }

  // error if didn't converge
  if(!converged){
    std::cout << "\tSolve for f,chimax FAILED (max error = " << maxError << ") after " << numIter << " iterations.\n";
  }

  *results[0] <<= apply_pointwise(factory<InterpEval>(rhoEval_), *f, *chimax);
  *results[1] <<= *chimax;

}

#endif // DensityFromZChi_h
