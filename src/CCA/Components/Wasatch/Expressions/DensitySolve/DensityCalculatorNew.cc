#include <CCA/Components/Wasatch/Expressions/DensitySolve/DensityCalculatorNew.h>
// #include <CCA/Components/Wasatch/Expressions/DensitySolve/NewtonUpdate.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolve/Residual.h>
// #include <CCA/Components/Wasatch/Expressions/DensitySolve/RelativeError.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>

#include <sci_defs/uintah_defs.h>

namespace WasatchCore{
namespace DelMe{

  using Expr::tag_list;

  template< typename FieldT >
  class OneVarNewtonSolve : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS(FieldT, xOld_, y_, dydx_)

    OneVarNewtonSolve( const Expr::Tag& xOldTag,
                       const Expr::Tag& yTag,
                       const Expr::Tag& dydxTag )
    : Expr::Expression<FieldT>()
    {
       this->set_gpu_runnable(true);
       xOld_ = this->template create_field_request<FieldT>(xOldTag);
       y_    = this->template create_field_request<FieldT>(yTag   );
       dydx_ = this->template create_field_request<FieldT>(dydxTag);
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a OneVarNewtonSolve expression
       *  @param resultTag tag to updated value of x for function y(x)
       *  @param xOldTag tag to old value of x
       *  @param yTag the tag to field for y(x)
       *  @param dydxTag tag to field for derivative of function y with respect to x
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& xOldTag,
               const Expr::Tag& yTag,
               const Expr::Tag& dydxTag )
      : ExpressionBuilder( resultTag ),
        xOldTag_( xOldTag ),
        yTag_   ( yTag    ),
        dydxTag_( dydxTag )
      {}

      Expr::ExpressionBase* build() const{
        return new OneVarNewtonSolve( xOldTag_, yTag_, dydxTag_ );
      }

    private:
      const Expr::Tag xOldTag_, yTag_, dydxTag_;
    };

    ~OneVarNewtonSolve(){}

    void evaluate(){
      using namespace SpatialOps;
      FieldT& xNew = this->value();
      xNew <<= xOld_->field_ref() - (y_->field_ref() / dydx_->field_ref());
    };
  };

  //===================================================================

  template< typename FieldT >
  DensFromMixfrac<FieldT>::
  DensFromMixfrac( const InterpT& rhoEval,
                  const Expr::Tag& rhoOldTag,
                  const Expr::Tag& rhoFTag, //rhoFTag will NOT be used if weakform is true.
                  const Expr::Tag& fTag,
                  const bool weakForm,
                  const double rTol,
                  const unsigned maxIter)
    : Expr::Expression<FieldT>(),
      DensityCalculatorBase( "DensFromMixFrac", tag_list(fTag), rTol, maxIter ),
      rhoEval_  ( rhoEval ),
      dRhodFTag_     ( "solver_d_rho_d_f"     , Expr::STATE_NONE ),
      dResidualdFTag_( "solver_d_residual_d_f", Expr::STATE_NONE ),
      bounds_   ( rhoEval.get_bounds()[0] ),
      weak_     ( weakForm )
  {
    assert(phiOldTags_  .size() == 1);
    assert(phiNewTags_  .size() == 1);
    assert(residualTags_.size() == 1);

    this->set_gpu_runnable(false);
    if (weak_) {
      f_ = this->template create_field_request<FieldT>(fTag);
    } else {
      rhoF_ = this->template create_field_request<FieldT>(rhoFTag);
      rhoOld_ = this->template create_field_request<FieldT>(rhoOldTag);
    }
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensFromMixfrac<FieldT>::
  ~DensFromMixfrac()
  {}

  //--------------------------------------------------------------------

  template< typename FieldT >
  Expr::IDSet 
  DensFromMixfrac<FieldT>::
  register_local_expressions()
  {
    Expr::IDSet rootIDs;
    Expr::ExpressionFactory& factory = *helper_.factory_;

    Expr::ExpressionID id;

    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;
    typedef typename TabPropsEvaluator<FieldT>::Builder TPEval;

    factory.register_expression(new PlcHldr( rhoFTag_ ));
    factory.register_expression(new PlcHldr( fTag_ ));

    // compute residual
    factory.register_expression( new typename Residual<FieldT>::
                                 Builder( residualTags_,
                                          rhoPhiTags_,
                                          phiOldTags_,
                                          densityTag_ )
                                );

    // compute density from lookup table
    factory.register_expression( new TPEval( densityTag_, 
                                             rhoEval_,
                                             Expr::tag_list(fTag_)
                                            )
                                );
    // compute d(rho)/d(f) from lookup table
    factory.register_expression( new TPEval( dRhodFTag_, 
                                             rhoEval_,
                                             Expr::tag_list(fTag_),
                                             fTag_
                                            )
                                );

    factory.register_expression( new typename OneVarNewtonSolve<FieldT>::
                                 Builder( phiNewTags_[0], // tag for updated mixture fraction
                                          phiOldTags_[0], // tag to old mixture fraction
                                          residualTags_[0],
                                          dResidualdFTag_ )
                            );

    return rootIDs;
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensFromMixfrac<FieldT>::
  set_initial_guesses()
  {
      Expr::UintahFieldManager<FieldT>& fieldManager = helper_.fml_ -> template field_manager<FieldT>();
      FieldT& fOld = fieldManager.field_ref( fTag_ );
      fOld <<= f_->field_ref();

      FieldT& rhoF = fieldManager.field_ref( rhoFTag_ );
      rhoF <<= rhoF_->field_ref();
  }

  //--------------------------------------------------------------------


  template< typename FieldT >
  void
  DensFromMixfrac<FieldT>::
  evaluate()
  {
    typedef typename Expr::Expression<FieldT>::ValVec SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();
    
    FieldT& rho = *results[0];
    if (!weak_) rho <<= rhoOld_->field_ref();
    
    FieldT& badPts = *results[1];
    FieldT& drhodf = *results[2];
    badPts <<= 0.0;
    
    
    // if( nbad>0 && maxIter_ != 0){
    //   std::cout << "\tConvergence failed at " << nbad << " points.\n";
    // }
  }

  //--------------------------------------------------------------------

  // template<typename FieldT>
  // void
  // DensFromMixfrac<FieldT>::
  // calc_jacobian_and_res( const DensityCalculatorBase::DoubleVec& passThrough,
  //                        const DensityCalculatorBase::DoubleVec& soln,
  //                        DensityCalculatorBase::DoubleVec& jac,
  //                        DensityCalculatorBase::DoubleVec& res )
  // {
  //   const double rhoF = passThrough[0];
  //   const double& f = soln[0];
  //   const double rhoCalc = rhoEval_.value( &f );
  //   jac[0] = rhoCalc + f * rhoEval_.derivative( &f, 0 );
  //   res[0] = f * rhoCalc - rhoF;
  // }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensFromMixfrac<FieldT>::
  Builder::Builder( const InterpT& rhoEval,
                    const Expr::TagList& resultsTag,
                    const Expr::Tag& rhoOldTag,
                    const Expr::Tag& rhoFTag,
                    const Expr::Tag& fTag,
                    const bool weakForm,
                    const double rtol,
                    const unsigned maxIter)
    : ExpressionBuilder( resultsTag ),
      rhoEval_  (rhoEval.clone() ),
      rhoOldTag_(rhoOldTag       ),
      rhoFTag_  (rhoFTag         ),
      fTag_     (fTag            ),
      weakForm_ (weakForm        ),
      rtol_     (rtol            ),
      maxIter_  (maxIter         )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensFromMixfrac       <SpatialOps::SVolField>;

}
}
