#include <CCA/Components/Wasatch/Expressions/DensitySolve/DensityCalculatorNew.h>
// #include <CCA/Components/Wasatch/Expressions/DensitySolve/NewtonUpdate.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolve/Residual.h>
// #include <CCA/Components/Wasatch/Expressions/DensitySolve/RelativeError.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <expression/ClipValue.h>

#include <sci_defs/uintah_defs.h>

namespace WasatchCore{
namespace DelMe{

  using Expr::tag_list;

    /**
   * \class OneVarNewtonSolve
   * @brief computes updated mixture fraction, \f[ f \f] given an old value of \f[ f \f],
   *        density (\f[ \rho \f]), \f[ \frac{\rho}{f} \f], and a residual with the following
   *        definition: 
   *        \f[ r(f) = (\rho f) - f G_\rho\f].
   * 
   */
  template< typename FieldT >
  class OneVarNewtonSolve : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS(FieldT, fOld_, rhoOld_, dRhodF_, residual_)

    OneVarNewtonSolve( const Expr::Tag& fOldTag,
                       const Expr::Tag& rhoOldTag,
                       const Expr::Tag& dRhodFTag,
                       const Expr::Tag& residualTag )
    : Expr::Expression<FieldT>()
    {
       this->set_gpu_runnable(true);
       fOld_     = this->template create_field_request<FieldT>( fOldTag     );
       rhoOld_   = this->template create_field_request<FieldT>( rhoOldTag   );
       dRhodF_   = this->template create_field_request<FieldT>( dRhodFTag   );
       residual_ = this->template create_field_request<FieldT>( residualTag );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a OneVarNewtonSolve expression
       *  @param resultTag tag to updated value of mixture fraction
       *  @param fOldTag tag to old value of mixture fraction
       *  @param rhoTag the tag to field for density
       *  @param dRhodFTag tag to field for derivative of density with respect to mixture fraction
       *  @param residualTag tag for residual \f[ r(f) = (\rho f) - f G_\rho \f]
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& fOldTag,
               const Expr::Tag& rhoOldTag,
               const Expr::Tag& dRhodFTag,
               const Expr::Tag& residualTag )
      : ExpressionBuilder( resultTag ),
        fOldTag_    ( fOldTag     ),
        rhoOldTag_  ( rhoOldTag   ),
        dRhodFTag_  ( dRhodFTag   ),
        residualTag_( residualTag )
      {}

      Expr::ExpressionBase* build() const{
        return new OneVarNewtonSolve( fOldTag_, rhoOldTag_, dRhodFTag_, residualTag_ );
      }

    private:
      const Expr::Tag fOldTag_, rhoOldTag_, dRhodFTag_, residualTag_;
    };

    ~OneVarNewtonSolve(){}

    void evaluate(){
      using namespace SpatialOps;
      FieldT& fNew = this->value();

      const FieldT& fOld   = fOld_    ->field_ref();
      const FieldT& rhoOld = rhoOld_  ->field_ref();
      const FieldT& dRhodF = dRhodF_  ->field_ref();
      const FieldT& res    = residual_->field_ref();
      fNew <<= fOld + res / (rhoOld + fOld*dRhodF);
    };
  };

  //===================================================================

  template< typename FieldT >
  DensFromMixfrac<FieldT>::
  DensFromMixfrac( const InterpT& rhoEval,
                  const Expr::Tag& rhoOldTag,
                  const Expr::Tag& rhoFTag, //rhoFTag will NOT be used if weakform is true.
                  const Expr::Tag& fOldTag,
                  const bool weakForm,
                  const double rTol,
                  const unsigned maxIter)
    : Expr::Expression<FieldT>(),
      DensityCalculatorBase( "DensFromMixFrac", tag_list(fOldTag), rTol, maxIter ),
      rhoEval_  ( rhoEval ),
      dRhodFTag_( "solver_d_rho_d_f" , Expr::STATE_NONE ),
      bounds_   ( rhoEval.get_bounds()[0] ),
      weak_     ( weakForm )
  {
    assert(phiOldTags_  .size() == 1);
    assert(phiNewTags_  .size() == 1);
    assert(residualTags_.size() == 1);

    this->set_gpu_runnable(true);

      fOld_   = this->template create_field_request<FieldT>( fOldTag   );
      rhoF_   = this->template create_field_request<FieldT>( rhoFTag   );
      rhoOld_ = this->template create_field_request<FieldT>( rhoOldTag );
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

    const Expr::Tag& fOldTag = phiOldTags_[0];
    const Expr::Tag& fNewTag = phiOldTags_[0];
    const Expr::Tag& rhoFTag = rhoPhiTags_[0];

    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;
    typedef typename TabPropsEvaluator<FieldT>::Builder TPEval;

    factory.register_expression(new PlcHldr( rhoFTag ));
    factory.register_expression(new PlcHldr( fOldTag ));

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
                                             phiOldTags_
                                            )
                                );
    // compute d(rho)/d(f) from lookup table
    factory.register_expression( new TPEval( dRhodFTag_, 
                                             rhoEval_,
                                             phiOldTags_,
                                             fOldTag
                                            )
                                );
    id = 
    factory.register_expression( new typename OneVarNewtonSolve<FieldT>::
                                 Builder( fNewTag,
                                          fOldTag,
                                          densityTag_,
                                          dRhodFTag_,
                                          residualTags_[0] )
                            );
    rootIDs.insert(id);

      // clip updated mixture fraction
    const Expr::Tag fClipTag = Expr::Tag(fNewTag.name()+"_clip", Expr::STATE_NONE);
    factory.register_expression( new typename Expr::ClipValue<FieldT>::
                                 Builder( fClipTag, bounds_.second, bounds_.first ) 
                                );
     factory.attach_modifier_expression( fClipTag, fNewTag );

    return rootIDs;
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensFromMixfrac<FieldT>::
  set_initial_guesses()
  {
      Expr::UintahFieldManager<FieldT>& fieldManager = helper_.fml_ -> template field_manager<FieldT>();
      FieldT& fOld = fieldManager.field_ref( phiOldTags_[0] );
      fOld <<= fOld_->field_ref();

      FieldT& rhoF = fieldManager.field_ref( rhoPhiTags_[0] );
      rhoF <<= rhoF_->field_ref();
  }

  //--------------------------------------------------------------------


  template< typename FieldT >
  void
  DensFromMixfrac<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    typedef typename Expr::Expression<FieldT>::ValVec SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();

    FieldT& rho = *results[0];
    rho <<= rhoOld_->field_ref();
    
    FieldT& badPts = *results[1];
    FieldT& drhodf = *results[2];

    std::cout<< "\nIn DensFromMixfrac::evaluate()...";

    // setup() needs to be run here because we need fields to be defined before a local patch can be created
    if( !setupHasRun_ ){ setup();}

    Expr::FieldManagerList* fml = helper_.fml_;
    newtonSolveTreePtr_->bind_fields( *fml );
    newtonSolveTreePtr_->lock_fields( *fml ); // is this needed?

    set_initial_guesses();

    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    unsigned numIter = 0;
    bool converged = false;

    const double absTol = rTol_/get_normalization_factor(0);

    while(numIter< maxIter_ && !converged)
    {
      ++numIter;
      newtonSolveTreePtr_->execute_tree();

      FieldT& fOld = fieldTManager.field_ref( phiOldTags_  [0] );
      FieldT& fNew = fieldTManager.field_ref( phiNewTags_  [0] );

      // update fOld for next iteration
      fOld <<= fNew;

      const FieldT& res  = fieldTManager.field_ref( residualTags_[0] );
      converged = nebo_max(abs(res)) < absTol;
    }

    // copy local fields to fields visible to uintah
    rho    <<= fieldTManager.field_ref( densityTag_ );
    drhodf <<= fieldTManager.field_ref( dRhodFTag_ );
    if(converged)
    {
      badPts <<= 0.0;
    }
    else
    {
      const FieldT& res  = fieldTManager.field_ref( residualTags_[0] );
      badPts <<= cond(abs(res) > absTol, res)
                     (0.0);
    }
    
    
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
                    const Expr::Tag& fOldTag,
                    const bool weakForm,
                    const double rtol,
                    const unsigned maxIter)
    : ExpressionBuilder( resultsTag ),
      rhoEval_  (rhoEval.clone() ),
      rhoOldTag_(rhoOldTag       ),
      rhoFTag_  (rhoFTag         ),
      fOldTag_  (fOldTag         ),
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
