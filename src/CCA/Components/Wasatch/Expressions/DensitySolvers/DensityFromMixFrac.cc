#ifdef __CUDACC__
#define ENABLE_CUDA
#endif

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromMixFrac.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/Residual.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <expression/ClipValue.h>

#include <sci_defs/uintah_defs.h>

namespace WasatchCore{

  using Expr::tag_list;

    /**
   * \class OneVarNewtonSolve
   * @brief computes updated mixture fraction, \f[ f \f] given an old value of \f[ f \f],
   *        density (\f[ \rho \f]), \f[ \frac{\rho}{f} \f], and a residual with the following
   *        definition: 
   *        \f[ r(f) = f G_\rho - (\rho f)\f].
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
       *  @param residualTag tag for residual \f[ r(f) = f G_\rho - (\rho f)\f]
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
      fNew <<= fOld - res / (rhoOld + fOld*dRhodF);
    };
  };

  //===================================================================

  template< typename FieldT >
  DensityFromMixFrac<FieldT>::
  DensityFromMixFrac( const InterpT& rhoEval,
                      const Expr::Tag& rhoOldTag,
                      const Expr::Tag& rhoFTag,
                      const Expr::Tag& fOldTag,
                      const double rTol,
                      const unsigned maxIter)
    : DensityCalculatorBase<FieldT>( rTol, 
                                     maxIter,
                                     rhoOldTag, 
                                     tag_list(fOldTag),
                                     tag_list(fOldTag) ),
      rhoEval_  ( rhoEval ),
      fOldTag_  ( this->betaOldTags_ [0] ),
      fNewTag_  ( this->betaNewTags_ [0] ),
      dRhodFTag_( this->dRhodPhiTags_[0] ),
      rhoFTag_  ( this->rhoPhiTags_  [0] ),
      fBounds_   ( rhoEval.get_bounds()[0] )
  {
    assert(this->phiOldTags_  .size() == 1);
    assert(this->phiNewTags_  .size() == 1);
    assert(this->residualTags_.size() == 1);

    this->set_gpu_runnable(true);

      fOld_   = this->template create_field_request<FieldT>( fOldTag   );
      rhoF_   = this->template create_field_request<FieldT>( rhoFTag   );
      rhoOld_ = this->template create_field_request<FieldT>( rhoOldTag );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromMixFrac<FieldT>::
  ~DensityFromMixFrac()
  {}

  //--------------------------------------------------------------------

  template< typename FieldT >
  Expr::IDSet 
  DensityFromMixFrac<FieldT>::
  register_local_expressions()
  {
    Expr::IDSet rootIDs;
    Expr::ExpressionFactory& factory = *(this->helper_.factory_);

    Expr::ExpressionID id;

    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;
    typedef typename TabPropsEvaluator<FieldT>::Builder TPEval;

    factory.register_expression(new PlcHldr( rhoFTag_            ));
    factory.register_expression(new PlcHldr( fOldTag_            ));
    factory.register_expression(new PlcHldr( this->densityOldTag_));

    // compute residual
    factory.register_expression( new typename Residual<FieldT>::
                                 Builder( this->residualTags_,
                                          this->rhoPhiTags_,
                                          this->phiOldTags_,
                                          this->densityOldTag_ )
                                );

    // compute d(rho)/d(f) from lookup table
    factory.register_expression( new TPEval( dRhodFTag_, 
                                             rhoEval_,
                                             this->betaOldTags_,
                                             fOldTag_
                                            )
                                );

    factory.register_expression( new typename OneVarNewtonSolve<FieldT>::
                                 Builder( fNewTag_,
                                          fOldTag_,
                                          this->densityOldTag_,
                                          dRhodFTag_,
                                          this->residualTags_[0] )
                            );

      // clip updated mixture fraction
    const Expr::Tag fClipTag = Expr::Tag(fNewTag_.name()+"_clip", Expr::STATE_NONE);
    factory.register_expression( new typename Expr::ClipValue<FieldT>::
                                 Builder( fClipTag, fBounds_.first, fBounds_.second ) 
                                );
     factory.attach_modifier_expression( fClipTag, fNewTag_ );

    // compute density from lookup table
    id = 
    factory.register_expression( new TPEval( this->densityNewTag_, 
                                             rhoEval_,
                                             this->betaNewTags_
                                            )
                                );
    rootIDs.insert(id);

    return rootIDs;
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensityFromMixFrac<FieldT>::
  set_initial_guesses()
  {
      Expr::UintahFieldManager<FieldT>& fieldTManager = this->helper_.fml_-> template field_manager<FieldT>();

      FieldT& rhoOld = fieldTManager.field_ref( this->densityOldTag_);
      rhoOld <<= rhoOld_->field_ref();

      FieldT& fOld = fieldTManager.field_ref( fOldTag_ );
      fOld <<= fOld_->field_ref();

      FieldT& rhoF = fieldTManager.field_ref( rhoFTag_ );
      rhoF <<= rhoF_->field_ref();
  }

  //--------------------------------------------------------------------


  template< typename FieldT >
  void
  DensityFromMixFrac<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    typedef typename Expr::Expression<FieldT>::ValVec SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();

    FieldT& rho    = *results[0];
    FieldT& dRhodF = *results[1];

    this->newton_solve();

    // copy local fields to fields visible to uintah
    Expr::FieldManagerList* fml = this->helper_.fml_;
    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    // copy local fields to fields visible to uintah
    rho    <<= fieldTManager.field_ref( this->densityNewTag_ );
    dRhodF <<= fieldTManager.field_ref( dRhodFTag_ );

    this->unlock_fields();
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromMixFrac<FieldT>::
  Builder::Builder( const Expr::Tag rhoNewTag,
                    const Expr::Tag dRhodFTag,
                    const InterpT&  rhoEval,
                    const Expr::Tag rhoOldTag,
                    const Expr::Tag rhoFTag,
                    const Expr::Tag fOldTag,
                    const double rtol,
                    const unsigned maxIter )
    : ExpressionBuilder( tag_list(rhoNewTag, dRhodFTag) ),
      rhoEval_  (rhoEval.clone() ),
      rhoOldTag_(rhoOldTag       ),
      rhoFTag_  (rhoFTag         ),
      fOldTag_  (fOldTag         ),
      rtol_     (rtol            ),
      maxIter_  (maxIter         )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensityFromMixFrac<SpatialOps::SVolField>;

}
