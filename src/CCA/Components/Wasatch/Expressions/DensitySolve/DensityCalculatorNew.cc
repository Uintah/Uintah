#include <CCA/Components/Wasatch/Expressions/DensitySolve/DensityCalculatorNew.h>
 #include <CCA/Components/Wasatch/Expressions/DensitySolve/Residual.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <CCA/Components/Wasatch/TagNames.h>
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
    : DensityCalculatorBase<FieldT>( rhoOldTag, 
                                     tag_list(TagNames::self().derivative_tag(rhoOldTag,fOldTag)),
                                     tag_list(fOldTag), 
                                     rTol, 
                                     maxIter ),
      rhoEval_  ( rhoEval ),
      fOldTag_  ( this->phiOldTags_[0] ),
      fNewTag_  ( this->phiNewTags_[0] ),
      dRhodFTag_( this->dRhodPhiTags_[0] ),
      rhoFTag_  ( this->rhoPhiTags_[0] ),
      bounds_   ( rhoEval.get_bounds()[0] ),
      weak_     ( weakForm )
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
  // template< typename FieldT >
  // void
  // DensFromMixfrac<FieldT>::
  // bind_operators( const SpatialOps::OperatorDatabase& opDB )
  // { 
  //   DensityCalculatorBase::bind_operators(opDB); 
  // };

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
                                             this->phiOldTags_,
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
                                 Builder( fClipTag, 0, 1 ) 
                                );
     factory.attach_modifier_expression( fClipTag, fNewTag_ );

    // compute density from lookup table
    id = 
    factory.register_expression( new TPEval( this->densityNewTag_, 
                                             rhoEval_,
                                             this->phiNewTags_
                                            )
                                );
    rootIDs.insert(id);

    return rootIDs;
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensFromMixfrac<FieldT>::
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

    // setup() needs to be run here because we need fields to be defined before a local patch can be created
    if( !this->setupHasRun_ ){ this->setup();}

    Expr::FieldManagerList* fml = this->helper_.fml_;

    Expr::ExpressionTree& newtonSolveTree = *(this->newtonSolveTreePtr_);
    newtonSolveTree.bind_fields( *fml );
    newtonSolveTree.lock_fields( *fml ); // this is needed... why?

    set_initial_guesses();

    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    unsigned numIter = 0;
    bool converged = false;

    const double absTol = this->rTol_/get_normalization_factor(0);

    while(numIter< this->maxIter_ && !converged)
    {
      ++numIter;
      newtonSolveTree.execute_tree();

      FieldT& fOld = fieldTManager.field_ref( fOldTag_ );
      FieldT& fNew = fieldTManager.field_ref( fNewTag_ );

      FieldT& rhoOld = fieldTManager.field_ref( this->densityOldTag_ );
      FieldT& rhoNew = fieldTManager.field_ref( this->densityNewTag_ );

      // update fOld and rhoOld for next iteration
      fOld   <<= fNew;
      rhoOld <<= rhoNew;

      const FieldT& res  = fieldTManager.field_ref( this->residualTags_[0] );
      converged = nebo_max(abs(res)) < absTol;
    }

    if(converged)
    {
      Expr::ExpressionTree& dRhodFTree = *(this->dRhodPhiTreePtr_);
      dRhodFTree.bind_fields( *fml );
      dRhodFTree.lock_fields( *fml );
      dRhodFTree.execute_tree();
      // copy local fields to fields visible to uintah
      badPts <<= 0.0;
      rho    <<= fieldTManager.field_ref( this->densityNewTag_ );
      drhodf <<= fieldTManager.field_ref( dRhodFTag_ );

      dRhodFTree.unlock_fields( *fml );
    }
    else
    {
      const FieldT& res  = fieldTManager.field_ref( this->residualTags_[0] );
      badPts <<= cond(abs(res) > absTol, 1)
                     (0.0);
      const double nbad  = nebo_sum(badPts);

      badPts <<= cond(badPts > 0, res)
                (0.0);
      std::cout << "\tConvergence failed at " << (int)nbad << " points.\n";
    }
    newtonSolveTree.unlock_fields( *fml );
    
  }

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
