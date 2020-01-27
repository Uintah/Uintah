#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromMixFracAndHeatLoss.h>
 #include <CCA/Components/Wasatch/Expressions/DensitySolvers/Residual.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <expression/ClipValue.h>

#include <sci_defs/uintah_defs.h>

namespace WasatchCore{

  using Expr::tag_list;

  template< typename FieldT >
  DensityFromMixFracAndHeatLoss<FieldT>::
  DensityFromMixFracAndHeatLoss(  const InterpT& rhoEval,
                                  const InterpT& enthEval,
                                  const Expr::Tag& rhoOldTag,
                                  const Expr::Tag& rhoFTag,
                                  const Expr::Tag& rhoHTag,
                                  const Expr::Tag& fOldTag,
                                  const Expr::Tag& gammaOldTag,
                                  const double rtol,
                                  const unsigned maxIter )
    : DensityCalculatorBase<FieldT>( rTol, 
                                     maxIter,
                                     rhoOldTag, 
                                     tag_list(fOldTag, Expr::Tag('h',Expr::STATE_NONE)),
                                     tag_list(fOldTag, gammaOldTag) ),
      rhoEval_  ( rhoEval  ),
      enthEval_ ( enthEval ),
      fOldTag_  ( this->phiOldTags_[0] ),
      gammaOldTag_( this->betaOldTags_[1] ),
      fNewTag_  ( this->phiNewTags_[0] ),
      gammaNewTag_( this->betaNewTags_[1] ),
      dRhodFTag_( this->dRhodPhiTags_[0] ),
      rhoFTag_  ( this->rhoPhiTags_[0] ),
      bounds_   ( rhoEval.get_bounds()[0] )
  {
    assert(this->phiOldTags_  .size() == 2);
    assert(this->phiNewTags_  .size() == 2);
    assert(this->betaOldTags_ .size() == 2);
    assert(this->betaNewTags_ .size() == 2);
    assert(this->residualTags_.size() == 2);

    this->set_gpu_runnable(true);

      fOld_     = this->template create_field_request<FieldT>( fOldTag     );
      gammaOld_ = this->template create_field_request<FieldT>( gammaOldTag );
      rhoF_     = this->template create_field_request<FieldT>( rhoFTag     );
      rhoH_     = this->template create_field_request<FieldT>( rhoHTag     );
      rhoOld_   = this->template create_field_request<FieldT>( rhoOldTag   );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromMixFracAndHeatLoss<FieldT>::
  ~DensityFromMixFracAndHeatLoss()
  {}

  //--------------------------------------------------------------------

  template< typename FieldT >
  Expr::IDSet 
  DensityFromMixFracAndHeatLoss<FieldT>::
  register_local_expressions()
  {
    Expr::IDSet rootIDs;
    Expr::ExpressionFactory& factory = *(this->helper_.factory_);

    Expr::ExpressionID id;

    // define tags that will only be used here
    const Expr::Tag dRhodGammaTag("solver_d_rho_d_gamma", Expr::STATE_NONE);
    const Expr::Tag dHdGammaTag  ("solver_d_h_d_gamma"  , Expr::STATE_NONE);
    const Expr::Tag dHdFTag      ("solver_d_h_d_f"      , Expr::STATE_NONE);

    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;
    typedef typename TabPropsEvaluator<FieldT>::Builder TPEval;

    factory.register_expression(new PlcHldr( rhoFTag_            ));
    factory.register_expression(new PlcHldr( rhoHTag_            ));
    factory.register_expression(new PlcHldr( fOldTag_            ));
    factory.register_expression(new PlcHldr( gammaOldTag_        ));
    factory.register_expression(new PlcHldr( this->densityOldTag_));

    // compute residuals
    factory.register_expression( new typename Residual<FieldT>::
                                 Builder( this->residualTags_,
                                          this->rhoPhiTags_,
                                          this->phiOldTags_,
                                          this->densityOldTag_ )
                                );

    // compute \f\frac{\partial \rho}{\partial f}\f$ from lookup table
    factory.register_expression( new TPEval( dRhodFTag_, 
                                             rhoEval_,
                                             this->betaOldTags_,
                                             fOldTag_
                                            )
                                );


    // compute \f\frac{\partial \rho}{\partial \gamma}\f$ from lookup table
    factory.register_expression( new TPEval( dRhodGammaTag, 
                                             rhoEval_,
                                             this->betaOldTags_,
                                             gammaOldTag_
                                            )
                                );

    // compute \f\frac{\partial h}{\partial f}\f$ from lookup table
    factory.register_expression( new TPEval( dHdFTag, 
                                             enthEval_,
                                             this->betaOldTags_,
                                             fOldTag_
                                            )
                                );

    // compute \f\frac{\partial \rho}{\partial \h}\f$
    //
    // here
    // 

    // compute \f\frac{\partial \rho}{\partial \gamma}\f$ from lookup table

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
  DensityFromMixFracAndHeatLoss<FieldT>::
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
  DensityFromMixFracAndHeatLoss<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    typedef typename Expr::Expression<FieldT>::ValVec SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();

    FieldT& rho    = *results[0];
    FieldT& drhodf = *results[1];
    FieldT& badPts = *results[2];

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
      const double nbad = nebo_sum(badPts);

      badPts <<= cond(badPts > 0, res)
                (0.0);
      std::cout << "\tConvergence failed at " << (int)nbad << " points.\n";
    }
    newtonSolveTree.unlock_fields( *fml );
    
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromMixFracAndHeatLoss<FieldT>::
  Builder::Builder( const Expr::Tag rhoNewTag,
                    const Expr::Tag dRhodFTag,
                    const Expr::Tag dRhodHTag,
                    const Expr::Tag badBtsTag,
                    const InterpT&  rhoEval,
                    const InterpT&  enthEval,
                    const Expr::Tag rhoOldTag,
                    const Expr::Tag rhoFTag,
                    const Expr::Tag rhoHTag,
                    const Expr::Tag fOldTag,
                    const Expr::Tag gammaOldTag,
                    const double rtol,
                    const unsigned maxIter )
    : ExpressionBuilder( tag_list(rhoNewTag, dRhodFTag, dRhodHTag, badPtsTag) ),
      rhoEval_    (rhoEval.clone() ),
      enthEval_   (rhoEval.clone() ),
      rhoOldTag_  (rhoOldTag       ),
      rhoFTag_    (rhoFTag         ),
      rhoHTag_    (rhoHTag         ),
      fOldTag_    (fOldTag         ),
      gammaOldTag_(gammaOldTag     ),
      rtol_       (rtol            ),
      maxIter_    (maxIter         )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensityFromMixFracAndHeatLoss<SpatialOps::SVolField>;

}
