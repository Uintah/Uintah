#ifdef __CUDACC__
#define ENABLE_CUDA
#endif

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromMixFracAndHeatLoss.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/Residual.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/NewtonUpdate.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/QuotientFunction.h>


#include <expression/ClipValue.h>
#include <expression/matrix-assembly/MapUtilities.h>

#include <spatialops/structured/SpatialFieldStore.h>

#include <sci_defs/uintah_defs.h>


namespace WasatchCore{

  using Expr::tag_list;
  typedef std::pair<double, double> BoundsT;

      /**
   * \class MixFracHeatLossJacobianAndResidual
   * \brief computes elements of the Jacobian matrix and residual needed to iteratively solve for
   *        density from mixture fraction and heat loss
   * 
   */
  template< typename FieldT >
  class MixFracHeatLossJacobianAndResidual : public Expr::Expression<FieldT>
  {
    const double fMin_, fMax_, fTol_;
    DECLARE_FIELDS(FieldT, rho_, f_, h_, rhoF_, rhoH_)
    DECLARE_FIELDS(FieldT, dRhodF_, dRhodGamma_, dHdF_, dHdGamma_)

    MixFracHeatLossJacobianAndResidual( const Expr::Tag& rhoTag,
                                        const Expr::Tag& fTag,
                                        const Expr::Tag& hTag,
                                        const Expr::Tag& rhoFTag,
                                        const Expr::Tag& rhoHTag,
                                        const Expr::Tag& dRhodFTag,
                                        const Expr::Tag& dRhodGammaTag,
                                        const Expr::Tag& dHdFTag,
                                        const Expr::Tag& dHdGammaTag,
                                        const BoundsT&   fBounds  )
    : Expr::Expression<FieldT>(),
      fMin_( fBounds.first  ),
      fMax_( fBounds.second ),
      fTol_( 1e-3 )
        {
       this->set_gpu_runnable(true);
       rho_        = this->template create_field_request<FieldT>( rhoTag        );
       f_          = this->template create_field_request<FieldT>( fTag          );
       h_          = this->template create_field_request<FieldT>( hTag          );
       rhoF_       = this->template create_field_request<FieldT>( rhoFTag       );
       rhoH_       = this->template create_field_request<FieldT>( rhoHTag       );
       dRhodF_     = this->template create_field_request<FieldT>( dRhodFTag     );
       dRhodGamma_ = this->template create_field_request<FieldT>( dRhodGammaTag );
       dHdF_       = this->template create_field_request<FieldT>( dHdFTag       );
       dHdGamma_   = this->template create_field_request<FieldT>( dHdGammaTag   );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a MixFracHeatLossJacobian expression
       *  @param jacobianTags tags to Jacobian elements (result elements[0-3]) to be computed 
       *  @param residualTags tags to residuals (result elements[4-5]) to be computed 
       *  @param rhoTag tag to density
       *  @param fTag the tag to mixture fraction
       *  @param hTag the tag to enthalpy
       *  @param rhoFTag the tag to density-weighted mixture fraction
       *  @param rhoHTag the tag to density-weighted enthalpy
       *  @param dRhodFTag tag to \f[ \frac{\partial \rho}{\partial f} \f]
       *  @param dRhodGammaTag tag to \f[ \frac{\partial \rho}{\partial \gamma} \f]
       *  @param dRhodFTag tag to \f[ \frac{\partial h}{\partial f} \f]
       *  @param dRhodGammaTag tag to \f[ \frac{\partial h}{\partial \gamma} \f]
       *  @param fBounds a pair with the min/max of mixture fraction
       */
      Builder( const Expr::TagList jacobianTags,
               const Expr::TagList residualTags,
               const Expr::Tag     rhoTag,
               const Expr::Tag     fTag,
               const Expr::Tag     hTag,
               const Expr::Tag     rhoFTag,
               const Expr::Tag     rhoHTag,
               const Expr::Tag     dRhodFTag,
               const Expr::Tag     dRhodGammaTag,
               const Expr::Tag     dHdFTag,
               const Expr::Tag     dHdGammaTag,
               const  BoundsT      fBounds )
      : ExpressionBuilder( result_tags(jacobianTags, residualTags) ),
        rhoTag_       ( rhoTag      ),
        fTag_         ( fTag        ),
        hTag_         ( hTag        ),
        rhoFTag_      ( rhoFTag     ),
        rhoHTag_      ( rhoHTag     ),
        dRhodFTag_    ( dRhodFTag   ),
        dRhodGammaTag_( dRhodFTag   ),
        dHdFTag_      ( dHdFTag     ),
        dHdGammaTag_  ( dHdGammaTag ),
        fBounds_      ( fBounds     )
      {
        assert(jacobianTags.size() == 4);
        assert(residualTags.size() == 2);
      }

      Expr::ExpressionBase* build() const{
        return new MixFracHeatLossJacobianAndResidual( rhoTag_, 
                                                       fTag_, 
                                                       hTag_, 
                                                       rhoFTag_, 
                                                       rhoHTag_,
                                                       dRhodFTag_, 
                                                       dRhodGammaTag_, 
                                                       dHdFTag_, 
                                                       dHdGammaTag_, 
                                                       fBounds_ );
      }

    private:
      const Expr::Tag rhoTag_, fTag_, hTag_, rhoFTag_, rhoHTag_,
                  dRhodFTag_, dRhodGammaTag_, dHdFTag_, dHdGammaTag_;
                  
      const BoundsT fBounds_;

      static Expr::TagList result_tags( const Expr::TagList jacTags,
                                        const Expr::TagList resTags)
      {
        Expr::TagList resultTags = jacTags;
        resultTags.insert(resultTags.end(), resTags.begin(), resTags.end());
        return resultTags;
      }
    };

    ~MixFracHeatLossJacobianAndResidual(){}

    void evaluate(){
      using namespace SpatialOps;
      typename Expr::Expression<FieldT>::ValVec&  results = this->get_value_vec();

      const FieldT& rho       = rho_        ->field_ref();
      const FieldT& f          = f_         ->field_ref();
      const FieldT& h          = h_         ->field_ref();
      const FieldT& rhoF       = rhoF_      ->field_ref();
      const FieldT& rhoH       = rhoH_      ->field_ref();
      const FieldT& dRhodF     = dRhodF_    ->field_ref();
      const FieldT& dRhodGamma = dRhodGamma_->field_ref();
      const FieldT& dHdF       = dHdF_      ->field_ref();
      const FieldT& dHdGamma   = dHdGamma_  ->field_ref();

      SpatFldPtr<FieldT> atBounds = SpatialFieldStore::get<FieldT>( rho );
      *atBounds <<= cond( abs(f-fMin_) < fTol_, 1 )
                        ( abs(f-fMax_) < fTol_, 1 )
                        ( 0 );
      
      // \f[ J_{f,f} = \frac{\partial \rho f}{\partial f}     \f]
      *results[0] <<= rho + f*dRhodF;              
      
      // \f[ J_{f,\gamma} = \frac{\partial \rho f}{\partial gamma} \f]
      *results[1] <<= cond( *atBounds > 0, 0 )
                          ( f*dRhodGamma );

      // \f[ J_{h,f}      = \frac{\partial \rho h}{\partial f}     \f]
      *results[2] <<= cond( *atBounds > 0, 0 )
                          ( rho*dHdF + h*dRhodF );

      // \f[ J_{h,\gamma} = \frac{\partial \rho h}{\partial gamma} \f]
      *results[3] <<= cond( *atBounds > 0, 1 )
                         ( rho*dHdGamma + h*dRhodGamma );

      // mixture fraction residual
      *results[4] <<= rho*f - rhoF;

      // mixture fraction residual
      *results[5] <<= cond( *atBounds > 0, 0 )
                          ( rho*h - rhoH  );
    };
  };

  //===================================================================

  template< typename FieldT >
  DensityFromMixFracAndHeatLoss<FieldT>::
  DensityFromMixFracAndHeatLoss(  const InterpT& rhoEval,
                                  const InterpT& enthEval,
                                  const Expr::Tag& rhoOldTag,
                                  const Expr::Tag& rhoFTag,
                                  const Expr::Tag& rhoHTag,
                                  const Expr::Tag& fOldTag,
                                  const Expr::Tag& hOldTag,
                                  const Expr::Tag& gammaOldTag,
                                  const double rTol,
                                  const unsigned maxIter )
    : DensityCalculatorBase<FieldT>( rTol, 
                                     maxIter,
                                     rhoOldTag, 
                                     tag_list(fOldTag, hOldTag),
                                     tag_list(fOldTag, gammaOldTag) ),
      rhoEval_    ( rhoEval                 ),
      enthEval_   ( enthEval                ),
      fOldTag_    ( this->betaOldTags_ [0]  ),
      hOldTag_    ( this->phiOldTags_  [1]  ),
      gammaOldTag_( this->betaOldTags_ [1]  ),
      fNewTag_    ( this->betaNewTags_ [0]  ),
      hNewTag_    ( this->phiNewTags_  [1]  ),
      gammaNewTag_( this->betaNewTags_ [1]  ),
      dRhodFTag_  ( this->dRhodPhiTags_[0]  ),
      dRhodHTag_  ( this->dRhodPhiTags_[1]  ),
      rhoFTag_    ( this->rhoPhiTags_  [0]  ),
      rhoHTag_    ( this->rhoPhiTags_  [1]  ),
      fBounds_    ( rhoEval.get_bounds()[0] ),
      gammaBounds_( rhoEval.get_bounds()[1] )
  {
    assert(this->phiOldTags_  .size() == 2);
    assert(this->phiNewTags_  .size() == 2);
    assert(this->betaOldTags_ .size() == 2);
    assert(this->betaNewTags_ .size() == 2);
    assert(this->residualTags_.size() == 2);

    this->set_gpu_runnable(true);
    fOld_     = this->template create_field_request<FieldT>( fOldTag     );
    hOld_     = this->template create_field_request<FieldT>( hOldTag     );
    gammaOld_ = this->template create_field_request<FieldT>( gammaOldTag ); 
    rhoF_     = this->template create_field_request<FieldT>( rhoFTag     );
    rhoH_     = this->template create_field_request<FieldT>( rhoHTag     ); 
    rhoOld_   = this->template create_field_request<FieldT>( rhoOldTag   ); 

    // set taglist for Jacobian matrix elements
    const std::string jacRowPrefix = "residual_jacobian_";
    const std::vector<std::string> jacRowNames = {jacRowPrefix + "f", jacRowPrefix + "h"};
    const std::vector<std::string> jacColNames = {"f", "gamma"};

    jacobianTags_ = Expr::matrix::matrix_tags( jacRowNames,"_",jacColNames);
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
    typedef typename Expr::ClipValue<FieldT>::Builder Clip;

    factory.register_expression(new PlcHldr( rhoFTag_            ));
    factory.register_expression(new PlcHldr( rhoHTag_            ));
    factory.register_expression(new PlcHldr( fOldTag_            ));
    factory.register_expression(new PlcHldr( hOldTag_            ));
    factory.register_expression(new PlcHldr( gammaOldTag_        ));
    factory.register_expression(new PlcHldr( this->densityOldTag_));

    id = 
    factory.register_expression( new TPEval( hNewTag_, 
                                             enthEval_,
                                             this->betaNewTags_
                                            )
                                );
    rootIDs.insert(id);
    

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

    // compute \f\frac{\partial h}{\partial f}\f$ from lookup table
    factory.register_expression( new TPEval( dHdGammaTag, 
                                             enthEval_,
                                             this->betaOldTags_,
                                             gammaOldTag_
                                            )
                                );

    // compute \f\frac{\partial \rho}{\partial \h}\f$
    factory.register_expression( new typename QuotientFunction<FieldT>::
                                     Builder( dRhodHTag_,
                                              dRhodGammaTag,
                                              dHdGammaTag ));

    // compute jacobian elements and residuals
    factory.register_expression( new typename MixFracHeatLossJacobianAndResidual<FieldT>::
                                     Builder( jacobianTags_,
                                              this->residualTags_,
                                              this->densityOldTag_,
                                              fOldTag_,
                                              hOldTag_,
                                              rhoFTag_,
                                              rhoHTag_,
                                              dRhodFTag_,
                                              dRhodGammaTag,
                                              dHdFTag,
                                              dHdGammaTag,
                                              fBounds_ ));

    factory.register_expression( new typename NewtonUpdate<FieldT>::
                                     Builder( this->betaNewTags_,
                                              this->residualTags_,
                                              jacobianTags_,
                                              this->betaOldTags_ ));


    // clip updated mixture fraction
    const Expr::Tag fClipTag = Expr::Tag(fNewTag_.name()+"_clip", Expr::STATE_NONE);
    factory.register_expression( new Clip( fClipTag, fBounds_.first, fBounds_.second ));
    factory.attach_modifier_expression( fClipTag, fNewTag_ );

    // clip updated heatLoss
    const Expr::Tag gammaClipTag = Expr::Tag(gammaNewTag_.name()+"_clip", Expr::STATE_NONE);
    factory.register_expression( new Clip( gammaClipTag, gammaBounds_.first, gammaBounds_.second ));
    factory.attach_modifier_expression( gammaClipTag, gammaNewTag_ );

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
  DensityFromMixFracAndHeatLoss<FieldT>::
  set_initial_guesses()
  {
      Expr::UintahFieldManager<FieldT>& fieldTManager = this->helper_.fml_-> template field_manager<FieldT>();
      
      FieldT& rhoOld = fieldTManager.field_ref( this->densityOldTag_);
      rhoOld <<= rhoOld_->field_ref();

      FieldT& fOld = fieldTManager.field_ref( fOldTag_ );
      fOld <<= fOld_->field_ref();

      FieldT& hOld = fieldTManager.field_ref( hOldTag_ );
      hOld <<= hOld_->field_ref();

      FieldT& gammaOld = fieldTManager.field_ref( gammaOldTag_ );
      gammaOld <<= gammaOld_->field_ref();

      FieldT& rhoF = fieldTManager.field_ref( rhoFTag_ );
      rhoF <<= rhoF_->field_ref();

      FieldT& rhoH = fieldTManager.field_ref( rhoHTag_ );
      rhoH <<= rhoH_->field_ref();
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
    FieldT& gamma  = *results[1];
    FieldT& dRhodF = *results[2];
    FieldT& dRhodH = *results[3];

    this->newton_solve();

    Expr::FieldManagerList* fml = this->helper_.fml_;
    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    // copy local fields to fields visible to uintah
    rho    <<= fieldTManager.field_ref( this->densityNewTag_ );
    gamma  <<= fieldTManager.field_ref( this->gammaNewTag_   );
    dRhodF <<= fieldTManager.field_ref( dRhodFTag_ );
    dRhodH <<= fieldTManager.field_ref( dRhodHTag_ );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromMixFracAndHeatLoss<FieldT>::
  Builder::Builder( const Expr::Tag rhoNewTag,
                    const Expr::Tag gammaNewTag,
                    const Expr::Tag dRhodFTag,
                    const Expr::Tag dRhodHTag,
                    const InterpT&  rhoEval,
                    const InterpT&  enthEval,
                    const Expr::Tag rhoOldTag,
                    const Expr::Tag rhoFTag,
                    const Expr::Tag rhoHTag,
                    const Expr::Tag fOldTag,
                    const Expr::Tag hOldTag,
                    const Expr::Tag gammaOldTag,
                    const double rTol,
                    const unsigned maxIter)
    : ExpressionBuilder( tag_list( rhoNewTag, 
                                   gammaNewTag, 
                                   dRhodFTag, 
                                   dRhodHTag  ) ),
      rhoEval_    (rhoEval.clone()  ),
      enthEval_   (enthEval.clone() ),
      rhoOldTag_  (rhoOldTag        ),
      rhoFTag_    (rhoFTag          ),
      rhoHTag_    (rhoHTag          ),
      fOldTag_    (fOldTag          ),
      hOldTag_    (hOldTag          ),
      gammaOldTag_(gammaOldTag      ),
      rtol_       (rTol             ),
      maxIter_    (maxIter          )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensityFromMixFracAndHeatLoss<SpatialOps::SVolField>;

}
