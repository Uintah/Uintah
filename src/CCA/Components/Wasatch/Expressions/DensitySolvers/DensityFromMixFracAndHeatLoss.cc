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

      /**
   * \class MixFracHeatLossJacobian
   * \brief computes elements of the Jacobian matrix needed to iteratively solve for
   *        density from mixture fraction and heat loss
   * 
   */
  template< typename FieldT >
  class MixFracHeatLossJacobian : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS(FieldT, rho_, f_, h_, dRhodF_, dRhodGamma_, dHdF_, dHdGamma_)

    MixFracHeatLossJacobian( const Expr::Tag& rhoTag,
                             const Expr::Tag& fTag,
                             const Expr::Tag& hTag,
                             const Expr::Tag& dRhodFTag,
                             const Expr::Tag& dRhodGammaTag,
                             const Expr::Tag& dHdFTag,
                             const Expr::Tag& dHdGammaTag )
    : Expr::Expression<FieldT>()
    {
       this->set_gpu_runnable(true);
       rho_        = this->template create_field_request<FieldT>( rhoTag        );
       f_          = this->template create_field_request<FieldT>( fTag          );
       h_          = this->template create_field_request<FieldT>( hTag          );
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
       *  @param jacobianTags tags to Jacobian elements to be computed
       *  @param rhoTag tag to density
       *  @param fTag the tag to mixture fraction
       *  @param hTag the tag to enthalpy
       *  @param dRhodFTag tag to \f[ \frac{\partial \rho}{\partial f} \f]
       *  @param dRhodGammaTag tag to \f[ \frac{\partial \rho}{\partial \gamma} \f]
       *  @param dRhodFTag tag to \f[ \frac{\partial h}{\partial f} \f]
       *  @param dRhodGammaTag tag to \f[ \frac{\partial h}{\partial \gamma} \f]
       */
      Builder( const Expr::TagList& jacobianTags,
               const Expr::Tag&     rhoTag,
               const Expr::Tag&     fTag,
               const Expr::Tag&     hTag,
               const Expr::Tag&     dRhodFTag,
               const Expr::Tag&     dRhodGammaTag,
               const Expr::Tag&     dHdFTag,
               const Expr::Tag&     dHdGammaTag )
      : ExpressionBuilder( jacobianTags ),
        rhoTag_       ( rhoTag      ),
        fTag_         ( fTag        ),
        hTag_         ( hTag        ),
        dRhodFTag_    ( dRhodFTag   ),
        dRhodGammaTag_( dRhodFTag   ),
        dHdFTag_      ( dHdFTag     ),
        dHdGammaTag_  ( dHdGammaTag )
      {
        assert(jacobianTags.size() == 4);
      }

      Expr::ExpressionBase* build() const{
        return new MixFracHeatLossJacobian( rhoTag_, fTag_, hTag_, dRhodFTag_, dRhodGammaTag_, dHdFTag_, dHdGammaTag_ );
      }

    private:
      const Expr::Tag rhoTag_, fTag_, hTag_, dRhodFTag_, dRhodGammaTag_, dHdFTag_, dHdGammaTag_;
    };

    ~MixFracHeatLossJacobian(){}

    void evaluate(){
      using namespace SpatialOps;
      typename Expr::Expression<FieldT>::ValVec&  jacobElems = this->get_value_vec();

      const FieldT& rho       = rho_        ->field_ref();
      const FieldT& f          = f_         ->field_ref();
      const FieldT& h          = h_         ->field_ref();
      const FieldT& dRhodF     = dRhodF_    ->field_ref();
      const FieldT& dRhodGamma = dRhodGamma_->field_ref();
      const FieldT& dHdF       = dHdF_      ->field_ref();
      const FieldT& dHdGamma   = dHdGamma_  ->field_ref();
      
      *jacobElems[0] <<= rho + f*dRhodF;              // \f[ = \frac{\partial \rho f}{\partial f}     \f]
      *jacobElems[1] <<= f*dRhodGamma;                // \f[ = \frac{\partial \rho f}{\partial gamma} \f]
      *jacobElems[2] <<= rho*dHdF + h*dRhodF;         // \f[ = \frac{\partial \rho h}{\partial f}     \f]
      *jacobElems[3] <<= rho*dHdGamma + h*dRhodGamma; // \f[ = \frac{\partial \rho h}{\partial gamma} \f]

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
                                  const Expr::Tag& gammaOldTag,
                                  const double rTol,
                                  const unsigned maxIter )
    : DensityCalculatorBase<FieldT>( rTol, 
                                     maxIter,
                                     rhoOldTag, 
                                     tag_list(fOldTag, Expr::Tag("h", Expr::STATE_NONE)),
                                     tag_list(fOldTag, gammaOldTag) ),
      rhoEval_    ( rhoEval                 ),
      enthEval_   ( enthEval                ),
      fOldTag_    ( this->betaOldTags_ [0]  ),
      gammaOldTag_( this->betaOldTags_ [1]  ),
      fNewTag_    ( this->betaNewTags_ [0]  ),
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
    const Expr::Tag hOldTag      (this->phiOldTags_[1]);
    const Expr::Tag hNewTag      (this->phiNewTags_[1]);
    const Expr::Tag dRhodGammaTag("solver_d_rho_d_gamma", Expr::STATE_NONE);
    const Expr::Tag dHdGammaTag  ("solver_d_h_d_gamma"  , Expr::STATE_NONE);
    const Expr::Tag dHdFTag      ("solver_d_h_d_f"      , Expr::STATE_NONE);

    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;
    typedef typename TabPropsEvaluator<FieldT>::Builder TPEval;
    typedef typename Expr::ClipValue<FieldT>::Builder Clip;

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

    // compute new enthalpy from lookup table ...
    // \todo: create a 'initialGuessGraphHelper' and a "register_inital_guess_exppressions()"
    // method to eliminate the need for this table lookup. 
    // Doing so will also eliminate the need to recompute residuals for relative error 
    // calculation
    id = 
    factory.register_expression( new TPEval( hNewTag, 
                                             enthEval_,
                                             this->betaNewTags_
                                            )
                                );
    rootIDs.insert(id);
    
    // compute new enthalpy from lookup table 
    factory.register_expression( new TPEval( hOldTag, 
                                             enthEval_,
                                             this->betaOldTags_
                                            )
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

    // compute jacobian elements
    factory.register_expression( new typename MixFracHeatLossJacobian<FieldT>::
                                     Builder( jacobianTags_,
                                              this->densityOldTag_,
                                              fOldTag_,
                                              hOldTag,
                                              dRhodFTag_,
                                              dRhodGammaTag,
                                              dHdFTag,
                                              dHdGammaTag ));

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
    // setup() needs to be run here because we need fields to be defined before a local patch can be created
    if( !this->setupHasRun_ ){ this->setup();}

    set_initial_guesses();

    const double maxError = this->newton_solve();

    Expr::FieldManagerList* fml = this->helper_.fml_;
    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    // copy local fields to fields visible to uintah
    rho    <<= fieldTManager.field_ref( this->densityNewTag_ );
    gamma  <<= fieldTManager.field_ref( this->gammaNewTag_   );
    dRhodF <<= fieldTManager.field_ref( dRhodFTag_ );
    dRhodH <<= fieldTManager.field_ref( dRhodHTag_ );

    this->unlock_fields();
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
                    const Expr::Tag gammaOldTag,
                    const double rTol,
                    const unsigned maxIter)
    : ExpressionBuilder( tag_list(rhoNewTag, gammaNewTag, dRhodFTag, dRhodHTag ) ),
      rhoEval_    (rhoEval.clone()  ),
      enthEval_   (enthEval.clone() ),
      rhoOldTag_  (rhoOldTag        ),
      rhoFTag_    (rhoFTag          ),
      rhoHTag_    (rhoHTag          ),
      fOldTag_    (fOldTag          ),
      gammaOldTag_(gammaOldTag      ),
      rtol_       (rTol             ),
      maxIter_    (maxIter          )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensityFromMixFracAndHeatLoss<SpatialOps::SVolField>;

}
