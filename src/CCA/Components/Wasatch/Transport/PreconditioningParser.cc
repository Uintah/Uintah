/*
 * The MIT License
 *
 * Copyright (c) 2016-2018 The University of Utah
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

#include <sci_defs/wasatch_defs.h>
#include <CCA/Components/Wasatch/Transport/PreconditioningParser.h>

#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <expression/ExprLib.h>
#include <expression/Expression.h>

#ifdef HAVE_POKITT
#include <pokitt/CanteraObjects.h>
#endif

namespace WasatchCore {


  /**
   *  \class ArtCompPGSPressure
   *  \author Tony Saad
   *  \date May, 2016
   *  \brief When using Artifical Compressibility to scale the acoustic speed, this expression
   *           computes the extra term required in the momentum equations. This term will
   *           be added as a dependency to the momentum RHS:
   *   \f[
   *     \left(1 - \frac{1}{\alpha^2} \right) \frac{\partial p}{\partial x_i}
   *   \f]
   *   for the ith momentum equation. When added to the momentum RHS, the net contribution/scaling to the
   *   momentum RHS is:
   *   \f[
   *     - \frac{\partial p}{\partial x_i} + \left(1 - \frac{1}{\alpha^2}\right) \frac{\partial p}{\partial x_i} = - \frac{1}{\alpha^2} \frac{\partial p}{\partial x_i}
   *   \f]
   * For further details, see Wang, Y., & Trouvé, A. (2004). Artificial acoustic
   *  stiffness reduction in fully compressible, direct numerical simulation of combustion.
   *  Combust. Theory Modelling, 8(3), 633–660.
   */
  template< typename MomDirT >
  class ArtCompPGSPressure : public Expr::Expression<SVolField>
  {
    DECLARE_FIELDS( SVolField, p_ )
    const double alpha_;

    typedef typename SpatialOps::OperatorTypeBuilder< typename GradOpSelector<SVolField, MomDirT>::Gradient, SVolField, SVolField >::type GradT;
    const GradT* grad_;

    ArtCompPGSPressure( const Expr::Tag& pTag,
                        const double alpha )
    : Expr::Expression<SVolField>(),
      alpha_(alpha)
    {
      this->set_gpu_runnable( true );
      p_ = this->template create_field_request<SVolField>( pTag );
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag pTag_;
      const double alpha_;
    public:

      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& pTag,
               const double alpha,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        pTag_( pTag ),
        alpha_( alpha )
      {}

      Expr::ExpressionBase* build() const{
        return new ArtCompPGSPressure<MomDirT>( pTag_, alpha_ );
      }

    };  /* end of Builder class */

    ~ArtCompPGSPressure(){}

    void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
      grad_ = opDB.retrieve_operator<GradT>();
    }

    void evaluate()
    {
      SVolField& result = this->value();
      const double fac = 1.0-1.0/(alpha_*alpha_);
      result <<= fac * ( *grad_ )( p_->field_ref() );
    }
  };

  //====================================================================

   /**
    *  \class ArtCompPGSEnergy
    *  \author Tony Saad
    *  \date May, 2016
    *  \brief When using the PGS (Pressure Gradient Scaling) model in Artificial
    *   Compressibility to rescale the acoustic speed in compressible flows, this
    *   expression computes the term required in the total internal energy equation.
    *   It will be added as a dependency to the total internal energy RHS.
    *   \f[
    *     (1 - \frac{1}{\alpha^2}) u_j \frac{\partial p}{\partial x_j}
    *   \f]
    *  For further details, see Wang, Y., & Trouvé, A. (2004). Artificial acoustic
    *  stiffness reduction in fully compressible, direct numerical simulation of combustion.
    *  Combust. Theory Modelling, 8(3), 633–660.
    */
   class ArtCompPGSEnergy : public Expr::Expression<SVolField>
   {
     DECLARE_FIELDS( SVolField, p_, u_, v_, w_ )
     const bool doX_, doY_, doZ_;
     const double alpha_;

     typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientX, SVolField, SVolField >::type GradXT;
     typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientY, SVolField, SVolField >::type GradYT;
     typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientZ, SVolField, SVolField >::type GradZT;
     const GradXT* dpdx_;
     const GradYT* dpdy_;
     const GradZT* dpdz_;

     ArtCompPGSEnergy( const Expr::Tag& pTag,
                       const Expr::TagList& velTags,
                       const double alpha )
     : Expr::Expression<SVolField>(),
       doX_( velTags[0] != Expr::Tag() ),
       doY_( velTags[1] != Expr::Tag() ),
       doZ_( velTags[2] != Expr::Tag() ),
       alpha_( alpha )
     {
       this->set_gpu_runnable( true );

       p_ = create_field_request<SVolField>( pTag );

       if( doX_ ) u_ = create_field_request<SVolField>( velTags[0] );
       if( doY_ ) v_ = create_field_request<SVolField>( velTags[1] );
       if( doZ_ ) w_ = create_field_request<SVolField>( velTags[2] );
     }


   public:

     class Builder : public Expr::ExpressionBuilder
     {
       const Expr::Tag pTag_;
       const Expr::TagList velTags_;
       const double alpha_;
     public:
       Builder( const Expr::Tag& resultTag,
                const Expr::Tag& pTag,
                const Expr::TagList& velTags,
                const double alpha,
                const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
       : ExpressionBuilder( resultTag, nghost ),
         pTag_   ( pTag    ),
         velTags_( velTags ),
         alpha_  ( alpha   )
       {
         assert( velTags.size() == 3 );
       }

       Expr::ExpressionBase* build() const{
         return new ArtCompPGSEnergy( pTag_,velTags_, alpha_ );
       }

     };  /* end of Builder class */

     ~ArtCompPGSEnergy(){}

     void bind_operators( const SpatialOps::OperatorDatabase& opDB )
     {
       if( doX_ ) dpdx_ = opDB.retrieve_operator<GradXT>();
       if( doY_ ) dpdy_ = opDB.retrieve_operator<GradYT>();
       if( doZ_ ) dpdz_ = opDB.retrieve_operator<GradZT>();
     }

     void evaluate()
     {
       SVolField& result = this->value();

       const SVolField& p = p_->field_ref();
       const double fac = 1.0 - 1.0/(alpha_*alpha_);
       if( doX_ && doY_ && doZ_ ){
         result <<= fac * (   u_->field_ref() * (*dpdx_)( p )
                            + v_->field_ref() * (*dpdy_)( p )
                            + w_->field_ref() * (*dpdz_)( p )
                          );
       }
       else{
         if( doX_ ) result <<= fac * u_->field_ref() * (*dpdx_)( p );
         else result <<= 0.0;
         if( doY_ ) result <<= result + fac * v_->field_ref() * (*dpdy_)( p );
         if( doZ_ ) result <<= result + fac * w_->field_ref() * (*dpdz_)( p );
       }
     }
   };

  //============================================================================

   /**
    *  \class ArtCompASREnergy
    *  \author Tony Saad
    *  \date May, 2016
    *  \brief When using the ASR (Acoustic Stiffness Reduction) Artificial Compressibility to rescale the acoustic speed in compressible flows,
    *    this expression computes the term required in the total internal energy equation. It will be added
    *    as a dependency to the total internal energy RHS.
    *    \f[
    *      \left(1 - \frac{1}{\alpha^2}\right) \frac{\gamma p}{\gamma - 1} \frac{\partial u_j}{\partial x_j}
    *      - \left(1 - \frac{1}{\alpha^2}\right) \left(\tau_{ij}\frac{\partial u_i}{\partial x_j} - \frac{\partial q_j}{\partial x_j} \right)
    *    \f]
    *    where
    *    \f[
    *     q_j = - k \frac{\partial T}{\partial x_j}
    *    \f]
    *
    *  For further details, see Wang, Y., & Trouvé, A. (2004). Artificial acoustic
    *  stiffness reduction in fully compressible, direct numerical simulation of combustion.
    *  Combust. Theory Modelling, 8(3), 633–660.
    */
   template< typename VolFieldT >
   class ArtCompASREnergy : public Expr::Expression<VolFieldT>
   {
     typedef typename SpatialOps::FaceTypes<VolFieldT>::XFace XFace;
     typedef typename SpatialOps::FaceTypes<VolFieldT>::YFace YFace;
     typedef typename SpatialOps::FaceTypes<VolFieldT>::ZFace ZFace;

     DECLARE_FIELDS( VolFieldT, xvel_, yvel_, zvel_, visc_, dil_, p_, mw_, cp_, temp_ )
     DECLARE_FIELDS( XFace, strainxx_, strainxy_, strainxz_, diffFluxX_ )
     DECLARE_FIELDS( YFace, strainyx_, strainyy_, strainyz_, diffFluxY_ )
     DECLARE_FIELDS( ZFace, strainzx_, strainzy_, strainzz_, diffFluxZ_ )
     DECLARE_VECTOR_OF_FIELDS( VolFieldT, specEnthalpies_ )
     DECLARE_VECTOR_OF_FIELDS( VolFieldT, specRxnRates_ )
     DECLARE_VECTOR_OF_FIELDS( XFace, xSpecFluxes_ )
     DECLARE_VECTOR_OF_FIELDS( YFace, ySpecFluxes_ )
     DECLARE_VECTOR_OF_FIELDS( ZFace, zSpecFluxes_ )

     const bool doX_, doY_, doZ_;
     const double alpha_;

     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::InterpC2FX  XInterp;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::InterpC2FY  YInterp;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::InterpC2FZ  ZInterp;
     XInterp* xInterp_;
     YInterp* yInterp_;
     ZInterp* zInterp_;

     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::DivX  DivX;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::DivY  DivY;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::DivZ  DivZ;
     DivX* divX_;
     DivY* divY_;
     DivZ* divZ_;

     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::GradX  DDXT;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::GradY  DDYT;
     typedef typename SpatialOps::BasicOpTypes<VolFieldT>::GradZ  DDZT;
     DDXT* ddx_;
     DDYT* ddy_;
     DDZT* ddz_;

     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,XFace>::type  SVol2XFluxInterpT;
     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,YFace>::type  SVol2YFluxInterpT;
     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,ZFace>::type  SVol2ZFluxInterpT;
     const SVol2XFluxInterpT* sVol2XFluxInterpOp_;
     const SVol2YFluxInterpT* sVol2YFluxInterpOp_;
     const SVol2ZFluxInterpT* sVol2ZFluxInterpOp_;

     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,XFace,VolFieldT>::type  XFlux2SVolInterpT;
     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,YFace,VolFieldT>::type  YFlux2SVolInterpT;
     typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ZFace,VolFieldT>::type  ZFlux2SVolInterpT;
     const XFlux2SVolInterpT* xFluxInterpOp_;
     const YFlux2SVolInterpT* yFluxInterpOp_;
     const ZFlux2SVolInterpT* zFluxInterpOp_;

     typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientX, VolFieldT, VolFieldT >::type GradXT;
     typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientY, VolFieldT, VolFieldT >::type GradYT;
     typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientZ, VolFieldT, VolFieldT >::type GradZT;
     const GradXT* dpdx_;
     const GradYT* dpdy_;
     const GradZT* dpdz_;

     ArtCompASREnergy( const Expr::Tag& xvelTag, const Expr::Tag& yvelTag, const Expr::Tag& zvelTag,
                       const Expr::Tag& diffFluxXTag, const Expr::Tag& diffFluxYTag, const Expr::Tag& diffFluxZTag,
                       const Expr::Tag& viscTag,
                       const Expr::Tag& dilataionTag,
                       const Expr::Tag& pressureTag,
                       const Expr::Tag& strainxxTag, const Expr::Tag& strainyxTag, const Expr::Tag& strainzxTag,
                       const Expr::Tag& strainxyTag, const Expr::Tag& strainyyTag, const Expr::Tag& strainzyTag,
                       const Expr::Tag& strainxzTag, const Expr::Tag& strainyzTag, const Expr::Tag& strainzzTag,
                       const Expr::Tag& mwTag, const Expr::Tag& cpTag, const Expr::Tag& tempTag,
                       const Expr::TagList& specEnthTags,
                       const Expr::TagList& xSpecDiffFluxTags,
                       const Expr::TagList& ySpecDiffFluxTags,
                       const Expr::TagList& zSpecDiffFluxTags,
                       const Expr::TagList& specRxnTags,
                       const double alpha )
     : Expr::Expression<VolFieldT>(),
       doX_   ( xvelTag != Expr::Tag() ),
       doY_   ( yvelTag != Expr::Tag() ),
       doZ_   ( zvelTag != Expr::Tag() ),
       alpha_ ( alpha                  )
     {
       this->set_gpu_runnable( true );

       visc_ = this->template create_field_request<VolFieldT>( viscTag      );
       dil_  = this->template create_field_request<VolFieldT>( dilataionTag );
       p_    = this->template create_field_request<VolFieldT>( pressureTag  );

       if( doX_ ){
         xvel_      = this->template create_field_request<VolFieldT>( xvelTag      );
         strainxx_  = this->template create_field_request<XFace    >( strainxxTag  );
         diffFluxX_ = this->template create_field_request<XFace    >( diffFluxXTag );
         if( doY_ ) strainyx_ = this->template create_field_request<YFace>( strainyxTag );
         if( doZ_ ) strainzx_ = this->template create_field_request<ZFace>( strainzxTag );
       }
       if( doY_ ){
         yvel_      = this->template create_field_request<VolFieldT>( yvelTag      );
         strainyy_  = this->template create_field_request<YFace    >( strainyyTag  );
         diffFluxY_ = this->template create_field_request<YFace    >( diffFluxYTag );
         if( doX_ ) strainxy_ = this->template create_field_request<XFace>( strainxyTag );
         if( doZ_ ) strainzy_ = this->template create_field_request<ZFace>( strainzyTag );
       }
       if( doZ_ ){
         zvel_      = this->template create_field_request<VolFieldT>( zvelTag      );
         strainzz_  = this->template create_field_request<ZFace    >( strainzzTag  );
         diffFluxZ_ = this->template create_field_request<ZFace    >( diffFluxZTag );
         if( doX_ ) strainxz_ = this->template create_field_request<XFace>( strainxzTag );
         if( doY_ ) strainyz_ = this->template create_field_request<YFace>( strainyzTag );
       }

       if( cpTag != Expr::Tag() ){
         mw_   = this->template create_field_request<VolFieldT>( mwTag );
         cp_   = this->template create_field_request<VolFieldT>( cpTag );
#        ifdef HAVE_POKITT
         temp_ = this->template create_field_request<VolFieldT>( tempTag );
         this->template create_field_vector_request<VolFieldT>( specEnthTags, specEnthalpies_ );
         if( doX_ ){
           assert( xSpecDiffFluxTags.size() == CanteraObjects::number_species() );
           this->template create_field_vector_request<XFace>( xSpecDiffFluxTags, xSpecFluxes_ );
         }
         if( doY_ ){
           this->template create_field_vector_request<YFace>( ySpecDiffFluxTags, ySpecFluxes_ );
           assert( ySpecDiffFluxTags.size() == CanteraObjects::number_species() );
         }
         if( doZ_ ){
           this->template create_field_vector_request<ZFace>( zSpecDiffFluxTags, zSpecFluxes_ );
           assert( zSpecDiffFluxTags.size() == CanteraObjects::number_species() );
         }
         if( specRxnTags.size() > 0 ){
           assert( specRxnTags.size() == CanteraObjects::number_species() );
           this->template create_field_vector_request<VolFieldT>( specRxnTags, specRxnRates_ );
         }
#        endif
       }
     }

   public:

     class Builder : public Expr::ExpressionBuilder
     {
       const Expr::TagList velTags_, diffFluxTags_;
       const Expr::Tag viscTag_, dilTag_, pressureTag_, mwTag_, cpTag_, tempTag_;
       const Expr::TagList specEnthTags_, xSpecDiffFluxTags_, ySpecDiffFluxTags_, zSpecDiffFluxTags_, specRxnTags_;
       const Expr::Tag strainxxTag_, strainyxTag_, strainzxTag_;
       const Expr::Tag strainxyTag_, strainyyTag_, strainzyTag_;
       const Expr::Tag strainxzTag_, strainyzTag_, strainzzTag_;
       const double alpha_;
     public:
       /**
        *  @brief Build a ArtCompASREnergy expression
        *  @param resultTag the tag for the value that this expression computes
        */
        Builder( const Expr::Tag& resultTag,
                 const Expr::TagList& velTags,
                 const Expr::TagList& diffFluxTags,
                 const Expr::Tag& viscTag,
                 const Expr::Tag& dilataionTag,
                 const Expr::Tag& pressureTag,
                 const Expr::Tag& strainxxTag, const Expr::Tag& strainyxTag, const Expr::Tag& strainzxTag, // X-momentum
                 const Expr::Tag& strainxyTag, const Expr::Tag& strainyyTag, const Expr::Tag& strainzyTag, // Y-momentum
                 const Expr::Tag& strainxzTag, const Expr::Tag& strainyzTag, const Expr::Tag& strainzzTag, // Z-momentum
                 const Expr::Tag& mwTag,
                 const Expr::Tag& cpTag, // if empty, we will assume \gamma=cp/cv = 1.4 and get cp from that.
                 const Expr::Tag& tempTag,
                 const Expr::TagList& specEnthTags,
                 const Expr::TagList& xSpecDiffFluxTags,
                 const Expr::TagList& ySpecDiffFluxTags,
                 const Expr::TagList& zSpecDiffFluxTags,
                 const Expr::TagList& specRxnTags,
                 const double alpha,
                 const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          velTags_     ( velTags ),
          diffFluxTags_( diffFluxTags),
          viscTag_     ( viscTag ),
          dilTag_      ( dilataionTag ),
          pressureTag_ ( pressureTag ),
          mwTag_       ( mwTag ),
          cpTag_       ( cpTag ),
          tempTag_     ( tempTag ),
          specEnthTags_( specEnthTags ),
          xSpecDiffFluxTags_( xSpecDiffFluxTags ), ySpecDiffFluxTags_( ySpecDiffFluxTags ), zSpecDiffFluxTags_( zSpecDiffFluxTags ),
          specRxnTags_ ( specRxnTags ),
          strainxxTag_ ( strainxxTag ), strainyxTag_( strainyxTag ), strainzxTag_( strainzxTag ),
          strainxyTag_ ( strainxyTag ), strainyyTag_( strainyyTag ), strainzyTag_( strainzyTag ),
          strainxzTag_ ( strainxzTag ), strainyzTag_( strainyzTag ), strainzzTag_( strainzzTag ),
          alpha_(alpha)
        {
          assert( velTags.size()      == 3 );
          assert( diffFluxTags.size() == 3 );
        }

        Expr::ExpressionBase* build() const{
          return new ArtCompASREnergy<VolFieldT>( velTags_[0], velTags_[1], velTags_[2],
                                                  diffFluxTags_[0], diffFluxTags_[1], diffFluxTags_[2],
                                                  viscTag_, dilTag_, pressureTag_,
                                                  strainxxTag_, strainyxTag_, strainzxTag_,
                                                  strainxyTag_, strainyyTag_, strainzyTag_,
                                                  strainxzTag_, strainyzTag_, strainzzTag_,
                                                  mwTag_, cpTag_, tempTag_, specEnthTags_,
                                                  xSpecDiffFluxTags_, ySpecDiffFluxTags_, zSpecDiffFluxTags_,
                                                  specRxnTags_, alpha_ );
        }

     };  /* end of Builder class */

     ~ArtCompASREnergy(){}

     void bind_operators( const SpatialOps::OperatorDatabase& opDB )
     {
       if( doX_ ){
         xInterp_            = opDB.retrieve_operator<XInterp          >();
         divX_               = opDB.retrieve_operator<DivX             >();
         sVol2XFluxInterpOp_ = opDB.retrieve_operator<SVol2XFluxInterpT>();
         dpdx_               = opDB.retrieve_operator<GradXT           >();
         ddx_                = opDB.retrieve_operator<DDXT             >();
         xFluxInterpOp_      = opDB.retrieve_operator<XFlux2SVolInterpT>();
       }
       if( doY_ ){
         yInterp_            = opDB.retrieve_operator<YInterp          >();
         divY_               = opDB.retrieve_operator<DivY             >();
         sVol2YFluxInterpOp_ = opDB.retrieve_operator<SVol2YFluxInterpT>();
         dpdy_               = opDB.retrieve_operator<GradYT           >();
         ddy_                = opDB.retrieve_operator<DDYT             >();
         yFluxInterpOp_      = opDB.retrieve_operator<YFlux2SVolInterpT>();
       }

       if( doZ_ ){
         zInterp_            = opDB.retrieve_operator<ZInterp          >();
         divZ_               = opDB.retrieve_operator<DivZ             >();
         sVol2ZFluxInterpOp_ = opDB.retrieve_operator<SVol2ZFluxInterpT>();
         dpdz_               = opDB.retrieve_operator<GradZT           >();
         ddz_                = opDB.retrieve_operator<DDZT             >();
         zFluxInterpOp_      = opDB.retrieve_operator<ZFlux2SVolInterpT>();
       }
     }

     void evaluate()
     {
       VolFieldT& result = this->value();
       const VolFieldT& visc = visc_->field_ref();
       const VolFieldT& dil  = dil_ ->field_ref();
       const VolFieldT& p    = p_   ->field_ref();

       const double scale1 = 1.0 - 1.0/(alpha_*alpha_);

       SpatialOps::SpatFldPtr<VolFieldT> scale2 = SpatialOps::SpatialFieldStore::get<VolFieldT>( p );
       if( cp_ ){
         const double gasConstant = 8314.459848;  // universal R = J/(kmol K).
         // scale2 is gamma/(gamma-1), which is also equal to (cp*MW/R)
         *scale2 <<= cp_->field_ref() * mw_->field_ref() / gasConstant;
       }
       else{
         const double gamma = 1.4;  // assume diatomic gas, for which gamma=1.4.
         *scale2 <<= gamma/(gamma-1.0);  // jcs note this is also equal to (c_p*MW/R)
       }

       if( doX_ && doY_ && doZ_ ){

         const XFace& strainxx = strainxx_->field_ref();
         const YFace& strainyx = strainyx_->field_ref();
         const ZFace& strainzx = strainzx_->field_ref();
         const XFace& strainxy = strainxy_->field_ref();
         const YFace& strainyy = strainyy_->field_ref();
         const ZFace& strainzy = strainzy_->field_ref();
         const XFace& strainxz = strainxz_->field_ref();
         const YFace& strainyz = strainyz_->field_ref();
         const ZFace& strainzz = strainzz_->field_ref();

         const VolFieldT& xvel = xvel_->field_ref();
         const VolFieldT& yvel = yvel_->field_ref();
         const VolFieldT& zvel = zvel_->field_ref();

         const XFace& diffFluxX = diffFluxX_->field_ref();
         const YFace& diffFluxY = diffFluxY_->field_ref();
         const ZFace& diffFluxZ = diffFluxZ_->field_ref();

         result <<= scale1 * *scale2 * p * dil
             - scale1 * 2.0 * visc
               * (  (*xFluxInterpOp_) ( (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil)) * (*ddx_)(xvel) + strainxy * (*ddx_)(yvel) + strainxz * (*ddx_)(zvel) )
                  + (*yFluxInterpOp_) (  strainyx * (*ddy_)(xvel) + (strainyy - 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*ddy_)(yvel) + strainyz * (*ddy_)(zvel) )
                  + (*zFluxInterpOp_) (  strainzx * (*ddz_)(xvel) + strainzy * (*ddz_)(yvel) + (strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*ddz_)(zvel) )
                 )
             + scale1 * ( (*divX_)(diffFluxX) + (*divY_)(diffFluxY) + (*divZ_)(diffFluxZ) );
       }
       else{
         result <<= 0.0;
         if( doX_ ){ // 1D x or 2D xy or 2D xz
           const XFace&     strainxx = strainxx_  ->field_ref();
           const XFace&     diffFluxX = diffFluxX_->field_ref();
           const VolFieldT& xvel      = xvel_     ->field_ref();

           result <<= result - (*xFluxInterpOp_) ( (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil)) * (*ddx_)(xvel) );

           if( doY_ ){
             const VolFieldT& yvel  = yvel_     ->field_ref();
             const YFace& diffFluxY = diffFluxY_->field_ref();
             const YFace& strainyx  = strainyx_ ->field_ref();
             const XFace& strainxy  = strainxy_ ->field_ref();
             const YFace& strainyy  = strainyy_ ->field_ref();

             result <<= result - (  (*xFluxInterpOp_) ( strainxy * (*ddx_)(yvel)  )
                                  + (*yFluxInterpOp_) ( strainyx * (*ddy_)(xvel)  ) );
           }
           if( doZ_ ){
             const VolFieldT& zvel = zvel_->field_ref();
             const ZFace& strainzx = strainzx_->field_ref();
             const XFace& strainxz = strainxz_->field_ref();

             result <<= result - ( (*xFluxInterpOp_) ( strainxz * (*ddx_)(zvel) )
                                 + (*zFluxInterpOp_) ( strainzx * (*ddz_)(xvel) ) );
           }
         } // doX_
         else if( doY_ ){ // 1D y or 2D yz
           const YFace& strainyy = strainyy_->field_ref();
           const VolFieldT& yvel = yvel_->field_ref();
           result <<= result - ( (*yFluxInterpOp_) (  (strainyy - 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*ddy_)(yvel) ) );

           if( doZ_ ){
             const VolFieldT& zvel = zvel_->field_ref();
             const ZFace& strainzy = strainzy_->field_ref();
             const ZFace& strainzz = strainzz_->field_ref();
             result <<= result - (*zFluxInterpOp_) ( strainzy * (*ddz_)(yvel) + (strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*ddz_)(zvel) );
           }
         }
         else if( doZ_ ){ // 1D z
           const VolFieldT& zvel = zvel_->field_ref();
           const ZFace& strainzz = strainzz_->field_ref();
           result <<= result - (*zFluxInterpOp_) ( (strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*ddz_)(zvel) );
         }

         result <<= scale1 * *scale2 * p * dil - scale1 * 2.0 * visc * result;
         if( doX_ ){
           const XFace& diffFluxX = diffFluxX_->field_ref();
           result <<= result + scale1 * (*divX_)(diffFluxX);
         }
         if( doY_ ){
           const YFace& diffFluxY = diffFluxY_->field_ref();
           result <<= result + scale1 * (*divY_)(diffFluxY);
         }
         if( doZ_ ){
           const ZFace& diffFluxZ = diffFluxZ_->field_ref();
           result <<= result + scale1 * (*divZ_)(diffFluxZ);
         }
       }

       // sum in species terms as appropriate
#      ifdef HAVE_POKITT
       if( cp_ ){
         const VolFieldT& mw   = mw_  ->field_ref();
         const VolFieldT& cp   = cp_  ->field_ref();
         const VolFieldT& temp = temp_->field_ref();
         const std::vector<double>& specMW = CanteraObjects::molecular_weights();
         const int nspec = CanteraObjects::number_species();
         if( doX_ && doY_ && doZ_ ){
           for( int i=0; i<nspec; ++i ){
             const VolFieldT& hi = specEnthalpies_[i]->field_ref();
             if( specRxnRates_.size() == nspec ){
               result <<= result + scale1 * ( hi - mw * cp * temp / specMW[i] )
                                 * (
                                     -(*divX_)( xSpecFluxes_[i]->field_ref() )
                                     -(*divY_)( ySpecFluxes_[i]->field_ref() )
                                     -(*divZ_)( zSpecFluxes_[i]->field_ref() )
                                     +specRxnRates_[i]->field_ref()
                                   );
             }
             else{
               result <<= result + scale1 * ( hi - mw * cp * temp / specMW[i] )
                                 * (
                                     -(*divX_)( xSpecFluxes_[i]->field_ref() )
                                     -(*divY_)( ySpecFluxes_[i]->field_ref() )
                                     -(*divZ_)( zSpecFluxes_[i]->field_ref() )
                                   );
             }
           }
         }
         else{
           for( int i=0; i<nspec; ++i ){
             const VolFieldT& hi = specEnthalpies_[i]->field_ref();
             // reuse scale2 from above
             (*scale2) <<= scale1 * ( hi - mw * cp * temp / specMW[i] );
             if( doX_ ) result <<= result - (*scale2) * (*divX_)( xSpecFluxes_[i]->field_ref() );
             if( doY_ ) result <<= result - (*scale2) * (*divY_)( ySpecFluxes_[i]->field_ref() );
             if( doZ_ ) result <<= result - (*scale2) * (*divZ_)( zSpecFluxes_[i]->field_ref() );
             if( specRxnRates_.size() == nspec ) result <<= result + (*scale2) * specRxnRates_[i]->field_ref();
           }
         } // 1D or 2D
       } // species terms
#      endif
     }
   };

  //============================================================================

  PreconditioningParser::PreconditioningParser( Uintah::ProblemSpecP& wasatchSpec,
                                                GraphCategories& gc )
  : wasatchSpec_( wasatchSpec ),
    factory_( gc[ADVANCE_SOLUTION]->exprFactory )
  {
    method_ = NO_PRECONDITIONING;  // populate this here.

    Uintah::ProblemSpecP precondParams = wasatchSpec_->findBlock("Preconditioning");

    if( !precondParams ) return; // no preconditioner block

    if( precondParams->findBlock("ArtificialCompressibility") ){
      Uintah::ProblemSpecP acParams = precondParams->findBlock("ArtificialCompressibility");
      std::string acModel;
      acParams->getAttribute("model",acModel);
      if( acModel == "ASR" ){
        method_ = ACOUSTIC_STIFFNESS_REDUCTION;
        setup_asr( acParams );
      }
      else if( acModel == "PGS" ){
        method_ = PRESSURE_GRADIENT_SCALING;
        setup_pgs( acParams );
      }
    }

    // OTHER PRECONDITIONER SETUP WILL EVENTUALLY GO HERE...
  }

  //---------------------------------------------------------------------------

  PreconditioningParser::~PreconditioningParser()
  {}

  //---------------------------------------------------------------------------

  Expr::TagList PreconditioningParser::vel_tags()
  {
    std::string xmom, ymom, zmom, xvel, yvel, zvel;
    Uintah::ProblemSpecP momSpec = wasatchSpec_->findBlock("MomentumEquations");
    momSpec->getWithDefault( "X-Velocity", xvel, "" );
    momSpec->getWithDefault( "Y-Velocity", yvel, "" );
    momSpec->getWithDefault( "Z-Velocity", zvel, "" );

    return tag_list(
        (xvel=="") ? Expr::Tag() : Expr::Tag(xvel,Expr::STATE_NONE),
        (yvel=="") ? Expr::Tag() : Expr::Tag(yvel,Expr::STATE_NONE),
        (zvel=="") ? Expr::Tag() : Expr::Tag(zvel,Expr::STATE_NONE)
        );
  }

  //---------------------------------------------------------------------------

  void PreconditioningParser::setup_pgs( Uintah::ProblemSpecP pgsParams )
  {
    const TagNames& tags = TagNames::self();
    double alpha = 10.0;
    pgsParams->getAttribute("coef",alpha);

    //-------------------------------
    // add term to momentum equation(s):
    Uintah::ProblemSpecP momSpec = wasatchSpec_->findBlock("MomentumEquations");
    std::string xmom, ymom, zmom;
    momSpec->getWithDefault( "X-Momentum", xmom, "" );
    momSpec->getWithDefault( "Y-Momentum", ymom, "" );
    momSpec->getWithDefault( "Z-Momentum", zmom, "" );

    if( xmom != "" ){
      const Expr::Tag pgsPressureTag( "PGS_pressure_" + xmom, Expr::STATE_NONE );
      typedef typename ArtCompPGSPressure<SpatialOps::XDIR>::Builder ACPGSPressure;
      factory_->register_expression( scinew ACPGSPressure( pgsPressureTag, tags.pressure, alpha ) );
      factory_->attach_dependency_to_expression( pgsPressureTag,
                                                 Expr::Tag( xmom+"_rhs", Expr::STATE_NONE ),
                                                 Expr::ADD_SOURCE_EXPRESSION );
    }
    if( ymom != "" ){
      const Expr::Tag pgsPressureTag( "PGS_pressure_" + ymom, Expr::STATE_NONE );
      typedef typename ArtCompPGSPressure<SpatialOps::YDIR>::Builder ACPGSPressure;
      factory_->register_expression( scinew ACPGSPressure( pgsPressureTag, tags.pressure, alpha ) );
      factory_->attach_dependency_to_expression( pgsPressureTag,
                                                 Expr::Tag( ymom+"_rhs", Expr::STATE_NONE ),
                                                 Expr::ADD_SOURCE_EXPRESSION );
    }
    if( zmom != "" ){
      const Expr::Tag pgsPressureTag( "PGS_pressure_" + zmom, Expr::STATE_NONE );
      typedef typename ArtCompPGSPressure<SpatialOps::ZDIR>::Builder ACPGSPressure;
      factory_->register_expression( scinew ACPGSPressure( pgsPressureTag, tags.pressure, alpha ) );
      factory_->attach_dependency_to_expression( pgsPressureTag,
                                                 Expr::Tag( zmom+"_rhs", Expr::STATE_NONE ),
                                                 Expr::ADD_SOURCE_EXPRESSION );
    }

    //-------------------------------
    // add term to the energy equation
    std::string totEnerg;
    wasatchSpec_->findBlock("EnergyEquation")->get("SolutionVariable",totEnerg);
    const Expr::Tag pgsEnergyTag( "AC_Energy", Expr::STATE_NONE );
    typedef ArtCompPGSEnergy::Builder ACEnergy;
    factory_->register_expression( scinew ACEnergy( pgsEnergyTag, tags.pressure, vel_tags(), alpha ) );
    factory_->attach_dependency_to_expression( pgsEnergyTag,
                                               Expr::Tag( totEnerg+"_rhs", Expr::STATE_NONE ),
                                               Expr::ADD_SOURCE_EXPRESSION );
  }

  //---------------------------------------------------------------------------

  void PreconditioningParser::setup_asr( Uintah::ProblemSpecP asrParams )
  {
    double alpha = 10.0;
    asrParams->getAttribute( "coef", alpha );
    const TagNames& tags = TagNames::self();
    Expr::TagList diffFluxTags;
    Expr::TagList xSpecFluxTags, ySpecFluxTags, zSpecFluxTags, specRxnTags, specEnthTags;
    Expr::Tag cpTag; // empty tag is special for ASR - uses gamma=1.4.
#   ifdef HAVE_POKITT
    Uintah::ProblemSpecP speciesSpec = wasatchSpec_->findBlock("SpeciesTransportEquations");
    if( speciesSpec ){
      cpTag = tags.heatCapacity;
      // names for the energy diffusive flux change when we are doing species transport (uggh)
      diffFluxTags = Expr::tag_list( tags.xHeatFlux, tags.yHeatFlux, tags.zHeatFlux );

      for( int i=0; i<CanteraObjects::number_species(); ++i ){
        const std::string specName = CanteraObjects::species_name(i);
        specEnthTags.push_back( Expr::Tag( specName+"_"+tags.enthalpy.name(), Expr::STATE_NONE ) );
        xSpecFluxTags.push_back( Expr::Tag( specName + tags.diffusiveflux + "X", Expr::STATE_NONE ) );
        ySpecFluxTags.push_back( Expr::Tag( specName + tags.diffusiveflux + "Y", Expr::STATE_NONE ) );
        zSpecFluxTags.push_back( Expr::Tag( specName + tags.diffusiveflux + "Z", Expr::STATE_NONE ) );
      }
      if( speciesSpec->findBlock("DetailedKinetics") ){
        for( int i=0; i<CanteraObjects::number_species(); ++i ){
          specRxnTags.push_back( Expr::Tag( "rr_"+CanteraObjects::species_name(i), Expr::STATE_NONE ) );
        }
      }
    }
    else
#   endif
    {
      const std::string& temperature = tags.temperature.name();
      diffFluxTags = Expr::tag_list( Expr::Tag( temperature + "_diffVelocity_X", Expr::STATE_NONE ),
                                     Expr::Tag( temperature + "_diffVelocity_Y", Expr::STATE_NONE ),
                                     Expr::Tag( temperature + "_diffVelocity_Z", Expr::STATE_NONE ) );
    }
    const Expr::Tag viscTag = parse_nametag( wasatchSpec_->findBlock("MomentumEquations")->findBlock("Viscosity")->findBlock("NameTag") );
    const Expr::Tag asrEnergyTag( "ASR_Energy", Expr::STATE_NONE );

    typedef ArtCompASREnergy<SpatialOps::SVolField>::Builder ASR;
    factory_->register_expression( scinew ASR( asrEnergyTag, vel_tags(), diffFluxTags,
                                               viscTag, tags.dilatation, tags.pressure,
                                               tags.strainxx, tags.strainyx, tags.strainzx,
                                               tags.strainxy, tags.strainyy, tags.strainzy,
                                               tags.strainxz, tags.strainyz, tags.strainzz,
                                               tags.mixMW, cpTag, tags.temperature, specEnthTags,
                                               xSpecFluxTags, ySpecFluxTags, zSpecFluxTags,
                                               specRxnTags, alpha ) );
    std::string totEnerg;
    wasatchSpec_->findBlock("EnergyEquation")->get("SolutionVariable",totEnerg);
    factory_->attach_dependency_to_expression( asrEnergyTag,
                                               Expr::Tag( totEnerg+"_rhs", Expr::STATE_NONE ),
                                               Expr::ADD_SOURCE_EXPRESSION );
  }
  //---------------------------------------------------------------------------

} /* namespace WasatchCore */
