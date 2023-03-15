/*
 * The MIT License
 *
 * Copyright (c) 2010-2023 The University of Utah
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

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSHelper.h>

#include <pokitt/CanteraObjects.h>

#ifndef PressureMaterialDerivativePartial_h
#define PressureMaterialDerivativePartial_h

using ScalarEOS::resolve_tag;
using ScalarEOS::resolve_tag_list;

/**
 *  \class PressureMaterialDerivativePartial
 *  \author Josh McConnell
 *  \date   October 2018
 *
 *  \brief Computes
 *  \f[
 *   \widehat{\frac{DP}{Dt}}
 *   &=& \frac{D \P}{D t} + \frac{\gamma \rho RT}{M} \nabla \cdot \mathbf{u}
 *   &=& \gamma RT\left[ \frac{1}{M}S_{\rho}-\frac{D}{Dt} \left( \frac{1}{M} \right) \right]
 *       + \left( \gamma-1 \right)\left( \rho c_p \widehat{\frac{DT}{Dt}} \right)
 *  \f]
 *
 *  where
 *  \f[
 *    \rho c_p \widehat{\frac{DT}{Dt}}
 *    &=& \rho c_p \frac{DT}{Dt} - \frac{DP}{Dt}
 *    &=& -\nabla \cdot (\mathbf{q}) + S_{\rho h}
 *        - \sum_{i=1}^{n_{s}-1}
 *        \left( h_{i}-h_{n_{s}} \right) \left( -\nabla \cdot \mathbf{J}_{\rho Y_i}+S_{\rho Y_{i}} \right),
 *  \f]
 *   \f$\mathbf{q}\f$ is the heat flux, \f$P\f$ is pressure, \f$\gamma = \frac{c_p M}{c_p M - R}\f$,
 *   \f$\rho\f$ is density, \f$\mathbf{u}\f$ is velocity, \f$\c_p\f$ is heat capacity, and \f$\R\f$ is the universal
 *   gas constant.
 */
template<typename FieldT>
class PressureMaterialDerivativePartial : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFluxT;
  typedef typename FaceTypes::YFace YFluxT;
  typedef typename FaceTypes::ZFace ZFluxT;

  typedef typename SpatialOps::BasicOpTypes<FieldT> OpTypes;
  typedef typename OpTypes::DivX DivX;
  typedef typename OpTypes::DivY DivY;
  typedef typename OpTypes::DivZ DivZ;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  SVolToFieldTInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,XVolField,XFluxT>::type  XVolToXFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,YVolField,YFluxT>::type  YVolToYFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ZVolField,ZFluxT>::type  ZVolToZFluxInterpT;

  typedef typename  boost::shared_ptr< const Expr::FieldRequest<FieldT> > volRequestT;

  typedef boost::shared_ptr< const Expr::FieldRequest<XFluxT> > XFluxRequestT;
  typedef boost::shared_ptr< const Expr::FieldRequest<YFluxT> > YFluxRequestT;
  typedef boost::shared_ptr< const Expr::FieldRequest<ZFluxT> > ZFluxRequestT;

  DECLARE_FIELDS( FieldT, density_, enthalpy_, temperature_, gamma_ )
  DECLARE_FIELD( FieldT, enthalpySrc_ )

  DECLARE_VECTOR_OF_FIELDS( FieldT, speciesSrcs_       )
  DECLARE_VECTOR_OF_FIELDS( FieldT, speciesMassFracs_  )
  DECLARE_VECTOR_OF_FIELDS( FieldT, speciesEnthalpies_ )

  DECLARE_FIELD( XFluxT, enthalpyDiffFluxX_ )
  DECLARE_FIELD( YFluxT, enthalpyDiffFluxY_ )
  DECLARE_FIELD( ZFluxT, enthalpyDiffFluxZ_ )

  DECLARE_VECTOR_OF_FIELDS( XFluxT, speciesDiffFluxX_ )
  DECLARE_VECTOR_OF_FIELDS( YFluxT, speciesDiffFluxY_ )
  DECLARE_VECTOR_OF_FIELDS( ZFluxT, speciesDiffFluxZ_ )

  DECLARE_FIELD( SVolField, volfrac_   )
  DECLARE_FIELD( XVolField, xAreaFrac_ )
  DECLARE_FIELD( YVolField, yAreaFrac_ )
  DECLARE_FIELD( ZVolField, zAreaFrac_ )

  const DivX*                divOpX_;
  const DivY*                divOpY_;
  const DivZ*                divOpZ_;
  const SVolToFieldTInterpT* volFracInterpOp_;
  const XVolToXFluxInterpT*  xVolToXFluxInterpOp_;
  const YVolToYFluxInterpT*  yVolToYFluxInterpOp_;
  const ZVolToZFluxInterpT*  zVolToZFluxInterpOp_;

  PressureMaterialDerivativePartial( const Expr::Tag&        densityTag,
                                     const Expr::Tag&        temperatureTag,
                                     const Expr::Tag&        gammaTag,
                                     const Expr::TagList&    specEnthalpyTags,
                                     const FieldTagInfo&     enthalpyInfo,
                                     const FieldTagListInfo& speciesInfo );

  const Expr::Tag enthalpyDiffFluxXTag_, enthalpyDiffFluxYTag_, enthalpyDiffFluxZTag_, enthalpySrcTag_,
                  volFracTag_, xAreaFracTag_, yAreaFracTag_, zAreaFracTag_;

  const bool doXDir_, doYDir_, doZDir_,
             haveEnthalpySrc_, haveSpeciesSrcs_, is3d_,
             haveVolFrac_, haveXAreaFrac_, haveYAreaFrac_, haveZAreaFrac_;

  const double gasConst_;
  const int nSpec_;
  std::vector<double> mwInvTerm_;

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a PressureMaterialDerivativePartial expression
     *  @param dpdtPartTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag&        dpdtPartTag,
             const Expr::Tag&        densityTag,
             const Expr::Tag&        temperatureTag,
             const Expr::Tag&        gammaTag,
             const Expr::TagList&    specEnthalpyTags,
             const FieldTagInfo&     enthalpyInfo,
             const FieldTagListInfo& speciesInfo );

    Expr::ExpressionBase* build() const{
      return new PressureMaterialDerivativePartial( densityTag_,
                                                    temperatureTag_,
                                                    gammaTag_,
                                                    specEnthalpyTags_,
                                                    enthalpyInfo_,
                                                    speciesInfo_ );
    }

  private:
    const Expr::Tag densityTag_, temperatureTag_, gammaTag_;
    const Expr::TagList specEnthalpyTags_;

    const FieldTagInfo     enthalpyInfo_;
    const FieldTagListInfo speciesInfo_;
  };

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void generic_density_weighted_material_derivative( FieldT&             result,
                                                     const FieldT&       phi,
                                                     const XFluxRequestT phiXFluxReq,
                                                     const YFluxRequestT phiYFluxReq,
                                                     const ZFluxRequestT phiZFluxReq,
                                                     const volRequestT   phiSrcReq );
  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
PressureMaterialDerivativePartial<FieldT>::
PressureMaterialDerivativePartial( const Expr::Tag&        densityTag,
                                   const Expr::Tag&        temperatureTag,
                                   const Expr::Tag&        gammaTag,
                                   const Expr::TagList&    specEnthalpyTags,
                                   const FieldTagInfo&     enthalpyInfo,
                                   const FieldTagListInfo& speciesInfo )
: Expr::Expression<FieldT>(),
  enthalpyDiffFluxXTag_( resolve_tag( DIFFUSIVE_FLUX_X, enthalpyInfo )         ),
  enthalpyDiffFluxYTag_( resolve_tag( DIFFUSIVE_FLUX_Y, enthalpyInfo )         ),
  enthalpyDiffFluxZTag_( resolve_tag( DIFFUSIVE_FLUX_Z, enthalpyInfo )         ),
  enthalpySrcTag_      ( resolve_tag( SOURCE_TERM     , enthalpyInfo   )       ),
  volFracTag_          ( resolve_tag( VOLUME_FRAC     , enthalpyInfo   )       ),
  xAreaFracTag_        ( resolve_tag( AREA_FRAC_X     , enthalpyInfo   )       ),
  yAreaFracTag_        ( resolve_tag( AREA_FRAC_Y     , enthalpyInfo   )       ),
  zAreaFracTag_        ( resolve_tag( AREA_FRAC_Z     , enthalpyInfo   )       ),
  doXDir_              ( enthalpyDiffFluxXTag_ != Expr::Tag()                  ),
  doYDir_              ( enthalpyDiffFluxYTag_ != Expr::Tag()                  ),
  doZDir_              ( enthalpyDiffFluxZTag_ != Expr::Tag()                  ),
  haveEnthalpySrc_     ( enthalpySrcTag_         != Expr::Tag()                ),
  haveSpeciesSrcs_     ( !(resolve_tag_list(SOURCE_TERM, speciesInfo).empty()) ),
  is3d_                ( doXDir_ && doYDir_ && doZDir_                         ),
  haveVolFrac_         ( volFracTag_   != Expr::Tag()                          ),
  haveXAreaFrac_       ( xAreaFracTag_ != Expr::Tag()                          ),
  haveYAreaFrac_       ( yAreaFracTag_ != Expr::Tag()                          ),
  haveZAreaFrac_       ( zAreaFracTag_ != Expr::Tag()                          ),
  gasConst_            ( CanteraObjects::gas_constant()                        ),
  nSpec_               ( CanteraObjects::number_species()                      )

{
  const Expr::Tag& enthalpyTag = resolve_tag( PRIMITIVE_VARIABLE , enthalpyInfo );
  assert(enthalpyTag != Expr::Tag() );

  /* ensure that species diffusive fluxes are enabled in the same directions
   * for species and enthalpy
   */
  bool xWrong = false;
  bool yWrong = false;
  bool zWrong = false;

  const Expr::TagList speciesMassFracTags  = resolve_tag_list(PRIMITIVE_VARIABLE, speciesInfo);
  const Expr::TagList speciesDiffFluxXTags = resolve_tag_list(DIFFUSIVE_FLUX_X  , speciesInfo);
  const Expr::TagList speciesDiffFluxYTags = resolve_tag_list(DIFFUSIVE_FLUX_Y  , speciesInfo);
  const Expr::TagList speciesDiffFluxZTags = resolve_tag_list(DIFFUSIVE_FLUX_Z  , speciesInfo);
  const Expr::TagList speciesSrcTags       = resolve_tag_list(SOURCE_TERM       , speciesInfo);

  if( doXDir_ == speciesDiffFluxXTags.empty() ) xWrong = true;
  if( doYDir_ == speciesDiffFluxYTags.empty() ) yWrong = true;
  if( doZDir_ == speciesDiffFluxZTags.empty() ) zWrong = true;

  if( xWrong || yWrong || zWrong ){
    std::ostringstream msg;
    msg << std::endl
        << __FILE__ << " : " << __LINE__
        << std::endl
        << "A conductive heat flux tag was specified, but not species diffusive "
        << "flux tags for the (or vice versa) for the following directions: \n"
        << "{"
        << (xWrong ? "X" : "")
        << (yWrong ? "Y" : "")
        << (zWrong ? "Z" : "")
        << "} \n";
    msg << "\n\nX: -----------------------------\n"
        << "species diffFlux tags are valid:" << (!speciesDiffFluxXTags.empty() ? "yes" : "no")
        << std::endl
        << "enthalpy diffFlux tag is valid :" << (doXDir_                       ? "yes" : "no")
        << "\n\nY: -----------------------------\n"
        << "species diffFlux tags are valid:" << (!speciesDiffFluxYTags.empty() ? "yes" : "no")
        << std::endl
        << "enthalpy diffFlux tag is valid : " << (doYDir_                      ? "yes" : "no")
        << "\n\nZ: -----------------------------\n"
        << "species diffFlux tags are valid:" << (!speciesDiffFluxZTags.empty() ? "yes" : "no")
        << std::endl
        << "enthalpy diffFlux tag is valid : " << (doZDir_                      ? "yes" : "no")
        << std::endl;

    throw std::invalid_argument(msg.str());
  }

  // ensure area fractions are correctly specified for 3D cases
  const bool haveAreaFrac = haveXAreaFrac_ || haveYAreaFrac_ || haveZAreaFrac_;
  if( is3d_ && haveAreaFrac ){
    if( !( haveXAreaFrac_ && haveYAreaFrac_ && haveZAreaFrac_ ) ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "In 3D, it is expected that if one area fraction is provided, they all are..."
          << std::endl << std::endl;
      throw std::invalid_argument( msg.str() );
    }
  }

  const std::vector<double>& mw = CanteraObjects::molecular_weights();
  mwInvTerm_.clear();
  for(int j=0; j<nSpec_-1; ++j)
  mwInvTerm_.push_back( 1.0/mw[j] - 1.0/mw[nSpec_-1] );

  this->set_gpu_runnable(true);

  density_      = this->template create_field_request<FieldT>( densityTag     );
  enthalpy_     = this->template create_field_request<FieldT>( enthalpyTag    );
  temperature_  = this->template create_field_request<FieldT>( temperatureTag );
  gamma_        = this->template create_field_request<FieldT>( gammaTag       );

  this->template create_field_vector_request<FieldT>( specEnthalpyTags   , speciesEnthalpies_ );
  this->template create_field_vector_request<FieldT>( speciesMassFracTags, speciesMassFracs_  );

  if(doXDir_){
    enthalpyDiffFluxX_ = this->template create_field_request<XFluxT>( enthalpyDiffFluxXTag_ );
    this->template create_field_vector_request<XFluxT>( speciesDiffFluxXTags, speciesDiffFluxX_ );
  }
  if(doYDir_){
    enthalpyDiffFluxY_ = this->template create_field_request<YFluxT>( enthalpyDiffFluxYTag_ );
    this->template create_field_vector_request<YFluxT>( speciesDiffFluxYTags, speciesDiffFluxY_ );
  }
  if(doZDir_){
    enthalpyDiffFluxZ_ = this->template create_field_request<ZFluxT>( enthalpyDiffFluxZTag_ );
    this->template create_field_vector_request<ZFluxT>( speciesDiffFluxZTags, speciesDiffFluxZ_ );
  }
  if( haveEnthalpySrc_ ) enthalpySrc_ = this->template create_field_request<FieldT>( enthalpySrcTag_ );
  if( haveSpeciesSrcs_ ) this->template create_field_vector_request<FieldT>( speciesSrcTags, speciesSrcs_ );

  if ( haveVolFrac_    ) volfrac_  = this->template create_field_request<FieldT>( volFracTag_ );

  if ( doXDir_ && haveXAreaFrac_ )  xAreaFrac_  = this->template create_field_request<XVolField>( xAreaFracTag_);
  if ( doYDir_ && haveYAreaFrac_ )  yAreaFrac_  = this->template create_field_request<YVolField>( yAreaFracTag_);
  if ( doZDir_ && haveZAreaFrac_ )  zAreaFrac_  = this->template create_field_request<ZVolField>( zAreaFracTag_);
}

//--------------------------------------------------------------------

template< typename FieldT >
void PressureMaterialDerivativePartial<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doXDir_ ) divOpX_ = opDB.retrieve_operator<DivX>();
  if( doYDir_ ) divOpY_ = opDB.retrieve_operator<DivY>();
  if( doZDir_ ) divOpZ_ = opDB.retrieve_operator<DivZ>();

  if( haveVolFrac_ ) volFracInterpOp_ = opDB.retrieve_operator<SVolToFieldTInterpT>();

  if( doXDir_ && haveXAreaFrac_ ) xVolToXFluxInterpOp_  = opDB.retrieve_operator<XVolToXFluxInterpT  >();
  if( doYDir_ && haveYAreaFrac_ ) yVolToYFluxInterpOp_  = opDB.retrieve_operator<YVolToYFluxInterpT  >();
  if( doZDir_ && haveZAreaFrac_ ) zVolToZFluxInterpOp_  = opDB.retrieve_operator<ZVolToZFluxInterpT  >();
}
//--------------------------------------------------------------------

template< typename FieldT >
void
PressureMaterialDerivativePartial<FieldT>::
generic_density_weighted_material_derivative( FieldT&             result,
                                              const FieldT&       phi,
                                              const XFluxRequestT phiXFluxReq,
                                              const YFluxRequestT phiYFluxReq,
                                              const ZFluxRequestT phiZFluxReq,
                                              const volRequestT   phiSrcReq ){
	/**
	 *  \brief Computes
	 *  \f[
	 *   \rho \frac{D \phi}{D t} =  - \nabla \cdot \mathbf{J}_{\rho \phi} + S_{\rho \phi},
	 *  \f]
	 *  where  \f$\mathbf{J}_{\rho \phi}\f$ and \f$S_{\rho \phi}\f$ are the diffusive flux and of source term,
	 *  respectively of \f$\rho \phi\f$.
	 */

 if(is3d_){
   // inline entire diffusive flux gradient calculation
   const XFluxT& xDiffFlux  =  phiXFluxReq->field_ref();
   const YFluxT& yDiffFlux  =  phiYFluxReq->field_ref();
   const ZFluxT& zDiffFlux  =  phiZFluxReq->field_ref();

   if( haveXAreaFrac_ ){ // previous error checking enforces that y and z area fractions are also present
     const XVolField& xAreaFrac = xAreaFrac_->field_ref();
     const YVolField& yAreaFrac = yAreaFrac_->field_ref();
     const ZVolField& zAreaFrac = zAreaFrac_->field_ref();
     result <<= -(*divOpX_)( (*xVolToXFluxInterpOp_)(xAreaFrac) * ( xDiffFlux ) )
                -(*divOpY_)( (*yVolToYFluxInterpOp_)(yAreaFrac) * ( yDiffFlux ) )
                -(*divOpZ_)( (*zVolToZFluxInterpOp_)(zAreaFrac) * ( zDiffFlux ) );
   }
   else{
     result <<= -(*divOpX_)( xDiffFlux )
                -(*divOpY_)( yDiffFlux )
                -(*divOpZ_)( zDiffFlux );
   }
 }// 3D case
 else{
   // handle 2D and 1D cases - not quite as efficient since we won't be
   // running as many production scale calculations in these configurations

   if (doXDir_) {
     if( haveXAreaFrac_ ) result <<= -(*divOpX_)( (*xVolToXFluxInterpOp_)(xAreaFrac_->field_ref()) * phiXFluxReq->field_ref() );
     else                 result <<= -(*divOpX_)( phiXFluxReq->field_ref() );
   } else{
     result <<= 0.0; // zero so that we can sum in Y and Z contributions as necessary
   }

   if (doYDir_) {
     if( haveYAreaFrac_ ) result <<= result -(*divOpY_)( (*yVolToYFluxInterpOp_)(yAreaFrac_->field_ref()) * phiYFluxReq->field_ref() );
     else                 result <<= result -(*divOpY_)( phiYFluxReq->field_ref() );
   }

   if (doZDir_) {
     if( haveZAreaFrac_ ) result <<= result -(*divOpZ_)( (*zVolToZFluxInterpOp_)(zAreaFrac_->field_ref()) * phiZFluxReq->field_ref() );
     else                 result <<= result -(*divOpZ_)( phiZFluxReq->field_ref() );
   }
 } // 2D and 1D cases

 // add contribution from phi source term if it exists
 if( phiSrcReq != nullptr ) result <<= result + phiSrcReq->field_ref();

}

//--------------------------------------------------------------------

template< typename FieldT >
void
PressureMaterialDerivativePartial<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& dPdtPart = this->value();

  const FieldT& density     = density_    ->field_ref();
  const FieldT& enthalpy    = enthalpy_   ->field_ref();
  const FieldT& temperature = temperature_->field_ref();
  const FieldT& gamma       = gamma_      ->field_ref();

  /**
   *  Begin by calculating
   *  \f[
   *   -\frac{\rho}{M^2}\frac{DM}{Dt}
   *   &=& \rho\frac{D}{Dt} \left( \frac{1}{M} \right)
   *   &=& \sum_{i=1}^{n_s-1}\left \frac{1}{M_i} - \frac{1}{M_n_s} \right) \rho\frac{DY_i}{Dt}
   *  \f]
   *   where \f$M_i\f$ and \f$Y_i\f$ are the mass fraction and molecular weight of species \f$i\f$,
   *   \f$M\f$ is the mixture molecular weight, \f$\rho\f$ is the density and \f$\n_s\f$ is the
   *   number of species considered.
   */


  SpatFldPtr<FieldT> gammaRTPtr = SpatialFieldStore::get<FieldT>( density );
  FieldT& gammaRT = *gammaRTPtr;
  gammaRT <<= gamma * temperature * gasConst_;

   SpatFldPtr<FieldT> gammaMinusOnePtr = SpatialFieldStore::get<FieldT>( density );
   FieldT& gammaMinusOne = *gammaMinusOnePtr;
   gammaMinusOne <<= gamma - 1;

   SpatFldPtr<FieldT> rhsTermPtr = SpatialFieldStore::get<FieldT>( density );
   FieldT& speciesRHSTerm = *rhsTermPtr;

  dPdtPart <<= 0;

  const FieldT& hn = speciesEnthalpies_[nSpec_-1]->field_ref();

  for(int i = 0; i<nSpec_-1; ++i){
    const FieldT& yi = speciesMassFracs_ [i]->field_ref();
    const FieldT& hi = speciesEnthalpies_[i]->field_ref();

    // obtain \f$ \rho\frac{DY_i}{Dt} = -\graddot\mathbf{J}_{\rho Y_{i}} + S_{\rho Y_{i}} - Y_{i}S_{\rho} \f$
    generic_density_weighted_material_derivative( speciesRHSTerm                                   ,
                                                  yi                                               ,
                                                  doXDir_          ? speciesDiffFluxX_[i] : nullptr,
                                                  doYDir_          ? speciesDiffFluxY_[i] : nullptr,
                                                  doZDir_          ? speciesDiffFluxZ_[i] : nullptr,
                                                  haveSpeciesSrcs_ ? speciesSrcs_     [i] : nullptr );

    dPdtPart <<= dPdtPart + ( gammaRT * mwInvTerm_[i] + gammaMinusOne * (hn - hi) ) * speciesRHSTerm;
  }

  /**
   *  At this point, we have
   *  \f[
   *     - \sum_{i=1}^{n_{s}-1}
   *     \left\{
   *       \gamma RT \left( \frac{1}{M_i} - \frac{1}{M_n_s} \right)
   *       - \left( \gamma -1 \right) \left( h_i - h_n_s \right)
   *     \right\}
   *     \left(-\nabla \cdot \mathbf{J}_{\rho Y_{i}} + S_{\rho Y_{i}} \right),
   *  \f]
   *
   *  The next step is to obtain
   *  \f[
   *      \left( \gamma -1 \right) left(-\nabla \cdot \mathbf{J}_{\rho Y_{i}} + S_{\rho Y_{i}} \right),
   *  \f]
   *
   */

  FieldT& enthalpyRHSTerm = *rhsTermPtr;

  /**
   *  Obtain the 'generic' part of the enthalpy material derivative,
   *  \f[
   *   \rho\frac{Dh}{Dt} - \frac{DP}{Dt} = -\graddot\mathbf{q} + S_{\rho h}
   *  \f]
   *   Note that the term \f$h S_{\rho}\f$ is not included.
   */
  generic_density_weighted_material_derivative( enthalpyRHSTerm,
                                                enthalpy,
                                                enthalpyDiffFluxX_,
                                                enthalpyDiffFluxY_,
                                                enthalpyDiffFluxZ_,
                                                enthalpySrc_ );

  dPdtPart <<= dPdtPart + gammaMinusOne * enthalpyRHSTerm;
}

//--------------------------------------------------------------------

template<typename FieldT>
PressureMaterialDerivativePartial<FieldT>::
Builder::Builder( const Expr::Tag&        dpdtPartTag,
                  const Expr::Tag&        densityTag,
                  const Expr::Tag&        temperatureTag,
                  const Expr::Tag&        gammaTag,
                  const Expr::TagList&    specEnthalpyTags,
                  const FieldTagInfo&     enthalpyInfo,
                  const FieldTagListInfo& speciesInfo )
  : ExpressionBuilder( dpdtPartTag ),
    densityTag_            ( densityTag             ),
    temperatureTag_        ( temperatureTag         ),
    gammaTag_              ( gammaTag               ),
    specEnthalpyTags_      ( specEnthalpyTags       ),
    enthalpyInfo_          ( enthalpyInfo           ),
    speciesInfo_           ( speciesInfo            )
{}

//====================================================================

#endif // PressureMaterialDerivativePartial_h

