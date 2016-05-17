/**
 *  \file   TotalInternalEnergyTransportEquation.cc
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2016 The University of Utah
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
 *
 */
#include <CCA/Components/Wasatch/Transport/TotalInternalEnergyTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/KineticEnergy.h>

namespace WasatchCore {

  //====================================================================

  /**
   *  \class ArtCompPGSEnergy
   *  \author Tony Saad
   *  \date May, 2016
   *  \brief When using the PGS (Pressure Gradient Scaling) model in Artificial Compressibility to rescale the acoustic speed in compressible flows,
   this expression computes the term required in the total internal energy equation. It will be added
   as a dependency to the total internal energy RHS.
     \f[
       (1 - \frac{1}{\alpha^2}) u_j \frac{\partial p}{\partial x_j}
     \f]
   */
  template< typename FieldT >
  class ArtCompPGSEnergy
  : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS( FieldT, p_, u_, v_, w_ )
    const bool doX_, doY_, doZ_;
    const double alpha_;
    
    
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientX, SVolField, SVolField >::type GradXT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientY, SVolField, SVolField >::type GradYT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientZ, SVolField, SVolField >::type GradZT;
    const GradXT* dpdx_;
    const GradYT* dpdy_;
    const GradZT* dpdz_;

    ArtCompPGSEnergy( const Expr::Tag& pTag,
                   const Expr::TagList& velTags,
                   const double alpha)
    : Expr::Expression<FieldT>(),
    doX_( velTags[0] != Expr::Tag() ),
    doY_( velTags[1] != Expr::Tag() ),
    doZ_( velTags[2] != Expr::Tag() ),
    alpha_(alpha)
    {
      this->set_gpu_runnable( true );
      
      p_ = this->template create_field_request<FieldT>( pTag );
      
      if( doX_ ) u_ = this->template create_field_request<FieldT>( velTags[0] );
      if( doY_ ) v_ = this->template create_field_request<FieldT>( velTags[1] );
      if( doZ_ ) w_ = this->template create_field_request<FieldT>( velTags[2] );
    }
    
    
  public:
    
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag pTag_;
      const Expr::TagList velTags_;
      double alpha_;
    public:
      Builder( const Expr::Tag& resultTag,
              const Expr::Tag& pTag,
              const Expr::TagList& velTags,
              const double alpha,
              const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
      pTag_( pTag ),
      velTags_( velTags ),
      alpha_(alpha)
      {
        assert( velTags.size() == 3 );
      }
      
      Expr::ExpressionBase* build() const{
        return new ArtCompPGSEnergy<FieldT>( pTag_,velTags_, alpha_ );
      }
      
    };  /* end of Builder class */
    
    ~ArtCompPGSEnergy(){}
    
    void bind_operators( const SpatialOps::OperatorDatabase& opDB )
    {
      if(doX_) dpdx_ = opDB.retrieve_operator<GradXT>();
      if(doY_) dpdy_ = opDB.retrieve_operator<GradYT>();
      if(doZ_) dpdz_ = opDB.retrieve_operator<GradZT>();
    }

    void evaluate()
    {
      FieldT& result = this->value();
      
      const FieldT& p = p_->field_ref();
      const double a2 = alpha_*alpha_;
      if( doX_ && doY_ && doZ_ ){
        result <<= (1.0 - 1.0/a2) * (   u_->field_ref() * ( *dpdx_ )( p_->field_ref() )
                                      + v_->field_ref() * ( *dpdy_ )( p_->field_ref() )
                                      + w_->field_ref() * ( *dpdz_ )( p_->field_ref() )
                                    );
      }
      else{
        if (doX_) result <<= (1.0 - 1.0/a2) * u_->field_ref() * ( *dpdx_ )( p_->field_ref() );
        else result <<= 0.0;
        if (doY_) result <<= result + (1.0 - 1.0/a2) * v_->field_ref() * ( *dpdy_ )( p_->field_ref() );
        if (doZ_) result <<= result + (1.0 - 1.0/a2) * w_->field_ref() * ( *dpdz_ )( p_->field_ref() );
      }
    }
  };

  //============================================================================

  /**
   *  \class  TemperaturePurePerfectGas
   *  \author James C. Sutherland
   *  \date   November 26, 2015
   *
   *  \brief For a pure, calorically perfect, diatomic gas, this computes the
   *    temperature, in Kelvin, from the total specific internal energy and
   *    the specific kinetic energy.
   *
   *  The total internal energy is \f$e_T = e - \frac{\mathbf{u}\cdot\mathbf{u}}{2}\f$.
   *  For a pure, calorically ideal gas, we can write \f$c_{V} = \frac{5}{2R}\f$
   *  such that \f$e = c_{V}T=\frac{5}{2R}T\f$ or
   *  \f[
   *    T=\frac{2R}{5}\left(e_{T}-\frac{1}{2}\mathbf{u}\cdot\mathbf{u}\right).
   *  \f]
   *  See also https://en.wikipedia.org/wiki/Perfect_gas
   */
  template< typename FieldT >
  class TemperaturePurePerfectGas : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS( FieldT, totalInternalEnergy_, kineticEnergy_ )

    TemperaturePurePerfectGas( const Expr::Tag& totalInternalEnergyTag,
                               const Expr::Tag& kineticEnergyTag )
      : Expr::Expression<FieldT>()
    {
      this->set_gpu_runnable( true );

      totalInternalEnergy_ = this->template create_field_request<FieldT>( totalInternalEnergyTag );
      kineticEnergy_       = this->template create_field_request<FieldT>( kineticEnergyTag       );
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag totalInternalEnergyTag_, kineticEnergyTag_;
    public:
      /**
       *  @brief Build a TemperaturePurePerfectGas expression
       *  @param resultTag the tag for the temperature (K)
       *  @param totalInternalEnergyTag the total specific internal energy, \f$e_T\f$.
       *  @oaram kineticEnergyTag the specific kinetic energy, \f$\mathbf{u}\cdot\mathbf{u}/2\f$.
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& totalInternalEnergyTag,
               const Expr::Tag& kineticEnergyTag,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          totalInternalEnergyTag_( totalInternalEnergyTag ),
          kineticEnergyTag_      ( kineticEnergyTag       )
      {}

      Expr::ExpressionBase* build() const{
        return new TemperaturePurePerfectGas( totalInternalEnergyTag_,kineticEnergyTag_ );
      }

    };  /* end of Builder class */

    ~TemperaturePurePerfectGas(){}

    void evaluate()
    {
      const FieldT& totalInternalEnergy = totalInternalEnergy_->field_ref();
      const FieldT& kineticEnergy       = kineticEnergy_      ->field_ref();
      const double gasConstant = 8.314459848/0.028966;  // universal R = J/(mol K). Specific R* = R/Mw = J/k
      this->value() <<= 2.0/(5.0*gasConstant) * ( totalInternalEnergy - kineticEnergy ); // total internal energy = J/Kg (per unit mass)
    }

  };

  //============================================================================

  /**
   *  \class TotalInternalEnergy_PurePerfectGas_IC
   *  \author James C. Sutherland
   *  \date November, 2015
   *  \brief Computes total internal energy from the temperature and velocity for a pure, perfect diatomic gas.
   */
  template< typename FieldT >
  class TotalInternalEnergy_PurePerfectGas_IC
   : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS( FieldT, temperature_, xvel_, yvel_, zvel_ )
    const bool doX_, doY_, doZ_;

    TotalInternalEnergy_PurePerfectGas_IC( const Expr::Tag& temperatureTag,
                                           const Expr::TagList& velTags )
      : Expr::Expression<FieldT>(),
        doX_( velTags[0] != Expr::Tag() ),
        doY_( velTags[1] != Expr::Tag() ),
        doZ_( velTags[2] != Expr::Tag() )
    {
      this->set_gpu_runnable( true );

      temperature_ = this->template create_field_request<FieldT>( temperatureTag );

      if( doX_ ) xvel_ = this->template create_field_request<FieldT>( velTags[0] );
      if( doY_ ) yvel_ = this->template create_field_request<FieldT>( velTags[1] );
      if( doZ_ ) zvel_ = this->template create_field_request<FieldT>( velTags[2] );
    }


  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag temperatureTag_;
      const Expr::TagList velTags_;
    public:
      /**
       *  @brief Build a TotalInternalEnergy_PurePerfectGas_IC expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& temperatureTag,
               const Expr::TagList& velTags,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          temperatureTag_( temperatureTag ),
          velTags_( velTags )
      {
        assert( velTags.size() == 3 );
      }

      Expr::ExpressionBase* build() const{
        return new TotalInternalEnergy_PurePerfectGas_IC<FieldT>( temperatureTag_,velTags_ );
      }

    };  /* end of Builder class */

    ~TotalInternalEnergy_PurePerfectGas_IC(){}

    void evaluate()
    {
      FieldT& result = this->value();

      const FieldT& temperature = temperature_->field_ref();

      const double gasConstant = 8.314459848/0.028966;  // Universal R = J/(mol K). Specific R* = R/Mw = J/(Kg K)

      if( doX_ && doY_ && doZ_ ){
        const FieldT& xvel = xvel_->field_ref();
        const FieldT& yvel = yvel_->field_ref();
        const FieldT& zvel = zvel_->field_ref();
        result <<= 5.0*gasConstant/2.0 * temperature + 0.5 * ( xvel*xvel + yvel*yvel + zvel*zvel );
      }
      else{
        result <<= 5.0*gasConstant/2.0 * temperature;
        if( doX_ ){
          const FieldT& xvel = xvel_->field_ref();
          result <<= result + 0.5 * xvel * xvel;
        }
        if( doY_ ){
          const FieldT& yvel = yvel_->field_ref();
          result <<= result + 0.5 * yvel * yvel;
        }
        if( doZ_ ){
          const FieldT& zvel = zvel_->field_ref();
          result <<= result + 0.5 * zvel * zvel;
        }
      }
    }
  };

  //============================================================================

  /**
   *  \class ViscousDissipation
   */
  template< typename VolFieldT >
  class ViscousDissipation
   : public Expr::Expression<VolFieldT>
  {
    typedef typename SpatialOps::FaceTypes<VolFieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<VolFieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<VolFieldT>::ZFace ZFace;

    DECLARE_FIELDS( VolFieldT, xvel_, yvel_, zvel_, visc_, dil_, p_ )
    DECLARE_FIELDS( XFace, strainxx_, strainxy_, strainxz_ )
    DECLARE_FIELDS( YFace, strainyx_, strainyy_, strainyz_ )
    DECLARE_FIELDS( ZFace, strainzx_, strainzy_, strainzz_ )

    const bool doX_, doY_, doZ_;

    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FX  XInterp;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FY  YInterp;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FZ  ZInterp;
    XInterp* xInterp_;
    YInterp* yInterp_;
    ZInterp* zInterp_;

    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivX  DivX;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivY  DivY;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivZ  DivZ;
    DivX* divX_;
    DivY* divY_;
    DivZ* divZ_;
    
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,XFace>::type  SVol2XFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,YFace>::type  SVol2YFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,ZFace>::type  SVol2ZFluxInterpT;
    const SVol2XFluxInterpT* sVol2XFluxInterpOp_;
    const SVol2YFluxInterpT* sVol2YFluxInterpOp_;
    const SVol2ZFluxInterpT* sVol2ZFluxInterpOp_;


    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientX, SVolField, SVolField >::type GradXT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientY, SVolField, SVolField >::type GradYT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientZ, SVolField, SVolField >::type GradZT;
    const GradXT* dpdx_;
    const GradYT* dpdy_;
    const GradZT* dpdz_;

    ViscousDissipation( const Expr::Tag& xvelTag, const Expr::Tag& yvelTag, const Expr::Tag& zvelTag,
                        const Expr::Tag viscTag,
                        const Expr::Tag& dilataionTag,
                        const Expr::Tag& strainxxTag, const Expr::Tag& strainyxTag, const Expr::Tag& strainzxTag,
                        const Expr::Tag& strainxyTag, const Expr::Tag& strainyyTag, const Expr::Tag& strainzyTag,
                        const Expr::Tag& strainxzTag, const Expr::Tag& strainyzTag, const Expr::Tag& strainzzTag )
      : Expr::Expression<VolFieldT>(),
        doX_( xvelTag != Expr::Tag() ),
        doY_( yvelTag != Expr::Tag() ),
        doZ_( zvelTag != Expr::Tag() )
    {
      this->set_gpu_runnable( true );

      visc_ = this->template create_field_request<VolFieldT>( viscTag      );
      dil_  = this->template create_field_request<VolFieldT>( dilataionTag );
      p_  = this->template create_field_request<VolFieldT>( Expr::Tag("pressure",Expr::STATE_NONE) );
      
      if( doX_ ){
        xvel_     = this->template create_field_request<VolFieldT>( xvelTag );
        strainxx_ = this->template create_field_request<XFace>( strainxxTag );
        if( doY_ ) strainyx_ = this->template create_field_request<YFace>( strainyxTag );
        if( doZ_ ) strainzx_ = this->template create_field_request<ZFace>( strainzxTag );
      }
      if( doY_ ){
        yvel_     = this->template create_field_request<VolFieldT>( yvelTag );
        strainyy_ = this->template create_field_request<YFace>( strainyyTag );
        if( doX_ ) strainxy_ = this->template create_field_request<XFace>( strainxyTag );
        if( doZ_ ) strainzy_ = this->template create_field_request<ZFace>( strainzyTag );
      }
      if( doZ_ ){
        zvel_     = this->template create_field_request<VolFieldT>( zvelTag );
        strainzz_ = this->template create_field_request<ZFace>( strainzzTag );
        if( doX_ ) strainxz_ = this->template create_field_request<XFace>( strainxzTag );
        if( doY_ ) strainyz_ = this->template create_field_request<YFace>( strainyzTag );
      }
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::TagList velTags_;
      const Expr::Tag viscTag_, dilTag_;
      const Expr::Tag strainxxTag_, strainyxTag_, strainzxTag_;
      const Expr::Tag strainxyTag_, strainyyTag_, strainzyTag_;
      const Expr::Tag strainxzTag_, strainyzTag_, strainzzTag_;
    public:
      /**
       *  @brief Build a ViscousDissipation expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::TagList& velTags,
               const Expr::Tag& viscTag,
               const Expr::Tag& dilataionTag,
               const Expr::Tag& strainxxTag, const Expr::Tag& strainyxTag, const Expr::Tag& strainzxTag, // X-momentum
               const Expr::Tag& strainxyTag, const Expr::Tag& strainyyTag, const Expr::Tag& strainzyTag, // Y-momentum
               const Expr::Tag& strainxzTag, const Expr::Tag& strainyzTag, const Expr::Tag& strainzzTag, // Z-momentum
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          velTags_( velTags ),
          viscTag_( viscTag ),
          dilTag_ ( dilataionTag ),
          strainxxTag_( strainxxTag ), strainyxTag_( strainyxTag ), strainzxTag_( strainzxTag ),
          strainxyTag_( strainxyTag ), strainyyTag_( strainyyTag ), strainzyTag_( strainzyTag ),
          strainxzTag_( strainxzTag ), strainyzTag_( strainyzTag ), strainzzTag_( strainzzTag )
      {
        assert( velTags.size() == 3 );
      }

      Expr::ExpressionBase* build() const{
        return new ViscousDissipation<VolFieldT>( velTags_[0], velTags_[1], velTags_[2],
                                                  viscTag_, dilTag_,
                                                  strainxxTag_, strainyxTag_, strainzxTag_,
                                                  strainxyTag_, strainyyTag_, strainzyTag_,
                                                  strainxzTag_, strainyzTag_, strainzzTag_ );
      }

    };  /* end of Builder class */

    ~ViscousDissipation(){}

    void bind_operators( const SpatialOps::OperatorDatabase& opDB )
    {
      if( doX_ ) xInterp_ = opDB.retrieve_operator<XInterp>();
      if( doY_ ) yInterp_ = opDB.retrieve_operator<YInterp>();
      if( doZ_ ) zInterp_ = opDB.retrieve_operator<ZInterp>();

      if( doX_ ) divX_    = opDB.retrieve_operator<DivX   >();
      if( doY_ ) divY_    = opDB.retrieve_operator<DivY   >();
      if( doZ_ ) divZ_    = opDB.retrieve_operator<DivZ   >();
      
      if( doX_ ) sVol2XFluxInterpOp_ = opDB.retrieve_operator<SVol2XFluxInterpT>();
      if( doY_ ) sVol2YFluxInterpOp_ = opDB.retrieve_operator<SVol2YFluxInterpT>();
      if( doZ_ ) sVol2ZFluxInterpOp_ = opDB.retrieve_operator<SVol2ZFluxInterpT>();
      
      if(doX_) dpdx_ = opDB.retrieve_operator<GradXT>();
      if(doY_) dpdy_ = opDB.retrieve_operator<GradYT>();
      if(doZ_) dpdz_ = opDB.retrieve_operator<GradZT>();
    }

    void evaluate()
    {
      VolFieldT& result = this->value();
      const VolFieldT& visc = visc_->field_ref();
      const VolFieldT& dil = dil_->field_ref();
      const double alpha = 10.0;
      const double gamma = 1.4;
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

        result <<= (*divX_)( 2.0*(*xInterp_)(visc) * (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil) + strainxy + strainxz) * (*xInterp_)( xvel ) )
                 + (*divY_)( 2.0*(*yInterp_)(visc) * (strainyx + strainyy - 1.0/3.0*(*sVol2YFluxInterpOp_)(dil) + strainyz) * (*yInterp_)( yvel ) )
                 + (*divZ_)( 2.0*(*zInterp_)(visc) * (strainzx + strainzy + strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*zInterp_)( zvel ) );
      }
      else{
        result <<= 0.0; // accumulate in as needed
        if( doX_ ){ // 1D x or 2D xy or 2D xz
          const XFace& strainxx = strainxx_->field_ref();

          result <<= (*divX_)( 2.0*(*xInterp_)(visc) * (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil)) * (*xInterp_)(xvel_->field_ref()) );
          
          if( doY_ ){
            const YFace& strainyx = strainyx_->field_ref();
            const XFace& strainxy = strainxy_->field_ref();
            const YFace& strainyy = strainyy_->field_ref();

            result <<= result + (*divX_)( 2.0*(*xInterp_)(visc) * strainxy * (*xInterp_)(xvel_->field_ref()) )
                              + (*divY_)( 2.0*(*yInterp_)(visc) * strainyx * (*yInterp_)(yvel_->field_ref()) );
          }
          if( doZ_ ){
            const ZFace& strainzx = strainzx_->field_ref();
            const XFace& strainxz = strainxz_->field_ref();
            const ZFace& strainzz = strainzz_->field_ref();

            result <<= result + (*divX_)( 2.0*(*xInterp_)(visc) * strainxz * (*xInterp_)(xvel_->field_ref()) )
                              + (*divZ_)( 2.0*(*zInterp_)(visc) * strainzx * (*zInterp_)(zvel_->field_ref()) );
          }
        } // doX_
        else if( doY_ ){ // 1D y or 2D yz
          const YFace& strainyy = strainyy_->field_ref();

          result <<= result + (*divY_)( 2.0*(*yInterp_)(visc) * (strainyy- 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*yInterp_)( yvel_->field_ref() ) );

          if( doZ_ ){
            const ZFace& strainzy = strainzy_->field_ref();
            const YFace& strainyz = strainyz_->field_ref();
            const ZFace& strainzz = strainzz_->field_ref();

            result <<= result + (*divY_)( 2.0*(*yInterp_)(visc) * strainyz * (*yInterp_)( yvel_->field_ref() ) )
                              + (*divZ_)( 2.0*(*zInterp_)(visc) * strainzy * (*zInterp_)(zvel_->field_ref()) );
          }
        }
        else if( doZ_ ){ // 1D z
          const ZFace& strainzz = strainzz_->field_ref();
          result <<= result + (*divZ_)( 2.0*(*zInterp_)(visc) * (strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*zInterp_)( zvel_->field_ref() ) );
        }
      }
    }
  };
  
  //============================================================================
  
  /**
   *  \class ArtCompASREnergy
   *  \author Tony Saad
   *  \date May, 2016
   *  \brief When using the ASR (Acoustic Stiffness Reduction) Artificial Compressibility to rescale the acoustic speed in compressible flows,
   this expression computes the term required in the total internal energy equation. It will be added
   as a dependency to the total internal energy RHS.
   \f[
     (1 - \frac{1}{\alpha^2}) \frac{\gamma p}{\gamma - 1} \frac{\partial u_j}{\partial x_j} - (1 - \frac{1}{\alpha^2})(\tau_{ij}\frac{\partial u_i}{\partial x_j} - \frac{\partial q_j}{\partial x_j} )
   \f]
   where
   \f[
    q_j = - k \frac{\partial T}{\partial x_j}
   \f]
   */
  template< typename VolFieldT >
  class ArtCompASREnergy
  : public Expr::Expression<VolFieldT>
  {
    typedef typename SpatialOps::FaceTypes<VolFieldT>::XFace XFace;
    typedef typename SpatialOps::FaceTypes<VolFieldT>::YFace YFace;
    typedef typename SpatialOps::FaceTypes<VolFieldT>::ZFace ZFace;
    
    DECLARE_FIELDS( VolFieldT, xvel_, yvel_, zvel_, visc_, dil_, p_ )
    DECLARE_FIELDS( XFace, strainxx_, strainxy_, strainxz_, diffFluxX_ )
    DECLARE_FIELDS( YFace, strainyx_, strainyy_, strainyz_, diffFluxY_ )
    DECLARE_FIELDS( ZFace, strainzx_, strainzy_, strainzz_, diffFluxZ_ )
    
    const bool doX_, doY_, doZ_;
    const double alpha_;
    
    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FX  XInterp;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FY  YInterp;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::InterpC2FZ  ZInterp;
    XInterp* xInterp_;
    YInterp* yInterp_;
    ZInterp* zInterp_;
    
    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivX  DivX;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivY  DivY;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::DivZ  DivZ;
    DivX* divX_;
    DivY* divY_;
    DivZ* divZ_;
    
    typedef typename SpatialOps::BasicOpTypes<SVolField>::GradX  DDXT;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::GradY  DDYT;
    typedef typename SpatialOps::BasicOpTypes<SVolField>::GradZ  DDZT;
    DDXT* ddx_;
    DDYT* ddy_;
    DDZT* ddz_;
    
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,XFace>::type  SVol2XFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,YFace>::type  SVol2YFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,ZFace>::type  SVol2ZFluxInterpT;
    const SVol2XFluxInterpT* sVol2XFluxInterpOp_;
    const SVol2YFluxInterpT* sVol2YFluxInterpOp_;
    const SVol2ZFluxInterpT* sVol2ZFluxInterpOp_;

    
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, XFace,SVolField>::type  XFlux2SVolInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, YFace,SVolField>::type  YFlux2SVolInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, ZFace,SVolField>::type  ZFlux2SVolInterpT;
    const XFlux2SVolInterpT* xFluxInterpOp_;
    const YFlux2SVolInterpT* yFluxInterpOp_;
    const ZFlux2SVolInterpT* zFluxInterpOp_;

    
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientX, SVolField, SVolField >::type GradXT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientY, SVolField, SVolField >::type GradYT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientZ, SVolField, SVolField >::type GradZT;
    const GradXT* dpdx_;
    const GradYT* dpdy_;
    const GradZT* dpdz_;
    
    ArtCompASREnergy(  const Expr::Tag& xvelTag, const Expr::Tag& yvelTag, const Expr::Tag& zvelTag,
                       const Expr::Tag& diffFluxXTag, const Expr::Tag& diffFluxYTag, const Expr::Tag& diffFluxZTag,
                       const Expr::Tag viscTag,
                       const Expr::Tag& dilataionTag,
                       const Expr::Tag& pressureTag,
                       const Expr::Tag& strainxxTag, const Expr::Tag& strainyxTag, const Expr::Tag& strainzxTag,
                       const Expr::Tag& strainxyTag, const Expr::Tag& strainyyTag, const Expr::Tag& strainzyTag,
                       const Expr::Tag& strainxzTag, const Expr::Tag& strainyzTag, const Expr::Tag& strainzzTag,
                       const double alpha)
    : Expr::Expression<VolFieldT>(),
      doX_   ( xvelTag != Expr::Tag() ),
      doY_   ( yvelTag != Expr::Tag() ),
      doZ_   ( zvelTag != Expr::Tag() ),
      alpha_ ( alpha                  )
    {
      this->set_gpu_runnable( true );
      
      visc_ = this->template create_field_request<VolFieldT>( viscTag      );
      dil_  = this->template create_field_request<VolFieldT>( dilataionTag );
      p_  = this->template create_field_request<VolFieldT>( pressureTag );
      
      if( doX_ ){
        xvel_     = this->template create_field_request<VolFieldT>( xvelTag );
        strainxx_ = this->template create_field_request<XFace>( strainxxTag );
        diffFluxX_ = this->template create_field_request<XFace> (diffFluxXTag);
        if( doY_ ) strainyx_ = this->template create_field_request<YFace>( strainyxTag );
        if( doZ_ ) strainzx_ = this->template create_field_request<ZFace>( strainzxTag );
      }
      if( doY_ ){
        yvel_     = this->template create_field_request<VolFieldT>( yvelTag );
        strainyy_ = this->template create_field_request<YFace>( strainyyTag );
        diffFluxY_ = this->template create_field_request<YFace> (diffFluxYTag);
        if( doX_ ) strainxy_ = this->template create_field_request<XFace>( strainxyTag );
        if( doZ_ ) strainzy_ = this->template create_field_request<ZFace>( strainzyTag );
      }
      if( doZ_ ){
        zvel_     = this->template create_field_request<VolFieldT>( zvelTag );
        strainzz_ = this->template create_field_request<ZFace>( strainzzTag );
        if( doX_ ) strainxz_ = this->template create_field_request<XFace>( strainxzTag );
        if( doY_ ) strainyz_ = this->template create_field_request<YFace>( strainyzTag );
        diffFluxZ_ = this->template create_field_request<ZFace> (diffFluxZTag);
      }
    }
    
  public:
    
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::TagList velTags_, diffFluxTags_;
      const Expr::Tag viscTag_, dilTag_, pressureTag_;
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
              const double alpha,
              const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        velTags_     ( velTags ),
        diffFluxTags_( diffFluxTags),
        viscTag_     ( viscTag ),
        dilTag_      ( dilataionTag ),
        pressureTag_ (pressureTag),
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
                                                      strainxzTag_, strainyzTag_, strainzzTag_, alpha_ );
      }
      
    };  /* end of Builder class */
    
    ~ArtCompASREnergy(){}
    
    void bind_operators( const SpatialOps::OperatorDatabase& opDB )
    {
      if( doX_ ) xInterp_ = opDB.retrieve_operator<XInterp>();
      if( doY_ ) yInterp_ = opDB.retrieve_operator<YInterp>();
      if( doZ_ ) zInterp_ = opDB.retrieve_operator<ZInterp>();
      
      if( doX_ ) divX_    = opDB.retrieve_operator<DivX   >();
      if( doY_ ) divY_    = opDB.retrieve_operator<DivY   >();
      if( doZ_ ) divZ_    = opDB.retrieve_operator<DivZ   >();
      
      if( doX_ ) sVol2XFluxInterpOp_ = opDB.retrieve_operator<SVol2XFluxInterpT>();
      if( doY_ ) sVol2YFluxInterpOp_ = opDB.retrieve_operator<SVol2YFluxInterpT>();
      if( doZ_ ) sVol2ZFluxInterpOp_ = opDB.retrieve_operator<SVol2ZFluxInterpT>();
      
      if(doX_) dpdx_ = opDB.retrieve_operator<GradXT>();
      if(doY_) dpdy_ = opDB.retrieve_operator<GradYT>();
      if(doZ_) dpdz_ = opDB.retrieve_operator<GradZT>();
      
      if(doX_) ddx_ = opDB.retrieve_operator<DDXT>();
      if(doY_) ddy_ = opDB.retrieve_operator<DDYT>();
      if(doZ_) ddz_ = opDB.retrieve_operator<DDZT>();

      if(doX_) xFluxInterpOp_ = opDB.retrieve_operator<XFlux2SVolInterpT>();
      if(doY_) yFluxInterpOp_ = opDB.retrieve_operator<YFlux2SVolInterpT>();
      if(doZ_) zFluxInterpOp_ = opDB.retrieve_operator<ZFlux2SVolInterpT>();
    }
    
    void evaluate()
    {
      VolFieldT& result = this->value();
      const VolFieldT& visc = visc_->field_ref();
      const VolFieldT& dil = dil_->field_ref();
      const VolFieldT& p   = p_->field_ref();
      
      const double gamma = 1.4;
      const double scale = (1.0 - 1.0/alpha_/alpha_);
      
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
        
        result <<=   scale * gamma / ( gamma - 1.0) * p * dil
                   - scale * 2.0 * visc * (  (*xFluxInterpOp_) ( (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil)) * (*ddx_)(xvel) + strainxy * (*ddx_)(yvel) + strainxz * (*ddx_)(zvel) )
                                           + (*yFluxInterpOp_) ( (strainyx * (*ddy_)(xvel) + strainyy - 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*ddy_)(yvel) + strainyz * (*ddy_)(zvel) )
                                           + (*zFluxInterpOp_) ( (strainzx * (*ddz_)(xvel) + strainzy * (*ddz_)(yvel) + strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*ddz_)(zvel) )
                                           - (*divX_)(diffFluxX) - (*divY_)(diffFluxY) - (*divZ_)(diffFluxZ)
                                         );
      }
      else{
        result <<=   scale * gamma / ( gamma - 1.0) * p * dil;
        if( doX_ ){ // 1D x or 2D xy or 2D xz
          const XFace& strainxx = strainxx_->field_ref();
          const VolFieldT& xvel = xvel_->field_ref();
          const XFace& diffFluxX = diffFluxX_->field_ref();

          result <<= result
                    - scale * 2.0 * visc * (  (*xFluxInterpOp_) ( (strainxx - 1.0/3.0*(*sVol2XFluxInterpOp_)(dil)) * (*ddx_)(xvel)  )
                                ) + scale * (*divX_)(diffFluxX);
          
          if( doY_ ){
            const VolFieldT& yvel = yvel_->field_ref();
            const YFace& diffFluxY = diffFluxY_->field_ref();
            const YFace& strainyx = strainyx_->field_ref();
            const XFace& strainxy = strainxy_->field_ref();
            const YFace& strainyy = strainyy_->field_ref();
            
            result <<= result
                      - scale * 2.0 * visc * (  (*xFluxInterpOp_) ( strainxy * (*ddx_)(yvel)  )
                                              + (*yFluxInterpOp_) ( (strainyx * (*ddy_)(xvel) + strainyy - 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*ddy_)(yvel) )
                                              ) + scale * (*divY_)(diffFluxY);

          }
          if( doZ_ ){
            const ZFace& strainzx = strainzx_->field_ref();
            const XFace& strainxz = strainxz_->field_ref();
            const ZFace& strainzz = strainzz_->field_ref();
            
            result <<= result + (*divX_)( 2.0*(*xInterp_)(visc) * strainxz * (*xInterp_)(xvel_->field_ref()) )
            + (*divZ_)( 2.0*(*zInterp_)(visc) * strainzx * (*zInterp_)(zvel_->field_ref()) );
          }
        } // doX_
        else if( doY_ ){ // 1D y or 2D yz
          const YFace& strainyy = strainyy_->field_ref();
          
          result <<= result + (*divY_)( 2.0*(*yInterp_)(visc) * (strainyy- 1.0/3.0*(*sVol2YFluxInterpOp_)(dil)) * (*yInterp_)( yvel_->field_ref() ) );
          
          if( doZ_ ){
            const ZFace& strainzy = strainzy_->field_ref();
            const YFace& strainyz = strainyz_->field_ref();
            const ZFace& strainzz = strainzz_->field_ref();
            
            result <<= result + (*divY_)( 2.0*(*yInterp_)(visc) * strainyz * (*yInterp_)( yvel_->field_ref() ) )
            + (*divZ_)( 2.0*(*zInterp_)(visc) * strainzy * (*zInterp_)(zvel_->field_ref()) );
          }
        }
        else if( doZ_ ){ // 1D z
          const ZFace& strainzz = strainzz_->field_ref();
          result <<= result + (*divZ_)( 2.0*(*zInterp_)(visc) * (strainzz - 1.0/3.0*(*sVol2ZFluxInterpOp_)(dil)) * (*zInterp_)( zvel_->field_ref() ) );
        }
      }
    }
  };
  
  //============================================================================

  TotalInternalEnergyTransportEquation::
  TotalInternalEnergyTransportEquation( const std::string e0Name,
                                        Uintah::ProblemSpecP momentumSpec,
                                        GraphCategories& gc,
                                        const Expr::Tag& densityTag,
                                        const Expr::Tag& temperatureTag,
                                        const Expr::Tag& pressureTag,
                                        const Expr::TagList& velTags,
                                        const Expr::TagList& bodyForceTags,
                                        const Expr::Tag& viscTag,
                                        const Expr::Tag& dilTag,
                                        const TurbulenceParameters& turbulenceParams )
    : ScalarTransportEquation<SVolField>( e0Name,
                                          momentumSpec->findBlock("EnergyEquation"),
                                          gc, densityTag,
                                          false, /* variable density */
                                          turbulenceParams,
                                          false /* don't call setup */ ),
      kineticEnergyTag_( "kinetic energy", Expr::STATE_NONE ),
      temperatureTag_( temperatureTag ),
      pressureTag_( TagNames::self().pressure ),
      velTags_( velTags ),
      bodyForceTags_(bodyForceTags)
  {
    const TagNames& tags = TagNames::self();
    Expr::ExpressionFactory& solnFactory = *gc[ADVANCE_SOLUTION]->exprFactory;

    //----------------------------------------------------------
    // kinetic energy
    solnFactory.register_expression( scinew KineticEnergy<MyFieldT,MyFieldT,MyFieldT,MyFieldT>::Builder( kineticEnergyTag_, velTags[0], velTags[1], velTags[2] ) );

    //----------------------------------------------------------
    // temperature calculation
    typedef TemperaturePurePerfectGas<MyFieldT>::Builder SimpleTemperature;
    solnFactory.register_expression( scinew SimpleTemperature( temperatureTag, primVarTag_, kineticEnergyTag_ ) );

    //----------------------------------------------------------
    // viscous dissipation
    typedef ViscousDissipation<MyFieldT>::Builder ViscDissip;
    const Expr::Tag visDisTag("viscous_dissipation",Expr::STATE_NONE);
    solnFactory.register_expression( scinew ViscDissip( visDisTag,
                                                        velTags, viscTag, dilTag,
                                                        tags.strainxx, tags.strainyx, tags.strainzx,
                                                        tags.strainxy, tags.strainyy, tags.strainzy,
                                                        tags.strainxz, tags.strainyz, tags.strainzz ) );

    //----------------------------------------------------------

    
    setup();  // base setup stuff (register RHS, etc.)

    // attach viscous dissipation expression to the RHS as a source
    solnFactory.attach_dependency_to_expression(visDisTag, this->rhsTag_);
    
    //----------------------------------------------------------
    // register artifical compressibility if any
    typedef ArtCompPGSEnergy<MyFieldT>::Builder ACEnergy;
    if (momentumSpec->findBlock("ArtificialCompressibility")) {
      Uintah::ProblemSpecP acSpec = momentumSpec->findBlock("ArtificialCompressibility");
      std::string model;
      acSpec->getAttribute("model",model);
      double alpha = 10.0;
      acSpec->getAttribute("coef",alpha);
      if (model == "PGS") {
        const Expr::Tag acEnergyTag("AC_Energy",Expr::STATE_NONE);
        solnFactory.register_expression( scinew ACEnergy( acEnergyTag,
                                                         pressureTag, velTags, alpha ) );
        
        // attach PGS energy preconditioning expression to the RHS as a source
        solnFactory.attach_dependency_to_expression(acEnergyTag, this->rhsTag_);
      } else if (model == "ASR") {
        // ASR
        typedef ArtCompASREnergy<MyFieldT>::Builder ASR;
        const Expr::Tag asrTag("ASR_Energy",Expr::STATE_NONE);
        
        const Expr::TagList diffFluxTags = tag_list(Expr::Tag("T_diffVelocity_X", Expr::STATE_NONE),
                                                    Expr::Tag("T_diffVelocity_Y", Expr::STATE_NONE),
                                                    Expr::Tag("T_diffVelocity_Z", Expr::STATE_NONE));
        
        solnFactory.register_expression( scinew ASR( asrTag,
                                                     velTags, diffFluxTags, viscTag, dilTag, pressureTag,
                                                     tags.strainxx, tags.strainyx, tags.strainzx,
                                                     tags.strainxy, tags.strainyy, tags.strainzy,
                                                     tags.strainxz, tags.strainyz, tags.strainzz, alpha ) );
        solnFactory.attach_dependency_to_expression(asrTag,    this->rhsTag_);
      }
    }

    //----------------------------------------------------------
    // body force tags
    if( bodyForceTags.size() != 3 ){
      std::ostringstream msg;
      msg << "ERROR: You must specify three tags for the body force." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );

    }
    typedef ExprAlgebra<SVolField> ExprAlgbr;
    const Expr::Tag rhoTag(this->densityTag_.name(), Expr::STATE_DYNAMIC);
    if( bodyForceTags_[0] != Expr::Tag() ){
      const Expr::Tag xBodyForceWorkTag("xBodyForceWork", Expr::STATE_NONE);
      const Expr::TagList theTagList( tag_list( rhoTag, velTags_[0], bodyForceTags_[0] ) );
      solnFactory.register_expression( new ExprAlgbr::Builder( xBodyForceWorkTag,
                                                               theTagList,
                                                                ExprAlgbr::PRODUCT ) );
      solnFactory.attach_dependency_to_expression(xBodyForceWorkTag, this->rhsTag_);
    }
    
    if( bodyForceTags_[1] != Expr::Tag() ){
      const Expr::Tag yBodyForceWorkTag("yBodyForceWork", Expr::STATE_NONE);
      const Expr::TagList theTagList( tag_list( rhoTag, velTags_[1], bodyForceTags_[1] ) );
      solnFactory.register_expression( new ExprAlgbr::Builder( yBodyForceWorkTag,
                                                              theTagList,
                                                              ExprAlgbr::PRODUCT ) );
      solnFactory.attach_dependency_to_expression(yBodyForceWorkTag, this->rhsTag_);
    }

    if( bodyForceTags_[2] != Expr::Tag() ){
      const Expr::Tag zBodyForceWorkTag("zBodyForceWork", Expr::STATE_NONE);
      const Expr::TagList theTagList( tag_list( rhoTag, velTags_[2], bodyForceTags_[2] ) );
      solnFactory.register_expression( new ExprAlgbr::Builder( zBodyForceWorkTag,
                                                              theTagList,
                                                              ExprAlgbr::PRODUCT ) );
      solnFactory.attach_dependency_to_expression(zBodyForceWorkTag, this->rhsTag_);
    }

  }

  //---------------------------------------------------------------------------

  TotalInternalEnergyTransportEquation::~TotalInternalEnergyTransportEquation()
  {}

  //---------------------------------------------------------------------------

  Expr::ExpressionID
  TotalInternalEnergyTransportEquation::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    
    // initial condition for total internal energy
    typedef TotalInternalEnergy_PurePerfectGas_IC<SVolField>::Builder SimpleEnergyIC;
    icFactory.register_expression( scinew SimpleEnergyIC( primVarTag_, temperatureTag_, velTags_ ) );
    
    // register expression to calculate the momentum initial condition from the initial conditions on
    // velocity and density in the cases that we are initializing velocity in the input file
    typedef ExprAlgebra<SVolField> ExprAlgbr;
    const Expr::Tag rhoTag(this->densityTag_.name(), Expr::STATE_NONE);
    const Expr::TagList theTagList( tag_list( primVarTag_, rhoTag ) );
    return icFactory.register_expression( new ExprAlgbr::Builder( this->initial_condition_tag(),
                                                          theTagList,
                                                          ExprAlgbr::PRODUCT ) );

  }

  //---------------------------------------------------------------------------

  void
  TotalInternalEnergyTransportEquation::
  setup_diffusive_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    
    for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
        diffFluxParams != 0;
        diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") )
    {
      setup_diffusive_velocity_expression<MyFieldT>( diffFluxParams,
                                                    temperatureTag_,
                                                    turbDiffTag_,
                                                    factory,
                                                    info );
      
    }
  }

  //------------------------------------------------------------------

  void
  TotalInternalEnergyTransportEquation::
  setup_convective_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    const Expr::Tag solnVarTag( solnVarName_, Expr::STATE_DYNAMIC );

    // create an expression for (\rho e_0 + p), which is what we advect.
    const Expr::Tag combinedVarTag( solnVarName_ + "_and_" + pressureTag_.name(), Expr::STATE_NONE );
    factory.register_expression( scinew ExprAlgebra<MyFieldT>::Builder( combinedVarTag,
                                                                        tag_list(solnVarTag,pressureTag_),
                                                                        ExprAlgebra<MyFieldT>::SUM ) );

    for( Uintah::ProblemSpecP convFluxParams=params_->findBlock("ConvectiveFlux");
        convFluxParams != 0;
        convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") )
    {
      setup_convective_flux_expression<MyFieldT>( convFluxParams, combinedVarTag, factory, info );
    }
  }


  //---------------------------------------------------------------------------

  void
  TotalInternalEnergyTransportEquation::
  setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
  {
    // see if we have radiation - if so, plug it in
    if( params_->findBlock("RadiativeSourceTerm") ){
      info[SOURCE_TERM] = parse_nametag( params_->findBlock("RadiativeSourceTerm")->findBlock("NameTag") );
    }

    // populate any additional source terms pushed on this equation
    ScalarTransportEquation<SVolField>::setup_source_terms( info, srcTags );
  }

  //---------------------------------------------------------------------------

} /* namespace WasatchCore */
