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
    
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,XFace>::type  SVol2XFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,YFace>::type  SVol2YFluxInterpT;
    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VolFieldT,ZFace>::type  SVol2ZFluxInterpT;
    const SVol2XFluxInterpT* sVol2XFluxInterpOp_;
    const SVol2YFluxInterpT* sVol2YFluxInterpOp_;
    const SVol2ZFluxInterpT* sVol2ZFluxInterpOp_;


    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientX, VolFieldT, VolFieldT >::type GradXT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientY, VolFieldT, VolFieldT >::type GradYT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename SpatialOps::GradientZ, VolFieldT, VolFieldT >::type GradZT;
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
      p_    = this->template create_field_request<VolFieldT>( Expr::Tag("pressure",Expr::STATE_NONE) );
      
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
      if( doX_ ){
        xInterp_            = opDB.retrieve_operator<XInterp          >();
        divX_               = opDB.retrieve_operator<DivX             >();
        sVol2XFluxInterpOp_ = opDB.retrieve_operator<SVol2XFluxInterpT>();
        dpdx_               = opDB.retrieve_operator<GradXT           >();
      }
      if( doY_ ){
        yInterp_            = opDB.retrieve_operator<YInterp          >();
        divY_               = opDB.retrieve_operator<DivY             >();
        sVol2YFluxInterpOp_ = opDB.retrieve_operator<SVol2YFluxInterpT>();
        dpdy_               = opDB.retrieve_operator<GradYT           >();
      }
      if( doZ_ ){
        zInterp_            = opDB.retrieve_operator<ZInterp          >();
        divZ_               = opDB.retrieve_operator<DivZ             >();
        sVol2ZFluxInterpOp_ = opDB.retrieve_operator<SVol2ZFluxInterpT>();
        dpdz_               = opDB.retrieve_operator<GradZT           >();
      }
    }

    void evaluate()
    {
      VolFieldT& result = this->value();
      const VolFieldT& visc = visc_->field_ref();
      const VolFieldT& dil = dil_->field_ref();
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

            result <<= result + (*divX_)( 2.0*(*xInterp_)(visc) * strainxy * (*xInterp_)(xvel_->field_ref()) )
                              + (*divY_)( 2.0*(*yInterp_)(visc) * strainyx * (*yInterp_)(yvel_->field_ref()) );
          }
          if( doZ_ ){
            const ZFace& strainzx = strainzx_->field_ref();
            const XFace& strainxz = strainxz_->field_ref();

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

            result <<= result + (*divY_)( 2.0*(*yInterp_)(visc) * strainyz * (*yInterp_)( yvel_->field_ref() ) )
                              + (*divZ_)( 2.0*(*zInterp_)(visc) * strainzy * (*zInterp_)( zvel_->field_ref() ) );
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
                                        Uintah::ProblemSpecP energyEqnSpec,
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
                                          energyEqnSpec,
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
    solnFactory.register_expression( new KineticEnergy<MyFieldT,MyFieldT,MyFieldT,MyFieldT>::Builder( kineticEnergyTag_, velTags[0], velTags[1], velTags[2] ) );

    //----------------------------------------------------------
    // temperature calculation
    typedef TemperaturePurePerfectGas<MyFieldT>::Builder SimpleTemperature;
    solnFactory.register_expression( new SimpleTemperature( temperatureTag, primVarTag_, kineticEnergyTag_ ) );

    //----------------------------------------------------------
    // viscous dissipation
    typedef ViscousDissipation<MyFieldT>::Builder ViscDissip;
    const Expr::Tag visDisTag("viscous_dissipation",Expr::STATE_NONE);
    solnFactory.register_expression( new ViscDissip( visDisTag,
                                                        velTags, viscTag, dilTag,
                                                        tags.strainxx, tags.strainyx, tags.strainzx,
                                                        tags.strainxy, tags.strainyy, tags.strainzy,
                                                        tags.strainxz, tags.strainyz, tags.strainzz ) );

    //----------------------------------------------------------

    
    setup();  // base setup stuff (register RHS, etc.)

    // attach viscous dissipation expression to the RHS as a source
    solnFactory.attach_dependency_to_expression(visDisTag, this->rhsTag_);
    
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
    icFactory.register_expression( new SimpleEnergyIC( primVarTag_, temperatureTag_, velTags_ ) );
    
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
    factory.register_expression( new ExprAlgebra<MyFieldT>::Builder( combinedVarTag,
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
