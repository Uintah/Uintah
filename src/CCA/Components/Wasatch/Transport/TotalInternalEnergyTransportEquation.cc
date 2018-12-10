/**
 *  \file   TotalInternalEnergyTransportEquation.cc
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionsOneSided.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>
#ifdef HAVE_POKITT
#include <pokitt/SpeciesN.h>
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/thermo/InternalEnergy.h>
#include <pokitt/thermo/Temperature.h>
#include <pokitt/thermo/Enthalpy.h>
#include <pokitt/transport/HeatFlux.h>
#include <pokitt/transport/ThermalCondMix.h>
#include <pokitt/thermo/HeatCapacity_Cp.h>
#include <pokitt/thermo/HeatCapacity_Cv.h>
#endif

namespace WasatchCore {

  template< typename FaceT, typename GradT >
  struct EnergyBoundaryTyper
  {
    typedef SpatialOps::SVolField CellT;
    typedef SpatialOps::Divergence DivT;

    typedef typename SpatialOps::OperatorTypeBuilder<GradT, CellT, CellT>::type CellNeumannT;
    typedef typename SpatialOps::OperatorTypeBuilder<DivT, FaceT, CellT>::type FaceNeumannT;

    typedef typename SpatialOps::NeboBoundaryConditionBuilder<CellNeumannT> CellNeumannBCOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<FaceNeumannT> FaceNeumannBCOpT;

    typedef typename ConstantBCNew<CellT,CellNeumannBCOpT>::Builder ConstantCellNeumannBC;
    typedef typename ConstantBCNew<FaceT,FaceNeumannBCOpT>::Builder ConstantFaceNeumannBC;
  };

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
    DECLARE_FIELDS( FieldT, totalInternalEnergy_, kineticEnergy_, mmw_ )

    TemperaturePurePerfectGas( const Expr::Tag& totalInternalEnergyTag,
                               const Expr::Tag& kineticEnergyTag,
                               const Expr::Tag& mixMWTag )
      : Expr::Expression<FieldT>()
    {
      this->set_gpu_runnable( true );

      totalInternalEnergy_ = this->template create_field_request<FieldT>( totalInternalEnergyTag );
      kineticEnergy_       = this->template create_field_request<FieldT>( kineticEnergyTag       );
      mmw_                 = this->template create_field_request<FieldT>( mixMWTag               );
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag totalInternalEnergyTag_, kineticEnergyTag_, mmwTag_;
    public:
      /**
       *  @brief Build a TemperaturePurePerfectGas expression
       *  @param resultTag the tag for the temperature (K)
       *  @param totalInternalEnergyTag the total specific internal energy, \f$e_T\f$.
       *  @oaram kineticEnergyTag the specific kinetic energy, \f$\mathbf{u}\cdot\mathbf{u}/2\f$.
       *  @oaram mixMWTag the mixture molecular weight, (kg/kmol or g/mol).
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& totalInternalEnergyTag,
               const Expr::Tag& kineticEnergyTag,
               const Expr::Tag& mixMWTag,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          totalInternalEnergyTag_( totalInternalEnergyTag ),
          kineticEnergyTag_      ( kineticEnergyTag       ),
          mmwTag_                ( mixMWTag               )
      {}

      Expr::ExpressionBase* build() const{
        return new TemperaturePurePerfectGas( totalInternalEnergyTag_,kineticEnergyTag_,mmwTag_ );
      }

    };  /* end of Builder class */

    ~TemperaturePurePerfectGas(){}

    void evaluate()
    {
      const FieldT& totalInternalEnergy = totalInternalEnergy_->field_ref();
      const FieldT& kineticEnergy       = kineticEnergy_      ->field_ref();
      const FieldT& mixMW               = mmw_                ->field_ref();
      const double gasConstant = 8314.459848;  // universal R = J/(kmol K).
      this->value() <<= 2.0/5.0 * mixMW/gasConstant * ( totalInternalEnergy - kineticEnergy ); // total internal energy = J/kg (per unit mass)
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
    DECLARE_FIELDS( FieldT, temperature_, mmw_, xvel_, yvel_, zvel_ )
    const bool doX_, doY_, doZ_;

    TotalInternalEnergy_PurePerfectGas_IC( const Expr::Tag& temperatureTag,
                                           const Expr::TagList& velTags,
                                           const Expr::Tag& mixMWTag )
      : Expr::Expression<FieldT>(),
        doX_( velTags[0] != Expr::Tag() ),
        doY_( velTags[1] != Expr::Tag() ),
        doZ_( velTags[2] != Expr::Tag() )
    {
      this->set_gpu_runnable( true );

      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      mmw_         = this->template create_field_request<FieldT>( mixMWTag       );

      if( doX_ ) xvel_ = this->template create_field_request<FieldT>( velTags[0] );
      if( doY_ ) yvel_ = this->template create_field_request<FieldT>( velTags[1] );
      if( doZ_ ) zvel_ = this->template create_field_request<FieldT>( velTags[2] );
    }


  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag temperatureTag_, mmwTag_;
      const Expr::TagList velTags_;
    public:
      /**
       *  @brief Build a TotalInternalEnergy_PurePerfectGas_IC expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& temperatureTag,
               const Expr::TagList& velTags,
               const Expr::Tag& mixMWTag,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          temperatureTag_( temperatureTag ),
          mmwTag_( mixMWTag ),
          velTags_( velTags )
      {
        assert( velTags.size() == 3 );
      }

      Expr::ExpressionBase* build() const{
        return new TotalInternalEnergy_PurePerfectGas_IC<FieldT>( temperatureTag_,velTags_,mmwTag_ );
      }

    };  /* end of Builder class */

    ~TotalInternalEnergy_PurePerfectGas_IC(){}

    void evaluate()
    {
      FieldT& result = this->value();

      const FieldT& temperature = temperature_->field_ref();
      const FieldT& mmw         = mmw_        ->field_ref();

      const double gasConstant = 8314.459848;  // Universal R = J/(kmol K)

      if( doX_ && doY_ && doZ_ ){
        const FieldT& xvel = xvel_->field_ref();
        const FieldT& yvel = yvel_->field_ref();
        const FieldT& zvel = zvel_->field_ref();
        result <<= 5.0/2.0 * gasConstant/mmw * temperature + 0.5 * ( xvel*xvel + yvel*yvel + zvel*zvel );
      }
      else{
        result <<= 5.0/2.0 * gasConstant/mmw * temperature;
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
  TotalInternalEnergyTransportEquation( const std::string rhoe0Name,
                                        Uintah::ProblemSpecP wasatchSpec,
                                        Uintah::ProblemSpecP energyEqnSpec,
                                        GraphCategories& gc,
                                        const Expr::Tag& densityTag,
                                        const Expr::Tag& temperatureTag,
                                        const Expr::Tag& pressureTag,
                                        const Expr::TagList& velTags,
                                        const Expr::TagList& bodyForceTags,
                                        const Expr::Tag& viscTag,
                                        const Expr::Tag& dilTag,
                                        const TurbulenceParameters& turbulenceParams,
                                        WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                                        std::set<std::string>& persistentFields )
    : ScalarTransportEquation<SVolField>( rhoe0Name,
                                          energyEqnSpec,
                                          gc, densityTag,
                                          turbulenceParams,
                                          persistentFields,
                                          false /* don't call setup */ ),
      wasatchSpec_( wasatchSpec ),
      kineticEnergyTag_( "kinetic energy", Expr::STATE_NONE ),
      temperatureTag_( temperatureTag ),
      pressureTag_( TagNames::self().pressure ),
      velTags_( velTags ),
      bodyForceTags_(bodyForceTags),
      dualTimeMatrixInfo_( dualTimeMatrixInfo )
  {
#   ifdef HAVE_POKITT
    massFracTags_.clear();
    for( int i=0; i<CanteraObjects::number_species(); ++i ){
      massFracTags_.push_back( Expr::Tag( CanteraObjects::species_name(i), Expr::STATE_NONE ) );
    }
#   endif

    const TagNames& tags = TagNames::self();
    Expr::ExpressionFactory& solnFactory = *gc[ADVANCE_SOLUTION]->exprFactory;

    //----------------------------------------------------------
    // kinetic energy
    solnFactory.register_expression( scinew KineticEnergy<MyFieldT,MyFieldT,MyFieldT,MyFieldT>::Builder( kineticEnergyTag_, velTags[0], velTags[1], velTags[2] ) );

    //----------------------------------------------------------
    // temperature calculation
#   ifdef HAVE_POKITT
    if( wasatchSpec->findBlock("SpeciesTransportEquations") ){
      const Expr::Tag oldTempTag( temperatureTag.name(), Expr::STATE_N );
      solnFactory.register_expression( scinew Expr::PlaceHolder<SVolField>::Builder(oldTempTag) );
      typedef pokitt::TemperatureFromE0<MyFieldT>::Builder Temperature;
      solnFactory.register_expression( scinew Temperature( temperatureTag,
                                                           massFracTags_,
                                                           primVarTag_,
                                                           kineticEnergyTag_,
                                                           oldTempTag,
                                                           1.e-3,
                                                           5000.,
                                                           1000 ) );
      typedef pokitt::HeatCapacity_Cp<SVolField>::Builder Cp;
      solnFactory.register_expression( new Cp( tags.cp, temperatureTag, massFracTags_ ) );
      typedef pokitt::HeatCapacity_Cv<SVolField>::Builder Cv;
      solnFactory.register_expression( new Cv( tags.cv, temperatureTag, massFracTags_ ) );
      dualTimeMatrixInfo_.universalGasConstant = CanteraObjects::gas_constant();
    }
    else{
      // calorically perfect gas
      typedef TemperaturePurePerfectGas<MyFieldT>::Builder SimpleTemperature;
      solnFactory.register_expression( scinew SimpleTemperature( temperatureTag, primVarTag_, kineticEnergyTag_, tags.mixMW ) );

      const double gasConstant = 8314.459848;  // universal R = J/(kmol K).
      dualTimeMatrixInfo_.universalGasConstant = gasConstant;

      typedef Expr::LinearFunction<MyFieldT>::Builder SimpleEnthalpy;
      solnFactory.register_expression( scinew SimpleEnthalpy( tags.enthalpy, temperatureTag, 7.0/2.0*gasConstant, 0.0 ) );

      // register expressions for CP and CV
      typedef MultiplicativeInverse<MyFieldT>::Builder ReciprocalFunc;

      // register an expression for cp = 7/2 * R/Mw
      double slope = 7.0/2.0 * gasConstant;
      solnFactory.register_expression( new ReciprocalFunc( tags.cp, tags.mixMW, slope,0.0 ) );

      // register an expression for cv = 5/2 * R/Mw
      slope = 5.0/2.0 * gasConstant;
      solnFactory.register_expression( new ReciprocalFunc( tags.cv, tags.mixMW, slope,0.0 ) );
    }
#   else
    {
      // calorically perfect gas
      typedef TemperaturePurePerfectGas<MyFieldT>::Builder SimpleTemperature;
      solnFactory.register_expression( scinew SimpleTemperature( temperatureTag, primVarTag_, kineticEnergyTag_, tags.mixMW ) );

      const double gasConstant = 8314.459848;  // universal R = J/(kmol K).
      dualTimeMatrixInfo_.universalGasConstant = gasConstant;
      
      typedef Expr::LinearFunction<MyFieldT>::Builder SimpleEnthalpy;
      solnFactory.register_expression( scinew SimpleEnthalpy( tags.enthalpy, temperatureTag, 7.0/2.0*gasConstant, 0.0 ) );
      
      // register expressions for CP and CV
      typedef MultiplicativeInverse<MyFieldT>::Builder ReciprocalFunc;
      
      // register an expression for cp = 7/2 * R/Mw
      double slope = 7.0/2.0 * gasConstant;
      solnFactory.register_expression( new ReciprocalFunc( tags.cp, tags.mixMW, slope,0.0 ) );
      
      // register an expression for cv = 5/2 * R/Mw
      slope = 5.0/2.0 * gasConstant;
      solnFactory.register_expression( new ReciprocalFunc( tags.cv, tags.mixMW, slope,0.0 ) );

    }
#   endif

    dualTimeMatrixInfo_.totalEnergy    = Expr::Tag( rhoe0Name, Expr::STATE_DYNAMIC );
    dualTimeMatrixInfo_.cpHeatCapacity = tags.cp;
    dualTimeMatrixInfo_.cvHeatCapacity = tags.cv;

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
#   ifdef HAVE_POKITT
    if( wasatchSpec_->findBlock("SpeciesTransportEquations") ){
      typedef pokitt::TotalInternalEnergy<MyFieldT>::Builder EnergyIC;
      icFactory.register_expression( scinew EnergyIC( primVarTag_,
                                                      temperatureTag_,
                                                      massFracTags_,
                                                      velTags_ ) );
    }
    else
#   endif
    {
      typedef TotalInternalEnergy_PurePerfectGas_IC<SVolField>::Builder SimpleEnergyIC;
      icFactory.register_expression( scinew SimpleEnergyIC( primVarTag_, temperatureTag_, velTags_, TagNames::self().mixMW ) );
    }
    typedef ExprAlgebra<SVolField> ExprAlgbr;
    const Expr::Tag rhoTag( densityTag_.name(), Expr::STATE_NONE );
    return icFactory.register_expression( new ExprAlgbr::Builder( this->initial_condition_tag(),
                                                                  tag_list( primVarTag_, rhoTag ),
                                                                  ExprAlgbr::PRODUCT ) );
  }

  //---------------------------------------------------------------------------

  void
  TotalInternalEnergyTransportEquation::
  setup_diffusive_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    
#   ifdef HAVE_POKITT
    Uintah::ProblemSpecP speciesParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    if( speciesParams ){
      if( params_->findBlock("DiffusiveFlux") ){
        std::ostringstream msg;
        msg << "ERROR: When using species transport, the energy diffusive flux will be automatically calculated.\n"
            << "       Do not specify it in your input file." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      const TagNames& tagNames = TagNames::self();
      Expr::TagList specEnthTags, specEngyTags, xSpecFluxTags, ySpecFluxTags, zSpecFluxTags;
      for( int i=0; i<CanteraObjects::number_species(); ++i ){
        const std::string specName = CanteraObjects::species_name(i);
        specEnthTags.push_back( Expr::Tag( specName+"_"+tagNames.enthalpy.name(), Expr::STATE_NONE ) );
        specEngyTags.push_back( Expr::Tag( specName+"_energy"                   , Expr::STATE_NONE ) );

        xSpecFluxTags.push_back( Expr::Tag( specName + tagNames.diffusiveflux + "X", Expr::STATE_NONE ) );
        ySpecFluxTags.push_back( Expr::Tag( specName + tagNames.diffusiveflux + "Y", Expr::STATE_NONE ) );
        zSpecFluxTags.push_back( Expr::Tag( specName + tagNames.diffusiveflux + "Z", Expr::STATE_NONE ) );

        typedef pokitt::SpeciesEnthalpy<MyFieldT>::Builder SpecEnth;
        factory.register_expression( scinew SpecEnth( specEnthTags[i], temperatureTag_, i ) );
        typedef pokitt::SpeciesInternalEnergy<MyFieldT>::Builder SpecEngy;
        factory.register_expression( scinew SpecEngy( specEngyTags[i], temperatureTag_, i ) );
      }
      dualTimeMatrixInfo_.set_enthalpies( specEnthTags );
      dualTimeMatrixInfo_.set_energies( specEngyTags );

      typedef pokitt::ThermalConductivity<MyFieldT>::Builder ThermCond;
      factory.register_expression( scinew ThermCond( tagNames.thermalConductivity,
                                                     temperatureTag_,
                                                     massFracTags_,
                                                     tagNames.mixMW ) );

      dualTimeMatrixInfo_.conductivity = tagNames.thermalConductivity;

      // trigger creation of diffusive flux expression from species diffusive flux spec to determine which directions are active.
      for( Uintah::ProblemSpecP diffFluxParams=speciesParams->findBlock("DiffusiveFlux");
          diffFluxParams != nullptr;
          diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") )
      {
        std::string direction;
        diffFluxParams->getAttribute("direction",direction);
        for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
          const std::string dir(1,*it);
          if( dir == "X" ){
            typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<MyFieldT>::XFace >::Builder XFlux;
            const Expr::ExpressionID id = factory.register_expression( scinew XFlux( tagNames.xHeatFlux, temperatureTag_, tagNames.thermalConductivity, specEnthTags, xSpecFluxTags ) );
            factory.cleave_from_children( id );
            info[DIFFUSIVE_FLUX_X] = tagNames.xHeatFlux;
          }
          else if( dir == "Y" ){
            typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<MyFieldT>::YFace >::Builder YFlux;
            const Expr::ExpressionID id = factory.register_expression( scinew YFlux( tagNames.yHeatFlux, temperatureTag_, tagNames.thermalConductivity, specEnthTags, ySpecFluxTags ) );
            factory.cleave_from_children( id );
            info[DIFFUSIVE_FLUX_Y] = tagNames.yHeatFlux;
          }
          else if( dir == "Z" ){
            typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<MyFieldT>::ZFace >::Builder ZFlux;
            const Expr::ExpressionID id = factory.register_expression( scinew ZFlux( tagNames.zHeatFlux, temperatureTag_, tagNames.thermalConductivity, specEnthTags, zSpecFluxTags ) );
            factory.cleave_from_children( id );
            info[DIFFUSIVE_FLUX_Z] = tagNames.zHeatFlux;
          }
        }
      }
    }
    else
#   endif
    {
      const TagNames& tagNames = TagNames::self();
      dualTimeMatrixInfo_.mmw = tagNames.mixMW;
      for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
          diffFluxParams != nullptr;
          diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") ) {
        if( diffFluxParams->findAttribute("coefficient") ){
          double coef;
          diffFluxParams->getAttribute("coefficient",coef);
          factory.register_expression( scinew Expr::ConstantExpr<MyFieldT>::Builder( tagNames.thermalConductivity, coef ) );
          dualTimeMatrixInfo_.conductivity = tagNames.thermalConductivity;
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          dualTimeMatrixInfo_.conductivity = coef;
        }
        setup_diffusive_velocity_expression<MyFieldT>( diffFluxParams,
                                                       temperatureTag_,
                                                       turbDiffTag_,
                                                       factory,
                                                       info );
      }
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
    dualTimeMatrixInfo_.totalEnthalpy = combinedVarTag;
    dualTimeMatrixInfo_.totalEnergy = solnVarTag;

    for( Uintah::ProblemSpecP convFluxParams=params_->findBlock("ConvectiveFlux");
         convFluxParams != nullptr;
         convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") ) {
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
  
  void TotalInternalEnergyTransportEquation::
  setup_boundary_conditions( WasatchBCHelper& bcHelper, GraphCategories& graphCat )
  {
//    ScalarTransportEquation<SVolField>::setup_boundary_conditions(bcHelper,graphCat);
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);

    // strings for BC spec names, will depend on boundary face
    const std::string normalConvFluxNameBase = "rhoet_and_pressure_convFlux_";
    std::string heatFluxName;
    std::string heatFluxNameBase = "Temperature_diffVelocity_";
#   ifdef HAVE_POKITT
    Uintah::ProblemSpecP speciesParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    if( speciesParams ){
      const TagNames& tagNames = TagNames::self();
      const std::string str = tagNames.xHeatFlux.name();
      heatFluxNameBase = str.substr(0, str.size()-1);
    }
#   endif

    // set up the extra fields for setting BCs on primitives
    const Expr::Tag temporaryTTag( "temporary_Temperature_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryMMWTag( "temporary_mmw_for_bcs", Expr::STATE_NONE );

#   ifdef HAVE_POKITT
    Expr::TagList temporaryYTags;
    for( const auto& yTag : massFracTags_ ){
      temporaryYTags.push_back( Expr::Tag( "temporary_" + yTag.name() + "_for_bcs", Expr::STATE_NONE ) );
    }
#   endif
    Expr::TagList temporaryVelocityTags = {Expr::Tag(), Expr::Tag(), Expr::Tag()};
    for( int i=0; i<3; ++i ){
      if( velTags_[i].name() != "" ){
        temporaryVelocityTags[i] = Expr::Tag( "temporary_" + velTags_[i].name() + "_for_bcs", Expr::STATE_NONE );
      }
    }
    const Expr::Tag temporaryRhoTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryEtTag( "temporary_Et_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryRhoEtTag( "temporary_rhoEt_for_bcs", Expr::STATE_NONE );


    // get reference state
    double refTemperature = 300.;
    if (wasatchSpec_->findBlock("NSCBC")) {
      Uintah::ProblemSpecP nscbcXMLSpec = wasatchSpec_->findBlock("NSCBC");
      nscbcXMLSpec->getAttribute("Tfarfield", refTemperature);
    }

    if( !( advSlnFactory.have_entry( temporaryTTag ) ) ){
      advSlnFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryTTag, refTemperature ) );
    }
    if( !( initFactory.have_entry( temporaryTTag ) ) ){
      initFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryTTag, refTemperature ) );
    }
    if( wasatchSpec_->findBlock("SpeciesTransportEquations") ){
#ifdef HAVE_POKITT
      const int nspec = CanteraObjects::number_species();
      if( !( advSlnFactory.have_entry( temporaryYTags[nspec-1] ) ) ){
        advSlnFactory.register_expression( new pokitt::SpeciesN<MyFieldT>::Builder( temporaryYTags[nspec-1], temporaryYTags, pokitt::ERRORSPECN, DEFAULT_NUMBER_OF_GHOSTS ) );
      }
      if( !( initFactory.have_entry( temporaryYTags[nspec-1] ) ) ){
        initFactory.register_expression( new pokitt::SpeciesN<MyFieldT>::Builder( temporaryYTags[nspec-1], temporaryYTags, pokitt::ERRORSPECN, DEFAULT_NUMBER_OF_GHOSTS ) );
      }
      if( !( advSlnFactory.have_entry( temporaryMMWTag ) ) ){
        advSlnFactory.register_expression( new pokitt::MixtureMolWeight<MyFieldT>::Builder( temporaryMMWTag, temporaryYTags, pokitt::MASS ) );
      }
      if( !( initFactory.have_entry( temporaryMMWTag ) ) ){
        initFactory.register_expression( new pokitt::MixtureMolWeight<MyFieldT>::Builder( temporaryMMWTag, temporaryYTags, pokitt::MASS ) );
      }
#endif
    }
    else{
      if( !( advSlnFactory.have_entry( temporaryMMWTag ) ) ){
        advSlnFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryMMWTag, 28.966 ) ); // how to from input file?
      }
      if( !( initFactory.have_entry( temporaryMMWTag ) ) ){
        initFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryMMWTag, 28.966 ) ); // how to from input file?
      }
    }

    if( !( advSlnFactory.have_entry( temporaryRhoEtTag ) ) ){
      if( wasatchSpec_->findBlock("SpeciesTransportEquations") ){
#ifdef HAVE_POKITT
        typedef pokitt::TotalInternalEnergy<SVolField>::Builder TotalEnergy;
        advSlnFactory.register_expression( new TotalEnergy( temporaryEtTag, temporaryTTag, temporaryYTags, temporaryVelocityTags ) );
        typedef ExprAlgebra<SVolField>::Builder ExprAlg;
        advSlnFactory.register_expression( new ExprAlg( temporaryRhoEtTag, Expr::tag_list( temporaryRhoTag, temporaryEtTag ), ExprAlgebra<SVolField>::PRODUCT ) );
#endif
      }
      else
      {
        const TagNames& tagNames = TagNames::self();
        typedef TotalInternalEnergy_PurePerfectGas_IC<SVolField>::Builder TotalEnergy;
        advSlnFactory.register_expression( new TotalEnergy( temporaryEtTag, temporaryTTag, temporaryVelocityTags, tagNames.mixMW ) );
        typedef ExprAlgebra<SVolField>::Builder ExprAlg;
        advSlnFactory.register_expression( new ExprAlg( temporaryRhoEtTag, Expr::tag_list( temporaryRhoTag, temporaryEtTag ), ExprAlgebra<SVolField>::PRODUCT ) );
      }
    }
    if( !( initFactory.have_entry( temporaryRhoEtTag ) ) ){
      if( wasatchSpec_->findBlock("SpeciesTransportEquations") ){
#ifdef HAVE_POKITT
// AND IF HAVE SPECIES TRANSPORT EQUATIONS........
      typedef pokitt::TotalInternalEnergy<SVolField>::Builder TotalEnergy;
      initFactory.register_expression( new TotalEnergy( temporaryEtTag, temporaryTTag, temporaryYTags, temporaryVelocityTags ) );
      typedef ExprAlgebra<SVolField>::Builder ExprAlg;
      initFactory.register_expression( new ExprAlg( temporaryRhoEtTag, Expr::tag_list( temporaryRhoTag, temporaryEtTag ), ExprAlgebra<SVolField>::PRODUCT ) );
#endif
      }
      else{
        const TagNames& tagNames = TagNames::self();
        typedef TotalInternalEnergy_PurePerfectGas_IC<SVolField>::Builder TotalEnergy;
        initFactory.register_expression( new TotalEnergy( temporaryEtTag, temporaryTTag, temporaryVelocityTags, tagNames.mixMW ) );
        typedef ExprAlgebra<SVolField>::Builder ExprAlg;
        initFactory.register_expression( new ExprAlg( temporaryRhoEtTag, Expr::tag_list( temporaryRhoTag, temporaryEtTag ), ExprAlgebra<SVolField>::PRODUCT ) );
      }
    }
    
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;

      // a lambda to make decorated tags for boundary condition expressions
      //
      // param: exprName: a string, the name of the field on which we will impose the boundary condition
      // param: description: a string describing the boundary condition, such as "neumann-zero-for-outflow" or "dirichlet-for-inflow"
      // param: direction: a string for the direction of the boundary face, such as "X", "Y", or "Z"
      auto get_decorated_tag = [&myBndSpec](const std::string exprName, const std::string description, const std::string direction) -> Expr::Tag
      {
        return Expr::Tag( exprName + "_STATE_NONE_" + description + "_bc_" + myBndSpec.name + "_" + direction + "dir", Expr::STATE_NONE );
      };
      
      switch ( myBndSpec.type ){
        case OUTFLOW:
        case OPEN:{

          // tags for modifier expressions and strings for BC spec names, will depend on boundary face
          std::string normalConvFluxName;
          Expr::Tag neumannZeroConvectiveFluxTag,
                    neumannZeroHeatFluxTag,
                    neumannZeroEnergyTag;

          // build boundary conditions for x, y, and z faces
          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              std::string dir = "X";
              typedef EnergyBoundaryTyper<SpatialOps::SSurfXField, SpatialOps::GradientX> BCTypes;

              heatFluxName       = heatFluxNameBase + dir;
              normalConvFluxName = normalConvFluxNameBase + dir;

              neumannZeroEnergyTag         = get_decorated_tag( this->solnVarTag_.name(), "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName      , "neumann-zero", dir );
              neumannZeroHeatFluxTag       = get_decorated_tag( heatFluxName            , "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroHeatFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroHeatFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroEnergyTag         ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroEnergyTag         ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
            }
            break;
            case Uintah::Patch::yminus:
            case Uintah::Patch::yplus:
            {
              std::string dir = "Y";
              typedef EnergyBoundaryTyper<SpatialOps::SSurfYField, SpatialOps::GradientY> BCTypes;

              heatFluxName       = heatFluxNameBase + dir;
              normalConvFluxName = normalConvFluxNameBase + dir;

              neumannZeroEnergyTag         = get_decorated_tag( this->solnVarTag_.name(), "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName      , "neumann-zero", dir );
              neumannZeroHeatFluxTag       = get_decorated_tag( heatFluxName            , "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroHeatFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroHeatFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroEnergyTag         ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroEnergyTag         ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
            }
            break;
            case Uintah::Patch::zminus:
            case Uintah::Patch::zplus:
            {
              std::string dir = "Z";
              typedef EnergyBoundaryTyper<SpatialOps::SSurfZField, SpatialOps::GradientZ> BCTypes;

              heatFluxName       = heatFluxNameBase + dir;
              normalConvFluxName = normalConvFluxNameBase + dir;

              neumannZeroEnergyTag         = get_decorated_tag( this->solnVarTag_.name(), "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName      , "neumann-zero", dir );
              neumannZeroHeatFluxTag       = get_decorated_tag( heatFluxName            , "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroHeatFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroHeatFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroEnergyTag         ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroEnergyTag         ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroEnergyTag        , 0.0 ) );
            }
            break;
            default:
              break;
          }

          BndCondSpec energySpec   = {this->solnVarTag_.name(), neumannZeroEnergyTag.name()        , 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec convFluxSpec = {normalConvFluxName      , neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec heatFluxSpec = {heatFluxName            , neumannZeroHeatFluxTag.name()      , 0.0, NEUMANN, FUNCTOR_TYPE};

          bcHelper.add_boundary_condition( bndName, heatFluxSpec );
          bcHelper.add_boundary_condition( bndName, energySpec   );
          bcHelper.add_boundary_condition( bndName, convFluxSpec );
        }
        break;
        case USER:
        {
          // parse through the list of user specified BCs that are relevant to this transport equation
        }
          break;
        case VELOCITY:{

          Expr::Tag bcCopiedEnergyTag;

          // build boundary conditions for x, y, and z faces
          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              std::string dir = "X";
              typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
              bcCopiedEnergyTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedEnergyTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
              if( !advSlnFactory.have_entry( bcCopiedEnergyTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
            }
            break;
            case Uintah::Patch::yminus:
            case Uintah::Patch::yplus:
            {
              std::string dir = "Y";
              typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
              bcCopiedEnergyTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedEnergyTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
              if( !advSlnFactory.have_entry( bcCopiedEnergyTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
            }
            break;
            case Uintah::Patch::zminus:
            case Uintah::Patch::zplus:
            {
              std::string dir = "Z";
              typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
              bcCopiedEnergyTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedEnergyTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
              if( !advSlnFactory.have_entry( bcCopiedEnergyTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedEnergyTag, temporaryRhoEtTag ) );
            }
            break;
            default:
              break;
          }

          BndCondSpec energyDirichletBC = {this->solnVarTag_.name(), bcCopiedEnergyTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};

          bcHelper.add_boundary_condition( bndName, energyDirichletBC );

          // set zero rhoet RHS at inlets - for nscbc???
          //          BndCondSpec rhoetRHSBCSpec = {this->rhs_tag().name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
          //          bcHelper.add_boundary_condition(bndName, rhoetRHSBCSpec);
        }
        break;
        case WALL:
          // parse through the list of user specified BCs that are relevant to this transport equation
        default:
          break;
      }
    }

  }

  //---------------------------------------------------------------------------
  
  void TotalInternalEnergyTransportEquation::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    //ScalarTransportEquation<SVolField>::apply_boundary_conditions(graphHelper, bcHelper);
    const Category taskCat = ADVANCE_SOLUTION;
    bcHelper.apply_boundary_condition<MyFieldT>( solnvar_np1_tag(), taskCat );
    bcHelper.apply_boundary_condition<MyFieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell

    std::string heatFluxNameBase = "Temperature_diffVelocity_";
#   ifdef HAVE_POKITT
    Uintah::ProblemSpecP speciesParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    if( speciesParams ){
      const TagNames& tagNames = TagNames::self();
      const std::string str = tagNames.xHeatFlux.name();
      heatFluxNameBase = str.substr(0, str.size()-1);
    }
#   endif


    // bcs for hard inflow - set primitive and conserved variables
    bcHelper.apply_boundary_condition<SpatialOps::SVolField>(this->temperatureTag_, taskCat, true);
    const Expr::Tag temporaryTTag( "temporary_Temperature_for_bcs", Expr::STATE_NONE );
    bcHelper.apply_boundary_condition<MyFieldT>( temporaryTTag, taskCat, true );

    bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag(heatFluxNameBase + "X", Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag(heatFluxNameBase + "Y", Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag(heatFluxNameBase + "Z", Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>(Expr::Tag("rhoet_and_pressure_convFlux_X", Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>(Expr::Tag("rhoet_and_pressure_convFlux_Y", Expr::STATE_NONE), taskCat);
    bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>(Expr::Tag("rhoet_and_pressure_convFlux_Z", Expr::STATE_NONE), taskCat);
    bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBC::ENERGY, taskCat);
  }


} /* namespace WasatchCore */
