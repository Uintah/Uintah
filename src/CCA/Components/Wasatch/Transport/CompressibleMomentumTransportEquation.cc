/*
 * The MIT License
 *
 * Copyright (c) 2015 The University of Utah
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

/**
 *  \file   CompressibleMomentumTransportEquation.cc
 *  \date   Nov 20, 2015
 *  \author James C. Sutherland
 */

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquation.h>

namespace Wasatch{


  /**
   *  \class IdealGasPressure
   *  \author James C. Sutherland
   *  \date November, 2015
   *
   *  \brief Calculates the pressure from the ideal gas law: \f$p=\frac{\rho R T}{M}\f$
   *   where \f$M\f$ is the mixture molecular weight.
   */
  template< typename FieldT >
  class IdealGasPressure : public Expr::Expression<FieldT>
  {
    const double gasConstant_;
    DECLARE_FIELDS( FieldT, density_, temperature_, mixMW_ )

    IdealGasPressure( const Expr::Tag& densityTag,
                      const Expr::Tag& temperatureTag,
                      const Expr::Tag& mixMWTag,
                      const double gasConstant )
      : Expr::Expression<FieldT>(),
        gasConstant_( gasConstant )
    {
      density_     = this->template create_field_request<FieldT>( densityTag     );
      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      mixMW_       = this->template create_field_request<FieldT>( mixMWTag       );
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag densityTag_, temperatureTag_, mixMWTag_;
      const double gasConstant_;
    public:
      /**
       *  @brief Build a IdealGasPressure expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& densityTag,
               const Expr::Tag& temperatureTag,
               const Expr::Tag& mixMWTag,
               const double gasConstant,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          densityTag_( densityTag ),
          temperatureTag_( temperatureTag ),
          mixMWTag_( mixMWTag ),
          gasConstant_( gasConstant )
      {}

      Expr::ExpressionBase* build() const{
        return new IdealGasPressure<FieldT>( densityTag_, temperatureTag_, mixMWTag_, gasConstant_ );
      }

    };  /* end of Builder class */

    ~IdealGasPressure(){}

    void evaluate()
    {
      FieldT& result = this->value();
      const FieldT& density     = density_    ->field_ref();
      const FieldT& temperature = temperature_->field_ref();
      const FieldT& mixMW       = mixMW_      ->field_ref();
      result <<= density * gasConstant_ * temperature / mixMW;
    }
  };

  //============================================================================

  /**
   *  \class Density_IC
   */
  template< typename FieldT >
  class Density_IC
   : public Expr::Expression<FieldT>
  {
    const double gasConstant_;
    DECLARE_FIELDS( FieldT, temperature_, pressure_, mixMW_ )

    Density_IC( const Expr::Tag& temperatureTag,
                const Expr::Tag& pressureTag,
                const Expr::Tag& mixMWTag,
                const double gasConstant )
      : Expr::Expression<FieldT>(),
        gasConstant_( gasConstant )
    {
      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      pressure_    = this->template create_field_request<FieldT>( pressureTag    );
      mixMW_       = this->template create_field_request<FieldT>( mixMWTag       );
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const double gasConstant_;
      const Expr::Tag temperatureTag_, pressureTag_, mixMWTag_;
    public:
      /**
       *  @brief Build a Density_IC expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& temperatureTag,
               const Expr::Tag& pressureTag,
               const Expr::Tag& mixMWTag,
               const double gasConstant,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTag, nghost ),
          gasConstant_   ( gasConstant    ),
          temperatureTag_( temperatureTag ),
          pressureTag_   ( pressureTag    ),
          mixMWTag_      ( mixMWTag       )
      {}

      Expr::ExpressionBase* build() const{
        return new Density_IC<FieldT>( temperatureTag_,pressureTag_,mixMWTag_,gasConstant_ );
      }
    };  /* end of Builder class */

    ~Density_IC(){}

    void evaluate(){
      this->value() <<=  ( pressure_->field_ref() * mixMW_->field_ref() )/( gasConstant_ * temperature_->field_ref() );
    }
  };

  //============================================================================

  /**
   * \class ContinuityTransportEquation
   * \author James C. Sutherland
   * \date November, 2015
   *
   * \note here we derive off of TransportEquation because ScalarTransportEquation
   * requires input file specs, but we don't need that for the continuity equation.
   */
  class ContinuityTransportEquation : public TransportEquation
  {
    typedef SpatialOps::SVolField  FieldT;

    const Expr::Tag xVelTag_, yVelTag_, zVelTag_;
    const Expr::Tag densTag_, temperatureTag_, pressureTag_, mixMWTag_;
    const double gasConstant_;
  public:
    ContinuityTransportEquation( const Expr::Tag densityTag,
                                 const Expr::Tag temperatureTag,
                                 const Expr::Tag mixMWTag,
                                 const double gasConstant,
                                 GraphCategories& gc,
                                 const Expr::Tag xvel,
                                 const Expr::Tag yvel,
                                 const Expr::Tag zvel )
    : TransportEquation( gc, densityTag.name(), NODIR, false /* variable density */ ),
      xVelTag_( xvel ),
      yVelTag_( yvel ),
      zVelTag_( zvel ),
      densTag_       ( densityTag     ),
      temperatureTag_( temperatureTag ),
      mixMWTag_      ( mixMWTag       ),
      gasConstant_( gasConstant )
    {
      setup();
    }

    void setup_diffusive_flux( FieldTagInfo& rhsInfo ){}
    void setup_source_terms  ( FieldTagInfo& rhsInfo, Expr::TagList& srcTags ){}

    void setup_convective_flux( FieldTagInfo& rhsInfo )
    {
      if( xVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "X", densTag_,
                                                  Expr::Tag(), /* default tag name for conv. flux */
                                                  CENTRAL,
                                                  xVelTag_,
                                                  "",
                                                  *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                  rhsInfo );
      if( yVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "Y", densTag_,
                                                  Expr::Tag(), /* default tag name for conv. flux */
                                                  CENTRAL,
                                                  yVelTag_,
                                                  "",
                                                  *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                  rhsInfo );
      if( zVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "Z", densTag_,
                                                  Expr::Tag(), /* default tag name for conv. flux */
                                                  CENTRAL,
                                                  zVelTag_,
                                                  "",
                                                  *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                  rhsInfo );
    }

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& exprFactory )
    {
      typedef Density_IC<FieldT>::Builder DensIC;
      return exprFactory.register_expression( scinew DensIC( initial_condition_tag(),
                                                             temperatureTag_,
                                                             TagNames::self().pressure,
                                                             mixMWTag_,
                                                             gasConstant_) );
    }
  };

  //============================================================================

  CompressibleMomentumTransportEquation::
  CompressibleMomentumTransportEquation( const std::string velName,
                                         const std::string momName,
                                         const Expr::Tag densityTag,
                                         const Expr::Tag temperatureTag,
                                         const Expr::Tag mixMWTag,
                                         const double gasConstant,
                                         const Expr::Tag bodyForceTag,
                                         GraphCategories& gc,
                                         Uintah::ProblemSpecP params,
                                         TurbulenceParameters turbParams )
    : TransportEquation( gc, momName, NODIR, false )
  {
    // todo:
    //  - strain tensor
    //  - convective flux
    //  - turbulent viscosity
    //  - diffusive flux
    //  - buoyancy?

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    typedef IdealGasPressure<FieldT>::Builder Pressure;
    factory.register_expression( scinew Pressure(initial_condition_tag(),
                                                 TagNames::self().pressure,
                                                 temperatureTag,
                                                 mixMWTag,
                                                 gasConstant) );

    setup();
  }

  //----------------------------------------------------------------------------

  CompressibleMomentumTransportEquation::
  ~CompressibleMomentumTransportEquation()
  {}

  //============================================================================

} // namespace Wasatch



