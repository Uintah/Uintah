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

#include <CCA/Components/Wasatch/TagNames.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Transport/ParseEquation.h>

namespace WasatchCore{

  //============================================================================
  
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
   *  \author James C. Sutherland
   *  \date November, 2015
   *
   *  \brief Calculates initial condition for the density given an initial pressure and temperature.
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
   * \author James C. Sutherland, Tony Saad
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

    //----------------------------------------------------------------------------
    
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                   GraphCategories& graphCat )
    {
      // make logical decisions based on the specified boundary types
      BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
        const std::string& bndName = bndPair.first;
        const BndSpec& myBndSpec = bndPair.second;
        
        switch ( myBndSpec.type ){
          case WALL:
          {
            // first check if the user specified momentum boundary conditions at the wall
            if( myBndSpec.has_field(this->solution_variable_name()) || myBndSpec.has_field(this->solution_variable_name() + "_rhs") ){
              std::ostringstream msg;
              msg << "ERROR: You cannot specify any density-related boundary conditions at a stationary wall. "
              << "This error occured while trying to analyze boundary " << bndName
              << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
            
            // set zero Neumann for density on the wall
            BndCondSpec rhoBCSpec = {this->solution_variable_name(),"none" ,0.0,NEUMANN,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhoBCSpec);

            // set zero density_rhs inside the extra cell
            BndCondSpec rhoRHSBCSpec = {this->rhs_tag().name(), "none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhoRHSBCSpec);
          }
            break;
          case VELOCITY:
          case OUTFLOW:
          case OPEN:{
            // for constant density problems, on all types of boundary conditions, set the scalar rhs
            // to zero. The variable density case requires setting the scalar rhs to zero ALL the time
            // and is handled in the code above.
            if( isConstDensity_ ){
              if( myBndSpec.has_field(rhs_name()) ){
                std::ostringstream msg;
                msg << "ERROR: You cannot specify scalar rhs boundary conditions unless you specify USER "
                << "as the type for the boundary condition. Please revise your input file. "
                << "This error occured while trying to analyze boundary " << bndName
                << std::endl;
                throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
              }
              const BndCondSpec rhsBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
              bcHelper.add_boundary_condition(bndName, rhsBCSpec);
            }
            
            break;
          }
          case USER:{
            // parse through the list of user specified BCs that are relevant to this transport equation
            break;
          }
            
          default:
            break;
        }
      }
    }

    //----------------------------------------------------------------------------
    
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                           WasatchBCHelper& bcHelper )
    {
      const Category taskCat = INITIALIZATION;
      // apply velocity boundary condition, if specified
      bcHelper.apply_boundary_condition<FieldT>(this->initial_condition_tag(), taskCat);
      
    }
    
    //----------------------------------------------------------------------------
    
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                   WasatchBCHelper& bcHelper )
    {
      const Category taskCat = ADVANCE_SOLUTION;
      // set bcs for momentum - use the TIMEADVANCE expression
      bcHelper.apply_boundary_condition<FieldT>( this->solnvar_np1_tag(), taskCat );
      // set bcs for velocity
      bcHelper.apply_boundary_condition<FieldT>( this->rhs_tag(), taskCat, true );
    }
    
    void setup_diffusive_flux( FieldTagInfo& rhsInfo ){}
    void setup_source_terms  ( FieldTagInfo& rhsInfo, Expr::TagList& srcTags ){}

    //----------------------------------------------------------------------------
    
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags )
    {
      
      typedef ScalarRHS<FieldT>::Builder RHSBuilder;
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      
      info[PRIMITIVE_VARIABLE] = solnVarTag_;
      
      return factory.register_expression( scinew RHSBuilder( rhsTag_, info, Expr::TagList(), densTag_, false, true, Expr::Tag() ) );
    }

    //----------------------------------------------------------------------------
    
    void setup_convective_flux( FieldTagInfo& rhsInfo )
    {
      if( xVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "X", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 xVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
      if( yVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "Y", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 yVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
      if( zVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<FieldT>( "Z", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 zVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
    }

    //----------------------------------------------------------------------------
    
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
  
  /**
   * \class CompressibleMomentumTransportEquation
   * \date November, 2015
   * \author Tony Saad, James C. Sutherland
   *
   * \brief Construct a compressible momentum transport equation - assumes collocated grid arrangement.
   *
   */
  template <typename MomDirT>
  class CompressibleMomentumTransportEquation : public WasatchCore::MomentumTransportEquationBase<SVolField>
  {
    typedef SpatialOps::SVolField FieldT;

  public:
    CompressibleMomentumTransportEquation( const Direction momComponent,
                                           const std::string velName,
                                           const std::string momName,
                                           const Expr::Tag densityTag,
                                           const Expr::Tag temperatureTag,
                                           const Expr::Tag mixMWTag,
                                           const double gasConstant,
                                           const Expr::Tag bodyForceTag,
                                           const Expr::Tag srcTermTag,
                                           GraphCategories& gc,
                                           Uintah::ProblemSpecP params,
                                           TurbulenceParameters turbParams );

    ~CompressibleMomentumTransportEquation();

    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                   GraphCategories& graphCat );

    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                           WasatchBCHelper& bcHelper );

    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                   WasatchBCHelper& bcHelper );

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory )
    {
      // register an initial condition for da pressure
      if( !icFactory.have_entry( this->pressureTag_ ) ) {
        icFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( pressureTag_, 101325.00 ) ); // set the pressure in Pa.
      }

      if( icFactory.have_entry( this->thisVelTag_ ) ) {
        
        // register expression to calculate the momentum initial condition from the initial conditions on
        // velocity and density in the cases that we are initializing velocity in the input file
        typedef ExprAlgebra<SVolField> ExprAlgbr;
        const Expr::Tag rhoTag(this->densityTag_.name(), Expr::STATE_NONE);
        const Expr::TagList theTagList( tag_list( this->thisVelTag_, rhoTag ) );
        icFactory.register_expression( new ExprAlgbr::Builder( this->initial_condition_tag(),
                                                                       theTagList,
                                                                       ExprAlgbr::PRODUCT ) );
      }
      
      // multiply the initial condition by the volume fraction for embedded geometries
      const EmbeddedGeometryHelper& geomHelper = EmbeddedGeometryHelper::self();
      if( geomHelper.has_embedded_geometry() ) {
        //create modifier expression
        typedef ExprAlgebra<FieldT> ExprAlgbr;
        const Expr::TagList theTagList( tag_list( this->thisVolFracTag_ ) );
        Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE );
        icFactory.register_expression( new ExprAlgbr::Builder( modifierTag,
                                                                       theTagList,
                                                                       ExprAlgbr::PRODUCT,
                                                                       true ) );
        icFactory.attach_modifier_expression( modifierTag, this->initial_condition_tag() );
      }
      return icFactory.get_id( this->initial_condition_tag() );
    }

  protected:

    void setup_diffusive_flux( FieldTagInfo& ){}
    void setup_convective_flux( FieldTagInfo& ){}
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){}
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags );

  };

}



