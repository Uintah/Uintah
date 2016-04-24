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

#include <Core/Exceptions/ProblemSetupException.h>

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OutflowBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OpenBC.h>

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
      this->set_gpu_runnable(true);
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
      this->set_gpu_runnable(true);
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

  
  //----------------------------------------------------------------------------
  
  Expr::ExpressionID
  ContinuityTransportEquation::initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    typedef Density_IC<FieldT>::Builder DensIC;
    return exprFactory.register_expression( scinew DensIC( initial_condition_tag(),
                                                          temperatureTag_,
                                                          TagNames::self().pressure,
                                                          mixMWTag_,
                                                          gasConstant_) );
  }


  //============================================================================

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
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
                                         TurbulenceParameters turbParams )
  : MomentumTransportEquationBase<SVolField>(momComponent,
                                          velName,
                                          momName,
                                          densityTag,
                                          false,
                                          bodyForceTag,
                                          srcTermTag,
                                          gc,
                                          params,
                                          turbParams)
  {
    // todo:
    //  - strain tensor       // registered by MomentumTransportEquationBase
    //  - convective flux     // registered by the MomentumTransportEquationBase
    //  - turbulent viscosity // SHOULD be registered by the MomentumTransportEquationBase. NOT READY YET.
    //  - buoyancy? //        // body forces are handled in the momentumequationbase

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    typedef IdealGasPressure<FieldT>::Builder Pressure;
    if (!factory.have_entry(TagNames::self().pressure)) {
      factory.register_expression( scinew Pressure(TagNames::self().pressure,
                                                   densityTag,
                                                   temperatureTag,
                                                   mixMWTag,
                                                   gasConstant) );

    }

    setup();
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  Expr::ExpressionID  CompressibleMomentumTransportEquation<MomDirT>::
  setup_rhs( FieldTagInfo&,
            const Expr::TagList& srcTags )
  {
    
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    Expr::Tag volFracTag = vNames.vol_frac_tag<FieldT>();
    
    Expr::ExpressionFactory& factory = *this->gc_[ADVANCE_SOLUTION]->exprFactory;

    typedef typename MomRHS<SVolField, MomDirT>::Builder RHS;
    return factory.register_expression( scinew RHS( this->rhsTag_,
                                                   this->pressureTag_,
                                                   rhs_part_tag(this->solnVarTag_),
                                                   volFracTag ) );
  }

  //----------------------------------------------------------------------------

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
  ~CompressibleMomentumTransportEquation()
  {}

  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                 GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory = *(graphCat[INITIALIZATION]->exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    //
    // Add dummy modifiers on all patches. This is used to inject new dpendencies across all patches.
    // Those new dependencies, result for example from complicated boundary conditions added in this
    // function. NOTE: whenever you want to add a new complex boundary condition, please use this
    // functionality to inject new dependencies across patches.
    //
    {
      // add dt dummy modifier for outflow bcs...
      bcHelper.create_dummy_dependency<SpatialOps::SingleValueField, FieldT>(rhs_part_tag(this->solnVarName_), tag_list(tagNames.dt),ADVANCE_SOLUTION);
      
      // add momentum dummy modifiers
      const Expr::Tag momTimeAdvanceTag(this->solnvar_np1_tag());
      bcHelper.create_dummy_dependency<SVolField, FieldT>(momTimeAdvanceTag, tag_list(this->densityTag_),ADVANCE_SOLUTION);
      bcHelper.create_dummy_dependency<FieldT, FieldT>(momTimeAdvanceTag, tag_list(this->thisVelTag_),ADVANCE_SOLUTION);
      if( initFactory.have_entry(this->thisVelTag_) ){
        const Expr::Tag densityStateNone(this->densityTag_.name(), Expr::STATE_NONE);
        bcHelper.create_dummy_dependency<SVolField, FieldT>(momTimeAdvanceTag, tag_list(densityStateNone),INITIALIZATION);
        bcHelper.create_dummy_dependency<FieldT, FieldT>(momTimeAdvanceTag, tag_list(this->thisVelTag_),INITIALIZATION);
      }
      
      if( !this->is_constant_density() ){
        const Expr::Tag rhoTagInit(this->densityTag_.name(), Expr::STATE_NONE);
        const Expr::Tag rhoStarTag = tagNames.make_star(this->densityTag_); // get the tagname of rho*
        bcHelper.create_dummy_dependency<SVolField, SVolField>(rhoStarTag, tag_list(rhoTagInit), INITIALIZATION);
        const Expr::Tag rhoTagAdv(this->densityTag_.name(), Expr::STATE_NONE);
        bcHelper.create_dummy_dependency<SVolField, SVolField>(rhoStarTag, tag_list(rhoTagAdv), ADVANCE_SOLUTION);
      }
    }
    //
    // END DUMMY MODIFIER SETUP
    //
    
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() )
    {
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;
      
      const bool isNormal = is_normal_to_boundary(this->staggered_location(), myBndSpec.face);
      
      // variable density: add bcopiers on all boundaries
      if( !this->is_constant_density() ){
        // if we are solving a variable density problem, then set bcs on density estimate rho*
        const Expr::Tag rhoStarTag = tagNames.make_star(this->densityTag_); // get the tagname of rho*
        // check if this boundary applies a bc on the density
        if( myBndSpec.has_field(this->densityTag_.name()) ){
          // create a bc copier for the density estimate
          const Expr::Tag rhoStarBCTag( rhoStarTag.name() + "_" + bndName +"_bccopier", Expr::STATE_NONE);
          BndCondSpec rhoStarBCSpec = {rhoStarTag.name(), rhoStarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
          if( !initFactory.have_entry(rhoStarBCTag) ){
            const Expr::Tag rhoTag(this->densityTag_.name(), Expr::STATE_NONE);
            initFactory.register_expression ( new typename BCCopier<SVolField>::Builder(rhoStarBCTag, rhoTag) );
            bcHelper.add_boundary_condition(bndName, rhoStarBCSpec);
          }
          if( !advSlnFactory.have_entry(rhoStarBCTag) ){
            const Expr::Tag rhoTag(this->densityTag_.name(), Expr::STATE_NONE);
            advSlnFactory.register_expression ( new typename BCCopier<SVolField>::Builder(rhoStarBCTag, rhoTag) );
            bcHelper.add_boundary_condition(bndName, rhoStarBCSpec);
          }
        }
      }
      
      switch (myBndSpec.type) {
        case WALL:
        {
          // first check if the user specified momentum boundary conditions at the wall
          if( myBndSpec.has_field(this->thisVelTag_.name()) || myBndSpec.has_field(this->solnVarName_) ||
             myBndSpec.has_field(this->rhs_name()) || myBndSpec.has_field(this->solnVarName_ + "_rhs_part") ){
            std::ostringstream msg;
            msg << "ERROR: You cannot specify any momentum-related boundary conditions at a stationary wall. "
            << "This error occured while trying to analyze boundary " << bndName
            << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          }
          
          // set zero momentum on the wall
          BndCondSpec momBCSpec = {this->solution_variable_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);

          // set zero pressure gradient
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(),"none" ,0.0,NEUMANN,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          
          // set zero velocity
          BndCondSpec velBCSpec = {this->thisVelTag_.name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, velBCSpec);
          
          break;
        }
        case VELOCITY:
        case OUTFLOW:
        case OPEN:
        {
          std::ostringstream msg;
          msg << "ERROR: VELOCITY, OPEN, and OUTFLOW boundary conditions are not currently supported for compressible flows in Wasatch. " << bndName
          << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          break;
        }
        case USER:
        {
          // pass through the list of user specified BCs that are relevant to this transport equation
          break;
        }
          
        default:
          break;
      } // SWITCH BOUNDARY TYPE
    } // BOUNDARY LOOP
  }

  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                         WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
    
    // apply velocity boundary condition, if specified
    bcHelper.apply_boundary_condition<FieldT>(this->thisVelTag_, taskCat);
    
    // tsaad: boundary conditions will not be applied on the initial condition of momentum. This leads
    // to tremendous complications in our graphs. Instead, specify velocity initial conditions
    // and velocity boundary conditions, and momentum bcs will appropriately propagate.
    bcHelper.apply_boundary_condition<FieldT>(this->initial_condition_tag(), taskCat);
    
    if( !this->is_constant_density() ){
      const TagNames& tagNames = TagNames::self();
      
      // set bcs for density
      const Expr::Tag densTag( this->densityTag_.name(), Expr::STATE_NONE );
      bcHelper.apply_boundary_condition<SVolField>(densTag, taskCat);
      
      // set bcs for density_*
      const Expr::Tag densStarTag = tagNames.make_star(this->densityTag_, Expr::STATE_NONE);
      bcHelper.apply_boundary_condition<SVolField>(densStarTag, taskCat);
    }
  }
  
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                                 WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    // set bcs for momentum - use the TIMEADVANCE expression
    bcHelper.apply_boundary_condition<FieldT>( this->solnvar_np1_tag(), taskCat );
    // set bcs for velocity
    bcHelper.apply_boundary_condition<FieldT>( this->thisVelTag_, taskCat );
    // set bcs for partial rhs
    bcHelper.apply_boundary_condition<FieldT>( rhs_part_tag(mom_tag(this->solnVarName_)), taskCat, true);
    // set bcs for full rhs
    bcHelper.apply_boundary_condition<FieldT>( this->rhs_tag(), taskCat, true);
    // set bcs for pressure
    bcHelper.apply_boundary_condition<FieldT>( this->pressureTag_, taskCat);
  }

  //============================================================================

  template class CompressibleMomentumTransportEquation<SpatialOps::XDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::YDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::ZDIR>;
} // namespace Wasatch



