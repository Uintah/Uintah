/*
 * The MIT License
 *
 * Copyright (c) 2015-2017 The University of Utah
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
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OutflowBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OpenBC.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/Derivative.h>
// -- NSCBC Includes -- //
#include <nscbc/CharacteristicBCBuilder.h>
#include <nscbc/TagManager.h>
#include <nscbc/SpeedOfSound.h>

namespace WasatchCore{

  //====================================================================
  // Find out if a certain momentum component of MomDirT is perpendicular to face
  template <typename MomDirT> bool is_perpendicular(const Uintah::Patch::FaceType face)
  {return false;}
  
  template<>
  bool is_perpendicular<SpatialOps::XDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::xminus || face == Uintah::Patch::xplus) {
      return true;
    } else {
      return false;
    }
  }

  template<>
  bool is_perpendicular<SpatialOps::YDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::yminus || face == Uintah::Patch::yplus) {
      return true;
    } else {
      return false;
    }
  }

  template<>
  bool is_perpendicular<SpatialOps::ZDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::zminus || face == Uintah::Patch::zplus) {
      return true;
    } else {
      return false;
    }
  }

  //====================================================================
  // Typedef the normal direction based on DirT
  template< typename DirT>
  struct NormalDirTypeSelector
  {
  public:
    typedef typename SpatialOps::SSurfXField   NormalDirT;
  };

  template<>
  struct NormalDirTypeSelector<SpatialOps::XDIR>
  {
    typedef typename SpatialOps::SSurfXField   NormalDirT;
  };
  
  template<>
  struct NormalDirTypeSelector<SpatialOps::YDIR>
  {
    typedef typename SpatialOps::SSurfYField   NormalDirT;
  };

  template<>
  struct NormalDirTypeSelector<SpatialOps::ZDIR>
  {
    typedef typename SpatialOps::SSurfZField   NormalDirT;
  };

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
                      const Expr::Tag& mixMWTag )
    : Expr::Expression<FieldT>(),
      gasConstant_( 8314.459848 )  // gas constant J/(kmol K)
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
    public:
      /**
       *  @brief Build a IdealGasPressure expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& densityTag,
               const Expr::Tag& temperatureTag,
               const Expr::Tag& mixMWTag,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        densityTag_( densityTag ),
        temperatureTag_( temperatureTag ),
        mixMWTag_( mixMWTag )
      {}
      
      Expr::ExpressionBase* build() const{
        return new IdealGasPressure<FieldT>( densityTag_, temperatureTag_, mixMWTag_ );
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
  class Density_IC : public Expr::Expression<FieldT>
  {
    const double gasConstant_;
    DECLARE_FIELDS( FieldT, temperature_, pressure_, mixMW_ )
    
    Density_IC( const Expr::Tag& temperatureTag,
                const Expr::Tag& pressureTag,
                const Expr::Tag& mixMWTag )
    : Expr::Expression<FieldT>(),
      gasConstant_( 8314.459848 ) // gas constant J/(kmol K)
    {
      this->set_gpu_runnable(true);
      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      pressure_    = this->template create_field_request<FieldT>( pressureTag    );
      mixMW_       = this->template create_field_request<FieldT>( mixMWTag       );
    }
    
  public:
    
    class Builder : public Expr::ExpressionBuilder
    {
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
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        temperatureTag_( temperatureTag ),
        pressureTag_   ( pressureTag    ),
        mixMWTag_      ( mixMWTag       )
      {}
      
      Expr::ExpressionBase* build() const{
        return new Density_IC<FieldT>( temperatureTag_,pressureTag_,mixMWTag_ );
      }
    };  /* end of Builder class */
    
    ~Density_IC(){}
    
    void evaluate(){
      this->value() <<= ( pressure_->field_ref() * mixMW_->field_ref() )/( gasConstant_ * temperature_->field_ref() );
    }
  };
  
  //----------------------------------------------------------------------------
  
  Expr::ExpressionID
  ContinuityTransportEquation::initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    typedef Density_IC<MyFieldT>::Builder DensIC;
    return exprFactory.register_expression( scinew DensIC( initial_condition_tag(),
                                                           temperatureTag_,
                                                           TagNames::self().pressure,
                                                           mixMWTag_ ) );
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
                                         const Expr::Tag e0Tag, // total internal energy tag
                                         const Expr::Tag bodyForceTag,
                                         const Expr::Tag srcTermTag,
                                         GraphCategories& gc,
                                         Uintah::ProblemSpecP params,
                                         TurbulenceParameters turbParams )
  : MomentumTransportEquationBase<SVolField>( momComponent,
                                              velName,
                                              momName,
                                              densityTag,
                                              false,
                                              bodyForceTag,
                                              srcTermTag,
                                              gc,
                                              params,
                                              turbParams ),
   temperatureTag_(temperatureTag),
   mixMWTag_(mixMWTag),
   e0Tag_(e0Tag)
  {
    // todo:
    //  - strain tensor       // registered by MomentumTransportEquationBase
    //  - convective flux     // registered by the MomentumTransportEquationBase
    //  - turbulent viscosity // SHOULD be registered by the MomentumTransportEquationBase. NOT READY YET.
    //  - buoyancy? //        // body forces are handled in the momentumequationbase

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    typedef IdealGasPressure<FieldT>::Builder Pressure;
    if( !factory.have_entry(TagNames::self().pressure) ){
      const Expr::ExpressionID pid = factory.register_expression( scinew Pressure( TagNames::self().pressure,
                                                                                   densityTag,
                                                                                   temperatureTag,
                                                                                   mixMWTag ) );
      factory.cleave_from_parents( pid );
    }
    setup();
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  Expr::ExpressionID
  CompressibleMomentumTransportEquation<MomDirT>::
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

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    //
    // Add dummy modifiers on all patches. This is used to inject new dependencies across all patches.
    // Those new dependencies result for example from complicated boundary conditions added in this
    // function. NOTE: whenever you want to add a new complex boundary condition, please use this
    // functionality to inject new dependencies across patches.
    //
    {
      // add dt dummy modifier for outflow bcs...
      bcHelper.create_dummy_dependency<SpatialOps::SingleValueField, FieldT>(rhs_part_tag(this->solnVarName_), tag_list(tagNames.dt), ADVANCE_SOLUTION);
      
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
    int jobid = 0;

    // we will need the speed of sound
    if( !( advSlnFactory.have_entry(tagNames.soundspeed) ) ){
      typedef typename NSCBC::SpeedOfSound<SVolField>::Builder SoundSpeed;
      advSlnFactory.register_expression(
                                        new SoundSpeed( tagNames.soundspeed, this->pressureTag_, this->densityTag_, tagNames.cp, tagNames.cv )
                                        );
    }

    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() )
    {
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;
      
      const bool isNormal = is_normal_to_boundary(this->staggered_location(), myBndSpec.face);

      //============================================================================================
      // NSCBC TREATMENT
      std::map< NSCBC::TagName, Expr::Tag > tags;
      std::map< NSCBC::TagListName, Expr::TagList > tagLists;
      {
        int normVelIndex = 0; // velocity component normal to this boundary face u_i. This will be picked from velTags_[]
        switch (myBndSpec.face) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::xplus:
            normVelIndex = 0;
            break;
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
            normVelIndex = 1;
            break;
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
            normVelIndex = 2;
            break;
          default:
            break;
        }
        
        // we will need dudx, dvdy, and dwdz. Register here if they were not registered
        const Expr::TagList dVelTags = tag_list(tagNames.dudx, tagNames.dvdy, tagNames.dwdz);
        if( !( advSlnFactory.have_entry(tagNames.dudx) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::XDIR>::Builder dudx;
          advSlnFactory.register_expression(new dudx( tagNames.dudx, this->velTags_[0] ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dvdy) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::YDIR>::Builder dvdy;
          advSlnFactory.register_expression(new dvdy( tagNames.dvdy, this->velTags_[1] ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dwdz) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::ZDIR>::Builder dwdz;
          advSlnFactory.register_expression(new dwdz( tagNames.dwdz, this->velTags_[2] ));
        }
        
        // we will also need dpdx, dpdy, and dpdz. Register here if they were not registered
        const Expr::TagList dPTags = tag_list(tagNames.dpdx, tagNames.dpdy,tagNames.dpdz);
        if( !( advSlnFactory.have_entry(tagNames.dpdx) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::XDIR>::Builder dpdx;
          advSlnFactory.register_expression(new dpdx( tagNames.dpdx, this->pressureTag_ ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dpdy) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::YDIR>::Builder dpdy;
          advSlnFactory.register_expression(new dpdy( tagNames.dpdy, this->pressureTag_ ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dpdz) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::ZDIR>::Builder dpdz;
          advSlnFactory.register_expression(new dpdz( tagNames.dpdz, this->pressureTag_ ));
        }
        
        // now fill in the tas for the NSCBC tag manager
        tags[NSCBC::U]       = this->velTags_[0];
        tags[NSCBC::V]       = this->velTags_[1];
        tags[NSCBC::W]       = this->velTags_[2];
        tags[NSCBC::FACEVEL] = this->velTags_[normVelIndex];
        tags[NSCBC::DVELHARDINFLOW]    = dVelTags[normVelIndex];
        tags[NSCBC::T]       = this->temperatureTag_;
        tags[NSCBC::P]       = this->pressureTag_;
        tags[NSCBC::DPHARDINFLOW]   = dPTags[normVelIndex];
        tags[NSCBC::RHO]     = this->densityTag_;
        tags[NSCBC::MMW]     = this->mixMWTag_;
        tags[NSCBC::CP]      = tagNames.cp;
        tags[NSCBC::CV]      = tagNames.cv;
        tags[NSCBC::C]       = tagNames.soundspeed;
        tags[NSCBC::E0]      = this->e0Tag_;
        tagLists[NSCBC::H_N]     = tag_list(tagNames.enthalpy);
        
        // create the NSCBC tag manager
        NSCBC::TagManager nscbcTagMgr( tags, tagLists, false );
        
        bcHelper.setup_nscbc<MomDirT>(myBndSpec, nscbcTagMgr, jobid++);
      }
      //============================================================================================

      // variable density: add bccopiers on all boundaries
      if( !this->is_constant_density() ){
        // if we are solving a variable density problem, then set bcs on density estimate rho*
        const Expr::Tag rhoStarTag = tagNames.make_star(this->densityTag_); // get the tagname of rho*
        // check if this boundary applies a bc on the density
        if( myBndSpec.has_field(this->densityTag_.name()) ){
          // create a bc copier for the density estimate
          const Expr::Tag rhoStarBCTag( rhoStarTag.name() + "_" + bndName + "_bccopier", Expr::STATE_NONE);
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
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(),"none" ,0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          
          // set zero velocity
          BndCondSpec velBCSpec = {this->thisVelTag_.name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, velBCSpec);
          
          // set zero momentum RHS on the wall
          BndCondSpec momRHSBCSpec = {this->rhs_tag().name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momRHSBCSpec);

          break;
        }
        case VELOCITY: // we will use this as the hard inflow
        {
          // set zero momentum RHS at inlets
          BndCondSpec momRHSBCSpec = {this->rhs_tag().name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momRHSBCSpec);
          break;
        }
        case OUTFLOW:  // we will use OUTFLOW/OPEN as NONREFLECTING
        case OPEN:
        {          
          typedef typename SpatialOps::UnitTriplet<MomDirT>::type UnitTripletT;
          
          //Create the one sided stencil gradient
          Expr::Tag convModTag;
          Expr::Tag pModTag;
          
          Expr::ExpressionBuilder* builder1 = NULL;
          Expr::ExpressionBuilder* builder2 = NULL;
          
          switch (myBndSpec.face) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::yplus:
            case Uintah::Patch::zplus:
            {
              if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<typename UnitTripletT::Negate>,FieldT>::type OpT;
                
                convModTag = Expr::Tag(this->solnVarName_ + "_partial_rhs_mod_plus_side_" + bndName, Expr::STATE_NONE);
                typedef typename BCOneSidedConvFluxDiv<FieldT,OpT>::Builder builderT;
                builder1 = new builderT( convModTag, thisVelTag_, mom_tag(this->solnVarName_));
                
                pModTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_plus_side_" + bndName, Expr::STATE_NONE);
                typedef typename WasatchCore::BCOneSidedGradP<FieldT,OpT>::Builder pbuilderT;
                builder2 = new pbuilderT( pModTag, this->pressureTag_ );
              }
              break;
            }
            case Uintah::Patch::xminus:
            case Uintah::Patch::yminus:
            case Uintah::Patch::zminus:
            {
              if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<UnitTripletT>,FieldT>::type OpT;
                
                convModTag = Expr::Tag(this->solnVarName_ + "_partial_rhs_mod_minus_side_" + bndName, Expr::STATE_NONE);
                typedef typename WasatchCore::BCOneSidedConvFluxDiv<FieldT,OpT>::Builder builderT;
                builder1 = new builderT( convModTag, thisVelTag_, mom_tag(this->solnVarName_) );
                
                pModTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_minus_side_" + bndName, Expr::STATE_NONE);
                typedef typename WasatchCore::BCOneSidedGradP<FieldT,OpT>::Builder pbuilderT;
                builder2 = new pbuilderT( pModTag, this->pressureTag_ );
              }
              break;
            }
            default:
              break;
          }
          
          if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
            advSlnFactory.register_expression(builder1);
            BndCondSpec rhsConvFluxSpec = {this->rhs_tag().name() + "_partial", convModTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsConvFluxSpec);

            advSlnFactory.register_expression(builder2);
            BndCondSpec rhsGradPSpec = {this->rhs_tag().name(), pModTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsGradPSpec);
          }
          
          // construct an appropriate convective flux in the boundary by using one-sided Neumann interpolation
          Expr::Tag normalConvModTag;
          Expr::Tag normalStrainModTag;
          Expr::Tag momModTag;
          Expr::Tag pModTag2;
          Expr::Tag rhsModTag;
          
          switch (myBndSpec.face) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              normalConvModTag = Expr::Tag( this->normalConvFluxTag_.name() + "_" + Expr::context2str(this->normalConvFluxTag_.context()) + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
              normalStrainModTag = Expr::Tag( this->normalStrainTag_.name() + "_" + Expr::context2str(this->normalStrainTag_.context()) + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
              momModTag = Expr::Tag( this->solnVarName_ + "_" + Expr::context2str(this->solnVarTag_.context()) + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
              pModTag2 = Expr::Tag( this->pressureTag_.name() + "_" + Expr::context2str(this->pressureTag_.context()) + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
              
              typedef OpTypes<FieldT> Ops;
              typedef typename Ops::InterpC2FX   DirichletT;
              typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientX, SVolField, SVolField >::type NeumannT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,DiriOpT>::Builder constBCDirichletT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,NeumOpT>::Builder constBCNeumannT;
              
              // for normal fluxes
              typedef typename NormalDirTypeSelector<MomDirT>::NormalDirT FluxT;
              typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
              typedef typename ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
              
              if (!initFactory.have_entry(momModTag)) initFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              
              if (!advSlnFactory.have_entry(normalConvModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalConvModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(normalStrainModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalStrainModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(momModTag)) advSlnFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(pModTag2)) advSlnFactory.register_expression( new constBCNeumannT( pModTag2, 0.0 ) );
              break;
            }

            case Uintah::Patch::yplus:
            case Uintah::Patch::yminus:
            {
              normalConvModTag = Expr::Tag( this->normalConvFluxTag_.name() + "_" + Expr::context2str(this->normalConvFluxTag_.context()) + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
              normalStrainModTag = Expr::Tag( this->normalStrainTag_.name() + "_" + Expr::context2str(this->normalStrainTag_.context()) + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
              momModTag = Expr::Tag( this->solnVarName_ + "_" + Expr::context2str(this->solnVarTag_.context()) + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
              pModTag2 = Expr::Tag( this->pressureTag_.name() + "_" + Expr::context2str(this->pressureTag_.context()) + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
              
              typedef OpTypes<FieldT> Ops;
              typedef typename Ops::InterpC2FY   DirichletT;
              typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientY, SVolField, SVolField >::type NeumannT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,DiriOpT>::Builder constBCDirichletT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,NeumOpT>::Builder constBCNeumannT;
              
              // for normal fluxes
              typedef typename NormalDirTypeSelector<MomDirT>::NormalDirT FluxT;
              typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
              typedef typename WasatchCore::ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
              
              if (!initFactory.have_entry(momModTag)) initFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              
              if (!advSlnFactory.have_entry(normalConvModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalConvModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(normalStrainModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalStrainModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(momModTag)) advSlnFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(pModTag2)) advSlnFactory.register_expression( new constBCNeumannT( pModTag2, 0.0 ) );
              break;
            }

            case Uintah::Patch::zplus:
            case Uintah::Patch::zminus:
            {
              normalConvModTag = Expr::Tag( this->normalConvFluxTag_.name() + "_" + Expr::context2str(this->normalConvFluxTag_.context()) + "_bc_" + myBndSpec.name + "_Zdirbc", Expr::STATE_NONE );
              normalStrainModTag = Expr::Tag( this->normalStrainTag_.name() + "_" + Expr::context2str(this->normalStrainTag_.context()) + "_bc_" + myBndSpec.name + "_zdirbc", Expr::STATE_NONE );
              momModTag = Expr::Tag( this->solnVarName_ + "_" + Expr::context2str(this->solnVarTag_.context()) + "_bc_" + myBndSpec.name + "_zdirbc", Expr::STATE_NONE );
              pModTag2 = Expr::Tag( this->pressureTag_.name() + "_" + Expr::context2str(this->pressureTag_.context()) + "_bc_" + myBndSpec.name + "_zdirbc", Expr::STATE_NONE );
              
              typedef OpTypes<FieldT> Ops;
              typedef typename Ops::InterpC2FZ   DirichletT;
              typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientZ, SVolField, SVolField >::type NeumannT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,DiriOpT>::Builder constBCDirichletT;
              typedef typename WasatchCore::ConstantBCNew<FieldT,NeumOpT>::Builder constBCNeumannT;
              
              // for normal fluxes
              typedef typename NormalDirTypeSelector<MomDirT>::NormalDirT FluxT;
              typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
              typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
              typedef typename WasatchCore::ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
              
              if (!initFactory.have_entry(momModTag)) initFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              
              if (!advSlnFactory.have_entry(normalConvModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalConvModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(normalStrainModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( normalStrainModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(momModTag)) advSlnFactory.register_expression( new constBCNeumannT( momModTag, 0.0 ) );
              if (!advSlnFactory.have_entry(pModTag2)) advSlnFactory.register_expression( new constBCNeumannT( pModTag2, 0.0 ) );
              break;
            }

            default:
              break;
          }
          if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
            BndCondSpec convFluxBCSpec = {this->normalConvFluxTag_.name(),normalConvModTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, convFluxBCSpec);
            
            BndCondSpec stressBCSpec = {this->normalStrainTag_.name(),normalStrainModTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, stressBCSpec);
          }

          // set Neumann = 0 on momentum
          BndCondSpec momBCSpec = {this->solnVarName_, momModTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);

          // set Neumann = 0 on the pressure
          BndCondSpec pBCSpec = {this->pressureTag_.name(), pModTag2.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
          bcHelper.add_boundary_condition(bndName, pBCSpec);
          
          // If we need a boundary condition for the rhs
//          BndCondSpec rhsZeroSpec = {this->rhs_tag().name(), "none", 0.0, DIRICHLET, FUNCTOR_TYPE};
//          bcHelper.add_boundary_condition(bndName, rhsZeroSpec);

          break;
        }
        case USER:
        default:
        {
          std::ostringstream msg;
          msg << "ERROR: VELOCITY, OPEN, and OUTFLOW boundary conditions are not currently supported for compressible flows in Wasatch. " << bndName
              << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      } // SWITCH BOUNDARY TYPE
    } // BOUNDARY LOOP
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
    
    // apply velocity boundary condition, if specified
    bcHelper.apply_boundary_condition<FieldT>(this->thisVelTag_, taskCat, true);
    
    // tsaad: boundary conditions will not be applied on the initial condition of momentum. This leads
    // to tremendous complications in our graphs. Instead, specify velocity initial conditions
    // and velocity boundary conditions, and momentum bcs will appropriately propagate.
    bcHelper.apply_boundary_condition<FieldT>(this->initial_condition_tag(), taskCat, true);
    
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
  
  //----------------------------------------------------------------------------

  template<typename MomDirT> NSCBC::TransportVal NSCBCMomentum();
  
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::XDIR>(){return NSCBC::MOMENTUM_X;}
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::YDIR>(){return NSCBC::MOMENTUM_Y;}
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::ZDIR>(){return NSCBC::MOMENTUM_Z;}
  
  //----------------------------------------------------------------------------
  
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
    // set bcs for convective flux
    bcHelper.apply_boundary_condition< typename NormalDirTypeSelector<MomDirT>::NormalDirT >( this->normalConvFluxTag_, taskCat);
    // set bcs for strain
    bcHelper.apply_boundary_condition< typename NormalDirTypeSelector<MomDirT>::NormalDirT >( this->normalStrainTag_, taskCat);
    // apply NSCBC boundary conditions
    bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBCMomentum<MomDirT>(), taskCat);
  }

  //============================================================================
  
  template class CompressibleMomentumTransportEquation<SpatialOps::XDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::YDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::ZDIR>;
  template struct NormalDirTypeSelector<SpatialOps::XDIR>;
  template struct NormalDirTypeSelector<SpatialOps::YDIR>;
  template struct NormalDirTypeSelector<SpatialOps::ZDIR>;
} // namespace Wasatch



