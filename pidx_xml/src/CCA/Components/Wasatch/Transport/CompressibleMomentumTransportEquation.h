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
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionsOneSided.h>

#include <CCA/Components/Wasatch/TagNames.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>


namespace WasatchCore{

  
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
    typedef SpatialOps::SVolField  MyFieldT;
    
    const Expr::Tag xVelTag_, yVelTag_, zVelTag_;
    const Expr::Tag densTag_, temperatureTag_, pressureTag_, mixMWTag_;
  public:
    ContinuityTransportEquation( const Expr::Tag densityTag,
                                 const Expr::Tag temperatureTag,
                                 const Expr::Tag mixMWTag,
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
      mixMWTag_      ( mixMWTag       )
    {
      setup();
    }

    //----------------------------------------------------------------------------
    
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                   GraphCategories& graphCat )
    {
      Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
      Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);

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
          {}
            break;
          case OUTFLOW:
          case OPEN:{
            // for constant density problems, on all types of boundary conditions, set the scalar rhs
            // to zero. The variable density case requires setting the scalar rhs to zero ALL the time
            // and is handled in the code above.
            if( isConstDensity_ ) {
              if( myBndSpec.has_field(rhs_name()) ) {
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
            
            Expr::Tag rhsModTag;
            Expr::ExpressionBuilder* builder1 = NULL;
            typedef typename SpatialOps::UnitTriplet<SpatialOps::XDIR>::type UnitTripletT;

            Expr::Tag normalVelTag;
            switch (myBndSpec.face) {
              case Uintah::Patch::xplus: case Uintah::Patch::xminus: normalVelTag = xVelTag_; break;
              case Uintah::Patch::yplus: case Uintah::Patch::yminus: normalVelTag = yVelTag_; break;
              case Uintah::Patch::zplus: case Uintah::Patch::zminus: normalVelTag = zVelTag_; break;
              default: break;
            }
            
            
            switch (myBndSpec.face) {
              case Uintah::Patch::xplus:
              case Uintah::Patch::yplus:
              case Uintah::Patch::zplus:
              {
                rhsModTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_plus_side_" + bndName, Expr::STATE_NONE);
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<typename UnitTripletT::Negate>,MyFieldT>::type OpT;
                typedef typename BCOneSidedConvFluxDiv<MyFieldT,OpT>::Builder builderT;
                builder1 = new builderT( rhsModTag, normalVelTag, this->densTag_ );
                break;
              }
              case Uintah::Patch::xminus:
              case Uintah::Patch::yminus:
              case Uintah::Patch::zminus:
              {
                rhsModTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_minus_side_" + bndName, Expr::STATE_NONE);
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<UnitTripletT>,MyFieldT>::type OpT;
                typedef typename BCOneSidedConvFluxDiv<MyFieldT,OpT>::Builder builderT;
                builder1 = new builderT( rhsModTag, normalVelTag, this->densTag_ );
                break;
              }
              default:
                break;
            }
            advSlnFactory.register_expression(builder1);
            
            //-----------------------------------------------
            // Use Neumann zero on the normal convective fluxes
            std::string normalConvFluxName;
            Expr::Tag convModTag, rhoModTag;
            switch (myBndSpec.face) {
              case Uintah::Patch::xplus:
              case Uintah::Patch::xminus:
              {
                std::string dir = "X";
                normalConvFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;
                
                convModTag = Expr::Tag( normalConvFluxName + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
                rhoModTag = Expr::Tag( this->solnVarName_ + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_xdirbc", Expr::STATE_NONE );
                
                typedef OpTypes<MyFieldT> Ops;
                typedef typename Ops::InterpC2FX   DirichletT;
                typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientX, SVolField, SVolField >::type NeumannT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
                typedef typename ConstantBCNew<MyFieldT,NeumOpT>::Builder constBCNeumannT;
                
                // for normal fluxes
                typedef typename SpatialOps::SSurfXField FluxT;
                typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
                typedef typename ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
                
                if (!advSlnFactory.have_entry(convModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( convModTag, 0.0 ) );
                if (!initFactory  .have_entry(rhoModTag )) initFactory  .register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                if (!advSlnFactory.have_entry(rhoModTag )) advSlnFactory.register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                break;
              }
              case Uintah::Patch::yminus:
              case Uintah::Patch::yplus:
              {
                std::string dir = "Y";
                normalConvFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;

                convModTag = Expr::Tag( normalConvFluxName + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
                rhoModTag = Expr::Tag( this->solnVarName_ + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_ydirbc", Expr::STATE_NONE );
                
                typedef OpTypes<MyFieldT> Ops;
                typedef typename Ops::InterpC2FY   DirichletT;
                typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientY, SVolField, SVolField >::type NeumannT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
                typedef typename ConstantBCNew<MyFieldT,NeumOpT>::Builder constBCNeumannT;
                
                // for normal fluxes
                typedef typename SpatialOps::SSurfYField FluxT;
                typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
                typedef typename ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
                
                if (!advSlnFactory.have_entry(convModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( convModTag, 0.0 ) );
                if (!initFactory.have_entry(rhoModTag)) initFactory.register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                if (!advSlnFactory.have_entry(rhoModTag)) advSlnFactory.register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                break;
              }
              case Uintah::Patch::zminus:
              case Uintah::Patch::zplus:
              {
                std::string dir = "Z";
                normalConvFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;

                convModTag = Expr::Tag( normalConvFluxName + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_zdirbc", Expr::STATE_NONE );
                rhoModTag = Expr::Tag( this->solnVarName_ + "_STATE_NONE" + "_bc_" + myBndSpec.name + "_zdirbc", Expr::STATE_NONE );
                
                typedef OpTypes<MyFieldT> Ops;
                typedef typename Ops::InterpC2FZ   DirichletT;
                typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::GradientZ, SVolField, SVolField >::type NeumannT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<DirichletT> DiriOpT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannT> NeumOpT;
                typedef typename ConstantBCNew<MyFieldT,NeumOpT>::Builder constBCNeumannT;
                
                // for normal fluxes
                typedef typename SpatialOps::SSurfZField FluxT;
                typedef typename SpatialOps::OperatorTypeBuilder<Divergence,  FluxT, SpatialOps::SVolField >::type NeumannFluxT;
                typedef typename SpatialOps::NeboBoundaryConditionBuilder<NeumannFluxT> NeumFluxOpT;
                typedef typename ConstantBCNew<FluxT, NeumFluxOpT>::Builder constBCNeumannFluxT;
                
                if (!advSlnFactory.have_entry(convModTag)) advSlnFactory.register_expression( new constBCNeumannFluxT( convModTag, 0.0 ) );
                if (!initFactory.have_entry(rhoModTag)) initFactory.register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                if (!advSlnFactory.have_entry(rhoModTag)) advSlnFactory.register_expression( new constBCNeumannT( rhoModTag, 0.0 ) );
                break;
              }
              default:
                break;
            }
            BndCondSpec convFluxSpec = {normalConvFluxName, convModTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, convFluxSpec);
            //-----------------------------------------------
            
            // correct the convective flux using a 1-sided stencil. do this directly on the RHS.
            BndCondSpec rhsConvFluxSpec = {this->rhs_tag().name(), rhsModTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsConvFluxSpec);

            // set Neumann 0 on density
            BndCondSpec rhoBCSpec = {this->solnVarName_, rhoModTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhoBCSpec);
            
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
      bcHelper.apply_boundary_condition<MyFieldT>(this->initial_condition_tag(), taskCat);
      
    }
    
    //----------------------------------------------------------------------------
    
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                   WasatchBCHelper& bcHelper )
    {
      const Category taskCat = ADVANCE_SOLUTION;
      
      // set bcs for density at np1 - use the TIMEADVANCE expression
      bcHelper.apply_boundary_condition<MyFieldT>( this->solnvar_np1_tag(), taskCat );
      
      // set bcs for density convective flux
      std::string convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "X";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );

      convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "Y";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );
      
      convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "Z";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );

      // set bcs for density RHS
      bcHelper.apply_boundary_condition<MyFieldT>( this->rhs_tag(), taskCat );

      // set nscbcs for density
      bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBC::DENSITY, taskCat);
    }
    
    void setup_diffusive_flux( FieldTagInfo& rhsInfo ){}
    void setup_source_terms  ( FieldTagInfo& rhsInfo, Expr::TagList& srcTags ){}

    //----------------------------------------------------------------------------
    
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags )
    {
      
      typedef ScalarRHS<MyFieldT>::Builder RHSBuilder;
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      
      info[PRIMITIVE_VARIABLE] = solnVarTag_;
      
      return factory.register_expression( scinew RHSBuilder( rhsTag_, info, Expr::TagList(), densTag_, false, true, Expr::Tag() ) );
    }

    //----------------------------------------------------------------------------
    
    void setup_convective_flux( FieldTagInfo& rhsInfo )
    {
      if( xVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<MyFieldT>( "X", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 xVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
      if( yVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<MyFieldT>( "Y", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 yVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
      if( zVelTag_ != Expr::Tag() )
        setup_convective_flux_expression<MyFieldT>( "Z", densTag_,
                                                 Expr::Tag(), /* default tag name for conv. flux */
                                                 CENTRAL,
                                                 zVelTag_,
                                                 *gc_[ADVANCE_SOLUTION]->exprFactory,
                                                 rhsInfo );
    }

    //----------------------------------------------------------------------------
    
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& exprFactory );
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
  class CompressibleMomentumTransportEquation : public MomentumTransportEquationBase<SVolField>
  {
    typedef SpatialOps::SVolField FieldT;

  public:
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

    const Expr::Tag temperatureTag_, mixMWTag_, e0Tag_;
    void setup_diffusive_flux( FieldTagInfo& ){}
    void setup_convective_flux( FieldTagInfo& ){}
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){}
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags );

  };

}



