/*
 * The MIT License
 *
 * Copyright (c) 2015-2018 The University of Utah
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

#include <sci_defs/wasatch_defs.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

#ifdef HAVE_POKITT
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/thermo/Density.h>
#endif

namespace WasatchCore{



  template< typename FaceT, typename GradT >
  struct ContinuityBoundaryTyper
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
  //----------------------------------------------------------------------------

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
    double gasConstant_;
    DECLARE_FIELDS( FieldT, temperature_, pressure_, mixMW_ )

    Density_IC( const Expr::Tag& temperatureTag,
                const Expr::Tag& pressureTag,
                const Expr::Tag& mixMWTag,
                Uintah::ProblemSpecP wasatchSpec )
    : Expr::Expression<FieldT>(),
      gasConstant_( 8314.459848 ) // gas constant J/(kmol K)
    {
      this->set_gpu_runnable(true);
      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      pressure_    = this->template create_field_request<FieldT>( pressureTag    );
      mixMW_       = this->template create_field_request<FieldT>( mixMWTag       );

#ifdef HAVE_POKITT
      Uintah::ProblemSpecP speciesParams = wasatchSpec->findBlock("SpeciesTransportEquations");
      if( speciesParams ){
        gasConstant_ = CanteraObjects::gas_constant();
      }
#endif
    }

  public:

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag temperatureTag_, pressureTag_, mixMWTag_;
      Uintah::ProblemSpecP wasatchSpec_;
    public:
      /**
       *  @brief Build a Density_IC expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& temperatureTag,
               const Expr::Tag& pressureTag,
               const Expr::Tag& mixMWTag,
               Uintah::ProblemSpecP wasatchSpec,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        temperatureTag_( temperatureTag ),
        pressureTag_   ( pressureTag    ),
        mixMWTag_      ( mixMWTag       ),
        wasatchSpec_( wasatchSpec )
      {}

      Expr::ExpressionBase* build() const{
        return new Density_IC<FieldT>( temperatureTag_,pressureTag_,mixMWTag_,wasatchSpec_ );
      }
    };  /* end of Builder class */

    ~Density_IC(){}

    void evaluate(){
      this->value() <<= ( pressure_->field_ref() * mixMW_->field_ref() )/( gasConstant_ * temperature_->field_ref() );
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
    typedef SpatialOps::SVolField  MyFieldT;
    
    const Expr::Tag xVelTag_, yVelTag_, zVelTag_;
    const Expr::Tag densTag_, temperatureTag_, pressureTag_, mixMWTag_;
    Uintah::ProblemSpecP wasatchSpec_;
  public:
    ContinuityTransportEquation( const Expr::Tag densityTag,
                                 const Expr::Tag temperatureTag,
                                 const Expr::Tag mixMWTag,
                                 GraphCategories& gc,
                                 const Expr::Tag xvel,
                                 const Expr::Tag yvel,
                                 const Expr::Tag zvel,
                                 Uintah::ProblemSpecP wasatchSpec )
    : TransportEquation( gc, densityTag.name(), NODIR /* variable density */ ),
      xVelTag_( xvel ),
      yVelTag_( yvel ),
      zVelTag_( zvel ),
      densTag_       ( densityTag     ),
      temperatureTag_( temperatureTag ),
      mixMWTag_      ( mixMWTag       ),
      wasatchSpec_   ( wasatchSpec    )
    {
      setup();
    }

    //----------------------------------------------------------------------------
    
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                   GraphCategories& graphCat )
    {
      Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
      Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);

      // set up the species mass density field that will be computed with primitives on which BCs have been set, for nonreflecting inflow boundaries
      const Expr::Tag temporaryRhoTag( "temporary_rho_for_bcs", Expr::STATE_NONE ); // requires the rho temporary field that continuity must build
      const Expr::Tag temporaryPTag( "temporary_pressure_for_bcs", Expr::STATE_NONE );
      const Expr::Tag temporaryTemperatureTag( "temporary_Temperature_for_bcs", Expr::STATE_NONE );
      const Expr::Tag temporaryMMWTag( "temporary_mmw_for_bcs", Expr::STATE_NONE );

      if( !( advSlnFactory.have_entry( temporaryRhoTag ) ) ){
#ifdef HAVE_POKITT
        typedef pokitt::Density<SVolField>::Builder Density;
        advSlnFactory.register_expression( new Density( temporaryRhoTag, temporaryTemperatureTag, temporaryPTag, temporaryMMWTag ) );
#else
        typedef Density_IC<SVolField>::Builder Density;
        advSlnFactory.register_expression( new Density( temporaryRhoTag, temporaryTemperatureTag, temporaryPTag, temporaryMMWTag, wasatchSpec_ ) );
#endif
      }
      if( !( initFactory.have_entry( temporaryRhoTag ) ) ){
#ifdef HAVE_POKITT
        typedef pokitt::Density<SVolField>::Builder Density;
        initFactory.register_expression( new Density( temporaryRhoTag, temporaryTemperatureTag, temporaryPTag, temporaryMMWTag ) );
#else
        typedef Density_IC<SVolField>::Builder Density;
        initFactory.register_expression( new Density( temporaryRhoTag, temporaryTemperatureTag, temporaryPTag, temporaryMMWTag, wasatchSpec_ ) );
#endif
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

          case OUTFLOW:
          case OPEN:{
            // tags for modifier expressions and strings for BC spec names, will depend on boundary face
            std::string normalConvectiveFluxName;
            Expr::Tag neumannZeroConvectiveFluxTag,
                      neumannZeroDensityTag,
                      bcCopiedRhoTag;

            // build boundary conditions for x, y, and z faces
            switch( myBndSpec.face ){
              case Uintah::Patch::xplus:
              case Uintah::Patch::xminus:
              {
                std::string dir = "X";
                typedef ContinuityBoundaryTyper<SpatialOps::SSurfXField, SpatialOps::GradientX> BCTypes;

                normalConvectiveFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;

                neumannZeroConvectiveFluxTag = get_decorated_tag( this->solnVarName_      , "neumann-zero", dir );
                neumannZeroDensityTag        = get_decorated_tag( normalConvectiveFluxName, "neumann-zero", dir );

                if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
                if( !advSlnFactory.have_entry( neumannZeroDensityTag        ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
                if( !initFactory  .have_entry( neumannZeroDensityTag        ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
              }
              break;
              case Uintah::Patch::yminus:
              case Uintah::Patch::yplus:
              {
                std::string dir = "Y";
                typedef ContinuityBoundaryTyper<SpatialOps::SSurfYField, SpatialOps::GradientY> BCTypes;

                normalConvectiveFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;

                neumannZeroConvectiveFluxTag = get_decorated_tag( this->solnVarName_      , "neumann-zero", dir );
                neumannZeroDensityTag        = get_decorated_tag( normalConvectiveFluxName, "neumann-zero", dir );

                if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
                if( !advSlnFactory.have_entry( neumannZeroDensityTag        ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
                if( !initFactory  .have_entry( neumannZeroDensityTag        ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
              }
              break;
              case Uintah::Patch::zminus:
              case Uintah::Patch::zplus:
              {
                std::string dir = "Z";
                typedef ContinuityBoundaryTyper<SpatialOps::SSurfZField, SpatialOps::GradientZ> BCTypes;

                normalConvectiveFluxName = this->solnVarName_ + TagNames::self().convectiveflux + dir;

                neumannZeroConvectiveFluxTag = get_decorated_tag( this->solnVarName_      , "neumann-zero", dir );
                neumannZeroDensityTag        = get_decorated_tag( normalConvectiveFluxName, "neumann-zero", dir );

                if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
                if( !advSlnFactory.have_entry( neumannZeroDensityTag        ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
                if( !initFactory  .have_entry( neumannZeroDensityTag        ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroDensityTag       , 0.0 ) );
              }
              break;
              default:
                break;
            }

            // make boundary condition specifications, connecting the name of the field (first) and the name of the modifier expression (second)
            BndCondSpec convFluxSpec = {normalConvectiveFluxName, neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            BndCondSpec densitySpec  = {this->solnVarTag_.name(), neumannZeroDensityTag.name()       , 0.0, NEUMANN, FUNCTOR_TYPE};

            // add boundary condition specifications to this boundary
            bcHelper.add_boundary_condition( bndName, convFluxSpec );
            bcHelper.add_boundary_condition( bndName, densitySpec );
          }
          break;

          case VELOCITY:
          {
            Expr::Tag bcCopiedRhoTag;

            // build boundary conditions for x, y, and z faces
            switch( myBndSpec.face ){
              case Uintah::Patch::xplus:
              case Uintah::Patch::xminus:
              {
                std::string dir = "X";
                typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
                bcCopiedRhoTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
                if( !initFactory  .have_entry( bcCopiedRhoTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
                if( !advSlnFactory.have_entry( bcCopiedRhoTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
              }
              break;
              case Uintah::Patch::yminus:
              case Uintah::Patch::yplus:
              {
                std::string dir = "Y";
                typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
                bcCopiedRhoTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
                if( !initFactory  .have_entry( bcCopiedRhoTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
                if( !advSlnFactory.have_entry( bcCopiedRhoTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
              }
              break;
              case Uintah::Patch::zminus:
              case Uintah::Patch::zplus:
              {
                std::string dir = "Z";
                typedef typename BCCopier<SVolField>::Builder CopiedCellBC;
                bcCopiedRhoTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
                if( !initFactory  .have_entry( bcCopiedRhoTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
                if( !advSlnFactory.have_entry( bcCopiedRhoTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoTag, temporaryRhoTag ) );
              }
              break;
              default:
                break;
            }

            // make boundary condition specifications, connecting the name of the field (first) and the name of the modifier expression (second)
            BndCondSpec rhoDirichletBC = {this->solnVarTag_.name(), bcCopiedRhoTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};

            // add boundary condition specifications to this boundary
            bcHelper.add_boundary_condition( bndName, rhoDirichletBC );
          }
          break;

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

      // bcs for hard inflow - set primitive and conserved variables
      const Expr::Tag temporaryYTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
      bcHelper.apply_boundary_condition<MyFieldT>( temporaryYTag, taskCat, true );
    }
    
    //----------------------------------------------------------------------------
    
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                   WasatchBCHelper& bcHelper )
    {
      const Category taskCat = ADVANCE_SOLUTION;

      bcHelper.apply_boundary_condition<MyFieldT>( this->solnvar_np1_tag(), taskCat );

      bcHelper.apply_boundary_condition<MyFieldT>( this->rhs_tag(), taskCat, true );
      
      std::string convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "X";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfXField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );
      convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "Y";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfYField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );
      convFluxName = this->solnVarName_ + TagNames::self().convectiveflux + "Z";
      bcHelper.apply_boundary_condition<SpatialOps::SSurfZField>( Expr::Tag(convFluxName, Expr::STATE_NONE), taskCat );

      bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBC::DENSITY, taskCat);

      // bcs for hard inflow - set primitive and conserved variables
      const Expr::Tag temporaryYTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
      bcHelper.apply_boundary_condition<MyFieldT>( temporaryYTag, taskCat, true );
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
    CompressibleMomentumTransportEquation( Uintah::ProblemSpecP wasatchSpec,
                                           const Direction momComponent,
                                           const std::string velName,
                                           const std::string momName,
                                           const Expr::Tag densityTag,
                                           const Expr::Tag temperatureTag,
                                           const Expr::Tag mixMWTag,
                                           const Expr::Tag e0Tag, // total internal energy tag
                                           const Expr::Tag bodyForceTag,
                                           const Expr::TagList& srcTermTags,
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

    Uintah::ProblemSpecP wasatchSpec_;
    const Expr::Tag temperatureTag_, mixMWTag_, e0Tag_;
    void setup_diffusive_flux( FieldTagInfo& ){}
    void setup_convective_flux( FieldTagInfo& ){}
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){}
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags );

  };

}
