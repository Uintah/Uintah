/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <CCA/Components/Wasatch/Transport/LowMachMomentumTransportEquation.h>

// -- Uintah includes --//
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Solvers/HypreSolver.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/OldVariable.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

#include <CCA/Components/Wasatch/Expressions/Strain.h>
#include <CCA/Components/Wasatch/Expressions/Dilatation.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/TimeDerivative.h>
#include <CCA/Components/Wasatch/Expressions/MomentumPartialRHS.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorBase.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/DynamicSmagorinskyCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OutflowBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OpenBC.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/PressureSource.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/ContinuityResidual.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/KineticEnergy.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
//-- ExprLib Includes --//
#include <expression/ExprLib.h>

using std::string;

namespace WasatchCore{


  //==================================================================

  template< typename FieldT >
  LowMachMomentumTransportEquation<FieldT>::
  LowMachMomentumTransportEquation(const Direction momComponent,
                                   const std::string velName,
                             const std::string momName,
                             const Expr::Tag densTag,
                             const Expr::Tag bodyForceTag,
                             const Expr::TagList& srcTermTags,
                             GraphCategories& gc,
                             Uintah::ProblemSpecP params,
                             TurbulenceParameters turbulenceParams,
                             Uintah::SolverInterface& linSolver,
                             Uintah::MaterialManagerP materialManager)
    : MomentumTransportEquationBase<FieldT>(momComponent,
                                            velName,
                         momName,
                         densTag,
                         bodyForceTag,
                         srcTermTags,
                         gc,
                         params,
                         turbulenceParams)
  {
    std::string xmomname, ymomname, zmomname; // these are needed to construct fx, fy, and fz for pressure RHS
    bool doMom[3];
    doMom[0] = params->get( "X-Momentum", xmomname );
    doMom[1] = params->get( "Y-Momentum", ymomname );
    doMom[2] = params->get( "Z-Momentum", zmomname );

    const bool isConstDensity = this->isConstDensity_;
    GraphHelper& graphHelper   = *(gc[ADVANCE_SOLUTION  ]);
    Expr::ExpressionFactory& factory = *(graphHelper.exprFactory);

    const TagNames& tagNames = TagNames::self();
    
    const bool enablePressureSolve = !(params->findBlock("DisablePressureSolve"));
    const EmbeddedGeometryHelper& embedGeom = EmbeddedGeometryHelper::self();
    //__________________
    // Pressure source term        
    if( !factory.have_entry( tagNames.pressuresrc ) ){
      const Expr::Tag densNP1Tag  = Expr::Tag(densTag.name(), Expr::STATE_NP1);
      
      // register the expression for pressure source term
      Expr::TagList psrcTagList;
      psrcTagList.push_back(tagNames.pressuresrc);
      if( !isConstDensity ) {
        psrcTagList.push_back(tagNames.drhodt );
      }

      // create an expression for divu. In the case of variable density flows, the scalar equations
      // will add their contributions to this expression
      if( !factory.have_entry( tagNames.divu ) ) {
        typedef typename Expr::ConstantExpr<SVolField>::Builder divuBuilder;
        factory.register_expression( new divuBuilder(tagNames.divu, 0.0));
      }

      const Expr::ExpressionID
      psrcID = factory.register_expression( new typename PressureSource::Builder( psrcTagList,
                                                                                  this->momTags_,
                                                                                  this->oldMomTags_,
                                                                                  this->velTags_,
                                                                                  tagNames.divu,
                                                                                  isConstDensity,
                                                                                  densTag,
                                                                                  densNP1Tag ) );
      
      factory.cleave_from_parents( psrcID  );
      factory.cleave_from_children( psrcID  );
    }
    
    
    //__________________
    // pressure
    if( enablePressureSolve ){
      if( !factory.have_entry( this->pressureTag_ ) ){
        Uintah::ProblemSpecP pressureParams = params->findBlock( "Pressure" );
        
        bool usePressureRefPoint = false;
        double refPressureValue = 0.0;
        Uintah::IntVector refPressureLocation(0,0,0);
        if (pressureParams->findBlock("ReferencePressure")) {
          usePressureRefPoint = true;
          Uintah::ProblemSpecP refPressureParams = pressureParams->findBlock("ReferencePressure");
          refPressureParams->getAttribute("value", refPressureValue);
          refPressureParams->get("ReferenceCell", refPressureLocation);
        }

        bool enforceSolvability = pressureParams->findBlock("EnforceSolvability") ? true : false;

        bool use3DLaplacian = true;
        pressureParams->getWithDefault("Use3DLaplacian", use3DLaplacian, true);
        
        // ALAS, we cannot throw an error here because setupFrequency is parsed using getWithDefault
        // which means it will be specified in the input.xml file that is generated by uintah...
        if (pressureParams->findBlock("Parameters")->findBlock("setupFrequency")) {
          std::ostringstream msg;
          msg << "WARNING: Wasatch does NOT allow specification of setupFrequency for the pressure solver. "
          << "The setupFrequency will be determined by Wasatch."
          << std::endl;
          std::cout << msg.str();
          //throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        linSolver.readParameters( pressureParams, "" );
        linSolver.getParameters()->setSolveOnExtraCells( false );
        linSolver.getParameters()->setUseStencil4( false );
        linSolver.getParameters()->setSymmetric( this->is_constant_density() );
        linSolver.getParameters()->setOutputFileName( "WASATCH" );
        
        // matrix update in hypre: If we have a moving geometry, then update every timestep.
        // Otherwise, no update is needed since the coefficient matrix is constant
        const bool updateCoefFreq = ( !isConstDensity || embedGeom.has_moving_geometry() ) ? 1 : 0;
        linSolver.getParameters()->setSetupFrequency( 0 ); // matrix Sparsity will never change.
        linSolver.getParameters()->setUpdateCoefFrequency( updateCoefFreq ); // coefficients may change if we have variable density or moving geometries
        // if pressure expression has not be registered, then register it
        Expr::Tag fxt, fyt, fzt;
        if( doMom[0] )  fxt = rhs_part_tag( xmomname );
        if( doMom[1] )  fyt = rhs_part_tag( ymomname );
        if( doMom[2] )  fzt = rhs_part_tag( zmomname );

        Expr::TagList ptags;
        ptags.push_back( this->pressureTag_ );
        ptags.push_back( Expr::Tag( this->pressureTag_.name() + "_rhs", this->pressureTag_.context() ) );
        const Expr::Tag densityTag = isConstDensity ? this->densityTag_ : Expr::Tag(this->densityTag_.name(), Expr::STATE_NP1);
        
        Expr::ExpressionBuilder* pbuilder = new typename Pressure::Builder( ptags, fxt, fyt, fzt,
                                                                            tagNames.pressuresrc, tagNames.dt, embedGeom.vol_frac_tag<SVolField>(), densityTag,
                                                                            embedGeom.has_moving_geometry(), usePressureRefPoint, refPressureValue,
                                                                            refPressureLocation, use3DLaplacian,
                                                                            enforceSolvability, this->is_constant_density(),
                                                                            linSolver);
        this->pressureID_ = factory.register_expression( pbuilder );
        factory.cleave_from_children( this->pressureID_ );
        factory.cleave_from_parents ( this->pressureID_ );
      }
      else{
        this->pressureID_ = factory.get_id( this->pressureTag_ );
      }
    } else if( factory.have_entry( this->pressureTag_ ) ) {
      this->pressureID_ = factory.get_id( this->pressureTag_ );
    }
    
    this->setup();
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  LowMachMomentumTransportEquation<FieldT>::
  ~LowMachMomentumTransportEquation()
  {}

  //-----------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID  LowMachMomentumTransportEquation<FieldT>::
  setup_rhs( FieldTagInfo&,
             const Expr::TagList& srcTags )
  {
    const bool enablePressureSolve = !(this->params_->findBlock("DisablePressureSolve"));

    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    Expr::Tag volFracTag = vNames.vol_frac_tag<FieldT>();

    Expr::ExpressionFactory& factory = *this->gc_[ADVANCE_SOLUTION]->exprFactory;
    
    typedef typename MomRHS<FieldT, SpatialOps::NODIR>::Builder RHS;
    return factory.register_expression( scinew RHS( this->rhsTag_,
                                                    ( (enablePressureSolve || factory.have_entry( this->pressureTag_ )) ? this->pressureTag_ : Expr::Tag()),
                                                    rhs_part_tag(this->solnVarTag_),
                                                    volFracTag ) );
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void LowMachMomentumTransportEquation<FieldT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper, GraphCategories& graphCat )
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
        const Expr::Tag densityNP1Tag = Expr::Tag(this->densityTag_.name(), Expr::STATE_NP1); // get the tagname of rho*
        // check if this boundary applies a bc on the density
        if( myBndSpec.has_field(this->densityTag_.name()) ){
          // create a bc copier for the density estimate
          const Expr::Tag rhoStarBCTag( densityNP1Tag.name() + bndName +"_bccopier", Expr::STATE_NONE);
          BndCondSpec rhoStarBCSpec = {densityNP1Tag.name(), rhoStarBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
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

          BndCondSpec momBCSpec = {this->solution_variable_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          
          BndCondSpec velBCSpec = {this->thisVelTag_.name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, velBCSpec);          

          if( isNormal ){
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(mom_tag(this->solnVarName_))).name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
            BndCondSpec rhsFullBCSpec = {this->rhs_name(), "none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }

          break;
        }
        case VELOCITY:
        {
          if( !myBndSpec.has_field(this->thisVelTag_.name()) && !myBndSpec.has_field(this->solnVarName_) ) {
            // tsaad: If this VELOCITY boundary does NOT have this velocity AND does not have this momentum specified
            // then assume that they are zero and create boundary conditions for them accordingly
            BndCondSpec velBCSPec = {this->thisVelTag_.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, velBCSPec);
            BndCondSpec momBCSPec = {this->solnVarName_, "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, momBCSPec);
          } else if( myBndSpec.has_field(this->thisVelTag_.name()) && myBndSpec.has_field(this->solnVarName_) ) {
            // tsaad: If this VELOCITY boundary has both VELOCITY and MOMENTUM specified, then
            // throw an error.
            std::ostringstream msg;
            msg << "ERROR: You cannot specify both velocity and momentum boundary conditions at a VELOCITY boundary. "
            << "This error occured while trying to analyze boundary " << bndName
            << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          } else if( myBndSpec.has_field(this->thisVelTag_.name()) && !myBndSpec.has_field(this->solnVarName_) ) {
            // tsaad: If this VELOCITY boundary has ONLY velocity specified, then infer momentum bc
            const Expr::Tag momBCTag( this->solnVarName_ + "_bc_primvar_" + bndName, Expr::STATE_NONE);
            advSlnFactory.register_expression ( new typename BCPrimVar<FieldT>::Builder(momBCTag, this->thisVelTag_, this->densityTag_) );
            
            if( initFactory.have_entry(this->thisVelTag_) ){
              const Expr::Tag densityStateNone(this->densityTag_.name(), Expr::STATE_NONE);
              initFactory.register_expression ( new typename BCPrimVar<FieldT>::Builder(momBCTag, this->thisVelTag_, densityStateNone) );
            }

            BndCondSpec momBCSPec = {this->solnVarName_, momBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, momBCSPec);            
          }

          if( isNormal ){
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(mom_tag(this->solnVarName_))).name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
            BndCondSpec rhsFullBCSpec = {this->rhs_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          
          break;
        }
        case OUTFLOW:
        {
          if( isNormal ){
            // register outflow functor for this boundary. we'll register one functor per boundary
            const Expr::Tag outBCTag(bndName + "_outflow_bc", Expr::STATE_NONE);
            typedef typename OutflowBC<FieldT>::Builder Builder;
            //bcHelper.register_functor_expression( scinew Builder( outBCTag, this->thisVelTag_ ), ADVANCE_SOLUTION );
            advSlnFactory.register_expression( scinew Builder( outBCTag, this->solution_variable_tag() ) );
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(this->solution_variable_tag())).name(),outBCTag.name(), 0.0, DIRICHLET,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
            
          }
          else {
            BndCondSpec rhsFullBCSpec = {this->rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          
          // after the correction has been made, update the momentum and velocities in the extra cells using simple Neumann conditions
          BndCondSpec momBCSpec = {this->solnVarName_, "none", 0.0, NEUMANN, DOUBLE_TYPE};
          BndCondSpec velBCSpec = {this->thisVelTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          bcHelper.add_boundary_condition(bndName, velBCSpec);

          // Set the pressure to Dirichlet 0 (atmospheric conditions)
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          break;
        }
        case OPEN:
        {
          if( isNormal ){
            // register pressurebc functor for this boundary. we'll register one functor per boundary
            const Expr::Tag openBCTag(bndName + "_open_bc", Expr::STATE_NONE);
            typedef typename OpenBC<FieldT>::Builder Builder;
            advSlnFactory.register_expression( scinew Builder( openBCTag, this->solution_variable_tag() ) );
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(this->solution_variable_tag())).name(),openBCTag.name(), 0.0, DIRICHLET,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsPartBCSpec);
          }
          else {
            BndCondSpec rhsFullBCSpec = {this->rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }

          // after the correction has been made, update the momentum and velocities in the extra cells using simple Neumann conditions
          BndCondSpec momBCSpec = {this->solnVarName_, "none", 0.0, NEUMANN, DOUBLE_TYPE};
          BndCondSpec velBCSpec = {this->thisVelTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);
          bcHelper.add_boundary_condition(bndName, velBCSpec);

          // Set the pressure to Dirichlet 0 (atmospheric conditions)
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          break;
        }
        case SYMMETRY:
        {
          if( isNormal ){
            BndCondSpec momBCSpec = {this->solnVarName_, "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            BndCondSpec velBCSpec = {this->thisVelTag_.name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, momBCSpec);
            bcHelper.add_boundary_condition(bndName, velBCSpec);
            
            BndCondSpec rhsFullBCSpec = {this->rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
            
            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(this->solution_variable_tag())).name(),"none", 0.0, DIRICHLET,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          else {
            
            BndCondSpec momBCSpec = {this->solnVarName_, "none", 0.0, NEUMANN, DOUBLE_TYPE};
            BndCondSpec velBCSpec = {this->thisVelTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, momBCSpec);
            bcHelper.add_boundary_condition(bndName, velBCSpec);

            BndCondSpec rhsFullBCSpec = {this->rhs_name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);

            BndCondSpec rhsPartBCSpec = {(rhs_part_tag(this->solution_variable_tag())).name(),"none", 0.0, NEUMANN,FUNCTOR_TYPE};
            bcHelper.add_boundary_condition(bndName, rhsFullBCSpec);
          }
          
          // Set the pressure to Neumann 0
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(), "none", 0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
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
  
  //==================================================================
  
  template< typename FieldT >
  void LowMachMomentumTransportEquation<FieldT>::
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
    }
  }

  //==================================================================
  
  template< typename FieldT >
  void LowMachMomentumTransportEquation<FieldT>::
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
    // set bcs for partial full rhs - apply directly on the boundary
    bcHelper.apply_boundary_condition<FieldT>( this->rhs_tag(), taskCat, true);

    if( !this->is_constant_density() ){
      const TagNames& tagNames = TagNames::self();

      // set bcs for density
      const Expr::Tag densityNP1Tag( this->densityTag_.name(), Expr::STATE_NP1 );
      bcHelper.apply_boundary_condition<SVolField>( densityNP1Tag, taskCat );
    }
  }

  //==================================================================

  template< typename FieldT >
  Expr::ExpressionID
  LowMachMomentumTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    // register an initial condition for da pressure
    if( !icFactory.have_entry( this->pressureTag_ ) ) {
      icFactory.register_expression( new typename Expr::ConstantExpr<SVolField>::Builder( TagNames::self().pressure, 0.0 ) );
    }
    
    if( icFactory.have_entry( this->thisVelTag_ ) ) {
      typedef typename InterpolateExpression<SVolField, FieldT>::Builder Builder;
      Expr::Tag interpolatedDensityTag(this->densityTag_.name() +"_interp_" + this->dir_name(), Expr::STATE_NONE);
      icFactory.register_expression(scinew Builder(interpolatedDensityTag, Expr::Tag(this->densityTag_.name(),Expr::STATE_NONE)));
      
      // register expression to calculate the momentum initial condition from the initial conditions on
      // velocity and density in the cases that we are initializing velocity in the input file
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList( tag_list( this->thisVelTag_, interpolatedDensityTag ) );
      icFactory.register_expression( new typename ExprAlgbr::Builder( this->initial_condition_tag(),
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
      icFactory.register_expression( new typename ExprAlgbr::Builder( modifierTag,
                                                                      theTagList,
                                                                      ExprAlgbr::PRODUCT,
                                                                      true ) );
      icFactory.attach_modifier_expression( modifierTag, this->initial_condition_tag() );
    }
    return icFactory.get_id( this->initial_condition_tag() );
  }


  //------------------------------------------------------------------

  //==================================================================  
  // Explicit template instantiation
  template class LowMachMomentumTransportEquation< XVolField >;
  template class LowMachMomentumTransportEquation< YVolField >;
  template class LowMachMomentumTransportEquation< ZVolField >;
  //==================================================================

} // namespace WasatchCore
