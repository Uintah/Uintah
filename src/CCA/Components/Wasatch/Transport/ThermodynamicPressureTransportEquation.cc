/*
 * The MIT License
 *
 * Copyright (c) 2012-2023 The University of Utah
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

#include <CCA/Components/Wasatch/Transport/ThermodynamicPressureTransportEquation.h>

#include <expression/ExprLib.h>
//-- SpatialOps includes --//
#include <spatialops/structured/IndexTriplet.h>
//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/VelocityDotScalarGradient.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/Gradient.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/LowMachPressureDriftCorrection.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace WasatchCore{

 typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for pressure transport

 template<typename FaceT, typename VelT>
 void
 register_grad_p_expression( Expr::ExpressionFactory& factory,
                             const Expr::Tag&         gradPTag,
                             const Expr::Tag&         momRHSTag,
                             const Expr::Tag&         pTag )
 {
   if(momRHSTag != Expr::Tag())
   {
     typedef typename Gradient<FaceT, VelT>::Builder Grad;

     factory.register_expression( scinew Grad(gradPTag, pTag) );
   }
 }


 //------------------------------------------------------------------

 template<typename FaceT, typename VelT>
 void
 register_vel_dot_grad_p_expression( Expr::ExpressionFactory& factory,
                                     const Expr::Tag&         velDotGradPTag,
                                     const Expr::Tag&         velocityTag,
                                     const Expr::Tag&         pTag,
                                     const Expr::Tag&         pRHSTag )
 {
   if(velocityTag != Expr::Tag())
   {
     typedef typename VelocityDotScalarGradient<FaceT, VelT>::Builder VelDotGradP;

     factory.register_expression( scinew VelDotGradP(velDotGradPTag,
                                                     pTag,
                                                     velocityTag) );

     factory.attach_dependency_to_expression(velDotGradPTag,
                                             pRHSTag,
                                             Expr::SUBTRACT_SOURCE_EXPRESSION);
   }
 }

 //------------------------------------------------------------------


 // explicit template instantiation
#define INSTANTIATE_GRAD_P_EXPRS( FACET, VELT )         \
\
template void \
register_grad_p_expression<FACET, VELT>(  \
Expr::ExpressionFactory& factory,                       \
const Expr::Tag&         gradPTag,                      \
const Expr::Tag&         momRHSTag,                     \
const Expr::Tag&         pTag );                        \
\
template void \
register_vel_dot_grad_p_expression<FACET, VELT>(  \
Expr::ExpressionFactory& factory,                       \
const Expr::Tag&         velDotGrapPTag,                \
const Expr::Tag&         velocityTag,                   \
const Expr::Tag&         pTag,                          \
const Expr::Tag&         pRHSTag );

 INSTANTIATE_GRAD_P_EXPRS( SpatialOps::FaceTypes<FieldT>::XFace, SpatialOps::XVolField )
 INSTANTIATE_GRAD_P_EXPRS( SpatialOps::FaceTypes<FieldT>::YFace, SpatialOps::YVolField )
 INSTANTIATE_GRAD_P_EXPRS( SpatialOps::FaceTypes<FieldT>::ZFace, SpatialOps::ZVolField )

  //------------------------------------------------------------------

  ThermodymamicPressureTransportEquation::
  ThermodymamicPressureTransportEquation( Uintah::ProblemSpecP   wasatchSpec,
                                          GraphCategories&       gc,
                                          std::set<std::string>& persistentFields )
    : WasatchCore::TransportEquation( gc, TagNames::self().thermodynamicPressure.name(), NODIR ),
      wasatchSpec_     ( wasatchSpec      ),
      persistentFields_( persistentFields )
  {

    // Ensure enthalpy and species are being transported

    for( Uintah::ProblemSpecP transEqnParams=wasatchSpec->findBlock("TransportEquation");
         transEqnParams != nullptr;
         transEqnParams=transEqnParams->findNextBlock("TransportEquation") )
    {
      std::string eqnLabel;
      transEqnParams->getAttribute( "equation", eqnLabel );
      if( eqnLabel == "enthalpy" ) enthalpyParams_ = transEqnParams;
    }

     Uintah::ProblemSpecP speciesParams = wasatchSpec->findBlock("SpeciesTransportEquations");

    if( (!enthalpyParams_) || (!speciesParams) ){
      std::ostringstream msg;
      msg << "There was a problem setting up equation for '" << solnVarName_ << "'. Check your input file"
          << "and ensure a TransportEquation equation=\"enthalpy\"' block and a 'SpeciesTransportEquations'"
          << "block exist."
          << std::endl;
      throw Uintah::ProblemSetupException(msg.str(), __FILE__, __LINE__ );
    }

    this->initial_condition(*gc[INITIALIZATION]->exprFactory);
    setup();
  }

  Expr::ExpressionID
  ThermodymamicPressureTransportEquation::
  setup_rhs( FieldTagInfo&        info,
             const Expr::TagList& srcTags )
  {
    Expr::ExpressionFactory& solnFactory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    Expr::ExpressionFactory& initFactory = *gc_[INITIALIZATION  ]->exprFactory;

    const TagNames& tagNames = TagNames::self();

    // set rhs to pressure material derivative
    const Expr::Tag pMaterialDerivativeTag = tagNames.DPDt;

    typedef ExprAlgebra<FieldT> ExprAlgbr;
    Expr::ExpressionID rhsID =
    solnFactory.register_expression( scinew typename ExprAlgbr::
                                                     Builder( rhsTag_,
                                                              Expr::tag_list(pMaterialDerivativeTag),
                                                              ExprAlgbr::SUM ) );
    //retrieve momentum names;
    std::string xMomName, yMomName, zMomName;
    Uintah::ProblemSpecP momEqnParams = wasatchSpec_->findBlock("MomentumEquations");

    const bool doXMom = momEqnParams->get( "X-Momentum", xMomName );
    const bool doYMom = momEqnParams->get( "Y-Momentum", yMomName );
    const bool doZMom = momEqnParams->get( "Z-Momentum", zMomName );

    const Expr::Tag xMomRHSTag = doXMom ? Expr::Tag(xMomName + "_rhs", Expr::STATE_NONE) : Expr::Tag();
    const Expr::Tag yMomRHSTag = doYMom ? Expr::Tag(yMomName + "_rhs", Expr::STATE_NONE) : Expr::Tag();
    const Expr::Tag zMomRHSTag = doZMom ? Expr::Tag(zMomName + "_rhs", Expr::STATE_NONE) : Expr::Tag();

    Expr::TagList rhsTags;
    std::string dir;

    const std::string gradPName       = "gradient_"          + this->solution_variable_name();
    const std::string velDotGradPName = "velocity_dot_grad_" + this->solution_variable_name();

    Expr::Tag xVelTag;
    Expr::Tag yVelTag;
    Expr::Tag zVelTag;

    // build expressions to calculate u dot grad(P)
    for( Uintah::ProblemSpecP convFluxParams=enthalpyParams_->findBlock("ConvectiveFlux");
             convFluxParams != nullptr;
             convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") ){

      convFluxParams->getAttribute("direction",dir);

      Expr::Tag advVelocityTag;

      // get the tag for the advective velocity
      Uintah::ProblemSpecP advVelocityParam = convFluxParams->findBlock( "AdvectiveVelocity" );
      if( advVelocityParam ){
        advVelocityTag = parse_nametag( advVelocityParam->findBlock( "NameTag" ) );
      }

      const bool advectionEnabled = advVelocityTag != Expr::Tag();
      bool wrongDirSpecified = true;

      const Expr::Tag gradPTag      (gradPName       + "_" + dir, Expr::STATE_NONE);
      const Expr::Tag velDotGrapPTag(velDotGradPName + "_" + dir, Expr::STATE_NONE);
      //should a flux limiter be used when calculating u dot grad(P) ?

      if     ( dir == "X" ){
        wrongDirSpecified = false;

        xVelTag = advVelocityTag;

        typedef SpatialOps::FaceTypes<FieldT>::XFace XFaceT;
        typedef SpatialOps::XVolField XVolT;

        register_vel_dot_grad_p_expression<XFaceT, XVolT>( solnFactory,
                                                           velDotGrapPTag,
                                                           advVelocityTag,
                                                           solnVarTag_,
                                                           rhsTag_ );

        register_grad_p_expression        <XFaceT, XVolT>( solnFactory,
                                                           gradPTag,
                                                           xMomRHSTag,
                                                           solnVarTag_ );
      }
      else if( dir == "Y" ){
        wrongDirSpecified = false;

        yVelTag = advVelocityTag;

        typedef SpatialOps::FaceTypes<FieldT>::YFace YFaceT;
        typedef SpatialOps::YVolField YVolT;

        register_vel_dot_grad_p_expression<YFaceT, YVolT>( solnFactory,
                                                           velDotGrapPTag,
                                                           advVelocityTag,
                                                           solnVarTag_,
                                                           rhsTag_ );

        register_grad_p_expression        <YFaceT, YVolT>( solnFactory,
                                                           gradPTag,
                                                           xMomRHSTag,
                                                           solnVarTag_ );
      }
      else if( (dir == "Z") && advectionEnabled ){
        wrongDirSpecified = false;

        zVelTag = advVelocityTag;

        typedef SpatialOps::FaceTypes<FieldT>::ZFace ZFaceT;
        typedef SpatialOps::ZVolField ZVolT;

        register_vel_dot_grad_p_expression<ZFaceT, ZVolT>( solnFactory,
                                                           velDotGrapPTag,
                                                           advVelocityTag,
                                                           solnVarTag_,
                                                           rhsTag_ );

        register_grad_p_expression        <ZFaceT, ZVolT>( solnFactory,
                                                           gradPTag,
                                                           xMomRHSTag,
                                                           solnVarTag_ );
      }
      else if(wrongDirSpecified){
        std::ostringstream msg;
        msg << "Invalid direction selection detected constructing setting RHS of" << solnVarName_ << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    const Expr::Tag divuPDriftCorrectionTag( "divu_pressure_drift_correction", Expr::STATE_NONE);

    typedef typename LowMachPressureDriftCorrection<SVolField>::Builder PDriftCorrection;

    solnFactory.register_expression( scinew PDriftCorrection( divuPDriftCorrectionTag,
                                                              solnVarNP1Tag_,
                                                              tagNames.backgroundPressure,
                                                              tagNames.adiabaticIndex,
                                                              xVelTag,
                                                              yVelTag,
                                                              zVelTag,
                                                              tagNames.dt) );

    const Expr::Tag solnVarInitTag = this->initial_condition_tag();
    initFactory.register_expression( scinew PDriftCorrection( divuPDriftCorrectionTag,
                                                              solnVarInitTag,
                                                              tagNames.backgroundPressure,
                                                              tagNames.adiabaticIndex,
                                                              xVelTag,
                                                              yVelTag,
                                                              zVelTag,
                                                              tagNames.dt) );

    solnFactory.attach_dependency_to_expression( divuPDriftCorrectionTag,
                                                 tagNames.divu );

    initFactory.attach_dependency_to_expression( divuPDriftCorrectionTag,
                                                 tagNames.divu );

//    solnFactory.cleave_from_children(solnFactory.get_id(this->solnVarNP1Tag_));
    return rhsID;
  }

  //------------------------------------------------------------------

  void ThermodymamicPressureTransportEquation::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
   
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    //const Expr::Tag phiTag( this->solution_variable_name(), Expr::STATE_N );
    
    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ) {

      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList = tag_list( vNames.vol_frac_tag<SVolField>() );
      const Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE);
      factory.register_expression( new typename ExprAlgebra<FieldT>::Builder(modifierTag,
                                                                   theTagList,
                                                                   ExprAlgbr::PRODUCT,
                                                                   true) );
      
      factory.attach_modifier_expression( modifierTag, initial_condition_tag() );
    }
    
    if( factory.have_entry(initial_condition_tag()) ){
      bcHelper.apply_boundary_condition<FieldT>( initial_condition_tag(), taskCat );
    }
  }

  //------------------------------------------------------------------

  void ThermodymamicPressureTransportEquation::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;
      
      if (!isConstDensity_) {
        // for variable density problems, we must ALWAYS guarantee proper boundary conditions for
        // rhof_{n+1}. Since we apply bcs on rhof at the bottom of the graph, we can't apply
        // the same bcs on rhof (time advanced). Hence, we set rhof_rhs to zero always :)
        if( !myBndSpec.has_field(rhs_name()) ){ // if nothing has been specified for the RHS
          const BndCondSpec rhsBCSpec = {rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
          bcHelper.add_boundary_condition(bndName, rhsBCSpec);
        }
      }

      switch ( myBndSpec.type ){
        case WALL:
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

  //------------------------------------------------------------------
  
  void ThermodymamicPressureTransportEquation::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    // Should we always apply BCs to STATE_NP1 or STATE_N? Does it even matter?
    const Category taskCat = ADVANCE_SOLUTION;
    const Expr::Tag solnVarTag = flowTreatment_==LOWMACH ? solnvar_np1_tag() : solution_variable_tag();
    bcHelper.apply_boundary_condition<FieldT>( solnVarTag, taskCat );
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell
  }

  //------------------------------------------------------------------

  Expr::ExpressionID
  ThermodymamicPressureTransportEquation::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    // set the initial condition based on the background pressure
    const Expr::Tag& bgpTag = TagNames::self().backgroundPressure;

    if(!icFactory.have_entry(bgpTag)){
      std::ostringstream msg;
                    msg << "ERROR: "
                        <<" Transport equation for'"
                        <<this->solnVarName_
                        <<"' requires registration of an expression for a field with name '"
                        << bgpTag.name()
                        << "' and context '"
                        << Expr::context2str(bgpTag.context())
                        << "' for tasklist =  'initialization' and 'advance_solution' \n";
                        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    typedef ExprAlgebra<FieldT>::Builder Algebra;
    return icFactory.register_expression( scinew Algebra( initial_condition_tag(),
                                                          tag_list( TagNames::self().backgroundPressure ),
                                                          ExprAlgebra<FieldT>::SUM) );
  }

  //------------------------------------------------------------------

  //==================================================================

} // namespace WasatchCore
