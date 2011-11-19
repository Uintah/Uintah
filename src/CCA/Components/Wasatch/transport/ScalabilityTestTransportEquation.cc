//-- Wasatch includes --//
#include "ScalabilityTestTransportEquation.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalabilityTestSrc.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace Wasatch{
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void setup_diffusive_velocity_expression(std::string dir,  
                                       std::string thisPhiName,
                              Expr::ExpressionFactory& factory,
                               typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    
    Expr::Tag diffFluxTag;  // we will populate this.
    
    diffFluxTag = Expr::Tag( thisPhiName + "_diffFlux_" + dir, Expr::STATE_NONE );
    const Expr::Tag phiTag( thisPhiName, Expr::STATE_N );

    Expr::ExpressionBuilder* builder = NULL;
    
    if( dir=="X" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradX>::Builder Flux;
      builder = scinew Flux( phiTag, 1.0 );
    } else if( dir=="Y" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradY>::Builder Flux;
      builder = scinew Flux( phiTag, 1.0 );
    } else if( dir=="Z" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradZ>::Builder Flux;
      builder = scinew Flux( phiTag, 1.0 );
    }
    
    if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive velocity expression for '" << thisPhiName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
    factory.register_expression( diffFluxTag, builder );
      
    typename ScalarRHS<FieldT>::FieldSelector fs;
    if     ( dir=="X" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_X;
    else if( dir=="Y" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Y;
    else if( dir=="Z" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for diffusive flux expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    info[ fs ] = diffFluxTag;
  }
    
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void setup_convective_flux_expression(std::string dir,  
                                       std::string thisPhiName,
                                        Expr::Tag advVelocityTag,
                                       Expr::ExpressionFactory& factory,
                                       typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> Ops;
    
    Expr::Tag convFluxTag;

    if (advVelocityTag == Expr::Tag()) {
      std::ostringstream msg;
      msg << "ERROR: no advective velocity set for transport equation '" << thisPhiName << "'" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
        
      convFluxTag = Expr::Tag( thisPhiName + "_convective_flux_" + dir, Expr::STATE_NONE );
      const Expr::Tag phiTag( thisPhiName, Expr::STATE_N );
      Expr::ExpressionBuilder* builder = NULL;
      
      if( dir=="X" ){
        proc0cout << "SETTING UP X-CONVECTIVE-FLUX EXPRESSION USING CENTRAL DIFFERENCING"  << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,XVolField,typename FaceTypes<FieldT>::XFace>::type VelInterpOpT;                            
            typedef typename ConvectiveFlux< typename Ops::InterpC2FX, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP Y-CONVECTIVE-FLUX EXPRESSION USING CENTRAL DIFFERENCING"  << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,YVolField,typename FaceTypes<FieldT>::YFace>::type VelInterpOpT;
        
            typedef typename ConvectiveFlux< typename Ops::InterpC2FY, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            
      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP Z-CONVECTIVE-FLUX EXPRESSION USING CENTRAL DIFFERENCING"  << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,ZVolField,typename FaceTypes<FieldT>::ZFace>::type VelInterpOpT;
        
            typedef typename ConvectiveFlux< typename Ops::InterpC2FZ, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            
      }
      
      if( builder == NULL ){        
        std::ostringstream msg;
        msg << "ERROR: Could not build a convective flux expression for '" << thisPhiName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        
      }
      
      factory.register_expression( convFluxTag, builder );
    
    
    typename ScalarRHS<FieldT>::FieldSelector fs;
    if      ( dir=="X" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_X;
    else if ( dir=="Y" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_Y;
    else if ( dir=="Z" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for convective flux expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    info[ fs ] = convFluxTag;
  }
  
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::
  ScalabilityTestTransportEquation( const std::string basePhiName, 
                                   const std::string thisPhiName,
                          const Expr::ExpressionID rhsID )
  : Wasatch::TransportEquation( thisPhiName, rhsID,
                                get_staggered_location<FieldT>() )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::~ScalabilityTestTransportEquation()
  {}
  
  //------------------------------------------------------------------  
  
  template< typename FieldT >  
  void ScalabilityTestTransportEquation<FieldT>::
  setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                         const Uintah::PatchSet* const localPatches,
                                         const PatchInfoMap& patchInfoMap,
                                         const Uintah::MaterialSubset* const materials)
  {    
  }  
  
  //------------------------------------------------------------------
  
  template< typename FieldT >  
  void ScalabilityTestTransportEquation<FieldT>::setup_boundary_conditions(const GraphHelper& graphHelper,
                                                                  const Uintah::PatchSet* const localPatches,
                                                                  const PatchInfoMap& patchInfoMap,
                                                                  const Uintah::MaterialSubset* const materials)
  {
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionID
  ScalabilityTestTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_registry().get_id( Expr::Tag( this->solution_variable_name(),
                                                      Expr::STATE_N ) );
  }
    
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionID
  ScalabilityTestTransportEquation<FieldT>::
  get_rhs_expr_id(std::string thisPhiName, 
                  Expr::ExpressionFactory& factory, 
                  Uintah::ProblemSpecP params )
  {
    typename ScalarRHS<FieldT>::FieldTagInfo info;
        
    //_________________
    // Diffusive Velocities
    bool doDiffusion = true;
    params->get( "DoDiffusion", doDiffusion);
    if (doDiffusion) {
      setup_diffusive_velocity_expression<FieldT>( "X", thisPhiName, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Y", thisPhiName, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Z", thisPhiName, factory, info );      
    }
    //__________________
    // Convective Fluxes
    bool doConvection = true;
    params->get( "DoConvection", doConvection);
    if (doConvection) {
      
      Expr::Tag advVelTag = parse_nametag( params->findBlock("X-Velocity" )->findBlock( "NameTag" ) );
      setup_convective_flux_expression<FieldT>( "X", thisPhiName, advVelTag, factory, info );
      
      advVelTag = parse_nametag( params->findBlock("Y-Velocity" )->findBlock( "NameTag" ) );
      setup_convective_flux_expression<FieldT>( "Y", thisPhiName, advVelTag, factory, info );

      advVelTag = parse_nametag( params->findBlock("Z-Velocity" )->findBlock( "NameTag" ) );
      setup_convective_flux_expression<FieldT>( "Z", thisPhiName, advVelTag, factory, info );            
    }
    //_____________
    // Source Terms
    std::vector<Expr::Tag> srcTags;
    bool doSrc = true;
    params->get( "DoSourceTerm", doSrc);
    if (doSrc) {
      
      int nEqs=0;
      params->get( "NumberOfEquations", nEqs );
      
      std::string basePhiName;
      params->get( "SolutionVariable", basePhiName );
      const Expr::Tag basePhiTag ( basePhiName, Expr::STATE_N );
      
      params->get( "SolutionVariable", basePhiName );
      const Expr::Tag thisPhiTag ( thisPhiName, Expr::STATE_N );
      
      const Expr::Tag srcTag ( thisPhiName + "_src", Expr::STATE_NONE );
      typedef typename ScalabilityTestSrc<FieldT>::Builder coupledSrcTerm;
      factory.register_expression( srcTag, scinew coupledSrcTerm( basePhiTag, nEqs) );
      srcTags.push_back( srcTag );
    }
    
    const Expr::Tag densT = Expr::Tag();
    const bool tempConstDens = false;
    return factory.register_expression( Expr::Tag( thisPhiName + "_rhs", Expr::STATE_NONE ),
                                       scinew typename ScalarRHS<FieldT>::Builder(info,srcTags,densT,tempConstDens) );
  }
  
  //------------------------------------------------------------------
  
  //==================================================================
  // Explicit template instantiation
  template class ScalabilityTestTransportEquation< SVolField >;
  template class ScalabilityTestTransportEquation< XVolField >;
  template class ScalabilityTestTransportEquation< YVolField >;
  template class ScalabilityTestTransportEquation< ZVolField >;
  //==================================================================
  
  
} // namespace Wasatch
