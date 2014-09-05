//-- Wasatch includes --//
#include "MomentTransportEquation.h"
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/PBE/MonosurfaceGrowth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/BulkDiffusionGrowth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/UniformGrowth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/PBE/QuadratureClosure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/transport/ScalarTransportEquation.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Wasatch {
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void setup_growth_expression( Uintah::ProblemSpecP growthParams,
                               const std::string& basePhiName,
                               const std::string& thisPhiName,
                               const double momentOrder,
                               const int nEqs,
                               Expr::TagList& growthTags,
                               const Expr::TagList& weightsTagList,
                               const Expr::TagList& abscissaeTagList,                               
                               Expr::ExpressionFactory& factory )
  {
    Expr::Tag growthTag; // this tag will be populated
    Expr::ExpressionBuilder* builder = NULL;

    std::string growthModel;
    growthParams->get("GrowthModel",growthModel);
    
    // see if we have a basic expression set for the growth.
    Uintah::ProblemSpecP nameTagParam = growthParams->findBlock("NameTag");
    
    if( nameTagParam ){
      growthTag = parse_nametag( nameTagParam );      
      
    } else { // if no expression was specified, build one based on the given info
      growthTag = Expr::Tag( thisPhiName + "_growth_" + growthModel, Expr::STATE_NONE );
      // now check what model the user has specified
      if (growthModel == "CONSTANT") {
        std::stringstream currentMomentOrderStr;
        currentMomentOrderStr << momentOrder;
        const Expr::Tag phiTag( basePhiName + "_" + currentMomentOrderStr.str(), Expr::STATE_N );      
        
        if (growthParams->findBlock("ConstantGrowthCoefficient")) {
          typedef typename UniformGrowth<FieldT>::Builder UniformGrowth;
          double growthRateVal;
          growthParams->get("ConstantCoefficient",growthRateVal);
          builder = scinew UniformGrowth(phiTag, growthRateVal);
        }
      }  else if (growthModel == "MONOSURFACE") {
        std::stringstream nextMomentOrderStr;
        nextMomentOrderStr << momentOrder + 1;
        const Expr::Tag phiTag( basePhiName + "_" + nextMomentOrderStr.str(), Expr::STATE_N );      
        if (growthParams->findBlock("ConstantGrowthCoefficient")) {
          typedef typename MonosurfaceGrowth<FieldT>::Builder growth;
          double coef;
          growthParams->get("ConstantGrowthCoefficient",coef);
          builder = scinew growth(phiTag, coef, momentOrder);
        }
        if (nEqs == momentOrder + 1) {
          // register a moment closure expression
          const Expr::Tag momentClosureTag( basePhiName + "_" + nextMomentOrderStr.str(), Expr::STATE_N );
          typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
          factory.register_expression(momentClosureTag, scinew MomClosure(weightsTagList,abscissaeTagList,momentOrder+1));
        }
      } else if (growthModel == "BULK_DIFFUSION") {
        std::stringstream previousMomentOrderStr;
        previousMomentOrderStr << momentOrder - 1;
        const Expr::Tag phiTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );      
        if (growthParams->findBlock("ConstantGrowthCoefficient")) {
          typedef typename BulkDiffusionGrowth<FieldT>::Builder growth;
          double coef;
          growthParams->get("ConstantGrowthCoefficient",coef);
          std::cout << "coefficient = " << coef << std::endl;
          builder = scinew growth(phiTag, coef, momentOrder);
        }
        if ( momentOrder == 0) {
          // register a moment closure expression
          const Expr::Tag momentClosureTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );
          typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
          factory.register_expression(momentClosureTag, scinew MomClosure(weightsTagList,abscissaeTagList,momentOrder));
        }
      }
      
    }
    growthTags.push_back(growthTag);
    factory.register_expression( growthTag, builder );
  }
  
  //------------------------------------------------------------------
  
  std::string
  get_population_name( Uintah::ProblemSpecP params )
  {
    std::string phiName;
    params->get("PopulationName",phiName);
    return phiName;
  }  
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionID
  MomentTransportEquation<FieldT>::
  get_moment_rhs_id(Expr::ExpressionFactory& factory,
                    Uintah::ProblemSpecP params,
                    Expr::TagList& weightsTags,
                    Expr::TagList& abscissaeTags,                    
                    const double momentOrder)
  {
    std::stringstream momentOrderStr;
    momentOrderStr << momentOrder;
    const std::string PopulationName = get_population_name(params);
    int nEnv = 1;
    params->get( "NumberOfEnvironments", nEnv );
    const int nEqs = 2*nEnv;
    
    const std::string basePhiName = "m_" + PopulationName;
    const std::string thisPhiName = "m_" + PopulationName + "_" + momentOrderStr.str();    

    //____________
    // start setting up the right-hand-side terms: these include expressions
    // for growth, nucleation, birth, death, and any othery fancy source terms
    Expr::TagList rhsTags;
    typename ScalarRHS<FieldT>::FieldTagInfo info;  
    
    //____________
    // Growth
    for( Uintah::ProblemSpecP growthParams=params->findBlock("GrowthExpression");
        growthParams != 0;
        growthParams=growthParams->findNextBlock("GrowthExpression") ){
      setup_growth_expression <FieldT>( growthParams, 
                                       basePhiName, 
                                       thisPhiName, 
                                       momentOrder, 
                                       nEqs, 
                                       rhsTags,
                                       weightsTags,
                                       abscissaeTags, 
                                       factory );            
    }
    
    //_________________
    // Diffusive Fluxes
    for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
        diffFluxParams != 0;
        diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFluxExpression") ){
      
      setup_diffusive_flux_expression<FieldT>( diffFluxParams, thisPhiName, factory, info );
      
    }
    
    //__________________
    // Convective Fluxes
    for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
        convFluxParams != 0;
        convFluxParams=convFluxParams->findNextBlock("ConvectiveFluxExpression") ){
      
      setup_convective_flux_expression<FieldT>( convFluxParams, thisPhiName, factory, info );
      
    }
    //
    return factory.register_expression( Expr::Tag( thisPhiName + "_rhs", Expr::STATE_NONE ),
                                       scinew typename ScalarRHS<FieldT>::Builder(info,rhsTags ));    
  }  
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  MomentTransportEquation<FieldT>::
  MomentTransportEquation( const std::string thisPhiName,
                          const Expr::ExpressionID rhsID )
  : Wasatch::TransportEquation( thisPhiName, rhsID, 
                               get_staggered_location<FieldT>() )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  MomentTransportEquation<FieldT>::~MomentTransportEquation()
  {}
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionID
  MomentTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_registry().get_id( Expr::Tag( this->solution_variable_name(),
                                                      Expr::STATE_N ) );
  }
  
  //------------------------------------------------------------------
  
  template< typename FieldT >  
  void MomentTransportEquation<FieldT>::
  setup_boundary_conditions(const GraphHelper& graphHelper,
                            const Uintah::PatchSet* const localPatches,
                            const PatchInfoMap& patchInfoMap,
                            const Uintah::MaterialSubset* const materials)
  {  }
  
  //------------------------------------------------------------------
  
  //==================================================================
  // Explicit template instantiation
  template class MomentTransportEquation< SVolField >;
  template class MomentTransportEquation< XVolField >;
  template class MomentTransportEquation< YVolField >;
  template class MomentTransportEquation< ZVolField >;
  //==================================================================
        
} // namespace Wasatch

