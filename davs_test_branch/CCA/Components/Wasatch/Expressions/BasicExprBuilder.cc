//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>


//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/BasicExprBuilder.h>
#include <CCA/Components/Wasatch/ParseTools.h>


//-- ExprLib includes --//
#include <expression/ExprLib.h>


//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredTypes.h>


#include <string>

namespace Wasatch{

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_it( Uintah::ProblemSpecP params )
  {
    Expr::ExpressionBuilder* builder = NULL;

    std::string exprType;
    Uintah::ProblemSpecP valParams = params->get("value",exprType);
    if( params->findBlock("Constant") ){
      double val;  params->get("Constant",val);
      typedef typename Expr::ConstantExpr<FieldT>::Builder Builder;
      builder = new Builder( val );
    }
    else if( params->findBlock("LinearFunction") ){
      double slope, intercept;
      Uintah::ProblemSpecP valParams = params->findBlock("LinearFunction");
      valParams->require("slope",slope);
      valParams->require("intercept",intercept);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::LinearFunction<FieldT>::Builder Builder;
      builder = new Builder( indepVarTag, slope, intercept );
    }

    return builder;
  }

  //------------------------------------------------------------------

  void
  create_expressions_from_input( Uintah::ProblemSpecP parser,
                                 GraphCategories& gc )
  {
    Expr::ExpressionBuilder* builder = NULL;

    for( Uintah::ProblemSpecP exprParams = parser->findBlock("BasicExpression");
         exprParams != 0;
         exprParams = exprParams->findNextBlock("BasicExpression") ){

      std::string label, fieldType, taskListName;
      exprParams->getAttribute("label",label);
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);

      const Expr::Tag tag = parse_nametag( exprParams->findBlock("NameTag") );

      std::cout << "Creating BasicExpression '" << label
                << "' for variable '" << tag.name()
                << "' with state " << tag.context()
                << " on task list '" << taskListName << "'"
                << std::endl;

      if     ( fieldType == "Cell"  )  builder = build_it<SpatialOps::structured::SVolField  >( exprParams );
      else if( fieldType == "XFace" )  builder = build_it<SpatialOps::structured::SSurfXField>( exprParams );
      else if( fieldType == "YFace" )  builder = build_it<SpatialOps::structured::SSurfYField>( exprParams );
      else if( fieldType == "ZFace" )  builder = build_it<SpatialOps::structured::SSurfZField>( exprParams );

      // jcs what about other field types?  Consider specifying cell
      // type (scalar, x, y, z) and location (center,xface,yface,zface)

      Category cat;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( tag, builder );
    }

  }

  //------------------------------------------------------------------

}
