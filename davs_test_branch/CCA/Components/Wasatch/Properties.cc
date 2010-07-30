#include "Properties.h"
#include "GraphHelperTools.h"
#include "Expressions/TabPropsEvaluator.h"

#include <expression/ExpressionFactory.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>


namespace Wasatch{

  //====================================================================

  void parseTabProps( Uintah::ProblemSpecP& params,
                      GraphHelper& gh )
  {
    std::string fileName;
    params->get("FileNamePrefix",fileName);

    cout << "Loading TabProps file '" << fileName << "' ... " << flush;

    StateTable table;
    try{
      table.read_hdf5( fileName );
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << e.what() << endl << endl
          << "Could not open TabProps file '" << fileName << ".h5'" << endl
          << "Check to ensure that the file exists in the run dir." << endl
          << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    cout << "done" << endl;

    //___________________________________
    // set up any variable aliasing for
    // independent variables in the table
    typedef std::map<std::string,std::string> VarNameMap;
    VarNameMap ivarAliasMap;

    for( Uintah::ProblemSpecP aliasParams = params->findBlock("IndependentVariable");
         aliasParams != 0;
         aliasParams = aliasParams->findNextBlock("IndependentVariable") ){
      std::string ivarName, aliasName;
      aliasParams->getAttribute( "name", ivarName );
      aliasParams->getAttribute( "alias", aliasName );
      ivarAliasMap[ivarName] = aliasName;
    }

    //______________________________________________________________
    // NOTE: the independent variable names must be specified in the
    // exact order dictated by the table.  This order will determine
    // the ordering for the arguments to the evaluator later on.
    typedef std::vector<std::string> Names;
    Names ivarNames;
    const Names& ivars = table.get_indepvar_names();
    for( Names::const_iterator inm=ivars.begin(); inm!=ivars.end(); ++inm ){
      const VarNameMap::const_iterator ii = ivarAliasMap.find(*inm);
      ivarNames.push_back( ii==ivarAliasMap.end() ? *inm : ii->second );
    }

    //________________________________________________________________
    // create an expression for each property.  alternatively, we
    // could create an expression that evaluated all required
    // properties at once, since the expression has that capability...
    for( Uintah::ProblemSpecP dvarParams = params->findBlock("ExtractVariable");
         dvarParams != 0;
         dvarParams = dvarParams->findNextBlock("ExtractVariable") ){

      //_____________________________________________
      // extract dependent variable names to alias.
      std::string dvarName, dvarAlias;
      dvarParams->getAttribute("name", dvarName );
      dvarParams->getAttribute("alias",dvarAlias);

      if( !table.has_depvar(dvarName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      cout << "Constructing property evaluator for '" << dvarName
           << "' from file '" << fileName << "'." << endl;

      const BSpline* const spline = table.find_entry( dvarName );

      Expr::Context context = Expr::STATE_NONE;
      Expr::ExpressionBuilder* builder = NULL;

      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "Cell" );

      if( fieldType == "Cell" ){
        typedef TabPropsEvaluator<SpatialOps::structured::SVolField>::Builder PropEvaluator;
        builder = new PropEvaluator( spline->clone(), context, ivarNames );
      }
      else if( fieldType == "XFace" ){
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfXField>::Builder PropEvaluator;
        builder = new PropEvaluator( spline->clone(), context, ivarNames );
      }
      else if( fieldType == "YFace" ){
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfYField>::Builder PropEvaluator;
        builder = new PropEvaluator( spline->clone(), context, ivarNames );
      }
      else if( fieldType == "ZFace" ){
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfZField>::Builder PropEvaluator;
        builder = new PropEvaluator( spline->clone(), context, ivarNames );
      }

      //____________________________
      // register the expression
      gh.exprFactory->register_expression( Expr::Tag( dvarAlias, context ), builder );

    }

  }

  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& params,
                             GraphHelper& gh )
  {
    for( Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
         tabPropsParams != 0;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") ){
      parseTabProps( tabPropsParams, gh );
    }
  }

  //====================================================================

} // namespace Wasatch
