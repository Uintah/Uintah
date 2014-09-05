//--- Local (Wasatch) includes ---//
#include "Properties.h"
#include "GraphHelperTools.h"
#include "ParseTools.h"
#include "Expressions/TabPropsEvaluator.h"

//--- ExprLib includes ---//
#include <expression/ExpressionFactory.h>

//--- TabProps includes ---//
#include <tabprops/Archive.h>

//--- Uintah includes ---//
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <fstream>

using std::endl;
using std::flush;

namespace Wasatch{

  //====================================================================

  /**
   *  \ingroup WasatchParser
   *  \brief set up TabProps for use on the given GraphHelper
   *  \param params - the parser parameters for the TabProps specification.
   *  \param gh - the GraphHelper associated with this instance of TabProps.
   */
  void parse_tabprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh )
  {
    std::string fileName;
    params->get("FileNamePrefix",fileName);

    proc0cout << "Loading TabProps file '" << fileName << "' ... " << std::flush;

    StateTable table;
    try{
      std::ifstream inFile( (fileName+".tbl").c_str(), std::ios_base::in );
      InputArchive ia(inFile);
      ia >> BOOST_SERIALIZATION_NVP(table);
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << e.what() << std::endl << std::endl
          << "Could not open TabProps file '" << fileName << ".tbl'" << std::endl
          << "Check to ensure that the file exists in the run dir." << std::endl
          << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    proc0cout << "done" << std::endl;

    //___________________________________________________________
    // get information for the independent variables in the table
    typedef std::map<std::string,Expr::Tag> VarNameMap;
    VarNameMap ivarMap;

    for( Uintah::ProblemSpecP ivarParams = params->findBlock("IndependentVariable");
         ivarParams != 0;
         ivarParams = ivarParams->findNextBlock("IndependentVariable") ){
      std::string ivarTableName;
      const Expr::Tag ivarTag = parse_nametag( ivarParams->findBlock("NameTag") );
      ivarParams->get( "NameInTable", ivarTableName );
      ivarMap[ivarTableName] = ivarTag;
    }

    //______________________________________________________________
    // NOTE: the independent variable names must be specified in the
    // exact order dictated by the table.  This order will determine
    // the ordering for the arguments to the evaluator later on.
    typedef std::vector<std::string> Names;
    std::vector<Expr::Tag> ivarNames;
    const Names& ivars = table.get_indepvar_names();
    for( Names::const_iterator inm=ivars.begin(); inm!=ivars.end(); ++inm ){
      ivarNames.push_back( ivarMap[*inm] );
    }

    //________________________________________________________________
    // create an expression for each property.  alternatively, we
    // could create an expression that evaluated all required
    // properties at once, since the expression has that capability...
    for( Uintah::ProblemSpecP dvarParams = params->findBlock("ExtractVariable");
         dvarParams != 0;
         dvarParams = dvarParams->findNextBlock("ExtractVariable") ){

      //_______________________________________
      // extract dependent variable information
      const Expr::Tag dvarTag = parse_nametag( dvarParams->findBlock("NameTag") );

      std::string dvarTableName;
      dvarParams->get( "NameInTable", dvarTableName );
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarTableName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      proc0cout << "Constructing property evaluator for '" << dvarTag
                << "' from file '" << fileName << "'." << std::endl;

      const BSpline* const spline = table.find_entry( dvarTableName );

      Expr::ExpressionBuilder* builder = NULL;

      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SVolField>::Builder PropEvaluator;
        builder = scinew PropEvaluator( spline->clone(), ivarNames );
        break;
      }
      case XVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfXField>::Builder PropEvaluator;
        builder = scinew PropEvaluator( spline->clone(), ivarNames );
        break;
      }
      case YVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfYField>::Builder PropEvaluator;
        builder = scinew PropEvaluator( spline->clone(), ivarNames );
        break;
      }
      case ZVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfZField>::Builder PropEvaluator;
        builder = scinew PropEvaluator( spline->clone(), ivarNames );
        break;
      }
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type named '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
        throw std::runtime_error( msg.str() );
      }

      //____________________________
      // register the expression
      gh.exprFactory->register_expression( dvarTag, builder );

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
      parse_tabprops( tabPropsParams, gh );
    }
  }

  //====================================================================

} // namespace Wasatch
