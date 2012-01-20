//--- Local (Wasatch) includes ---//
#include "Properties.h"
#include "GraphHelperTools.h"
#include "ParseTools.h"
#include "Expressions/TabPropsEvaluator.h"
#include "Expressions/DensityCalculator.h"

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

#ifndef TabProps_BSPLINE
// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;
#endif

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
          << "Error reading TabProps file '" << fileName << ".tbl'" << std::endl
          << "Check to ensure that the file exists." << std::endl
          << "It is also possible that there was an error loading the file.  This could be caused" << std::endl
          << "by a file in a format incompatible with the version of TabProps linked in here." << std::endl
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
      const InterpT* const interp = table.find_entry( dvarTableName );
      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SVolField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case XVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfXField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case YVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfYField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case ZVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfZField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type named '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
        throw std::runtime_error( msg.str() );
      }
    }

    //________________________________________________________________
    // create an expression specifically for density.

    Uintah::ProblemSpecP densityParams = params->findBlock("ExtractDensity");
    if (densityParams != 0) {

      std::vector<Expr::Tag> rhoEtaNames;
      std::vector<Expr::Tag> reiEtaNames;
      for( Uintah::ProblemSpecP rhoEtaParams = densityParams->findBlock("DensityWeightedIVar");
          rhoEtaParams != 0;
          rhoEtaParams = rhoEtaParams->findNextBlock("DensityWeightedIVar") ){
        const Expr::Tag rhoEtaTag = parse_nametag( rhoEtaParams->findBlock("NameTag") );
        rhoEtaNames.push_back( rhoEtaTag );
        Uintah::ProblemSpecP reiEtaParams = rhoEtaParams->findBlock("RelatedIVar");
        const Expr::Tag reiEtaTag = parse_nametag( reiEtaParams->findBlock("NameTag") );
        reiEtaNames.push_back( reiEtaTag );
      }


      //_______________________________________
      // extract density variable information
      std::string dvarTableName;
      densityParams->get( "NameInTable", dvarTableName );
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarTableName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      const InterpT* const interp = table.find_entry( dvarTableName );

      //_____________________________________
      // register the expression for density
      typedef DensityCalculator<SpatialOps::structured::SVolField>::Builder DensCalc;
      gh.exprFactory->register_expression( scinew DensCalc( parse_nametag( densityParams->findBlock("NameTag") ),
                                                            interp->clone(), rhoEtaNames, reiEtaNames, ivarNames ) );

    }

  }
  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& params,
                             GraphCategories& gc )
  {
    //_________________________________________________________________________________
    // extracting the density tag in the cases that it is needed and also throwing the
    // error messages in different error conditions regarding to the input file

    Uintah::ProblemSpecP densityParams  = params->findBlock("Density");
    Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");

    if (tabPropsParams) {
      if (tabPropsParams->findBlock("ExtractDensity") && !densityParams) {
        std::ostringstream msg;
        msg << "ERROR: You need a tag for density when you want to extract it using TabProps." << endl
            << "       Please include the \"Density\" block in wasatch in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    for( Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
         tabPropsParams != 0;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") ){
      // determine which task list this goes on
      std::string taskListName;
      tabPropsParams->require("TaskList",taskListName);
      Category cat;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      parse_tabprops( tabPropsParams, *gc[cat] );
    }
  }

  //====================================================================

} // namespace Wasatch
