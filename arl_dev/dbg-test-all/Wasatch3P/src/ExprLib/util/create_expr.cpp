#include <CreateExpr.h>

#include <boost/program_options.hpp>

#include <string>
#include <iostream>
#include <fstream>

using namespace std;
namespace po = boost::program_options;

int main( int iarg, char* carg[] )
{
  string exprName, fileName, fieldName;

  po::options_description desc("Supported Options");
  desc.add_options()
    ( "help", "print help message\n" )
    ( "expr-name,e",
      po::value<string>(&exprName),
      "(REQUIRED) Set the name for the expression" )
    ( "file-name,n",
      po::value<string>(&fileName),
      "Set the name for the generated file (default is exprname" )
    ( "field-type,f",
      po::value<string>(&fieldName)->default_value("FieldT"),
      "Set the type of field this expression evaluates. Note that this will only be set as a template parameter if the -F option is also specified" )
    ( "field-as-template,F",
      "Triggers inclusion of the field type name as a template parameter. By default, the expression will not be templated on the field type." )
    ( "template-param,t",
      po::value<vector<string> >()->multitoken(),
      "Sets template parameters (space-separated). Note that if you want the Field type to be a template parameter, it must be explicitly declared as such by including it in this list. Alternatively, you can use the -F flag");

  po::variables_map args;

  try{
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    // resolve options
    if( args.count("help") ){
      cout << desc << endl
           << "---------------------------- Example Usage ----------------------------" << endl
           << endl
           << "Example 1:  create_expr -e Test" << endl
           << "  Creates an expression named 'Test' with a field type 'FieldT' that must be" << endl
           << "  defined via an appropriate typedef at the top of the file." << endl
           << endl
           << "Example 2:  create_expr -e test -f CellT -F" << endl
           << "  Creates an expression named 'test' in a file named 'test.h' with a Field type" << endl
           << "  'CellT'.  The 'test' expression will be templated on the CellT type." << endl
           << endl
           << "Example 3:  create_expr -e Test -F" << endl
           << "  Creates an expression named 'Test' templated on the field type 'FieldT'" << endl
           << endl
           << "-----------------------------------------------------------------------" << endl
           << endl;
      return -1;
    }

    // require this argument.
    if( !args.count("expr-name") ){
      cout << "ERROR: Name of the expression must be supplied as a command line argument" << endl
           << desc << endl;
      return -1;
    }

    if( !args.count("file-name") ){
      fileName = args["expr-name"].as<string>();
    }


    Info info( fieldName, exprName, fileName );

    if( args.count("field-as-template") )  info.set( Info::EXTRA_TEMPLATE_PARAMS, fieldName );

    if( args.count("template-param") ){
      const vector<string>& tps = args["template-param"].as<vector<string> >();
      for( vector<string>::const_iterator i=tps.begin(); i!=tps.end(); ++i ){
        info.set( Info::EXTRA_TEMPLATE_PARAMS, *i );
      }
    }

    info.finalize();
    CreateExpr ce( info );
    {
      std::ofstream fout( (fileName+".h").c_str(), ios_base::out );
      fout << ce.get_stream().str();
    }

    cout << info << endl
         << "File created as \'" << fileName << ".h\'" << endl;
  }
  catch( po::unknown_option& e ){
    cout << e.what() << endl;
  }
  catch( runtime_error& e ){
    cout << e.what() << endl;
    return -1;
  }
  catch(...){
    cout << "unknown error occurred.  Check input flags." << endl
         << desc << endl;
    return -1;
  }

  return 0;

}
