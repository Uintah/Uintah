#include "FlameMaster.h"

#include <iostream>
#include <stdexcept>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main( int iarg, char* carg[] )
{
  using namespace std;

  string tableName, path, prefix, mixFrac, chi;
  int order;
  bool allowClipping = true;

  po::options_description desc("Supported Options");
  desc.add_options()
              ( "help", "print help message" )
              ( "table-name",       po::value<string>(&tableName)->default_value("FMTable.tbl"), "The name for the table that is to be generated" )
              ( "path",             po::value<string>(&path     ), "Path to search for FlameMaster files (defaults to working directory)" )
              ( "prefix",           po::value<string>(&prefix   ), "Optional prefix for files to read.  For example, 'flm_298' would parse all files beginning with 'flm_298' (defaults to all files)" )
              ( "order",            po::value<int   >(&order    )->default_value(3), "Order for interpolation.")
              ( "disable-clipping", "don't clip values on independent variables prior to interpolation. The default behavior is to clip indpendent variables to avoid extrapolation" )
              ( "mixture-fraction", po::value<string>(&mixFrac  )->default_value("Z"), "Name of the mixture fraction variable in the flamelet files")
              ( "dissipation-rate", po::value<string>(&chi      )->default_value("stoichScalarDissRate"), "Name of the mixture fraction variable in the flamelet files");



  // parse the command line options
  try{
    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("help") ){
      cout << endl
          << "--------------------------------------------------------------------" << endl
          << "FlameMaster -> StateTable conversion utility." << endl
          << "Author:   James C. Sutherland (James.Sutherland@utah.edu)" << endl
          << endl
          << "TabProps Version (date): " << TabPropsVersionDate << endl
          << "TabProps Version (hash): " << TabPropsVersionHash << endl
          << "--------------------------------------------------------------------" << endl
          << endl
          << "This imports adiabatic flamelet files from FlameMaster and" << endl
          << "outputs a StateTable suitable for loading into a CFD solver." << endl
          << endl
          << desc << endl << endl
          << "Example: Generate a table from files in '~/tmp' beginning with 'flame_123'\n"
          << "         and using second order interpolants:" << endl
          << "   fm_loader --path=~/tmp --prefix=flame_123 --order=2" << endl
          << endl;
      return 1;
    }
    allowClipping = ( args.count("disable-clipping") > 0 );
  }
  catch( std::exception& err ){
    cout << err.what() << endl << desc << endl;
    return -1;
  }

  //-- Load the library from disk, generate the table, and write it out to disk
  try{
    FlameMasterLibrary fml( path, prefix, order, allowClipping );
    const StateTable& table = fml.generate_table("FMTable");
    table.output_table_info(std::cout);
    table.write_table("FMTable.tbl");
  }
  catch( std::exception& err ){
    cout << endl << "ERROR trapped.  Details follow... " << endl
         << err.what() << endl;
    return -1;
  }

  return 0;
}
