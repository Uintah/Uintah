/*
 * The MIT License
 *
 * Copyright (c) 2010-2014 The University of Utah
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
#include <tabprops/StateTable.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;
Interp4D i4d;
Interp5D i5d;


int main( int argc, const char* argv[] )
{
  using namespace std;

  string fname;

  // parse the command line options
  try{
    po::options_description desc("Supported Options");
    desc.add_options()
          ( "help", "print help message" );

    po::options_description hidden("Hidden options");
    hidden.add_options()
            ("file1", po::value<string>(&fname), "table file");

    po::positional_options_description p;
    p.add("file1", 1);

    po::options_description cmdline_options;
    cmdline_options.add(desc).add(hidden);

    po::variables_map args;
    po::store( po::command_line_parser(argc,argv).
               options(cmdline_options).positional(p).run(), args );
    po::notify(args);

    if( args.count("help") ){
      cout << "Usage:  tablecompare <file>\n"
          << desc << "\n";
      return -1;
    }

    if( fname.empty() ){
      cout << "Must provide names of input files for comparison!" << endl << desc << endl;
      return -1;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << "\n\nUsage:  tablecompare <file>\n\n";
    return -1;
  }

  try{
    StateTable tbl;
    try{
      tbl.read_table(fname);
    }
    catch( std::exception& e ){
      cout << endl << e.what() << endl << endl
           << "ERROR READING TABLE: '" << fname << "'" << endl
           << "check to ensure that this file exists." << endl
           << endl;
      return -1;
    }

    tbl.output_table_info( cout );

    const vector<string>& indepVarNames = tbl.get_indepvar_names();
    const vector<string>& dvarNames = tbl.get_depvar_names();
    const size_t nivar = indepVarNames.size();
    const size_t ndvar = dvarNames.size();
    std::vector<double> ivar(nivar,0.0);
    bool more = true;
    int niter = 0;

    while( more ){

      ++niter;

      for( size_t i=0; i<nivar; ++i ){
        cout << "Enter value for " << indepVarNames[i] << ": ";
        cin >> ivar[i];
      }

      cout << endl << endl;
      cout.width(35+nivar*2*20); cout.fill('='); cout << endl << "" << endl;
      cout.width(35); cout.fill(' '); cout << " ";
      for( size_t i=0; i<nivar; ++i ) cout << setw(20) << std::left << "d var" << setw(20) << std::left << "d^2 var";
      cout << endl << setw(20) << "Dependent Var" << setw(15) << "Value";
      for( size_t i=0; i<2*nivar; ++i ){
        cout.width(18); cout.fill('-'); cout << "" << "  ";
      }
      cout.width(35); cout.fill(' '); cout << endl << "";
      for( size_t i=0; i<2*nivar; ++i ){
        std::string tmp = "d " + indepVarNames[i/2];
        if( i%2 ) tmp += "^2";
        cout << setw(20) << std::left << tmp;
      }
      cout.width(35+nivar*2*20); cout.fill('='); cout << endl << "" << endl;
      cout.fill(' ');
      for( size_t i=0; i<ndvar; ++i ){
        const InterpT* const interp = tbl.find_entry( dvarNames[i] );
        const double dvar = interp->value(ivar);
        cout << setw(20) << dvarNames[i] << setw(15) << dvar;
        for( size_t j=0; j<nivar; ++j ){
          cout << setw(20) << interp->derivative(ivar,j)
               << setw(20) << interp->second_derivative(ivar,j,j);
        }
        cout << endl;
      }
      cout << endl << endl;

      if( niter>10 ) more = false;
    }

  }
  catch( std::runtime_error& e ){
    cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}
