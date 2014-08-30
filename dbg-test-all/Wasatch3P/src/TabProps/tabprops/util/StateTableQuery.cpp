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
#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>
#include <tabprops/Archive.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

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

    const vector<string> & indepVarNames = tbl.get_indepvar_names();
    const int nvar = indepVarNames.size();
    vector<double> hi(nvar), lo(nvar);
    vector<int> npts(nvar);
    vector<bool> logScale(nvar,false);

    for( int i=0; i<nvar; ++i ){
      cout << "Enter the number of points in the " << indepVarNames[i] << " dimension: ";
      cin >> npts[i];
      cout << "Enter the lower bound in the " << indepVarNames[i] << " dimension: ";
      cin >> lo[i];
      cout << "Enter the upper bound in the " << indepVarNames[i] << " dimension: ";
      cin >> hi[i];
      cout << endl;

      if( hi[i]/lo[i]>=100 ){
	cout << "  Log-scale for this variable? [y/n] ";
	string ans;
	cin >> ans;
	if( ans=="y" || ans=="Y" ){
	  logScale[i] = true;
	  hi[i] = std::log10( hi[i] );
	  lo[i] = std::log10( lo[i] );
	}
      }
    }

    const string prefix = fname.substr( 0, fname.find_first_of(".") );

    tbl.write_tecplot( npts, hi, lo, prefix );
    cout << "Tecplot file is called: " << prefix+".dat" << endl;

    tbl.write_matlab ( npts, hi, lo, logScale, prefix );
    cout << "Matlab file is called: '" << prefix+".m'" << endl;
  }
  catch( std::runtime_error & e )
  {
    cout << e.what() << std::endl;
    return -1;
  }
}
