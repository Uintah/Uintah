/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

/*
 *  grid_reader.cc: Reads in a binary grid.xml file and displays the actual XML
 *
 *  Written by:
 *   J. Davison de St. Germain
 *   SCI Institute
 *   University of Utah
 *   Aug. 2016
 *
 */

#include <iostream>

#include <stdio.h>

using namespace std;

/////////////////////////////////////////////////////////////////

void
usage( const std::string& badarg, const std::string& progname )
{
  if(badarg != "") {
    cerr << "Error parsing argument: " << badarg << "\n";
  }
  cout << "Usage: " << progname << " [options] <grid.xml>\n\n";
  cout << "Valid options are:\n";
  cout << "  -h[elp]\n";
  cout << "\n";
  exit( 1 );
}

int
main(int argc, char** argv)
{
  string filename;

  /*
   * Parse arguments
   */
  for( int pos = 1; pos < argc; pos++ ) {
    string arg = argv[ pos ];
    if( (arg == "-help") || (arg == "-h") ) {
      usage( "", argv[0] );
    } 
    else {
      filename = arg;
    }
  }

  if( filename == "" ) {
    cerr << "No grid file specified\n";
    usage( "", argv[0] );
  }


  

}
