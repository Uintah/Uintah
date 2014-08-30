/*
 * Copyright (c) 2014 The University of Utah
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

#include "PhilCoalTableImporter.h"

#include <tabprops/StateTable.h>

#include <stdexcept>
#include <iostream>
#include <fstream>

using namespace std;

void print_names( const vector<string>& names )
{
  for( vector<string>::const_iterator inm=names.begin(); inm!=names.end(); ++inm ){
    cout << *inm << endl;
  }
}

int main()
{
  try{
    PhilCoalTableReader ct( "oxyflam1.mix", "MyTable" );

    const vector<string>& dvars = ct.get_dvar_names();
    const vector<string>& ivars = ct.get_ivar_names();

    StateTable table;
    table.read_table( "MyTable.tbl" );
    const vector<string>& ivarsNew = table.get_indepvar_names();
    const vector<string>& dvarsNew = table.get_depvar_names();

    if( ivarsNew != ivars ){
      cout << "independent variables are corrupt" << endl;
      return -1;
    }

    // note that the table may re-order the entries...
    if( dvarsNew.size() != dvars.size() ){
      cout << "dependent variables are corrupt" << endl
           << "-----------------------" << endl
           << "original: " << dvars.size() << endl;
      print_names( dvars );
      cout << "-----------------------" << endl
           << "table:" << dvarsNew.size() << endl;
      print_names( dvarsNew );
      return -1;
    }
  }
  catch( runtime_error& e ){
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}
