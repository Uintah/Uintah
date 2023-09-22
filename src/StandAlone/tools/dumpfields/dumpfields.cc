/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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
 *  dumpfields.cc: Print out a uintah data archive
 *
 *  The fault of
 *   Andrew D. Brydon
 *   Los Alamos National Laboratory
 *   Mar 2004
 *
 *  Based on puda, written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 */

#include <cassert>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Endian.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cerrno>
#include <algorithm>

#include "utils.h"
#include "Args.h"
#include "FieldSelection.h"
#include "ScalarDiags.h"
#include "VectorDiags.h"
#include "TensorDiags.h"

#include "TextDumper.h"
#include "InfoDumper.h"
#include "HistogramDumper.h"
/*
#include "EnsightDumper.h"
#include "DXDumper.h"
*/

using namespace std;
using namespace Uintah;


// -----------------------------------------------------------------------------
// store tuple of (variable, it's type)
typedef pair<string, const Uintah::TypeDescription*> typed_varname;


static
void usage(const string& badarg, const string& progname)
{
  if(badarg != ""){
    cerr << "Error parsing argument: " << badarg << endl;
  }
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -basename [bnm]            alternate output basename\n";
  cerr << "  -format [fmt]              output format, one of (text,histogram,info)\n"; // ensight, dx
  cerr << "                             default is info\n";

  cerr << "  selection options:" << endl;
  cerr << FieldSelection::options() << endl;

  cerr << "  time options:" << endl;
  cerr << "      -timestep_low  [int]    output data sets starting at timestep [int]\n";
  cerr << "      -timestep_high [int]    output data sets up to timestep [int]\n";
  cerr << "      -timestep_inc  [int]    output data sets every [int] timesteps\n";

  cerr << "  info options:" << endl;
  cerr << InfoOpts::options() << endl;

  cerr << "  text options:" << endl;
  cerr << TextOpts::options() << endl;

  // cerr << "  ensight options:" << endl;
  // cerr << EnsightOpts::options() << endl;

  cerr << "  histogram options:" << endl;
  cerr << HistogramOpts::options() << endl;

  cerr << "  help options:" << endl;
  cerr << "      -help                  this help\n";
  cerr << "      -show_diags            print available diagnostic names\n";
  cerr << "      -show_tensor_ops       print known tensor transformations\n";
  cerr << "      -showFields            print available field names (requires archive name)\n";
  cerr << endl;
  exit(EXIT_SUCCESS);
}

//______________________________________________________________________
//
int
main(int argc, char** argv)
{
  try {
    /*
     * Parse arguments
     */
    Args args(argc, argv);

    // throw help early
    if(args.getLogical("help") || args.getLogical("h")){
      usage("", args.progname());
    }

    // global options
    //bool do_verbose = args.getLogical("verbose");

    // time stepping
    int timestep_lower = args.getInteger("timestep_low",  0);
    int timestep_upper = args.getInteger("timestep_high", INT_MAX);
    int timestep_inc   = args.getInteger("timestep_inc",  1);

    // general writing options
    string fmt          = args.getString("format",   "info");
    string basedir      = args.getString("basename", "");

    if( args.getLogical("show_diags") ) {
      cout << "Valid diagnostics: " << endl;
      describeScalarDiags( cout );
      describeVectorDiags( cout );
      describeTensorDiags( cout );
      cout << endl;
      exit(EXIT_SUCCESS);
    }

    if( args.getLogical("show_tensor_ops") ) {
      cout << "Valid tensor operations: " << endl;
      describeTensorDiags(cout);
      cout << endl;
      exit(EXIT_SUCCESS);
    }

    //
    // parse the name of the uda
    string uda = args.trailing();
    if( uda=="" ){
      usage("", args.progname());
    }

    if( basedir=="" ){
      basedir = uda.substr(0, uda.find('.'));
    }

    cout << "uda: " << uda << endl;
    DataArchive* da = scinew DataArchive(uda);

    // load list of possible variables from the data archive
    vector<string>  allVars;
    vector<int>     num_matls;
    vector<const Uintah::TypeDescription*> alltypes;

    da->queryVariables( allVars, num_matls, alltypes );

    ASSERTEQ( allVars.size(), alltypes.size() );

    if( args.getLogical("showFields") ) {
      cout << "Valid field names are: " << endl;

      for(vector<string>::const_iterator vit=allVars.begin();vit!=allVars.end();vit++) {
        if(*vit != "p.x"){
          cout << "   " << *vit << endl;
        }
      }
      cout << endl;
      exit(EXIT_SUCCESS);
    }

    // select appropriate fields, materials and diagnostics
    FieldSelection fldSelection(args, allVars);

    // build a specific dumper
    FieldDumper * dumper = 0;
    if( fmt=="text" ) {
      dumper = scinew TextDumper( da, basedir, args, fldSelection);
    }
    else if( fmt=="histogram" || fmt=="hist" ) {
      dumper = new HistogramDumper(da, basedir, args, fldSelection);
    }

#if 0
    }  /* untested
    else if(fmt=="ensight") {
      dumper = scinew EnsightDumper(da, basedir, args, fldSelection);
      */
    }
    else if(fmt=="dx" || fmt=="opendx") {
      dumper = scinew DXDumper(da, basedir, binary, onedim);
      */
#endif

    else if( fmt=="info" ) {
      dumper = new InfoDumper(da, basedir, args, fldSelection);
    }
    else {
      cerr << "Failed to find match to format '" + fmt + "'" << endl;
      usage("", argv[0]);
    }

    if( args.hasUnused() ) {
      cerr << "Unused options detected" << endl;
      vector<string> extraargs = args.unusedArgs();

      for(vector<string>::const_iterator ait=extraargs.begin();ait!=extraargs.end();ait++){
          cerr << "    " << *ait << endl;
      }
      usage("", argv[0]);
    }

    ///______________________________________________________________________
    //
    // load list of possible timesteps and times
    vector<int>    timesteps;
    vector<double> times;
    da->queryTimesteps(timesteps, times);
    ASSERTEQ(timesteps.size(), times.size());

    cout << "There are " << timesteps.size() << " timesteps:\n";



    if( timestep_lower<0 ){
      timestep_lower = 0;
    }
    if( timestep_upper>=(int)timesteps.size() ) {
      timestep_upper = (int)timesteps.size()-1;
    }
    if( timestep_inc<=0 ){
      timestep_inc   = 1;
    }

    //__________________________________
    //
    // build list of (variable, type tuples) for any fields in use
    list<typed_varname> dumpVars;
    int nVars = (int)allVars.size();

    for(int i=0;i<nVars;i++) {
      if( fldSelection.wantField(allVars[i]) ){
        dumpVars.push_back( typed_varname(allVars[i], alltypes[i]) );
      }
    }

    for(list<typed_varname>::const_iterator vit=dumpVars.begin();vit!=dumpVars.end();vit++) {
      const string fieldname = vit->first;
      const Uintah::TypeDescription* td  = vit->second;
      dumper->addField( fieldname, td );
    }

    //__________________________________
    //
    // loop over the times
    for(int i=timestep_lower;i<=timestep_upper;i+=timestep_inc) {
      cout << timesteps[i] << ": " << times[i] << endl;

      FieldDumper::Step * step_dumper = dumper->addStep(timesteps[i], times[i], i);

      step_dumper->storeGrid();

      for(list<typed_varname>::const_iterator vit=dumpVars.begin();vit!=dumpVars.end();vit++) {
        const string fieldname = vit->first;
        const Uintah::TypeDescription* td = vit->second;

        if( fieldname=="p.x" ){
          continue; // dont work with point field
        }

        step_dumper->storeField(fieldname, td);
      }
      cout << endl;

      dumper->finishStep(step_dumper);

      // FIXME:
      delete step_dumper;
    }
    delete dumper;
  }
  catch (ProblemSetupException & e) {
    cerr << endl;
    cerr << "----------------------------------------------------------------------" << endl;
    cerr << endl;
    cerr << "ERROR: " << e.message() << endl;
    cerr << endl;
    cerr << "----------------------------------------------------------------------" << endl;
    usage("", argv[0]);
    exit(EXIT_FAILURE);
  }
  catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  }
  catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
  exit(EXIT_SUCCESS);
}
