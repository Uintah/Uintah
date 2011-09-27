/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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
 *  Copyright (C) 2000 U of U
 */

#include <cassert>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/Matrix3.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Endian.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

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

using namespace SCIRun;
using namespace std;
using namespace Uintah;


// -----------------------------------------------------------------------------

// store tuple of (variable, it's type)
typedef pair<string, const Uintah::TypeDescription*> typed_varname;		  

static 
void usage(const string& badarg, const string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -basename [bnm]            alternate output basename\n";
  cerr << "  -format [fmt]              output format, one of (text,histogram,info)\n"; // ensight, dx
  cerr << "                             default is info\n";
  cerr << "  selection options:" << endl;
  cerr << FieldSelection::options() << endl;
  cerr << "  time options:" << endl;
  cerr << "      -datasetlow  [int]    output data sets starting at [int]\n";
  cerr << "      -datasethigh [int]    output data sets up to [int]\n";
  cerr << "      -datasetinc  [int]    output every [int] data sets\n";
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
  cerr << "      -showdiags             print available diagnostic names\n";
  cerr << "      -showtensorops         print known tensor transformations\n";
  cerr << "      -showfields            print available field names (requires archive name)\n";
  cerr << endl;
  exit(EXIT_SUCCESS);
}

int
main(int argc, char** argv)
{
  try {
    /*
     * Parse arguments
     */
    Args args(argc, argv);
  
    // throw help early
    if(args.getLogical("help") || args.getLogical("h"))
      usage("", args.progname());
    
    // global options
    //bool do_verbose = args.getLogical("verbose");
  
    // time stepping
    int time_step_lower = args.getInteger("datasetlow",  0);
    int time_step_upper = args.getInteger("datasethigh", INT_MAX);
    int time_step_inc   = args.getInteger("datasetinc",  1);

    // general writing options
    string fmt          = args.getString("format",   "info");
    string basedir      = args.getString("basename", "");
    
    if(args.getLogical("showdiags")) {
      cout << "Valid diagnostics: " << endl;
      describeScalarDiags(cout);
      describeVectorDiags(cout);
      describeTensorDiags(cout);
      cout << endl;
      exit(EXIT_SUCCESS);
    }
    
    if(args.getLogical("showtensorops")) {
      cout << "Valid tensor operations: " << endl;
      describeTensorDiags(cout);
      cout << endl;
      exit(EXIT_SUCCESS);
    }
    
    string filebase = args.trailing();
    if(filebase=="")
      usage("", args.progname());
    
    if(basedir=="")
      basedir = filebase.substr(0, filebase.find('.'));
    
    cout << "filebase: " << filebase << endl;
    DataArchive* da = scinew DataArchive(filebase);
    
    // load list of possible variables from the data archive
    vector<string> allvars;
    vector<const Uintah::TypeDescription*> alltypes;
    da->queryVariables(allvars, alltypes);
    ASSERTEQ(allvars.size(), alltypes.size());
    
    if(args.getLogical("showfields")) {
      cout << "Valid field names are: " << endl;
      for(vector<string>::const_iterator vit(allvars.begin());vit!=allvars.end();vit++) {
        if(*vit != "p.x") 
          cout << "   " << *vit << endl;
      }
      cout << endl;
      exit(EXIT_SUCCESS);
    }
    
    // select appropriate fields, materials and diagnostics
    FieldSelection fldselection(args, allvars);
    
    // build a specific dumper
    FieldDumper * dumper = 0;
    if(fmt=="text") {
      dumper = scinew TextDumper(da, basedir, args, fldselection);
      /* untested 
    } else if(fmt=="ensight") {
      dumper = scinew EnsightDumper(da, basedir, args, fldselection);
      */
    } else if(fmt=="histogram" || fmt=="hist") {
      dumper = new HistogramDumper(da, basedir, args, fldselection);
      /* untested
    } else if(fmt=="dx" || fmt=="opendx") {
      dumper = scinew DXDumper(da, basedir, binary, onedim);
      */
    } else if(fmt=="info") {
      dumper = new InfoDumper(da, basedir, args, fldselection);
    } else {
      cerr << "Failed to find match to format '" + fmt + "'" << endl;
      usage("", argv[0]);
    }
    
    if(args.hasUnused()) {
      cerr << "Unused options detected" << endl;
      vector<string> extraargs = args.unusedArgs();
      for(vector<string>::const_iterator ait(extraargs.begin());ait!=extraargs.end();ait++)
        {
          cerr << "    " << *ait << endl;
        }
      usage("", argv[0]);
    }
    
    // load list of possible indices and times
    vector<int>    index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    cout << "There are " << index.size() << " timesteps:\n";
    
    if(time_step_lower<0)                  time_step_lower = 0;
    if(time_step_upper>=(int)index.size()) time_step_upper = (int)index.size()-1;
    if(time_step_inc<=0)                   time_step_inc   = 1;
    
    // build list of (variable, type tuples) for any fields in use
    list<typed_varname> dumpvars;
    int nvars = (int)allvars.size();
    for(int i=0;i<nvars;i++) {
      if( fldselection.wantField(allvars[i]) )
        dumpvars.push_back( typed_varname(allvars[i], alltypes[i]) );
    }
    
    for(list<typed_varname>::const_iterator vit(dumpvars.begin());vit!=dumpvars.end();vit++) {
      const string fieldname = vit->first;
      const Uintah::TypeDescription* td  = vit->second;
      dumper->addField(fieldname, td);
    }
    
    // loop over the times
    for(int i=time_step_lower;i<=time_step_upper;i+=time_step_inc) {
      cout << index[i] << ": " << times[i] << endl;
        
      FieldDumper::Step * step_dumper = dumper->addStep(index[i], times[i], i);
        
      step_dumper->storeGrid();
        
      for(list<typed_varname>::const_iterator vit(dumpvars.begin());vit!=dumpvars.end();vit++) {
        const string fieldname = vit->first;
        if(fieldname=="p.x") continue; // dont work with point field
        
        const Uintah::TypeDescription* td      = vit->second;
        
        step_dumper->storeField(fieldname, td);
      }
      cout << endl;
	
      dumper->finishStep(step_dumper);
      
      // FIXME: 
      // delete step_dumper;
    }
      
    // delete dumper;
    
  } catch (ProblemSetupException & e) {
    cerr << endl;
    cerr << "----------------------------------------------------------------------" << endl;
    cerr << endl;
    cerr << "ERROR: " << e.message() << endl;
    cerr << endl;
    cerr << "----------------------------------------------------------------------" << endl;
    usage("", argv[0]);
    exit(EXIT_FAILURE);
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
  exit(EXIT_SUCCESS);
}
