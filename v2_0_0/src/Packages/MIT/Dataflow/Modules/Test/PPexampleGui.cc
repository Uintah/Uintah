/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  PPexampleGui.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/MIT/Dataflow/Modules/Test/PPexampleGui.h>
#include <iostream>

namespace MIT {

using namespace SCIRun;
using namespace std;

PPexampleGui::PPexampleGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
PPexampleGui::~PPexampleGui()
{
}

void 
PPexampleGui::tcl_command( TCLArgs &args, void *data)
{
  int i;
  if ( args[1] == "burning") {
    string_to_int(args[2],i);
    cerr << "burning " << i << endl;
    burning(i);
  }
  else if ( args[1] == "monitor" ) {
    string_to_int(args[2],i);
    monitor(i);
  }
  else if ( args[1] == "thin" ) {
    string_to_int(args[2],i);
    thin(i);
  }
  else if ( args[1] == "exec" ) {
    cerr << "PPexampleGui exec" << endl;
    go();
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT



