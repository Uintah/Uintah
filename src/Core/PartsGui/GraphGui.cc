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
 *  GraphGui.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/Signals.h>
#include <Core/Parts/GraphPart.h>
#include <Core/PartsGui/GraphGui.h>

namespace SCIRun {

GraphGui::GraphGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
GraphGui::~GraphGui()
{
}


void
GraphGui::add_values( vector<double> &v )
{
  return;
  cerr << "GraphGui ";
  for (unsigned i=0; i<v.size(); i++)
    cerr << v[i] << " ";
  cerr << endl;
}

void 
GraphGui::attach( PartInterface *interface )
{
  GraphPart *graph = (GraphPart *)interface;
  if ( !graph ) {
    cerr << "GraphGui[connect]: got the wrong interface type\n";
    return;
  }

  connect( graph->new_values, this, &GraphGui::add_values);
}

} // namespace MIT





