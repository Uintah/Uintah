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
 *  RtPDSimGui.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>
#include <Packages/MIT/Dataflow/Modules/Metropolis/RtPDSimPartGui.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/RtPDSimPart.h>

namespace MIT {

using namespace std;
using namespace SCIRun;

static GuiCreator<RtPDSimPartGui> Rt_creator("Rt");

RtPDSimPartGui::RtPDSimPartGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
RtPDSimPartGui::~RtPDSimPartGui()
{
}

void 
RtPDSimPartGui::set_df( float v )
{
  
  tcl_ << "set-df " << v;
  tcl_exec();
}


void 
RtPDSimPartGui::attach( PartInterface *interface)
{
  RtPDSimPart *pi = dynamic_cast<RtPDSimPart *>(interface);
  if ( !pi ) {
    cerr << "RtPDSimPartGui got wrong type of interface\n";
    return;
  }

  connect( pi->df_changed, this, &RtPDSimPartGui::set_df );
  connect( df, pi, &RtPDSimPart::df );

  set_df( pi->df() );
}

void 
RtPDSimPartGui::tcl_command( TCLArgs &args, void *data)
{
  double v;
  if ( args[1] == "df") {
    string_to_double(args[2], v);
    df(v);
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT

