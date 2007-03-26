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
 *  RUPDSimGui.cc
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
#include <Packages/MIT/Dataflow/Modules/Metropolis/RUPDSimPartGui.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/RUPDSimPart.h>

namespace MIT {

using namespace std;
using namespace SCIRun;

static GuiCreator<RUPDSimPartGui> RU_creator("RU");

RUPDSimPartGui::RUPDSimPartGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
RUPDSimPartGui::~RUPDSimPartGui()
{
}

void 
RUPDSimPartGui::set_par( const vector<double> &values )
{
  
  tcl_ << "set-par ";
  for (unsigned i=0; i<values.size(); i++)
    tcl_ << " " << values[i];
  tcl_exec();
}


void 
RUPDSimPartGui::attach( PartInterface *interface)
{
  RUPDSimPart *pi = dynamic_cast<RUPDSimPart *>(interface);
  if ( !pi ) {
    cerr << "RUPDSimPartGui got wrong type of interface\n";
    return;
  }

  connect( pi->par_changed, this, &RUPDSimPartGui::set_par );
  connect( pars, pi, &RUPDSimPart::pars );
  connect( par, pi, &RUPDSimPart::par );

  set_par( pi->par() );
}

void 
RUPDSimPartGui::tcl_command( TCLArgs &args, void *data)
{
  double v;
  if ( args[1] == "par-all" ) {
    vector<double> v(args.count()-1);
    for (int i=0; i<args.count()-1; i++) {
      string_to_double(args[i+1],v[i]);
    }
    pars(v);
  }
  else if ( args[1] == "par" ) {
    int i;
    double v;

    string_to_int( args[2], i );
    string_to_double( args[3], v );

    par( i, v );
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT

