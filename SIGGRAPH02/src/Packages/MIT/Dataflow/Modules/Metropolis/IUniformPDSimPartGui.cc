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
 *  IUPDSimGui.cc
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
#include <Packages/MIT/Dataflow/Modules/Metropolis/IUniformPDSimPartGui.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IUniformPDSimPart.h>

namespace MIT {

using namespace std;
using namespace SCIRun;

static GuiCreator<IUniformPDSimPartGui> IUniform_creator("IU");

IUniformPDSimPartGui::IUniformPDSimPartGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
IUniformPDSimPartGui::~IUniformPDSimPartGui()
{
}

void 
IUniformPDSimPartGui::set_par( const vector<double> &values )
{
  
  tcl_ << "set-par ";
  for (unsigned i=0; i<values.size(); i++)
    tcl_ << " " << values[i];
  tcl_exec();
}

void 
IUniformPDSimPartGui::set_center( const vector<double> &values )
{
  
  tcl_ << "set-center ";
  for (unsigned i=0; i<values.size(); i++)
    tcl_ << " " << values[i];
  tcl_exec();
}


void 
IUniformPDSimPartGui::attach( PartInterface *interface)
{
  IUniformPDSimPart *pi = dynamic_cast<IUniformPDSimPart *>(interface);
  if ( !pi ) {
    cerr << "IUniformPDSimPartGui got wrong type of interface\n";
    return;
  }

  connect( pi->par_changed, this, &IUniformPDSimPartGui::set_par );
  connect( pars, pi, &IUniformPDSimPart::pars );
  connect( par, pi, &IUniformPDSimPart::par );

  connect( pi->center_changed, this, &IUniformPDSimPartGui::set_center );
  connect( centers, pi, &IUniformPDSimPart::centers );
  connect( center, pi, &IUniformPDSimPart::center );

  set_par( pi->par() );
  set_center( pi->center() );
}

void 
IUniformPDSimPartGui::tcl_command( TCLArgs &args, void *data)
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
  else if ( args[1] == "center-all" ) {
    vector<double> v(args.count()-1);
    for (int i=0; i<args.count()-1; i++) {
      string_to_double(args[i+1],v[i]);
    }
    centers(v);
  }
  else if ( args[1] == "center" ) {
    int i;
    double v;

    string_to_int( args[2], i );
    string_to_double( args[3], v );

    center( i, v );
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT

