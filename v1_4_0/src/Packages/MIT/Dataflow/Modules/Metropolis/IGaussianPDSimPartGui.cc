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
 *  IGaussianPDSimGui.cc
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
#include <Packages/MIT/Dataflow/Modules/Metropolis/IGaussianPDSimPartGui.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IGaussianPDSimPart.h>

namespace MIT {

using namespace std;
using namespace SCIRun;

static GuiCreator<IGaussianPDSimPartGui> IGaussian_creator("IG");

IGaussianPDSimPartGui::IGaussianPDSimPartGui( const string &name, 
					      const string &script)
  : PartGui( name, script )
{
  set_id( name );
}
 
IGaussianPDSimPartGui::~IGaussianPDSimPartGui()
{
}

void 
IGaussianPDSimPartGui::set_mean( const vector<double> &values )
{
  
  tcl_ << "set-mean ";
  for (unsigned i=0; i<values.size(); i++)
    tcl_ << " " << values[i];
  tcl_exec();
}


void 
IGaussianPDSimPartGui::attach( PartInterface *interface)
{
  IGaussianPDSimPart *pi = dynamic_cast<IGaussianPDSimPart *>(interface);
  if ( !pi ) {
    cerr << "IGaussianPDSimPartGui got wrong type of interface\n";
    return;
  }

  connect( pi->mean_changed, this, &IGaussianPDSimPartGui::set_mean );
  connect( means, pi, &IGaussianPDSimPart::means );
  connect( mean, pi, &IGaussianPDSimPart::mean );

  set_mean( pi->mean() );
}

void 
IGaussianPDSimPartGui::tcl_command( TCLArgs &args, void *data)
{
  double v;
  if ( args[1] == "mean-all" ) {
    vector<double> v(args.count()-1);
    for (int i=0; i<args.count()-1; i++) {
      string_to_double(args[i+1],v[i]);
    }
    means(v);
  }
  else if ( args[1] == "mean" ) {
    int i;
    double v;

    string_to_int( args[2], i );
    string_to_double( args[3], v );

    mean( i, v );
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT

