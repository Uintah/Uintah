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
 *  SamplerGui.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/MIT/Dataflow/Modules/Metropolis/SamplerGui.h>
#include <iostream>

namespace MIT {

using namespace SCIRun;
using namespace std;

SamplerGui::SamplerGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id( name );
  
}
 
SamplerGui::~SamplerGui()
{
}

void
SamplerGui::set_iter( int i ) 
{
  tcl_ << "set-iter " << i;
  tcl_exec();
}

void
SamplerGui::set_kappa( double k ) 
{
  tcl_ << "set-kappa " << k;
  tcl_exec();
}

void
SamplerGui::set_nparms( int i ) 
{
  tcl_ << "set-nparms " << i;
  tcl_exec();
}

void
SamplerGui::set_theta( vector<double> *v)
{
  tcl_ << "set-theta";
  for (int i = 0; i < v->size(); i++)
    tcl_ << ' ' << (*v)[i];
  tcl_exec();
}

void
SamplerGui::set_sigma( vector<vector<double> > *v)
{
  tcl_ << "set-sigma";
  for (int i = 0; i < v->size(); i++)
    for (int j = 0; j < (*v)[i].size(); j++)
      tcl_ << ' ' << (*v)[i][j];
  tcl_exec();
}

void 
SamplerGui::done()
{
  tcl_ << "done";
  tcl_exec();
}

void 
SamplerGui::tcl_command( TCLArgs &args, void *data)
{
  int i, num;
  double d;
  if ( args[1] == "iterations") {
    string_to_int(args[2],i);
    num_iter(i);
  }
  else if ( args[1] == "sub" ) {
    string_to_int(args[2],i);
    subsample(i);
  }
  else if ( args[1] == "k" ) {
    string_to_double(args[2],d);
    kappa(d);
  }
  else if ( args[1] == "run" ) {
    go( 1 );
  }
  else if ( args[1] == "pause" ) {
    go( 2 );
  }
  else if ( args[1] == "stop" ) {
    go( 0 );
  }
  else if ( args[1] == "theta" ) {
    vector<double> t;
    string_to_int(args[2], num);
    t.resize(num);
    for (i = 0; i < num; i++)
    {
      string_to_double(args[i+3], d);
      t[i] = d;
    }
    theta(&t);
  }
  else if ( args[1] == "sigma" ) {
    
  }
  else
    PartGui::tcl_command( args, data);
};

} // namespace MIT

