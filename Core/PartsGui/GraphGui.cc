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

#include <Core/2d/LockedPolyline.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Graph.h>

#include <Core/Util/Signals.h>
#include <Core/Parts/GraphPart.h>
#include <Core/PartsGui/GraphGui.h>

namespace SCIRun {

GraphGui::GraphGui( const string &name, const string &script)
  : PartGui( name, script )
{
  monitor_ = scinew CrowdMonitor( id().c_str() );
  graph_ = scinew Graph( id()+"-Graph" );
  diagram_ = scinew Diagram("Metropolis");
  graph_->add("Theta", diagram_);
}
 
GraphGui::~GraphGui()
{
}


void
GraphGui::reset( int n )
{
  for (int i=poly_.size(); i<n; i++) {
    LockedPolyline *p = scinew LockedPolyline( i );
    p->set_lock( monitor_ );
    p->set_color( Color( drand48(), drand48(), drand48() ) );
    poly_.push_back( p );
    diagram_->add( p );
  }

  for (int i=0; i<n; i++)
    poly_[i]->clear();
}

void
GraphGui::add_values( vector<double> &v )
{
  for (unsigned i=0; i<v.size(); i++) 
    poly_[i]->add(v[i]);

  graph_->need_redraw();
}

void 
GraphGui::attach( PartInterface *interface )
{
  GraphPart *graph = dynamic_cast<GraphPart *>(interface);
  if ( !graph ) {
    cerr << "GraphGui[connect]: got the wrong interface type\n";
    return;
  }

  connect( graph->reset, this, &GraphGui::reset);
  connect( graph->new_values, this, &GraphGui::add_values);
}

void
GraphGui::set_window( const string &window )
{
  graph_->set_window( window );
}


} // namespace MIT





