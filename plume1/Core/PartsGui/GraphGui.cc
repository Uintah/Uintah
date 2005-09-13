/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#include <iostream>
#include <Core/2d/LockedPolyline.h>
#include <Core/2d/ParametricPolyline.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Graph.h>

#include <Core/Util/Signals.h>
#include <Core/Parts/GraphPart.h>
#include <Core/PartsGui/GraphGui.h>


//#include <typeinfo>

namespace SCIRun {

using std::cerr;

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

#ifdef CHRIS
void
GraphGui::reset( const vector<DrawObj*> &objs)
{
  for (unsigned loop=0; loop<objs.size(); ++loop) {
    //    cerr << "checking DrawObj type: ";
    //    const type_info &t=typeid(objs[loop]);
    //    cerr << t.name() << endl;
    if (LockedPolyline *p =  dynamic_cast<LockedPolyline*>(objs[loop])) {
      cerr << "item " << loop << " is type LockedPolyline." << endl;
      p = scinew LockedPolyline( loop );
      p->set_lock( monitor_ );
      p->set_color( Color( drand48(), drand48(), drand48() ) );
      poly_.push_back( p );
      diagram_->add( p );
      p->clear();
    } else if (ParametricPolyline *p = 
                 dynamic_cast<ParametricPolyline*>(objs[loop])) {
      cerr << "item " << loop << " is type ParametricPolyline." << endl;
      p = scinew ParametricPolyline( loop );
      p->set_lock( monitor_ );
      p->set_color( Color( drand48(), drand48(), drand48() ) );
      poly_.push_back( p );
      diagram_->add( p );
      p->clear();
    } else {
      cerr << "item " << loop << " is unknown type." << endl;
    }
  }
}
#else
void
GraphGui::reset( int n )
{
  int i;
  int size = int(poly_.size());
  for ( i=0; i<size; i++) 
    poly_[i]->clear();

  for (; i <n; i++) {
    LockedPolyline *p = scinew LockedPolyline(i);
    p->set_lock( monitor_ );
    Color c;
    // byte 0 is valid flag, bytes 1-3 are rgb
    vector<unsigned char> color;
    color.resize(4);
    color[0]=color[1]=color[2]=color[3]=0;
    get_property(i,"color",color);
    if (!color[0]) {
      vector<unsigned char> new_color;
      new_color.resize(3);
      new_color[0]=(unsigned char)(drand48()*255.);
      new_color[1]=(unsigned char)(drand48()*255.);
      new_color[2]=(unsigned char)(drand48()*255.);
      set_property(i,"color", new_color);
      p->set_color( Color( new_color[0]/255., new_color[1]/255., 
			   new_color[2]/255. ) );
    } else 
      p->set_color( Color( color[1]/255., color[2]/255., color[3]/255. ) );
    poly_.push_back( p );
    diagram_->add( p );
  } 
}
#endif

#ifdef CHRIS
void
GraphGui::add_values( unsigned item, const vector<double> &v )
{
  poly_[item]->add(v);

  if ( item == poly_.size()-1 )
    graph_->need_redraw();
}
#else
void
GraphGui::add_values( const vector<double> &v )
{
  for (unsigned i=0; i<v.size(); i++)
    poly_[i]->add(v[i]);

  graph_->need_redraw();
}
#endif

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

  PartGui::attach(interface);
}

void
GraphGui::set_window( const string &window )
{
  graph_->set_window( window );
}


} // namespace MIT





