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
 *  OpenGL.cc: Rendering for OpenGL windows
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/NotFinished.h>

#include <Core/2d/OpenGL.h>
#include <Core/2d/Polyline.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Axes.h>

#include <GL/gl.h>
#include <GL/glu.h>
#if !defined(__linux) && !defined(_WIN32)
#include <GL/gls.h>
#endif


#include <X11/X.h>
#include <X11/Xlib.h>


namespace SCIRun {


void 
Polyline::draw()
{
  glColor3f( color.r(), color.g(), color.b() );
  glBegin(GL_LINE_STRIP);
  for (int i=0; i<data_.size(); i++) 
    glVertex2f( i, data_[i] );
  glEnd();
}

void 
Diagram::draw()
{
  if ( graph_.size() == 0 ) return; 

  if ( select_mode == 2 ) { // select_mode = many
    if ( scale_mode == 1 ) { // scale_mode = all
      reset_bbox();
      
      if ( !graphs_bounds_.valid() ) return;
      
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho( graphs_bounds_.min().x(),  graphs_bounds_.max().x(),
	       graphs_bounds_.min().y(),  graphs_bounds_.max().y(),
	       -1, 1 );
      
      glColor3f( 0,0,0 );
      for (int i=0; i<graph_.size(); i++) 
	if ( graph_[i]->is_enabled() ) 
	  graph_[i]->draw();
    }
    else { // scale_mode == each
      for (int i=0; i<graph_.size(); i++) 
	if ( graph_[i]->is_enabled() ) {
	  BBox2d bbox;
	  graph_[i]->get_bounds( bbox );
	  if ( bbox.valid() ) {
	    glMatrixMode(GL_PROJECTION);
	    glLoadIdentity();
	    glOrtho( bbox.min().x(),  bbox.max().x(),
		     bbox.min().y(),  bbox.max().y(),
		     -1, 1 );
	    graph_[i]->draw();
	  }
	}
    }
  }
  else { // select_mode == one
    if ( graph_[selected_]->is_enabled() ) {
      BBox2d bbox;
      graph_[selected_]->get_bounds( bbox );
      if ( bbox.valid() ) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho( bbox.min().x(),  bbox.max().x(),
		 bbox.min().y(),  bbox.max().y(),
		 -1, 1 );
	graph_[selected_]->draw();
      }
    }
  }
}

void 
Axes::draw()
{
  glBegin(GL_LINE);
  glVertex2f(0.0f, 0.0f);
  glVertex2f(100.0f, 100.0f);
  glEnd();
}

} // End namespace SCIRun



