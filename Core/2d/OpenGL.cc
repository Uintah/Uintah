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

#define SMIDGE 0.05

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
  double smidgex;
  double smidgey;
  
  if ( graph_.size() == 0 ) return; 

  if ( select_mode == 2 ) { // select_mode = many
    if ( scale_mode == 1 ) { // scale_mode = all
      reset_bbox();
      
      if ( !graphs_bounds_.valid() ) return;
      
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();

      smidgex = (graphs_bounds_.max().x()-graphs_bounds_.min().x())*SMIDGE;
      smidgey = (graphs_bounds_.max().y()-graphs_bounds_.min().y())*SMIDGE;

      glOrtho( graphs_bounds_.min().x()-smidgex,
	       graphs_bounds_.max().x()+smidgex,
	       graphs_bounds_.min().y()-smidgey,
	       graphs_bounds_.max().y()+smidgey,
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
	  smidgex = (bbox.max().x()-bbox.min().x())*SMIDGE;
	  smidgey = (bbox.max().y()-bbox.min().y())*SMIDGE;
	  if ( bbox.valid() ) {
	    glMatrixMode(GL_PROJECTION);
	    glLoadIdentity();
	    glOrtho( bbox.min().x()-smidgex,  bbox.max().x()+smidgex,
		     bbox.min().y()-smidgey,  bbox.max().y()+smidgey,
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
      smidgex = (bbox.max().x()-bbox.min().x())*SMIDGE;
      smidgey = (bbox.max().y()-bbox.min().y())*SMIDGE;
      if ( bbox.valid() ) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho( bbox.min().x()-smidgex,  bbox.max().x()+smidgex,
		 bbox.min().y()-smidgey,  bbox.max().y()+smidgey,
		 -1, 1 );
	graph_[selected_]->draw();
      }
    }
  }
}

void 
Axes::draw()
{
  if (!initialized) {
    init_glprintf();
    initialized = true;
  }

  double pm[16];
  double smidge;
  glGetDoublev(GL_PROJECTION_MATRIX,pm);

  // set the projection to NDC
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  smidge = 2.*SMIDGE/(1.+2.*SMIDGE);

  glBegin(GL_LINES);
  glVertex2f(-1.0f,0.0f);
  glVertex2f(1.0f,0.0f);
  glVertex2f(-1.f+smidge,-1.0f);
  glVertex2f(-1.f+smidge,1.0f);
  glEnd();

  double hdelta = 2./num_h_tics;
  double vdelta = 2./num_v_tics;
  int loop;

  glBegin(GL_LINES);
  for (loop=0;loop<num_h_tics;++loop) {
    glVertex2d(loop*hdelta-1.,0.0161);
    glVertex2d(loop*hdelta-1.,-0.0161);
  }

  for (loop=0;loop<num_v_tics;++loop) {
    glVertex2d(-0.99+smidge,vdelta*loop-1);
    glVertex2d(-1.01+smidge,vdelta*loop-1);
  }
  glEnd();

  // restore the projection
  glLoadMatrixd(pm);
}

} // End namespace SCIRun



