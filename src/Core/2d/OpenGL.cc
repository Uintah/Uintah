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
#include <Core/2d/Hairline.h>
#include <Core/2d/BoxObj.h>
#include <Core/2d/Zoom.h>
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
Polyline::draw( bool )
{
  glColor3f( color_.r(), color_.g(), color_.b() );
  
  read_lock();

  glBegin(GL_LINE_STRIP);
  for (int i=0; i<data_.size(); i++) 
    glVertex2f( i, data_[i] );
  glEnd();

  read_unlock();
}
  
void 
Diagram::draw( bool pick)
{
  if ( poly_.size() == 0 ) return; 

  glMatrixMode(GL_PROJECTION);

  if  ( !pick ) {
    if ( select_mode_ == 2 ) { // select_mode_ = many
      if ( scale_mode_ == 1 ) { // scale_mode_ = all
	
	reset_bbox();
	
	if ( graphs_bounds_.valid() ) {
	
	  glPushMatrix();
	  glOrtho( graphs_bounds_.min().x(),  graphs_bounds_.max().x(),
		   graphs_bounds_.min().y(),  graphs_bounds_.max().y(),
		   -1, 1 );
	  
	  glColor3f( 0,0,0 );
	  for (int i=0; i<poly_.size(); i++) 
	    if ( active_[i] ) 
	      poly_[i]->draw();
	  glPopMatrix();
	}
      }
      else { // scale_mode == each
	for (int i=0; i<poly_.size(); i++) 
	  if ( active_[i] ) {
	    BBox2d bbox;
	    poly_[i]->get_bounds( bbox );
	    if ( bbox.valid() ) {
	      glMatrixMode(GL_PROJECTION);
	      glPushMatrix();
	      glOrtho( bbox.min().x(),  bbox.max().x(),
		       bbox.min().y(),  bbox.max().y(),
		       -1, 1 );
	      poly_[i]->draw();
	      glPopMatrix();
	    }
	  }
      }
    }
    else { // select_mode == one
      if ( active_[selected_] ) {
	BBox2d bbox;
	poly_[selected_]->get_bounds( bbox );
	if ( bbox.valid() ) {
	  glMatrixMode(GL_PROJECTION);
	  glPushMatrix();
	  glOrtho( bbox.min().x(),  bbox.max().x(),
		   bbox.min().y(),  bbox.max().y(),
		   -1, 1 );
	  poly_[selected_]->draw();
	  glPopMatrix();
	}
      }
    }  

    // display the widgets
    for (int i=0; i<widget_.size(); i++) 
      widget_[i]->draw();
  }
  else { // pick 
    // in mode we only draw the widgets
    for (int i=0; i<widget_.size(); i++) {
      glLoadName( i );
      widget_[i]->draw( true );
    }
  }
}

void
Hairline::draw( bool pick )
{
  HairObj::draw( pick );
  update();
}

void
HairObj::draw( bool ) 
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();

  glOrtho( 0, 1,  0, 1,  -1, 1 );

  glColor3f(0,0,0);
  glBegin(GL_LINES);
    glVertex2f( pos_, 0 );
    glVertex2f( pos_, 1 );
  glEnd();

  glPopMatrix();
}
  
  

void 
Axes::draw( bool )
{
  if (!initialized) {
    init_glprintf();
    initialized = true;
  }

  double smidge;

  // set the projection to NDC
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
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
  glPopMatrix();
}


void
BoxObj::draw( bool pick )
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();

  if ( pick )
    glOrtho( 0, 1,  1, 0,  -1, 1 );
  else
    glOrtho( 0, 1,  0, 1,  -1, 1 );

  glColor3f(0,0,0);

  glBegin(GL_LINE_LOOP);
    glVertex2f( screen_.min().x(), screen_.min().y());
    glVertex2f( screen_.max().x(), screen_.min().y());
    glVertex2f( screen_.max().x(), screen_.max().y());
    glVertex2f( screen_.min().x(), screen_.max().y());
  glEnd();

  glPopMatrix();
}

void
Zoom::draw( bool )
{
  cerr << "ZoomObj draw" << endl;
}


} // End namespace SCIRun



