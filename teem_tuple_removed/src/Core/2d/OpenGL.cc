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
#include <stdlib.h>

#include <Core/2d/OpenGL.h>
#include <Core/2d/glprintf.h>
#include <Core/2d/Polyline.h>
#include <Core/2d/ParametricPolyline.h>
#include <Core/2d/LockedPolyline.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Hairline.h>
#include <Core/2d/HistObj.h>
#include <Core/2d/BoxObj.h>
#include <Core/2d/Zoom.h>
#include <Core/2d/Axes.h>

#include <GL/gl.h>
#include <sci_glu.h>
#if HAVE_GL_GLS_H
#include <GL/gls.h>
#endif


#include <X11/X.h>
#include <X11/Xlib.h>

using namespace std;
using namespace SCIRun;

void 
Polyline::draw( bool )
{
  glColor3f( color_.r(), color_.g(), color_.b() );
  
  glBegin(GL_LINE_STRIP);
  for (unsigned i=0; i<data_.size(); i++) 
    glVertex2f( i, data_[i] );
  glEnd();
}

void 
LockedPolyline::draw( bool )
{
  glColor3f( color_.r(), color_.g(), color_.b() );
  
  read_lock();

  glBegin(GL_LINE_STRIP);
  for (unsigned i=0; i<data_.size(); i++) 
    glVertex2f( i, data_[i] );
  glEnd();

  read_unlock();
}

void
ParametricPolyline::draw( bool )
{
  glColor3f( color_.r(), color_.g(), color_.b() );

  glBegin(GL_LINE_STRIP);

  iter i = data_.begin();

  // we want to draw the points in parameter sorted order
  // maps iterate in sorted order of their key
  // the parameter of a parametric polyline is the map's key!
  while (i != data_.end()) 
    glVertex2f((*i).second.first,(*i++).second.second);


  glEnd();
}
  
void 
HistObj::draw( bool )
{
  glColor3f( color_.r(), color_.g(), color_.b() );
  glBegin(GL_QUADS);
  double pos = ref_min_;
  double dp = (ref_max_ - ref_min_)/bins_;
  for (unsigned i=0; i<data_.size(); i++) {
    glVertex2f(pos,0);
    glVertex2f(pos+dp,0);
    glVertex2f(pos+dp,data_[i]);
    glVertex2f(pos,data_[i]);
    pos += dp;
  }
  glEnd();

}

void 
Diagram::draw( bool pick)
{
  if ( poly_.size() == 0 ) return; 

  glMatrixMode(GL_PROJECTION);

  bool have_some = false;

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
	  
	  have_some = true;
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

	      have_some = true;
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

	  have_some = true;
	}
      }
    }  

    if ( have_some ) {
      // display the widgets
      for (int i=0; i<widget_.size(); i++) 
	widget_[i]->draw();
    }
  }
  else { // pick 
    // in this mode we only draw the widgets
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

#define CHAR_SIZE 0.04
#define CHAR_W CHAR_SIZE
#define CHAR_H CHAR_SIZE*1.61
  
void
XAxis::draw( bool pick )
{
  double loc[] = {0,0,0};
  double norm[] = {0,0,-1};
  double up[] = {0,1,0};
  int loop;
  
  if (!initialized_) {
    initialized_ = true;
    init_glprintf();
  }

  // draw the axes and tic marks
  XAxisObj::draw( pick );

  if (!pick) {
    // transform the positions to NDC (with y flip)
    double pos = (1.-pos_)*2.-1.;
    
    double delta = 2./(num_tics_+1);
    
    glTextNormal(norm);
    glTextAlign(GL_LEFT);
    glTextSize(CHAR_W,CHAR_H);

    // set the projection to NDC
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glOrtho(-1,1,-1,1,0,1);
    
    // draw the tic mark values
    up[0] = 0.544;
    up[1] = 0.839;
    up[2] = 0.;
    glTextUp(up);
    loc[1] = pos-0.0163;
    for (loop=1;loop<=num_tics_;++loop) {
      loc[0] = (loop*delta)-(.5*CHAR_W)-1.;
      glTextAnchor(loc);
      glprintf("%-1.3f",parent_->x_get_at((loc[0]+1.)/2.));

    }

    glPopMatrix();
  }
}

void
YAxis::draw( bool pick )
{
  double loc[] = {0,0,0};
  double norm[] = {0,0,-1};
  double up[] = {0,1,0};
  int loop;
  
  if (!initialized_) {
    initialized_ = true;
    init_glprintf();
  }

  // draw the axes and tic marks
  YAxisObj::draw( pick );

  if (!pick) {
    // transform the position to NDC 
    double pos = pos_*2.-1.;

    double delta = 2./(num_tics_+1);

    glTextNormal(norm);
    glTextAlign(GL_RIGHT);
    glTextSize(CHAR_W,CHAR_H);
    
    // set the projection to NDC
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glOrtho(-1,1,-1,1,0,1);
    
    // draw the tic mark values
    up[0] = 0.;
    up[1] = 1.;
    up[2] = 0.;      
    glTextUp(up);
    loc[0] = pos-0.01;
    for (loop=1;loop<=num_tics_;++loop) {
      loc[1] = (loop*delta)+(.5*CHAR_H)-1.;
      glTextAnchor(loc);
      glprintf("%-1.3f",parent_->y_get_at((loc[1]+1.)/2.));
    }
    
    glPopMatrix();
  }
}


void 
XAxisObj::draw( bool pick)
{
  double delta = 2./(num_tics_+1);
  int loop;

  // transform the pos to NDC (with y flip if not picking)
  double pos;
  if (!pick)
    pos = (1.-pos_)*2.-1.;
  else
    pos = pos_*2.-1.;

  // set the projection to NDC
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glOrtho(-1,1,-1,1,0,1);

  glColor3f(0.f,0.f,0.f);

  // draw the axes lines
  glBegin(GL_LINES);
    glVertex2f(-1.0f,pos);
    glVertex2f(1.0f,pos);
  glEnd();

  // draw the tic marks
  if (!pick) {
    glBegin(GL_LINES);
    for (loop=1;loop<=num_tics_;++loop) {
      glVertex2d(loop*delta-1.,pos+0.0161);
      glVertex2d(loop*delta-1.,pos-0.0161);
    }
    glEnd();
  }

  // restore the original projection
  glPopMatrix();
}


void 
YAxisObj::draw( bool pick)
{
  double delta = 2./(num_tics_+1);
  int loop;

  // transform the position to NDC
  double pos = pos_*2.-1.;

  // set the projection to NDC
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glOrtho(-1,1,-1,1,0,1);

  glColor3f(0.f,0.f,0.f);

  // draw the axes lines
  glBegin(GL_LINES);
    glVertex2f(pos,-1.0f);
    glVertex2f(pos,1.0f);
  glEnd();

  // draw the tic marks
  if (!pick) {
    glBegin(GL_LINES);
    for (loop=1;loop<=num_tics_;++loop) {
      glVertex2d(pos+0.01,delta*loop-1.);
      glVertex2d(pos-.01,delta*loop-1.);
    }
    glEnd();
  }

  // restore the original projection
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



