/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "GLAnimatedStreams.h"
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <sci_gl.h>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>

namespace Uintah {

using std::cerr;
using std::endl;

using namespace SCIRun;

const double GLAnimatedStreams::FADE = 9.0;
const int GLAnimatedStreams::MAXN = 10;

const int NUMSOLS = 1000;		// total number of solutions to draw

const int MAX_ITERS = 500;		// the max. # of iterations of a
				// solution
int STREAMER_LENGTH = 50;

const double RMAX = (double)(((int)RAND_MAX) + 1);
inline double dRANDOM() {
  return ((((double)rand()) + 0.5) / RMAX);
}


GLAnimatedStreams::GLAnimatedStreams() 
  : GeomObj(), mutex("GLAnimatedStreams Mutex"),
    vfh_(0), _cmapH(0), _pause(false), _normalsOn(false), _lighting(false),
    _use_dt(false), _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2), flow(0), _delta_T(-1), _normal_method(STREAM_LIGHT_WIRE),
    _usesWidget(false), widgetLocation(0,0,0)
{

  NOT_FINISHED("GLAnimatedStreams::GLAnimatedStreams(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLAnimatedStreams::GLAnimatedStreams(FieldHandle vfh, ColorMapHandle map)
  : GeomObj(),  mutex("GLAnimatedStreams Mutex"),
    vfh_(vfh), _cmapH(map), _pause(false), _normalsOn(false), _lighting(false),
    _use_dt(false), _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2), flow(0), _delta_T(-1), _normal_method(STREAM_LIGHT_WIRE),
    _usesWidget(false), widgetLocation(0,0,0)
{
  init();
}

GLAnimatedStreams::GLAnimatedStreams(const GLAnimatedStreams& copy)
  : GeomObj( copy ), mutex("GLAnimatedStreams Mutex"),
    vfh_(copy.vfh_), _cmapH(copy._cmapH),
    _pause(copy._pause), _normalsOn(copy._normalsOn),
    _lighting(copy._lighting), _use_dt(copy._use_dt),
    _stepsize(copy._stepsize), fx(copy.fx),
    tail(copy.tail), head(copy.head), _numStreams(copy._numStreams),
    _linewidth(copy._linewidth), flow(copy.flow), _delta_T(copy._delta_T), 
    _normal_method(copy._normal_method),
    _usesWidget(copy._usesWidget), widgetLocation(copy.widgetLocation)
{
  
} 

void GLAnimatedStreams::init()
{
  ResetStreams();
  initColorMap();
}

void GLAnimatedStreams::initColorMap() {
  BBox bb;
  bb  = vfh_->mesh()->get_bounding_box();
  maxwidth = std::max(bb.max().x() - bb.min().x(),
		      std::max(bb.max().y() - bb.min().y(),
			       bb.max().z() - bb.min().z()));
  maxspeed = maxwidth/50.0;
  minspeed = maxwidth/200.0;
  if(!_cmapH->IsScaled()){
    _cmapH->Scale(minspeed, maxspeed);
  }
}

void GLAnimatedStreams::SetColorMap( ColorMapHandle map) {
  mutex.lock();
  this->_cmapH = map;
  cmapHasChanged = true;
  initColorMap();
  mutex.unlock();
}

void GLAnimatedStreams::SetVectorField( FieldHandle vfh ){
  cerr << "GLAnimatedStreams::SetVectorField:start\n";
  
  mutex.lock();
  this->vfh_ = vfh;
  mutex.unlock();
  init();
}

// tries to keep going with a new vector field.  If the new field is
// incompatable with the old one then the effective call is to
// SetVectorField().
void GLAnimatedStreams::ChangeVectorField( FieldHandle new_vfh ) {
  cerr << "GLAnimatedStreams::ChangeVectorField:start\n";
  BBox old_bbox = vfh_->mesh()->get_bounding_box();
  BBox new_bbox = new_vfh->mesh()->get_bounding_box();
  if (old_bbox.min() == new_bbox.min() && old_bbox.max() == new_bbox.max()){
    // then the bounding boxes are the same
    mutex.lock();
    vfh_ = new_vfh;
    mutex.unlock();
  } else {
    // different bounding boxes - call SetVectorField
    SetVectorField(new_vfh);
  }
}

void GLAnimatedStreams::ResetStreams() {
  mutex.lock();

  if( fx )
    delete [] fx;
  if( tail ) 
    delete [] tail;
  if( head )
    delete [] head;
  
  fx = scinew Point[NUMSOLS];
  tail = scinew streamerNode*[NUMSOLS];
  head = scinew streamerNode*[NUMSOLS];

  cerr << "GLAnimatedStreams::ResetStreams:end\n";

  mutex.unlock();
}

void GLAnimatedStreams::IncrementFlow() {
  //  mutex.lock();
  flow++;
  cerr << "increment flow: " << flow << endl;
  //  mutex.unlock();
}

void GLAnimatedStreams::DecrementFlow() {
  //  mutex.lock();
  flow--;
  cerr << "decrement flow: " << flow << endl;
  //  mutex.unlock();
}

void GLAnimatedStreams::SetNormalMethod( int method ) {
  _normal_method = method;
}

GLAnimatedStreams::~GLAnimatedStreams()
{
  delete [] fx;
  delete [] tail;
  delete [] head;
}

GeomObj* 
GLAnimatedStreams::clone()
{
  return scinew GLAnimatedStreams( *this );
}

void 
GLAnimatedStreams::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  if( !pre_draw(di, mat, 0) ) return;
  mutex.lock();
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    setup();
    preDraw();
    draw();
    postDraw();
    cleanup();
  }
  // now increment the streams
  // this should be done only if the animation flag is on or there are values
  // in the flow variable
  if ( !_pause || flow) {
    AdvanceStreams(0,_numStreams);
    if (flow) DecrementFlow();
    // add streams
    _numStreams++;
    // but only up to a certain point
    if (_numStreams > NUMSOLS) _numStreams = NUMSOLS;
    // start the new solution if we're
    // adding one
    else { newStreamer(_numStreams-1); }     
  } // end if ( !_pause || flow)
  mutex.unlock();

}

void
GLAnimatedStreams::drawWireFrame()
{
  
}

void
GLAnimatedStreams::preDraw()
{
  // Open GL enable or diable;
  glLineWidth( _linewidth );
  glPushMatrix();

  //glTranslatef(0, 0, -1.5*maxwidth);

  				// draw messages
  //glRotatef(xrot, 0, 1, 0);
  //glRotatef(yrot, 1, 0, 0);

  
#if 0
  GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
  GLfloat mat_shininess[] = {50.0 };
  
  glShadeModel (GL_SMOOTH);
  glMaterialfv( GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv( GL_FRONT, GL_SHININESS, mat_shininess);
  
  GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  
  glEnable(GL_LIGHT0);
#endif
  gl_lighting_disabled = !((bool) glIsEnabled(GL_LIGHTING));
  if (_lighting && gl_lighting_disabled) {
    glEnable(GL_LIGHTING);
    //    GLfloat light_ambient[] = { 1.0,1.0,1.0,1.0 };
    GLfloat light_ambient[] = { 0.5, 0.5, 0.5, 1.0 };
    //    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
    //glEnable(GL_LIGHT_MODEL_AMBIENT);
    //    GLfloat matl_ambient[] = { 0.5, 0.5, 0.5, 1.0 };
    //glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,matl_ambient);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  }
}

void
GLAnimatedStreams::postDraw()
{
  // opposite preDraw
  glLineWidth( 1 );
  glPopMatrix();
  if (_lighting && gl_lighting_disabled) {
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
  }
}





void
GLAnimatedStreams::setup()
{

}


void
GLAnimatedStreams::cleanup()
{


}  


#define GLANIMATEDSTREAMS_VERSION 1

void GLAnimatedStreams::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("GLAnimatedStreams::io");
}


bool
GLAnimatedStreams::saveobj(std::ostream&, const string&, GeomSave*)
{
   NOT_FINISHED("GLAnimatedStreams::saveobj");
    return false;
}


//----------------------------------------------------------------------
// RUNGEKUTTA
//----------------------------------------------------------------------
				// move forward one single streamline
				// by one step
void
GLAnimatedStreams::RungeKutta(Point& x, double h) {
  //  cerr << "(h = " << h << ")";
  Vector m1,m2,m3,m4;
  Point  x1,x2,x3;
  double h2 = h/2.0;
  interpolate(vfh_, x,m1);
  x1 = x + m1*h2;
  interpolate(vfh_, x1,m2);
  x2 = x + m2*h2;
  interpolate(vfh_, x2,m3);
  x3 = x + m3*h;
  interpolate(vfh_, x3,m4);
  x += (m1 + m2 * 2.0 + m3 * 2.0 + m4) * _stepsize * (1.0/6.0); 
}


void 
GLAnimatedStreams::newStreamer(int whichStreamer)
{
  streamerNode* tempNode;
  double costheta;
  double sintheta;
  double phi;
  BBox bb;
  Point min, max;
  bb = vfh_->mesh()->get_bounding_box();
  min = bb.min(); max = bb.max();

  Vector toPoint(0,0,0);
  Point loc;
  // if we are using a widget then we want to choose a point near
  // the widget to start the stream.  This uses a simple rejection
  // test to determine the usability of the sample.
  if( bb.inside(widgetLocation)){
    double r = std::min(max.x() - min.x(),
			std::min(max.y() - min.y(), max.z() - min.z()));
    r /= 2.0;
    double scale;
    do{
      costheta = 1.0 - 2.0 * dRANDOM();
      sintheta = sqrt(1 - costheta * costheta);
      phi = 2 * M_PI * dRANDOM();
      toPoint = Vector(r * cos(phi) * sintheta,
		       r * sin(phi) * sintheta,
		       r * costheta);
      scale = dRANDOM();
      loc = widgetLocation + toPoint*(scale*scale);
    } while (!bb.inside( loc ));
  } else {
    // we just pick a random Point in the domain.
    toPoint = Vector( (max.x() - min.x()) * dRANDOM(),
		      (max.y() - min.y()) * dRANDOM(),
		      (max.z() - min.z()) * dRANDOM());
    loc = min + toPoint;
  }
  fx[whichStreamer] = loc;
  
  tempNode = scinew streamerNode;
  tempNode->position = fx[whichStreamer];

  tempNode->normal = Vector(0,0,0);
  tempNode->color = Material(Color(0,0,0));
  tempNode->tangent = Vector(0,0,0);
  tempNode->counter = 0;
  tempNode->next = 0;
  head[whichStreamer] = tail[whichStreamer] = tempNode;

}

  
void
GLAnimatedStreams::draw()
{
  // meat and potatoes
  Point min, max;
  BBox bounds;
  
  bounds = vfh_->mesh()->get_bounding_box();
  min = bounds.min(); max = bounds.max();
  
  int i;
  double nl;
  
  nl = maxwidth / 50.0;
  
  streamerNode* tempNode;
  
  // draw the streams as a linked list starting with tail and going to head
  for (i = 0; i < _numStreams; i++) {
    tempNode = tail[i];      
                                // skip two-- the normals for the
			        // first two iterations are always
				// funky
    tempNode = tempNode->next;
      if ( !tempNode ) continue;
      tempNode = tempNode->next;
      if ( !tempNode ) continue;

      glBegin(GL_LINE_STRIP);
      
      while ( tempNode ) {
	Color c = tempNode->color.diffuse;
	glColor3f(c.r(), c.g(), c.b());
	if (_lighting)
	  glNormal3f(tempNode->normal.x(), tempNode->normal.y(),
		     tempNode->normal.z());
	glVertex3f(tempNode->position.x(), tempNode->position.y(),
		   tempNode->position.z());

	if (_normalsOn) {
				// draw out to the normal point and
				// back (because we're using line
				// strips)
	  glVertex3f(tempNode->position.x()+tempNode->normal.x() * nl,
		     tempNode->position.y()+tempNode->normal.y() * nl,
		     tempNode->position.z()+tempNode->normal.z() * nl);
	  glVertex3f(tempNode->position.x(), tempNode->position.y(),
		   tempNode->position.z());
	}
	tempNode = tempNode->next;
      }
      glEnd();
  }
}

void GLAnimatedStreams::AdvanceStreams(int start, int end) {
  if (end > _numStreams)
    end = _numStreams;
  
  BBox bounds = vfh_->mesh()->get_bounding_box();
  GLfloat light_pos[4];
  glGetLightfv(GL_LIGHT0, GL_POSITION, light_pos);
  Vector Light(light_pos[0],light_pos[1],light_pos[2]);

  // if you are not using the iteration step method you should only go through
  // this loop once.
  double iterations;
  if (!_use_dt || _stepsize >= _delta_T) {
    iterations = 1;
  } else {
    iterations = _delta_T / _stepsize;
  }
  // iter starts at how many interations to do and goes to 0
  // iter can be used as a fraction when it's < 1 (it's still > 0, though)
  for (double iter = iterations; iter > 0; iter--) {
    cerr << "iter = " << iter << endl;
    for (int i = start; i < end; i++) {
	
      // flow forward
      Vector tangent;
      interpolate(vfh_, fx[i],tangent);
      // compute the next location
      if (!_use_dt) {
	RungeKutta(fx[i], _stepsize);
	//cerr << "Using step size\n";
      } else {
	if (_stepsize >= _delta_T) {
	  RungeKutta(fx[i], _delta_T);
	} else {
	  if (iter >= 1)
	    RungeKutta(fx[i], _stepsize);
	  else
	    RungeKutta(fx[i], _stepsize * iter);
	}
      }

      if( !bounds.inside(fx[i]) )
	head[i]->counter = MAX_ITERS;
      
      streamerNode* tempNode;
      if( head[i]->counter < MAX_ITERS) {
	head[i]->next = scinew streamerNode;
	head[i]->next->counter = head[i]->counter+1;
	tempNode = head[i];
	head[i] = head[i]->next;
	
	head[i]->position = fx[i];
	head[i]->tangent = tangent;

	// compute the normal
	switch (_normal_method) {
	case STREAM_LIGHT_WIRE:
	  {
	    head[i]->normal = Cross(Cross(head[i]->tangent,
					  Light - Vector(fx[i])),
				    head[i]->tangent);
	  }
	  break;
	case STREAM_LIGHT_CURVE:
	  {
	    Vector right = Cross( head[i]->tangent,  tempNode->tangent );
	    if (-1e-6 < right.length() && right.length() < 1e-6 ){
	      right = Cross( head[i]->tangent,  tempNode->normal );
	      head[i]->normal = Cross( right, head[i]->tangent );
	    } else {
	      head[i]->normal = Cross( right, head[i]->tangent );
	    }
	  }
	  break;
	}

	// normalize the normal
	double nl = 1.0/(head[i]->normal.length());
	head[i]->normal *= nl;
	//	head[i]->normal.normalize();
	
	double l = head[i]->tangent.length();
	
	if(-1e-6 < l && l < 1e-6 ){
	  head[i]->counter = MAX_ITERS;
	}
	
	//	  cerr << "l = " << l << endl;
	head[i]->color = *(_cmapH->lookup( l ).get_rep());
	head[i]->next = 0;
      }
      
      if (head[i]->counter-tail[i]->counter > STREAMER_LENGTH ||
	  head[i]->counter >= MAX_ITERS) {
	tempNode = tail[i];
	tail[i] = tail[i]->next;
	delete tempNode;
	if (tail[i] == NULL) {
	  newStreamer(i);
	}
      }
    } // for i = 0 .. _numStreams
  } // for iter = 0 .. _iterations
}

bool  
GLAnimatedStreams::interpolate(FieldHandle texfld_, const Point& p, Vector& val)
{

  const TypeDescription *td = 
    texfld_->get_type_description(Field::MESH_TD_E);
  if( td->get_name().find("LatVolField") != string::npos ) {
    VectorFieldInterfaceHandle vfi(texfld_->query_vector_interface());
    // use virtual field interpolation
    if( vfi.get_rep() != 0 ){
      return vfi->interpolate( val, p);
    }
    return false; // Added by Dd... if this is correct, remove this comment
  } else {
    return false;
  }
}

  
} // namespace Uintah
