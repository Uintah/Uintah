#include "GLAnimatedStreams.h"
#include "LevelField.h"
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <GL/gl.h>
#include <math.h>
#include <float.h>
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


GLAnimatedStreams::GLAnimatedStreams(int id) 
  : GeomObj( id ), mutex("GLAnimatedStreams Mutex"),
    vfh_(0), _cmapH(0), _pause(false), _normalsOn(false),
    _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2), flow(0), _usesWidget(false), widgetLocation(0,0,0)
{

  NOT_FINISHED("GLAnimatedStreams::GLAnimatedStreams(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLAnimatedStreams::GLAnimatedStreams(int id, 
				   FieldHandle vfh,
				   ColorMapHandle map)
  : GeomObj( id ),  mutex("GLAnimatedStreams Mutex"),
    vfh_(vfh), _cmapH(map), _pause(false), _normalsOn(false),
    _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2), flow(0), _usesWidget(false), widgetLocation(0,0,0)
{
  init();
}

GLAnimatedStreams::GLAnimatedStreams(const GLAnimatedStreams& copy)
  : GeomObj( copy.id ), mutex("GLAnimatedStreams Mutex"),
    vfh_(copy.vfh_), _cmapH(copy._cmapH), fx(copy.fx),
    head(copy.head), tail(copy.tail), _numStreams(copy._numStreams),
    _pause(copy._pause), _normalsOn(copy._normalsOn),
    _linewidth(copy._linewidth), flow(copy.flow),
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
  
  fx = new Point[NUMSOLS];
  tail = new streamerNode*[NUMSOLS];
  head = new streamerNode*[NUMSOLS];

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

#ifdef SCI_OPENGL
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
  mutex.unlock();

}
#endif

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
  //  glEnable(GL_LIGHTING);
  
  
}

void
GLAnimatedStreams::postDraw()
{
  // opposite preDraw
  glLineWidth( 1 );
  glPopMatrix();
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
  Vector m1,m2,m3,m4;
  Point  x1,x2,x3;
  double h2 = _stepsize/2.0;
  //int i;  <- Not used (Dd)
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
  //  if( _usesWidget && 
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
  
  tempNode = new streamerNode;
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
  Vector right;
  
  int i;
  double l, nl;
  
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

  // now increment the streams
  // this should be done only if the animation flag is on or there are values
  // in the flow variable
  if ( !_pause || flow) {
    for (i = 0; i < _numStreams; i++) {
      
				// flow forward	  
      RungeKutta(fx[i], _stepsize);
      if( !bounds.inside(fx[i]) )
	head[i]->counter = MAX_ITERS;
      
      if( head[i]->counter < MAX_ITERS) {
	head[i]->next = new streamerNode;
	head[i]->next->counter = head[i]->counter+1;
	tempNode = head[i];
	head[i] = head[i]->next;
	
	head[i]->position = fx[i];
	head[i]->tangent = head[i]->position - tempNode->position;
	
	right = Cross( head[i]->tangent,  tempNode->tangent );
	if (-1e-6 < right.length() && right.length() < 1e-6 ){
	  right = Cross( head[i]->tangent,  tempNode->normal );
	head[i]->normal = Cross( right, head[i]->tangent );
	} else {
	  head[i]->normal = Cross( right, head[i]->tangent );
	}

	l = 1.0/(head[i]->normal.length());
	
	head[i]->normal *= l;
	
	l = head[i]->tangent.length();
	
	if(-1e-6 < l && l < 1e-6 ){
	  head[i]->counter = MAX_ITERS;
	}
	
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
    }
    if (flow) DecrementFlow();
  } // end if ( !_pause || flow)

                                // add streams
  _numStreams++;
				// but only up to a certain point
  if (_numStreams > NUMSOLS) _numStreams = NUMSOLS;
  
				// start the new solution if we're
				// adding one
  else { newStreamer(_numStreams-1); }     
  
}

bool  
GLAnimatedStreams::interpolate(FieldHandle texfld_, const Point& p, Vector& val)
{
  const string field_type = texfld_->get_type_name(0);
  const string type = texfld_->get_type_name(1);
  if( texfld_->get_type_name(0) == "LevelField" ){
    // this should be faster than the virtual function call, but
    // it hasn't been tested.
    if (type == "Vector") {
      LevelField<Vector> *fld =
	dynamic_cast<LevelField<Vector>*>(texfld_.get_rep());
      return fld->interpolate(val ,p);
    } else {
      cerr << "Uintah::AnimatedStreams::interpolate:: error - unimplemented Field type: " << type << endl;
      return false;
    }
  } else if( texfld_->get_type_name(0) == "LatticeVol" ){
    VectorFieldInterface *vfi;
    // use virtual field interpolation
    if( vfi = texfld_->query_vector_interface()){
      return vfi->interpolate( val, p);
    }
    return false; // Added by Dd... if this is correct, remove this comment
  } else {
    return false;
  }
}

  
} // namespace Uintah
