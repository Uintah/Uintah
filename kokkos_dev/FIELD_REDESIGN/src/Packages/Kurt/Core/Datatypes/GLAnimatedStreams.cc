#include "GLAnimatedStreams.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
#include <GL/gl.h>
#include <math.h>
#include <float.h>
#include <iostream>

namespace SCICore {
namespace GeomSpace  {

using std::cerr;
using std::endl;

using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;

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
    _vfH(0), _cmapH(0), _pause(false), _normalsOn(false),
    _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2)
{

  NOT_FINISHED("GLAnimatedStreams::GLAnimatedStreams(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLAnimatedStreams::GLAnimatedStreams(int id, 
				   VectorFieldHandle vfh,
				   ColorMapHandle map)
  : GeomObj( id ),  mutex("GLAnimatedStreams Mutex"),
    _vfH(vfh), _cmapH(map), _pause(false), _normalsOn(false),
    _stepsize(0.1), fx(0), tail(0), head(0), _numStreams(0),
    _linewidth(2)
{
  init();
}

GLAnimatedStreams::GLAnimatedStreams(const GLAnimatedStreams& copy)
  : GeomObj( copy.id ), mutex("GLAnimatedStreams Mutex"),
    _vfH(copy._vfH), _cmapH(copy._cmapH), fx(copy.fx),
    head(copy.head), tail(copy.tail), _numStreams(copy._numStreams),
    _pause(copy._pause), _normalsOn(copy._normalsOn),
    _linewidth(copy._linewidth)
{
  
} 

void GLAnimatedStreams::init()
{

  if( fx )
    delete [] fx;
  if( tail ) 
    delete [] tail;
  if( head )
    delete [] head;
  
  fx = new Point[NUMSOLS];
  maxwidth = _vfH->longest_dimension();
  maxspeed = maxwidth/50.0;
  minspeed = maxwidth/200.0;
  if(!_cmapH->IsScaled()){
    _cmapH->Scale(minspeed, maxspeed);
  }
  tail = new streamerNode*[NUMSOLS];
  head = new streamerNode*[NUMSOLS];

  Point min, max;
  _vfH->get_bounds(min, max);

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
GLAnimatedStreams::saveobj(std::ostream&, const clString& format, GeomSave*)
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
  int i;
  _vfH->interpolate(x,m1);
  x1 = x + m1*h2;
  _vfH->interpolate(x1,m2);
  x2 = x + m2*h2;
  _vfH->interpolate(x2,m3);
  x3 = x + m3*h;
  _vfH->interpolate(x3,m4);
  x += (m1 + m2 * 2.0 + m3 * 2.0 + m4) * _stepsize * (1.0/6.0); 
}


void 
GLAnimatedStreams::newStreamer(int whichStreamer)
{
  streamerNode* tempNode;
  int i;
  Point min, max;
  _vfH->get_bounds(min, max);
  
  Vector toPoint = Vector( (max.x() - min.x()) * dRANDOM(),
			   (max.y() - min.y()) * dRANDOM(),
			   (max.z() - min.z()) * dRANDOM());

  fx[whichStreamer] = min + toPoint;

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
  _vfH->get_bounds(min, max);
  Vector right;

  BBox bounds;
  bounds.extend(min);
  bounds.extend(max);

  int i;
  double l, nl;

  nl = maxwidth / 50.0;
  
  streamerNode* tempNode;

  

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

                                // add streams
  _numStreams++;
				// but only up to a certain point
  if (_numStreams > NUMSOLS) _numStreams = NUMSOLS;
  
				// start the new solution if we're
				// adding one
  else { newStreamer(_numStreams-1); }     
  
}


  
} // namespace SCICore
} // namespace GeomSpace
