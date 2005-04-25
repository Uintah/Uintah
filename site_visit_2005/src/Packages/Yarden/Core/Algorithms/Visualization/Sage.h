/*
 *  Sage.h
 *      View Depended Iso Surface Extraction
 *      for Structures Grids (Bricks)
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef Sage_h
#define Sage_h

#include <stdio.h>
#include <iostream>

#include <Core/Containers/String.h>
#include <Core/Thread/Time.h>
#include <Dataflow/Network/Module.h> 
#include <Core/Datatypes/ScalarField.h> 
#include <Core/Datatypes/ScalarFieldRG.h> 
#include <Dataflow/Ports/ScalarFieldPort.h>  
#include <Core/Thread/Thread.h>

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/Pt.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/Trig.h>

#include <Packages/Yarden/Core/Algorithms/Visualization/Screen.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/mcube_scan.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/BonTree.h>


namespace Yarden {

using namespace SCIRun;

typedef Time  SysTime;

    
template < class AI >
class SageBase 
{
public:
  static SageBase<AI> *make(ScalarFieldHandle, AI *ai);

protected:
  AI *ai;
  int counter;

  Screen *screen;

  // View parameters
  //	 View *view;
  double znear, zfar;
  int xres, yres;
	 
  GeomGroup* group;
  GeomPts* points;

  Point eye;
  Vector U,V,W;
  Vector AU, AV, AW;
  Vector X,Y,Z;
	 
  double gx, gy, gz;
  double sx, sy, sz;

  // control parameters
  int scan, bbox_visibility, reduce, extract_all, min_size;

  bool region_on;
  double region_min0, region_min1;
  double region_max0, region_max1;

  SysTime::SysClock extract_time;
  SysTime::SysClock scan_time;
  int scan_number, tri_number;
public:
  SageBase( AI *ai) : ai(ai), region_on(false) {}
  virtual ~SageBase() {}

  void setScreen( Screen *s ) { screen = s; }
  void setView( const View &, double, double, int, int  );
  void setParameters( int, int, int, int, int);
  void setRegion( bool set ) { region_on = set; }
  void setRegion( double, double, double, double );
  virtual void search( double, GeomGroup*, GeomPts * ) {}
};
    
template < class T > class SageStack;
//    template < class T, class F > class BonTree::Tree;

template <class T, class F, class AI>
class Sage : public SageBase<AI> 
{
private:
  F *field;
  T min, max;

  SageStack<T> *stack;
  BonTree::Tree<T,F> *tree;

  int mask;
  int dx, dy, dz;
  int dim;

public:
  // no public variables

public:
  Sage<T,F,AI>( F *, AI *);
  virtual ~Sage();
	
  virtual void search( double, GeomGroup*, GeomPts *);

private:
  void project( const Point &, Pt &);
  void new_field();
  int adjust( double, double, int &);
  int extract( double, int, int, int, int, int, int );
  int bbox_projection( int, int, int, int, int, int,
		       double &, double &, double &, double &, 
		       double & );

};

} // End namespace Yarden
    


#ifdef __GNUG__
static int trunc(double v ) { return v > 0 ? int(floor(v)) : int(floor(v+1)); }
#endif

namespace Yarden {
    
using namespace SCIRun;

/*
     * SageBase class
     */

template < class AI >
SageBase<AI> *
SageBase<AI>::make( ScalarFieldHandle field, AI *ai)
{
  ScalarFieldRGBase *base = field->getRGBase();
  if ( !base ) return 0;
      
  if ( base->getRGDouble() ) 
    return new Sage<double,ScalarFieldRGdouble,AI>(base->getRGDouble(),ai);

  if ( base->getRGFloat() )  
    return new Sage<float, ScalarFieldRGfloat,AI> (base->getRGFloat(),ai);
      
  if ( base->getRGInt() )    
    return new Sage<int,   ScalarFieldRGint,AI> (base->getRGInt(),ai);
      
  if ( base->getRGShort() )  
    return new Sage<short, ScalarFieldRGshort,AI> (base->getRGShort(),ai);

#ifdef HAVE_USHORT      
  if ( base->getRGUshort() ) {
    return new Sage<unsigned short, ScalarFieldRGushort,AI> (base->getRGUshort(),ai);
  }
#endif
  if ( base->getRGChar() )   
    return new Sage<char,  ScalarFieldRGchar,AI> (base->getRGChar(),ai);
      
  if ( base->getRGUchar() )  
    return new Sage<uchar, ScalarFieldRGuchar,AI> (base->getRGUchar(),ai);
      
  return 0; 
}


template< class AI>
void SageBase<AI>::setView (const View &view, double znear, double zfar, 
			    int xres, int yres )
{
  //view = v;
  this->znear = znear;
  this->zfar = zfar;
  this->xres = xres;
  this->yres = yres;

  eye = view.eyep();

  Z = Vector(view.lookat()-eye);
  Z.normalize();
  X = Vector(Cross(Z, view.up()));
  X.normalize();
  Y = Vector(Cross(X, Z));
  //   yviewsize= 1./Tan(DtoR(view.fov()/2.));
  //   xviewsize=yviewsize*gd->yres/gd->xres;;
  double xviewsize= 1./Tan(DtoR(view.fov()/2.));
  double yviewsize=xviewsize*yres/xres;;
  U = X*xviewsize;
  V = Y*yviewsize;
  W = Z;
      
  X = X/xviewsize;
  Y = Y/yviewsize;
      
  AU = Abs(U);
  AU.x(AU.x() * sx );
  AU.y(AU.y() * sy );
  AU.z(AU.z() * sz );
      
  AV = Abs(V);
  AV.x(AV.x() * sx );
  AV.y(AV.y() * sy );
  AV.z(AV.z() * sz );
      
  AW = Abs(W);
  AW.x(AW.x() * sx );
  AW.y(AW.y() * sy );
  AW.z(AW.z() * sz );
}

template <class AI>
void SageBase<AI>::setParameters( int scan, int vis, int reduce, 
				  int all, int size)
{
  this->scan = scan;
  this->bbox_visibility = vis;
  this->reduce =  reduce;
  this->extract_all = all;
  this->min_size = size;
}

template <class AI>
void SageBase<AI>::setRegion( double x0, double x1, double y0,double y1)
{
  region_on = true;
  region_min0 = x0;
  region_min1 = x1;
  region_max0 = y0;
  region_max1 = y1;
}

/*
     * SageCell
     */

template < class T >
struct SageCell {
  //  BonTree::Node *node;
  BonTree::Node<T> *node;
  int i, j, k;
  int dx, dy, dz;
  int mask;
};
    
// Stack
template < class T >
class SageStack {
public:
  int size;
  int pos;
  int depth;
  int use;
  SageCell<T> *top;
  SageCell<T> *stack;
      
public:
  SageStack() { size = 0; pos = 0; depth = 0; top=0; stack = 0;}
  ~SageStack() { if ( stack ) delete stack; }
      
  void resize( int s );
  void push( BonTree::Node<T> *, int, int, int, int, int, int, int );
  void pop( BonTree::Node<T> *&, int &, int& , int &, int &, int &, int &, int &);
  int empty() { return pos==0; }
  void print() { printf("Stack max depth = %d / %d [%.2f]\n  # of op = %d\n",
			depth, size, 100.0*depth/size,use);}
  void reset() { top = stack; pos=0;}
};
    
    
template < class T>
void
SageStack<T>::resize( int s )
{
  if ( s > size ) {
    if ( stack ) delete stack;
    stack = scinew SageCell<T>[s];
    size = s;
  }
  pos = 0;
  depth = 0;
  use = 0;
  top = stack;
}
    
template <class T>
inline void
SageStack<T>::pop( BonTree::Node<T> *&node, int &i, int &j, int &k, 
		   int &dx, int &dy, int &dz, int &mask)
{
  if ( pos-- == 0 ) {
    cerr << " Stack underflow \n";
    abort();
  }
  node = top->node;
  i = top->i;
  j = top->j;
  k = top->k;
  dx = top->dx;
  dy = top->dy;
  dz = top->dz;
  mask = top->mask;
  top--;
}
    
template <class T>
inline void
SageStack<T>::push( BonTree::Node<T> *node, int i, int j, int k, int dx,
		    int dy, int dz,  int mask )
{
  if ( pos >= size-1 ) {
    cerr << " Stack overflow [" << pos << "]\n";
    abort();
  }
      
  top++;
  top->node = node;
  top->i = i;
  top->j = j;
  top->k = k;
  top->dx = dx;
  top->dy = dy;
  top->dz = dz;
  top->mask = mask;
  pos++;
  use++;
  if ( pos > depth ) depth = pos;
}
    
    
// Sage

template <class T, class F, class AI>
Sage<T,F,AI>::Sage( F *f, AI *ai ) : SageBase<AI>(ai), field(f)
{
  stack = new SageStack<T>;
  tree = new BonTree::Tree<T,F>;
      
  new_field();
}

template <class T, class F, class AI>
Sage<T,F,AI>::~Sage()
{
  delete stack;
  delete tree;
}
	    
    
    
template <class T, class F, class AI>
void Sage<T,F,AI>::project( const Point &p, Pt &q )
{
  Vector t = p - eye;
  double px = Dot(t, U );
  double py = Dot(t, V );
  double pz = Dot(t, W );
  q.x = (px/pz+1)*xres/2-0.5;
  q.y = (py/pz+1)*yres/2-0.5;
}


template <class T, class F, class AI>
void Sage<T,F,AI>::new_field()
{
  dx = field->grid.dim1()-1;
  dy = field->grid.dim2()-1;
  dz = field->grid.dim3()-1;
      
  int mdim = dx;
  if ( mdim < dy ) mdim = dy;
  if ( mdim < dz ) mdim = dz;

  Point bmin, bmax;
  field->get_bounds( bmin, bmax );
  gx = bmax.x() - bmin.x();
  gy = bmax.y() - bmin.y();
  gz = bmax.z() - bmin.z();
      
  sx = gx/dx;
  sy = gy/dy;
  sz = gz/dz;

  double dmin, dmax;
  field->get_minmax( dmin, dmax );
  min = T(dmin);
  max = T(dmax);
      
  tree->init( field );
  mask = tree->mask;
      
  stack->resize( mdim * 1000 ); // more then enough
  for (dim = 1; dim < mdim; dim <<=1);
}


int permutation[8][8] = {
  {0,4,1,2,6,3,5,7},
  {1,3,5,0,2,7,4,6},
  {2,3,6,0,4,1,7,5},
  {3,7,1,2,5,0,6,4},
  {4,6,0,5,7,2,1,3},
  {5,7,4,1,3,6,0,2},
  {6,7,2,4,5,0,3,1},
  {7,6,3,5,4,1,2,0}
};

template <class T, class F, class AI>
int Sage<T,F,AI>::adjust( double left, double right, int &x )
{
  double l = left -0.5;
  double r = right -0.5;
  int L = trunc(l);
  int R = trunc(r);
  if ( L == R )
    return 0;
  x =  right > R+0.5 ? R : L;
  return 1;
}
    
    
#define Deriv(u1,u2,u3,u4,d1,d2,d3,d4) \
            (((val[u1]+val[u2]+val[u3]+val[u4])-\
	     (val[d1]+val[d2]+val[d3]+val[d4]))/4.)
    
template <class T, class F, class AI>
void Sage<T,F,AI>::search( double iso, 
			   GeomGroup *group, GeomPts *points )
{
  cerr << "Sage search" << endl;
  this->group = group;
  this->points = points;

  stack->reset();
  stack->push( tree->tree[0], 0, 0, 0, dx-1, dy-1, dz-1, mask );

  screen->clear();

  SysTime::SysClock search_begin = SysTime::currentTicks();
  SysTime::SysClock projection_time = 0;
  SysTime::SysClock vis_time = 0;
      
  extract_time = 0;
  scan_time = 0;
  counter = 0;
  scan_number = 0;
  tri_number = 0;

  int prev_counter = 0;
  double size = 5*Pow(dx*dy*dz,2.0/3);
  while ( !stack->empty() ) {

    if ( counter >  prev_counter * 1.5 ) {
      double progress = counter/size;
      ai->update_progress( progress );
      prev_counter = counter;
    }
	   
    if ( ai->get_abort() ) 
      return; 
    
    int i, j, k;
    int dx, dy, dz;
    int mask;
    BonTree::Node<T> *node;
	
    stack->pop( node, i, j, k, dx, dy, dz, mask);
	
    if (  iso < node->min || node->max < iso ) 
      continue;
	
    
    int status = 1;
    if ( bbox_visibility  ) {
      double left, right, top, bottom;
      double pw;

      SysTime::SysClock  t =SysTime::currentTicks();
      status = bbox_projection( i, j, k, dx+1, dy+1, dz+1, 
				left, right, top, bottom, pw);
      projection_time += SysTime::currentTicks() - t;
 
      if ( status < 0 ) continue;
      if ( status > 0 ) {
	if ( reduce ) {
	  if ( (right-left) <= min_size  && (top-bottom) <= min_size ) {
	    int px,py;
	    if ( adjust( left, right, px ) && adjust( bottom, top, py ) ) {
	      if ( screen->cover_pixel(px,py) ) {
		double x = ((px+0.5)*2/xres-1);
		double y = ((py+0.5)*2/yres-1);
		double z = 1;
		    
		Point Q = eye+((X*x+Y*y+Z*z)*pw);
		double val[8];
		val[0]=field->grid(i,      j,      k);
		val[1]=field->grid(i+dx+1, j,      k);
		val[2]=field->grid(i+dx+1, j+dy+1, k);
		val[3]=field->grid(i,      j+dy+1, k);
		val[4]=field->grid(i,      j,      k+dz+1);
		val[5]=field->grid(i+dx+1, j,      k+dz+1);
		val[6]=field->grid(i+dx+1, j+dy+1, k+dz+1);
		val[7]=field->grid(i,      j+dy+1, k+dz+1);
		    
		Vector N( Deriv(0,3,4,7, 1,2,5,6),
			  Deriv(0,1,4,5, 2,3,6,7),
			  Deriv(0,1,2,3, 4,5,6,7));
		points->add( Q, 1, N );
	      }
	    }
	    continue;
	  }
	}
	int l = trunc(left);
	int r = trunc(right+1);
	int b = trunc(bottom);
	int t = trunc(top+1);
	SysTime::SysClock vis_begin = SysTime::currentTicks();
	int vis = screen->visible( l,b,r,t); //left, bottom, right, top );
	//SysTime::SysClock vis_end = SysTime::currentTicks();
	vis_time += SysTime::currentTicks()-vis_begin;
	    
	if ( !vis ) 
	  continue;
      }
    }
	
	
    if ( !node->child ) {
      if ( status == 0 ) continue;
	  
      int start  = (eye.x() > (i+2)*sx) ? 1 : 0;
      if ( eye.y() > (j+2)*sy ) start += 2;
      if ( eye.z() > (k+2)*sz ) start += 4;
	  
      int *order = permutation[start];
      for (int o=7; o>=0; o--)
	switch (order[o] ) {
	case 0:
	  extract( iso, i,j,k, 1, 1, 1 );
	  break;
	case 1:
	  extract( iso, i+1,j,k, 1, 1, 1 );
	  break;
	case 2:
	  extract( iso, i,j+1,k, 1, 1, 1 );
	  break;
	case 3:
	  extract( iso, i+1,j+1,k, 1, 1, 1 );
	  break;
	case 4:
	  extract( iso, i,j,k+1, 1, 1, 1 );
	  break;
	case 5:
	  extract( iso, i+1,j,k+1, 1, 1, 1 );
	  break;
	case 6:
	  extract( iso, i,j+1,k+1, 1, 1, 1 );
	  break;
	case 7:
	  extract( iso, i+1,j+1,k+1, 1, 1, 1 );
	  break;
	}
      continue;
    }
	
    int dx1, dy1, dz1;
    dx1 = dy1 = dz1 = 0;
    if ( mask & dx ) {
      dx1 = dx & ~mask;
      dx  = mask-1;
    }
    if ( mask & dy ) {
      dy1 = dy & ~mask;
      dy  = mask-1;
    }
    if ( mask & dz ) {
      dz1 = dz & ~mask;
      dz  = mask-1;
    }
    mask >>= 1;
    int start  = (eye.x() > (i+dx+1)*sx) ? 1 : 0;
    if ( eye.y() > (j+dy+1)*sy ) start += 2;
    if ( eye.z() > (k+dz+1)*sz ) start += 4;
	
    int *order = permutation[start];
	
    int type = node->type;
    BonTree::Node<T> *child = node->child;
	
    for (int o=7; o>=0 ; o-- ) {
      switch ( order[o] ) {
      case 0:
	stack->push( child, i, j, k, dx, dy, dz, mask );
	break;
      case 1:
	if ( !(type & 1) )
	  stack->push( child+1, i+dx+1, j, k, dx1, dy, dz, mask );
	break;
      case 2:
	if ( !(type & 2) )
	  stack->push( child+2, i, j+dy+1, k, dx, dy1, dz, mask );
	break;
      case 3:
	if ( !(type & 3) )
	  stack->push( child+3, i+dx+1, j+dy+1, k, dx1, dy1, dz, mask );
	break;
      case 4:
	if ( !(type & 4) )
	  stack->push( child+4, i, j, k+dz+1, dx, dy, dz1, mask );
	break;
      case 5:
	if ( !(type & 5) )
	  stack->push( child+5, i+dx+1, j, k+dz+1, dx1, dy, dz1, mask  );
	break;
      case 6:
	if ( !(type & 6) )
	  stack->push( child+6, i, j+dy+1, k+dz+1, dx, dy1, dz1, mask );
	break;
      case 7:
	if ( !(type & 7) )
	  stack->push( child+7, i+dx+1, j+dy+1, k+dz+1, dx1, dy1, dz1, mask );
	break;
      }
    }
  }
  SysTime::SysClock search_end = SysTime::currentTicks();
      
  printf("Search time = %.3f:\n"
	 "\t projection= %.3f\n"
	 "\t vis = %.3f\n"
	 "\t extract time = %.3f\n"
	 "\t\t scan: time = %.3f number %d   (%g)\n"
	 "\t triangles = %d (per scan %.2f)\n",
	 (search_end - (long long)search_begin)*SysTime::secondsPerTick(),
	 projection_time*SysTime::secondsPerTick(),
	 vis_time*SysTime::secondsPerTick(),
	 extract_time*SysTime::secondsPerTick(),
	 scan_time*SysTime::secondsPerTick(),
	 scan_number, scan_time*SysTime::secondsPerTick()/scan_number,
	 tri_number, tri_number/(double)scan_number);
}
    
template <class T, class F, class AI>
int Sage<T,F,AI>::extract( double iso, 
			   int i, int j, int k, int dx, int dy, int dz )
{
  SysTime::SysClock start = SysTime::currentTicks();
      
  double val[8];
  val[0]=field->grid(i,    j,    k)-iso;
  val[1]=field->grid(i+dx, j,    k)-iso;
  val[2]=field->grid(i+dx, j+dy, k)-iso;
  val[3]=field->grid(i,    j+dy, k)-iso;
  val[4]=field->grid(i,    j,    k+dz)-iso;
  val[5]=field->grid(i+dx, j,    k+dz)-iso;
  val[6]=field->grid(i+dx, j+dy, k+dz)-iso;
  val[7]=field->grid(i,    j+dy, k+dz)-iso;
  int mask=0;
  int idx;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }
  if (mask==0 || mask==255) {
    //printf("Extract nothing:: [%d %d %d] \n", i,j,k );
    extract_time += SysTime::currentTicks() - start;
    return 0;
  }

  if ( region_on ) {
    double min, max;
    min = max = val[0];
    for ( int i=1; i<8; i++) 
      if ( val[i] < min ) min = val[i];
      else if ( val[i] > max ) max = val[i];

    min += iso;
    max += iso;

    if ( min < region_min0 || min > region_min1 ||
	 max < region_max0 || max > region_max1 ) {
      extract_time += SysTime::currentTicks() - start;
      /* 	    cerr << "Sage region reject ["  */
      /* 		 << region_min0 << " - "<< region_min1 << " x "  */
      /* 		 << region_max0 << " - "<< region_max1 << " ]\n" */
      /* 		 << "  at : " << min << " " << max << endl; */
      return 0;
    }
  }
	
  counter++;
  // #ifdef VIS_WOMAM
  //  double ps = pixel_size[offset+k];
  //   double ps = 1;
      
  //   double x0 = i * ps + 256*(1-ps);
  //   double y0 = j * ps + 256*(1-ps);
  //   double z0 = offset+k ;
      
  //   double x1 = x0 + dx*ps;
  //   double y1 = y0 + dy*ps;
  //   double z1 = offset+k+dz ;
      
  // #else
      
  double x0 = i*sx;
  double x1 = (i+dx)*sx;
  double y0 = j*sy;
  double y1 = (j+dy)*sy;
  double z0 = k*sz;
  double z1 = (k+dz)*sz;
  // #endif
      
  Point vp[8];
  vp[0]=Point(x0, y0, z0);
  vp[1]=Point(x1, y0, z0);
  vp[2]=Point(x1, y1, z0);
  vp[3]=Point(x0, y1, z0);
  vp[4]=Point(x0, y0, z1);
  vp[5]=Point(x1, y0, z1);
  vp[6]=Point(x1, y1, z1);
  vp[7]=Point(x0, y1, z1);
      
      
  // >> Begin new projection
      
  TriangleCase *tcase=&tri_case[mask];
  int *vertex = tcase->vertex;
  Pt p[12];
  Point q[12];
      
  // interpolate and project vertices
  int v=0;
  for (int t=0; t<tcase->n; t++) {
    int id = vertex[v++];
    for ( ; id != -1; id=vertex[v++] ) {
      int v1 = edge_table[id][0];
      int v2 = edge_table[id][1];
      if ( val[v1]*val[v2] > 0 ) {
	printf("BUG at %d\n", mask );
	continue;
      }
      q[id] = Interpolate(vp[v1], vp[v2], val[v1]/(val[v1]-val[v2]));
      if ( scan ) project( q[id], p[id] );
    }
  }
      
  v = 0;
  int scan_edges[10];
  int double_edges[10];
      
  GeomTrianglesP *tmp = scinew GeomTrianglesP;
      
  int vis = 0;
      
  for ( int t=0; t<tcase->n; t++) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    int e=2;
	
    scan_edges[0] = v0;
    scan_edges[1] = v1;
    double_edges[0] = double_edges[1] = 1;
	
    for (; v2 != -1; v1=v2,v2=vertex[v++]) {
      double l= (p[v1].x-p[v0].x)*(p[v2].y-p[v0].y) 
	- (p[v1].y-p[v0].y)*(p[v2].x-p[v0].x);
      double_edges[e] = l > 0 ? 1 : -1;
      scan_edges[e] = v2;
      e++;
      tmp->add(q[v0], q[v1], q[v2]);
    }
	
    scan_edges[e] = scan_edges[0];
    double_edges[e] = double_edges[0] = double_edges[e-1];
    double_edges[1] = double_edges[2];
	
    if ( scan ) {
      SysTime::SysClock t = SysTime::currentTicks();
      vis += screen->scan(p, e,  scan_edges, double_edges);
      scan_time += SysTime::currentTicks() - t;
      scan_number++;
      tri_number += e-2;
    }
    else
      vis = 1;
  }
      
  if ( extract_all || vis ) {
    group->add(tmp );
  }
  else
    delete tmp;
      
  extract_time += SysTime::currentTicks() - start;
  return 1;
}
    

template <class T, class F, class AI>
int Sage<T,F,AI>::bbox_projection( int i, int j, int k, 
				   int dx, int dy, int dz,
				   double &left, double &right, 
				   double &top, double &bottom, 
				   double &pw )
{
      
  Vector p = Point((i+dx/2.)*sx,(j+dy/2.)*sy,(k+dz/2.)*sz)-eye;
      
  double lu = (dx*AU.x()+dy*AU.y()+dz*AU.z())/2;
  double lv = (dx*AV.x()+dy*AV.y()+dz*AV.z())/2;
  double lw = (dx*AW.x()+dy*AW.y()+dz*AW.z())/2;
      
  double pu = Dot(p,U);
  double pv = Dot(p,V);
  pw = Dot(p,W);
      
  if ( pw +lw < 0  )
    return -1;
  if ( pw -lw < 0 )
    return 0;
      
      
  double near = 1./(pw-lw);
  double far  = 1./(pw+lw);
      
  double q = pu-lu;
  left = (q* (q>0?far:near)+1)*(xres-1)/2;
  q = pu+lu;
  right =(q* (q<0?far:near)+1)*(xres-1)/2;
  q = pv-lv;
  bottom = (q* (q>0?far:near)+1)*(yres-1)/2;
  q = pv+lv;
  top = (q* (q<0?far:near)+1)*(yres-1)/2;
      
  //   glColor3f(0,1,0);
  //   glBegin(GL_LINE_LOOP);
  //   glVertex2i( left, bottom );
  //   glVertex2i( right, bottom );
  //   glVertex2i( right, top );
  //   glVertex2i( left, top );
  //   glEnd();
      
  //   printf("green : %.1f %.1f %.1f %.1f\n", left,right,bottom,top);
      
  return 1;
}


} // End namespace Yarden

#endif
