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
 *  Sage.h
 *      View Depended Iso Surface Extraction for Structures Grids (Bricks)
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef Sage_h
#define Sage_h

#include <stdio.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Thread/Time.h>
#include <Dataflow/Network/Module.h> 
#include <Core/Datatypes/Field.h> 
#include <Dataflow/Ports/FieldPort.h>  

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomPoint.h>
#include <Core/Geom/View.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/Trig.h>

#include <Core/Algorithms/Visualization/Screen.h>
#include <Core/Algorithms/Visualization/mc_table.h>
#include <Core/Algorithms/Visualization/Tree.h>

#include <Core/Util/DynamicLoader.h>
namespace SCIRun {

class Screen;

typedef Time  SysTime;

// SageBase

class SageAlg : public DynamicAlgoBase {
public:
  SageAlg();
  virtual ~SageAlg();

  virtual void release() = 0;
  virtual void set_field( Field * ) = 0;
  virtual void search( double, GeomGroup*, GeomPoints * ) {}

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

class SageBase  : public SageAlg
{
protected:
  int counter;

  Screen *screen;

  // View parameters
  //	 View *view;
  double znear, zfar;
  int xres, yres;
	 
  GeomGroup* group;
  GeomPoints* points;

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
  SageBase() : region_on(false) {}
  virtual ~SageBase() {}

  void setScreen( Screen *s ) { screen = s; }
  void setView( const View &, double, double, int, int  );
  void setParameters( int, int, int, int, int);
  void setRegion( bool set ) { region_on = set; }
  void setRegion( double, double, double, double );
  virtual void search( double, GeomGroup*, GeomPoints * ) {}
};
    

class SageStack;

template <class F>
class Sage : public SageBase 
{
  typedef typename F::mesh_type              mesh_type;
  typedef typename F::mesh_handle_type       mesh_handle_type;

  typedef typename F::value_type             value_type;
  typedef typename F::fdata_type             fdata_type;
  typedef typename F::mesh_type::Cell::index_type  cell_index_type;
  typedef typename mesh_type::Node::array_type     node_array_type;

private:
  F *field;
  mesh_handle_type mesh;
  value_type min, max;

  SageStack *stack;
  Tree<Cell<value_type> > *tree;

  int nx, ny, nz;
  int fx, fy, fz;
  int dim;

public:
  // no public variables

public:
  Sage();
  virtual ~Sage();
	
  virtual void release() {}
  virtual void set_field( Field *f ); 
  virtual void search( double, GeomGroup*, GeomPoints *);

private:
  void project( const Point &, Pt &);
  void new_field();
  int adjust( double, double, int &);
  void extract( double, int, int, int );
  int bbox_projection( int, int, int, int, int, int,
		       double &, double &, double &, double &, 
		       double & );

};



#ifdef __GNUG__
static int trunc(double v ) { return v > 0 ? int(floor(v)) : int(floor(v+1)); }
#endif


/*
 * SageBase class
 */
inline
void SageBase::setView (const View &view, double znear, double zfar, 
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

inline
void SageBase::setParameters( int scan, int vis, int reduce, 
			      int all, int size)
{
  this->scan = scan;
  this->bbox_visibility = vis;
  this->reduce =  reduce;
  this->extract_all = all;
  this->min_size = size;
}

inline
void SageBase::setRegion( double x0, double x1, double y0,double y1)
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

struct SageCell {
  int level;
  int i, j, k;

  SageCell() {}
};
    
// Stack
class SageStack {
public:
  int size;
  int pos;
  int depth;
  int use;
  SageCell *top;
  SageCell *stack;
      
public:
  SageStack() { size = 0; pos = 0; depth = 0; top=0; stack = 0;}
  ~SageStack() { if ( stack ) delete stack; }
      
  void resize( int s );
  void push( int, int, int, int );
  void pop( int &, int& , int &, int &);
  int empty() { return pos==0; }
  void print() { printf("Stack max depth = %d / %d [%.2f]\n  # of op = %d\n",
			depth, size, 100.0*depth/size,use);}
  void reset() { top = stack; pos=0;}
};
    
    
inline void
SageStack::resize( int s )
{
  if ( s > size ) {
    if ( stack ) delete stack;
    stack = scinew SageCell[s];
    size = s;
  }
  pos = 0;
  depth = 0;
  use = 0;
  top = stack;
}
    
inline void
SageStack::pop(int &level, int &i, int &j, int &k )
{
  if ( pos-- == 0 ) {
    cerr << " Stack underflow \n";
    abort();
  }
  level = top->level;
  i = top->i;
  j = top->j;
  k = top->k;
  top--;
}
    
inline void
SageStack::push( int level, int i, int j, int k )
{
  if ( pos >= size-1 ) {
    cerr << " Stack overflow [" << pos << "]\n";
    abort();
  }
      
  top++;
  top->level = level;
  top->i = i;
  top->j = j;
  top->k = k;
  pos++;
  use++;
  if ( pos > depth ) depth = pos;
}
    
    
// Sage

template <class Field>
Sage<Field>::Sage() : SageBase(), field(0), stack(0), tree(0)
{
}

template <class Field>
Sage<Field>::~Sage()
{
  delete stack;
  delete tree;
}
	    
template<class F>
void Sage<F>::set_field( Field *gf )
{
  F *f = dynamic_cast<F *>(gf);
  if ( f ) {
    cerr << "sage set field\n";
    field = f;
    if ( stack ) delete stack;
    stack = new SageStack;
    if ( tree ) delete tree;
    cerr << "allocate new tree\n";
    tree = new Tree<Cell<value_type> >;

    mesh = field->get_typed_mesh();

    nx = mesh->get_ni()-1;
    ny = mesh->get_nj()-1;
    nz = mesh->get_nk()-1;
    
    Point bmin = mesh->get_min();
    Point bmax = mesh->get_max();

    gx = bmax.x() - bmin.x();
    gy = bmax.y() - bmin.y();
    gz = bmax.z() - bmin.z();
    
    sx = gx/nx;
    sy = gy/ny;
    sz = gz/nz;
    
    cerr << "init tree" << nx << " " << ny << " " << nz << endl;
    tree->init( field, 4 );
    tree->get_factors( fx, fy, fz);
    cerr << "init tree done\n";
    //tree->show();
    stack->resize( /*mdim*/ 10 * 1000 ); // more then enough
  }
}

    
template <class Field>
void Sage<Field>::project( const Point &p, Pt &q )
{
  Vector t = p - eye;
  double px = Dot(t, U );
  double py = Dot(t, V );
  double pz = Dot(t, W );
  q.x = (px/pz+1)*xres/2-0.5;
  q.y = (py/pz+1)*yres/2-0.5;
}



#if 0
static 
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
#endif

template <class Field>
int Sage<Field>::adjust( double left, double right, int &x )
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
    
template <class Field>
void Sage<Field>::search( double iso, 
			   GeomGroup *group, GeomPoints *points )
{
  setParameters( 1, 1, 0, 1, 0 );
  this->group = group;
  this->points = points;

  SysTime::SysClock search_begin = SysTime::currentTicks();
  SysTime::SysClock projection_time = 0;
  SysTime::SysClock vis_time = 0;
      
  extract_time = 0;
  scan_time = 0;
  counter = 0;
  scan_number = 0;
  tri_number = 0;

  stack->reset();
  //screen->clear();
  
  int d1 = tree->tree[0].dim1();
  int d2 = tree->tree[0].dim2();
  int d3 = tree->tree[0].dim3();
  for (int i=0; i < d1; i++)
    for (int j=0; j < d2; j++)
      for (int k=0; k < d3; k++) {
	Cell<value_type> &cell = tree->tree[0](i,j,k);
	if (  cell.min() < iso && iso < cell.max()  ) 
	  stack->push( 0, i, j, k );
      }


  //double size = 5*Pow(dx*dy*dz,2.0/3);
  while ( !stack->empty() ) {

    int level, i, j, k;
    stack->pop( level, i, j, k );
	
    int status = 1;
	
    if ( tree->depth() == level ) {
      if ( status == 0 ) continue;
      
      int sx = i*fx;
      int ex = sx + fx;
      
      int sy = j*fy;
      int ey = sy + fy;
      
      int sz = k*fz;
      int ez = sz + fz;
      
      if (ex > nx ) ex = nx;
      if (ey > ny ) ey = ny;
      if (ez > nz ) ez = nz;

      for (int x=sx; x<ex; x++)
	for (int y=sy; y<ey; y++)
	  for (int z=sz; z<ez; z++)
	    extract( iso, x,y,z );
    }
    else {
      int sx = i*d1;
      int ex = sx + d1;
      
      int sy = j*d2;
      int ey = sy + d2;
      
      int sz = k*d3;
      int ez = sz + d3;

      for (int x=sx; x<ex; x++)
	for (int y=sy; y<ey; y++)
	  for (int z=sz; z<ez; z++) {
	    Cell<value_type> &cell = tree->tree[level+1](x,y,z);
	    if (  cell.min() < iso && iso < cell.max()  ) {
	      stack->push( level+1, x,y,z );
	    }
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
    
template <class Field>
void Sage<Field>::extract( double iso, int x, int y, int z )
{
  SysTime::SysClock start = SysTime::currentTicks();
      
  node_array_type node(8);
  Point vp[8];
  value_type value[8];
  int code = 0;

  mesh->get_nodes( node, cell_index_type(x,y,z) );

  for (int i=7; i>=0; i--) {
    field->value( value[i], node[i] );
    code = code*2+(value[i] < iso );
  }

  if ( code == 0 || code == 255 ) {
    extract_time += SysTime::currentTicks() - start;
    return;
  }

  // get the vertices locations
  for (int i=7; i>=0; i--) 
    mesh->get_point( vp[i], node[i] );

#if 0
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
#endif
	
  counter++;
      
  // >> Begin new projection
      
  TriangleCase *tcase=&tri_case[code];
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
      q[id] = Interpolate(vp[v1], vp[v2], 
			  (value[v1]-iso)/double(value[v1]-value[v2]));
      if ( scan ) project( q[id], p[id] );
    }
  }
      
  v = 0;
  //int scan_edges[10];
  //int double_edges[10];
      
  GeomTrianglesP *tmp = scinew GeomTrianglesP;
      
  int vis = 0;
  for ( int t=0; t<tcase->n; t++) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    int e=2;
	
    //scan_edges[0] = v0;
    //scan_edges[1] = v1;
    //double_edges[0] = double_edges[1] = 1;
	
    for (; v2 != -1; v1=v2,v2=vertex[v++]) {
      //double l= (p[v1].x-p[v0].x)*(p[v2].y-p[v0].y) 
      //  - (p[v1].y-p[v0].y)*(p[v2].x-p[v0].x);
      //double_edges[e] = l > 0 ? 1 : -1;
      //scan_edges[e] = v2;
      e++;
      tmp->add(q[v0], q[v1], q[v2]);
    }

#if 0	
    scan_edges[e] = scan_edges[0];
    double_edges[e] = double_edges[0] = double_edges[e-1];
    double_edges[1] = double_edges[2];
	
    if ( scan ) {
      SysTime::SysClock t = SysTime::currentTicks();
      //vis += screen->scan(p, e,  scan_edges, double_edges);
      scan_time += SysTime::currentTicks() - t;
      scan_number++;
      tri_number += e-2;
    }
    else
#endif
      vis = 1;
  }
      
  if ( extract_all || vis ) {
    group->add(tmp );
  }
  else
    delete tmp;
      
  extract_time += SysTime::currentTicks() - start;
}
    

template <class Field>
int Sage<Field>::bbox_projection( int i, int j, int k, 
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
}


} // namespace SCIRun

#endif // SAGE_h
