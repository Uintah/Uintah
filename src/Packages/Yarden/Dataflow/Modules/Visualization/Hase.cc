/*
 *  Hase.cc    View Depended Iso Surface Extraction
 *             for Structures Grids (Bricks)
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Dec 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#define DEF_CLOCK

#include <stdio.h>

#include <Core/Containers/String.h>
#include <Core/Util/Timer.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/SurfacePort.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomOpenGL.h>

#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <tcl.h>
#include <tk.h>

#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Trig.h>

#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>

#include <Packages/Yarden/Core/Algorithms/Visualization/mcube_scan.h>
#include <Packages/Yarden/Core/Datatypes/Screen.h>
#include <Packages/Yarden/Core/Datatypes/Clock.h>
#include <Packages/Yarden/Core/Datatypes/BonTree.h>

namespace Yarden {
using namespace SCIRun;

//#define DOUBLE
//#define VIS_WOMAN
#define FLOAT
  
#ifdef VIS_WOMAN
#define SHORT
#endif
  
#ifdef CHAR
  typedef unsigned char Value;
  typedef ScalarFieldRGchar FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGChar())
#endif
  
#ifdef FLOAT
  typedef float Value;
  typedef ScalarFieldRGfloat FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGFloat())
#endif
  
#ifdef SHORT
  typedef short Value;
  typedef ScalarFieldRGshort FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGShort())
#endif
  
#ifdef DOUBLE
  typedef double Value;
  typedef ScalarFieldRGdouble FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGDouble())
#endif
  
  double model[4][4], proj[4][4];
  int vp[4];
  
  double PM[4][4];
  double APM[4][4];

  double Cx, Cz, Cw, Cd;

  int vis_count;
  int offset = 0;
  extern int show;
  
  iotimer_t rebuild_start, rebuild_make,
    rebuild_draw, rebuild_get,
    rebuild_build, rebuild_last, rebuild_end;
  iotimer_t make_start, make_lock,make_info,make_get, make_make, make_mat,
    make_err, make_done;
  iotimer_t estimate_timer;

  extern "C" Tcl_Interp* the_interp;
  extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
  
  iotimer_t extract_timer, vis_timer, project_timer, copy_timer, hist_timer;
  iotimer_t draw_timer;
  CPUTimer t_extract;
  int bbox_count;
  int debug = 1;
  int scan_yes;
  int scan_no;
  int print_stat = 0;
  
  int trid = 0;
  Value MIN(Value p, Value q) {return p<q? p : q;}
  Value MAX(Value p, Value q) {return p>q? p : q;}
  
struct Table {
  int value;
  int bbox_visible;
  int reduce;
  int extract;
  int visible;
  iotimer_t time;
  iotimer_t cycle;
};

Table empty_table = {0, 0, 0, 0, 0, 0, 0 };
const int table_size = 1; // 000000;
Table table[ table_size];
int tpos;


GeometryData *gd;
View *view;

struct { 
  int n;
  int area;
  iotimer_t time;
  int vis;
  int empty;
} collect_copy[257];

// Statistics

class Statistics {
public:
  int value;
  int bbox;
  int extracted;
  int size;
  int bbox_draw;
  int bbox_draw1;
  
public:
  
  void reset( int s=0) {value = bbox = bbox_draw = bbox_draw1 = extracted = 0; size=s; }
  void print();
  Statistics() {reset();}
  ~Statistics() {}
};

void
Statistics::print()
{
  if (!print_stat )
    return;
  
  printf("Statistics  [%d]\n"
	 "  value: %d\n  bbox: %d\n"
	 "  bbox_draw: %d %d [%d]\n"
	 "  extracted: %d\n\n",
	 size, value, bbox, bbox_draw, bbox_draw1,
	 bbox_draw-bbox_draw1, extracted);
  
  
  FILE* file= fopen("data", "w" );
  if ( file ) {
    int bbox_visible, bbox_not_visible, reduce,
      extract, visible, not_visible;
    visible = bbox_visible = bbox_not_visible = reduce =
      extract = visible = not_visible = 0;
    iotimer_t time = 0;
    
    printf("table size: %d\n", tpos );
    for (int i=0; i<=tpos; i++ ) {
      if ( table[i].bbox_visible > 0 ) bbox_visible = table[i].bbox_visible;
      if ( table[i].bbox_visible < 0 ) bbox_not_visible = -table[i].bbox_visible;
      if ( table[i].visible == 1 ) visible++;
      if ( table[i].visible == -1 ) not_visible++;

      fprintf( file, "%d %2d  %2d %ll %ll\n",
	       table[i].value,
	       table[i].bbox_visible,
	       table[i].visible,
	       table[i].time>>3,
	       table[i].cycle>>3);
    }

    fclose(file);
    printf("done\n");
  }
}

#define BonNode BonTree::Node<Value>

// Cell
struct HaseCell {
  BonNode *node;
  int i, j, k;
  int dx, dy, dz;
  int mask;
};

// Stack
class HaseStack {
public:
  int size;
  int pos;
  int depth;
  int use;
  HaseCell *top;
  HaseCell *stack;

public:
  HaseStack() { size = 0; pos = 0; depth = 0; top=0; stack = 0;}
  ~HaseStack() { if ( stack ) delete stack; }

  void resize( int s );
  void push( BonNode *, int, int, int, int, int, int, int );
  void pop( BonNode *&, int &, int& , int &, int &, int &, int &, int &);
  int empty() { return pos==0; }
  void print() { printf("Stack max depth = %d / %d [%.2f]\n  # of op = %d\n",
			depth, size, 100.0*depth/size,use);}
  void reset() { top = stack; pos=0;}
};

void
HaseStack::resize( int s )
{
  if ( s > size ) {
    if ( stack ) delete stack;
    stack = scinew HaseCell[s];
    size = s;
  }
  pos = 0;
  depth = 0;
  use = 0;
  top = stack;
}

inline void
HaseStack::pop( BonNode *&node, int &i, int &j, int &k, 
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
  
inline void
HaseStack::push( BonNode *node, int i, int j, int k, int dx,
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


// Warp

struct Warp {
  double xscale;
  double yscale;
  double x;
  double y;
};

// Hase

class Hase : public Module 
{
  ScalarFieldIPort* infield;  // input scalar fields (bricks)
  ScalarFieldIPort* incolorfield;
  ColorMapIPort* incolormap;

  GeometryOPort* ogeom;       // input from salmon - view point

  Tk_Window tkwin;
  Window win;
  Display* dpy;
  GLXContext cx;

  //float* data;
  GLubyte *data;
  GuiDouble isoval;
  GuiDouble isoval_min, isoval_max;
  GuiInt tcl_value, tcl_bbox, tcl_visibility;
  GuiInt tcl_scan, tcl_depth, tcl_reduce, tcl_cover, tcl_all;
  GuiInt tcl_rebuild;
  GuiInt tcl_use_hw, tcl_estimate;
  GuiInt tcl_minmax, tcl_finish, tcl_max_area;

  int value, bbox_visibility, visibility, cutoff_depth;
  int scan, count_values, extract_all;
  int use_hw, minmax, finish, max_area;
  int estimate;

  int hw_init;

  int box_id;
  int surface_id;
  int surface_id1;
  int surface_id2;
  int surface_id3;
  int shadow_id;
  int points_id;
  MaterialHandle bone;
  MaterialHandle flesh;
  MaterialHandle matl;
  MaterialHandle matl1;
  MaterialHandle matl2;
  MaterialHandle matl3;
  MaterialHandle shadow_matl;
  MaterialHandle box_matl;
  MaterialHandle points_matl;
  
  GeomTrianglesP* group;
  GeomPts* points;
  GeomGroup* tgroup;
  GeomObj* topobj;
  
  GeomTrianglesP *triangles;
  GeomMaterial *surface;

  ScalarFieldHandle scalar_field;
  FIELD_TYPE *field;

  GeometryData *local_gd;
  Point eye;
  Warp *warp;
  
  int dx, dy, dz, dim;
  int mask;
  int field_generation;
  double iso_value, prev_value;
  int initialized;
  int reduce;
  int new_surface;
  
  double bbox_limit_x, bbox_limit_y;
  double bbox_limit_x1, bbox_limit_y1;
  double left, right, top, bottom;
  Value min, max;
  Point bmin, bmax;
  double gx, gy, gz;
  double sx, sy, sz;

  HaseStack stack;
  BonTree::Tree<Value, FIELD_TYPE> tree;
  Statistics statistics;
  int counter;

  char *gl_name;
  int init_gl;

  Vector U,V,W;
  Vector AU, AV, AW;
  Vector X,Y,Z;
  int xres, yres;

  Screen screen;
  int screen_id;

  int hist_size;
  unsigned short *hist;
  double pixel_size[2000];

public:

  Hase( const clString& id);
  virtual ~Hase();

  virtual void execute();

  void project( const Point &, Pt &);

  void new_field( FIELD_TYPE *field );
  void check( const Point &, double &, double &, double &, double & );
  void search();
  void search( double );
  int  extract( double, int, int, int, int, int, int );
  void redraw( int xres, int yres);
  void compute_depth( double& znear, double& zfar);
  void reset_view( GeometryData *);
  int hw_visible( double, double, double, double );

  void redraw_done();
  double hw_project( int i, int j, int k, int dx, int dy, int dz,
		   double &left, double &right, double &top, double &bottom );
  double bbox_projection( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom );
  double bbox( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom );
  int adjust( double, double, int &);
  void test();
  void test_project( int i, int j, int k, int dx, int dy, int dz );
};
  
extern "C" Module* make_Hase(const clString& id)
{
  return scinew Hase(id);
}

static clString module_name("Hase");
static clString box_name("HaseBox");
static clString surface_name("Hase");
static clString surface_name1("HaseScreen");
static clString surface_name2("HaseBack");
static clString surface_name3("HasePts");
static clString shadow_name("HaseShadow");

Hase::Hase(const clString& id)
  : Module("Hase", id, Filter ), isoval("isoval", id, this),
    isoval_min("isoval_min", id, this), isoval_max("isoval_max", id, this),
    tcl_bbox("bbox", id, this), 
    tcl_scan("scan", id, this),  tcl_value("value", id, this), 
    tcl_visibility("visibility", id, this),
    tcl_depth("cutoff_depth", id, this),
    tcl_reduce("reduce",id,this), tcl_cover("cover",id,this), 
    tcl_all("all",id,this),
    tcl_rebuild("rebuild",id,this),
    tcl_use_hw("use_hw",id,this),
    tcl_minmax("minmax",id,this),
    tcl_finish("finish",id,this),
    tcl_max_area("max_area",id,this),
    tcl_estimate("estimate",id,this)
{
  init_clock();
  printf( "Hase::Hase :: %d\n", tri_case[136].vertex[4]);
  // Create the input ports
  infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
  add_iport(infield);

  incolorfield=scinew ScalarFieldIPort(this, "Color Field",
				       ScalarFieldIPort::Atomic);
  add_iport(incolorfield);
  incolormap=scinew ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
  add_iport(incolormap);
    
  // Create the output port
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  
  Color Flesh = Color(1.0000, 0.4900, 0.2500);
  Color Bone = Color(0.9608, 0.8706, 0.7020);
  
  flesh = scinew Material( Flesh*.1, Flesh*.6, Flesh*.6, 20 );
  bone = scinew Material( Bone*.1, Bone*.6, Bone*.6, 20 );
    
  box_matl=scinew Material(Color(0.3,0.3,0.3), Color(.8,.8,.8), Color(.7,.7,.7), 20);
  points_matl=scinew Material(Color(0.3,0,0), Color(.8,0,0), Color(.7,.7,.7), 20);
  shadow_matl=scinew Material(Color(0,0.3,0.3), Color(0,.8,.8), Color(.7,.7,.7), 20);
  matl1=scinew Material(Color(0.3,0.0,0), Color(0.8,0,0), Color(.7,.7,.7), 20);
  matl2=scinew Material(Color(0.3,0,0.0), Color(.8,0,0), Color(.7,.7,.7), 20);
  matl3=scinew Material(Color(0,0.3,0.3), Color(0,.8,.8), Color(.7,.7,.7), 20);
  surface_id = 0;

  box_id = 0;
  shadow_id = 0;
  surface_id1 = 0;
  surface_id2 = 0;
  surface_id3 = 0;
  points_id = 0;
  field_generation = 0;

  local_gd = scinew GeometryData;
  local_gd->xres = 512;
  local_gd->yres = 512;
  local_gd->znear = 1;
  local_gd->zfar = 2;
  local_gd->view = scinew View( Point(0.65, 0.5, -4.5),
				Point(0.5,0.5,0.5),
				Vector(0,1,0),
				17 );
  
  hw_init = 1;
  minmax = 1;
  finish = 1;
  max_area = 25*25;
  //  int n =  4096;
  warp = 0;
  initialized = 0;

  screen.setup( 512, 512 );
  prev_value = -1;
  screen_id = 0;

  int i=0;
  for (i=0; i<209; i++)
    pixel_size[i] = .48828;
  for(; i<227; i++)
    pixel_size[i] = .72266;
  for(; i<249; i++)
    pixel_size[i] = .85937;
  for(; i<1106; i++)
    pixel_size[i] = .9375;
  for(; i<1110; i++)
    pixel_size[i] = .72266;
  for(; i<1117; i++)
    pixel_size[i] = .9375;
  for(; i<1734; i++)
    pixel_size[i] = .72266;

}

Hase::~Hase()
{
}

inline void
Hase::project( const Point &p, Pt &q )
{
  double dummy;
  gluProject( p.x(), p.y(), p.z(),
	     (double *)model, (double *)proj, vp, 
	      &q.x, &q.y, &dummy);
  q.x -= 0.5;
  q.y -= 0.5;

//   Vector t = p - eye;
//   double px = Dot(t, U );
//   double py = Dot(t, V );
//   double pz = Dot(t, W );
//   //cerr << "project: " << px << " " << py << " " << pz << endl;
//   q.x = (px/pz+1)*xres/2-0.5;
//   q.y = (py/pz+1)*yres/2-0.5;
}

void Hase::execute()
{
  cerr << "exec" << endl;
  iotimer_t start = read_time();
  extract_timer = vis_timer = project_timer = hist_timer = copy_timer = 
    draw_timer = estimate_timer = 0;

  if(!infield->get(scalar_field)) {
    error("No input field\n");
    return;
  }
  
  field = GET_FIELD(scalar_field); 
  if ( !field ) {
    error("\n\n>>>>> Can't Hase field type ");
    if ( scalar_field->getRGBase() ) cerr << "RGBase\n";
    if ( scalar_field->getUG() )   cerr << "UG\n";
    if ( scalar_field->getHUG() ) cerr << "RGBase\n";
    //    if ( scalar_field->getBG() ) cerr << "BG    \n";
    return;
  }
  
  if ( field->generation !=  field_generation ) {
    // new field
    cerr << "new field\n";
    new_field( field );
    return;
  }
  
  gd = ogeom->getData(0, GEOM_VIEW);
  if ( gd == NULL ) {
    cerr << "using prec view." << endl;
    gd = local_gd;
  }
  else {
    *local_gd->view = *gd->view;
    local_gd->zfar = gd->zfar;
    local_gd->znear = gd->znear;
    local_gd->xres = gd->xres;
    local_gd->yres = gd->yres;
  }
  

  view = gd->view;
  eye = view->eyep();
  xres = gd->xres;
  yres = gd->yres;

  Z = Vector(view->lookat()-eye);
  Z.normalize();
  X = Vector(Cross(Z, view->up()));
  X.normalize();
  Y = Vector(Cross(X, Z));
//   yviewsize= 1./Tan(DtoR(view->fov()/2.));
//   xviewsize=yviewsize*gd->yres/gd->xres;;
  double xviewsize= 1./Tan(DtoR(view->fov()/2.));
  double yviewsize=xviewsize*gd->yres/gd->xres;;
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
  //  printf(" res = %d %d  fov= %lf  viewsize= %lf %lf\n",
  //	 xres, yres, view->fov(), xviewsize, yviewsize );


  //  statistics.reset( dx*dy*dz);
  stack.reset();
  
  iso_value = isoval.get();
  use_hw = tcl_use_hw.get();
  finish = tcl_finish.get();
  estimate = tcl_estimate.get();
  max_area = tcl_max_area.get();
  max_area *= max_area;
  if ( use_hw ) {
    int m  = tcl_minmax.get();
    if ( m != minmax ) {
      hw_init = 1;
      minmax = m;
    }
  }
  reset_view( gd );
  search();  
}

void
Hase::new_field( FIELD_TYPE *field )
{
  dx = field->grid.dim1()-1;
  dy = field->grid.dim2()-1;
  dz = field->grid.dim3()-1;
  
  int mdim = dx;
  if ( mdim < dy ) mdim = dy;
  if ( mdim < dz ) mdim = dz;
  
  field->get_bounds( bmin, bmax );
  
  if ( box_id ) 
    ogeom->delObj(box_id);

  GeomBox *box = scinew GeomBox( bmin, bmax, 1 );
  GeomObj *bbox= scinew GeomMaterial( box, box_matl);
  box_id = ogeom->addObj( bbox, box_name );
  
  cerr << "Field bounds: " << bmin << "   " << bmax << "\n";
  gx = bmax.x() - bmin.x();
  gy = bmax.y() - bmin.y();
  gz = bmax.z() - bmin.z();
  
  sx = gx/dx;
  sy = gy/dy;
  sz = gz/dz;
  
  double dmin, dmax;
  field->get_minmax( dmin, dmax );
  min = Value(dmin);
  max = Value(dmax);

  isoval_min.set(min);
  isoval_max.set(max);
  isoval.set((min+max)/2);
  reset_vars();

  tree.init( field );
  mask = tree.mask;
  field_generation = field->generation;

  stack.resize( mdim * 1000 ); // more then enough
  for (dim = 1; dim < mdim; dim <<=1);
}






void
Hase::search()
{
  iotimer_t start = read_time();
  
  trid = 0;
  vis_count = 0;
  scan_yes = scan_no = 0;
  value = tcl_value.get();
  scan = tcl_scan.get();
  visibility = tcl_visibility.get();
  bbox_visibility = tcl_bbox.get();
  reduce =  tcl_reduce.get();
  screen.cover( tcl_cover.get() );
  extract_all = tcl_all.get();

  statistics.reset();
  new_surface = 1;

  group = scinew GeomTrianglesP;
  points = scinew GeomPts(10000);
  points->have_normal = 1;
  tgroup=scinew GeomGroup;
  topobj=tgroup;
  
  
  ScalarFieldHandle colorfield;
  int have_colorfield=incolorfield->get(colorfield); ColorMapHandle cmap;
  int have_colormap=incolormap->get(cmap);
  
  if ( new_surface ) {
    if(have_colormap && !have_colorfield){
      // Paint entire surface based on colormap
      topobj=scinew GeomMaterial(tgroup, cmap->lookup(iso_value));
    } else if(have_colormap && have_colorfield){
      // Nothing - done per vertex
    } else {
      // Default material
      topobj=scinew GeomMaterial(tgroup, iso_value < 800 ? flesh : bone);
    }
  }
  
  stack.push( tree.tree[0], 0, 0, 0, dx-1, dy-1, dz-1, mask );

  // SEARCH >>>
  iotimer_t start1 = read_time();

  screen.clear();

  search(iso_value);

  iotimer_t end1 = read_time();
  

  if ( points_id ) {
    ogeom->delObj(points_id);
    points_id = 0;
  }
  
  if(new_surface && surface_id )
      ogeom->delObj(surface_id);
  
  if ( new_surface ) {
    tgroup->add(group);
    if ( group->size() == 0 && points->pts.size() == 0 ) {
      if ( !box_id ) {
	GeomBox *box = scinew GeomBox( bmin, bmax, 1 );
	GeomObj *bbox= scinew GeomMaterial( box, box_matl);
	box_id = ogeom->addObj( bbox, box_name );
      }
    }
    else if ( box_id ) {
      ogeom->delObj(box_id);
      box_id = 0;
    }

    if ( tgroup->size() == 0 ) {
      delete tgroup;
      surface_id=0;
    } else {
      surface_id=ogeom->addObj( topobj, surface_name ); // , &lock );
    }
    if ( points->pts.size() > 0 )
      points_id =ogeom->addObj( scinew GeomMaterial( points, 
						     iso_value < 800 ? flesh 
						     : bone ),
				"Hase points");
  }
  else
    ogeom->flushViews();
  
  iotimer_t end = read_time();			
  printf("Scan: %d cells\n", statistics.extracted );
  printf("vis count = %d\n", vis_count);
//   printf("Scan : %d %d\n", scan_yes, scan_no );	
  
  printf(" Search Timers: \n\tinit %.3lf  \n"
	 "\tsearch %.3lf (vis=%.1lf proj=%.1lf  ext=%.1lf) \n"
	 "\tcopy = %.1lf hist = %.1lf draw = %.1lf\n"
	 "\tall %.3lf\n "
	 "\testimate %.3lf\n",
   	 (end-start -(end1-start1))*cycleval*1e-9,
   	 (end1-start1)*cycleval*1e-9, 
   	 vis_timer*cycleval*1e-9, 
	 project_timer*cycleval*1e-9,
	 (extract_timer-draw_timer)*cycleval*1e-9,
	 copy_timer*cycleval*1e-9, hist_timer*cycleval*1e-9,
	 draw_timer*cycleval*1e-9,
   	 (end-start)*cycleval*1e-9,
	 estimate_timer*cycleval*1e-9);

  iotimer_t t = 0;
  for (int j=0; j<256; j++)
    t+= collect_copy[j].time;

  printf(" %d points \n", points->pts.size());

  printf("copy :\n");
  for (int i=0; i<257; i++) 
    if ( collect_copy[i].n > 0 ) {
      printf("%3d: %5d vis=%3.0f%% empty=%3.0lf%%  area=%7.1lf "
	     "time=%6.1lf (%6.1lf) avg=%7.3lf \n",
	     i*2,collect_copy[i].n, 
	     collect_copy[i].vis*100.0/collect_copy[i].n,
	     collect_copy[i].empty*100.0/collect_copy[i].n,
	     double(collect_copy[i].area)/collect_copy[i].n,
	     collect_copy[i].time*cycleval*1e-9,
	     t*cycleval*1e-9,
	     double(collect_copy[i].time*cycleval*1e-9)/collect_copy[i].n);
      t -= collect_copy[i].time;
    }
}

int permutation[8][8] = {
  0,4,1,2,6,3,5,7,
  1,3,5,0,2,7,4,6,
  2,3,6,0,4,1,7,5,
  3,7,1,2,5,0,6,4,
  4,6,0,5,7,2,1,3,
  5,7,4,1,3,6,0,2,
  6,7,2,4,5,0,3,1,
  7,6,3,5,4,1,2,0,
};

inline int
Hase::adjust( double left, double right, int &x )
{
  int l = trunc(left-0.5);
  int r = trunc(right-0.5);
  x = l+1;
  return ( l < r );

//   if ( x == r )
//     return 0;
//   x =  l; // right > R+0.5 ? R : L;
//   return 1;
}


#define Deriv(u1,u2,u3,u4,d1,d2,d3,d4) (((val[u1]+val[u2]+val[u3]+val[u4])-\
					(val[d1]+val[d2]+val[d3]+val[d4]))/4.)
void
Hase::search( double v )
{

  while ( !stack.empty() ) {

    if ( abort_flag )
      return;
      
    int i, j, k;
    int dx, dy, dz;
    int mask;
    BonNode *node;

    iotimer_t end = read_time();
    
    stack.pop( node, i, j, k, dx, dy, dz, mask);

     if (  v < node->min || node->max < v ) {
       statistics.value++;
       continue;
     }

     if ( bbox_visibility  ) {
       double left, right, top, bottom;
       double pw = 1;

       iotimer_t es = read_time();
       if ( estimate ) {
 	 pw = bbox(i, j, k, dx+1, dy+1, dz+1, left, right, top, bottom );
       }
       else 
	 pw =hw_project( i, j, k, dx+1, dy+1, dz+1, left, right, top, bottom );

       iotimer_t ee = read_time();
       estimate_timer += ee-es;

       if ( reduce ) {
	 if ( (right-left) < 2 && (top-bottom) < 2  ) {
	   int px,py;
	   if ( adjust( left, right, px ) && adjust( bottom, top, py ) ) {
	     if ( screen.cover_pixel(px,py) ) {
	       double x, y, z;
	       double wx, wy, wz;

	       gluProject((i+(dx+1)/2.)*sx, (j+(dy+1)/2.)*sy, (k+(dz+1)/2.)*sz,
			  (double *)model, (double *)proj, vp,
			  &wx,&wy, &wz );
	       gluUnProject( px, py, wz,
   			     (double *)model, (double *)proj, vp, 
   			     &x, &y, &z );	       
 	       Point Q(x,y,z);
	       // back project
// 	       double Oz = (k+(dz+1)/2.)*sz;
// 	       double Ox = (py - Cx*px + Cz*Oz + Cw)*Cd;
// 	       double Oy = (px - PM[0][2]*Oz - PM[0][3]-PM[0][0]*Ox)/PM[0][1];
// 	       Point Q( Ox, Oy, Oz );
//   	       gluUnProject( px, py, pw, 
//   			     (double *)model, (double *)proj, vp, 
//   			     &x, &y, &z );	       
//     	       Point Q((i+(dx+1)/2.)*sx, (j+(dy+1)/2.)*sy, (k+(dz+1)/2.)*sz );
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

       int vis;
       vis_count++;
       {
	 int xres = 512;
	 if ( left < 0 ) left = 0;
	 else if ( left >= xres ) left = xres-1;
	 if ( right < 0 ) right = 0;
	 else if ( right >= xres ) right = xres-1;
       }
       if ( bottom < 0 ) bottom = 0;
       else if ( bottom >= yres ) bottom = yres-1;
       if ( top < 0 ) top = 0;
       else if ( top >= yres ) top = yres-1;
       iotimer_t vis_begin = read_time();
       if ( use_hw ) {
	 vis = hw_visible( left, bottom, right, top );
       }
       else {
	 vis = screen.visible( left,bottom,right,top); 
	 iotimer_t end=read_time();
	 double area = (right-left)*(top-bottom);
	 int a = int(sqrt(double(area))/2);
	 collect_copy[a].time += end-vis_begin;
	 collect_copy[a].area += area;
	 collect_copy[a].n++;
	 if ( vis ) collect_copy[a].vis++;

       }
       iotimer_t vis_end = read_time();
       vis_timer += vis_end-vis_begin;
       
       // show box
//        glMatrixMode(GL_PROJECTION);
//        glPushMatrix();
//        glLoadIdentity();
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glLoadIdentity();

//        double l = left*2/xres-1;
//        double r = right*2/xres-1;
//        double t = top*2/yres-1;
//        double b = bottom*2/yres-1;

//        glEnable(GL_COLOR_LOGIC_OP);
//        glColor3f(1,1,1);
//        glBegin(GL_LINE_LOOP);
//        glVertex2d(l,b);
//        glVertex2d(r,b);
//        glVertex2d(r,t);
//        glVertex2d(l,t);
//        glEnd();

//        printf("%5.1lf %5.1lf %5.1lf %5.1lf : %s", 
// 	      left,right, top, bottom,
// 	      vis?"Visible":"no");
//        scanf("%*c");

//        glBegin(GL_LINE_LOOP);
//        glVertex2d(l,b);
//        glVertex2d(r,b);
//        glVertex2d(r,t);
//        glVertex2d(l,t);
//        glEnd();
//        glDisable(GL_COLOR_LOGIC_OP);

//        glMatrixMode(GL_PROJECTION);
//        glPopMatrix();
//        glMatrixMode(GL_MODELVIEW);
//        glPopMatrix();

//        glColor3f(1,0,0);

       if ( !vis ) {
	 statistics.bbox++;
	 continue;
       }
     }
     
     if ( !node->child ) {
       if ( extract_all ) 
	 extract( v, i, j, k, dx+1, dy+1, dz+1 );
       else {
	 int start  = (eye.x() > (i+2)*sx) ? 1 : 0;
	 if ( eye.y() > (j+2)*sy ) start += 2;
	 if ( eye.z() > (k+2)*sz ) start += 4;
	 
	 int *order = permutation[start];
	 for (int o=7; o>=0; o--)
	   switch (order[o] ) {
	     case 0:
	       extract( v, i,j,k, 1, 1, 1 );
	       break;
	     case 1:
	       extract( v, i+1,j,k, 1, 1, 1 );
	       break;
	     case 2:
	       extract( v, i,j+1,k, 1, 1, 1 );
	       break;
	     case 3:
	       extract( v, i+1,j+1,k, 1, 1, 1 );
	       break;
	     case 4:
	       extract( v, i,j,k+1, 1, 1, 1 );
	       break;
	     case 5:
	       extract( v, i+1,j,k+1, 1, 1, 1 );
	       break;
	     case 6:
	       extract( v, i,j+1,k+1, 1, 1, 1 );
	       break;
	     case 7:
	       extract( v, i+1,j+1,k+1, 1, 1, 1 );
	       break;
	   }
       }
       continue;
     }
     
     int dx1, dy1, dz1;
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
     BonNode *child = node->child;
    
     for (int o=7; o>=0 ; o-- ) {
       switch ( order[o] ) {
	 case 0:
	   stack.push( child, i, j, k, dx, dy, dz, mask );
	   break;
	 case 1:
	   if ( !(type & 1) )
	     stack.push( child+1, i+dx+1, j, k, dx1, dy, dz, mask );
	   break;
	 case 2:
	   if ( !(type & 2) )
	     stack.push( child+2, i, j+dy+1, k, dx, dy1, dz, mask );
	   break;
	 case 3:
	   if ( !(type & 3) )
	     stack.push( child+3, i+dx+1, j+dy+1, k, dx1, dy1, dz, mask );
	   break;
	 case 4:
	   if ( !(type & 4) )
	     stack.push( child+4, i, j, k+dz+1, dx, dy, dz1, mask );
	   break;
	 case 5:
	   if ( !(type & 5) )
	     stack.push( child+5, i+dx+1, j, k+dz+1, dx1, dy, dz1, mask  );
	   break;
	 case 6:
	   if ( !(type & 6) )
	     stack.push( child+6, i, j+dy+1, k+dz+1, dx, dy1, dz1, mask );
	   break;
	 case 7:
	   if ( !(type & 7) )
	     stack.push( child+7, i+dx+1, j+dy+1, k+dz+1, dx1, dy1, dz1, mask );
	   break;
       }
    }
  }
}

int
Hase::extract( double iso, int i, int j, int k, int dx, int dy, int dz )
{
  iotimer_t start = read_time();

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
    extract_timer += read_time() - start;
    return 0;
  }

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
      q[id] = Interpolate(vp[v1], vp[v2], val[v1]/(val[v1]-val[v2]));
      if ( scan && !use_hw ) project( q[id], p[id] );
    }
  }
  
  v = 0;
  GeomTrianglesP *tmp = scinew GeomTrianglesP;
  
  int vis = 0;
  
  for ( int t=0; t<tcase->n; t++) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    
    if ( use_hw ) {
      if (finish) glFinish();
      iotimer_t s = read_time();
      glBegin(GL_TRIANGLE_FAN);
      glVertex3d(q[v0].x(), q[v0].y(), q[v0].z());
      glVertex3d(q[v1].x(), q[v1].y(), q[v1].z());
      int vv2=v2;
      int vv = v;
      for (; vv2 != -1; vv2=vertex[vv++]) {
	glVertex3d(q[vv2].x(), q[vv2].y(), q[vv2].z());
      }
      glEnd();
      if (finish) glFinish();
      iotimer_t e = read_time();
      
      draw_timer += e-s;

      for (; v2 != -1; v1=v2,v2=vertex[v++]) {
	tmp->add(q[v0], q[v1], q[v2]);
      }
      vis = 1;
    }
    else { // no hw
      int scan_edges[10];
      int double_edges[10];
      int e=2;
      
      scan_edges[0] = v0;
      scan_edges[1] = v1;
      double_edges[0] = double_edges[1] = 1;
      for (; v2 != -1; v1=v2,v2=vertex[v++]) {
	int l= (p[v1].x-p[v0].x)*(p[v2].y-p[v0].y) 
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
	iotimer_t s = read_time();
	vis += screen.scan(p, e,  scan_edges, double_edges);
	iotimer_t e = read_time();
	draw_timer += e-s;
      }
      else
	vis = 1;
    }
  }
  
  if ( extract_all || vis ) {
    tgroup->add(tmp );
  }
  else
    delete tmp;

  statistics.extracted++;
  extract_timer += read_time() - start;
  return 1;
}



double
Hase::bbox_projection( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  Vector p = Point((i+dx/2.)*sx,(j+dy/2.)*sy,(k+dz/2.)*sz)-eye;

  double lu = (dx*AU.x()+dy*AU.y()+dz*AU.z())/2;
  double lv = (dx*AV.x()+dy*AV.y()+dz*AV.z())/2;
  double lw = (dx*AW.x()+dy*AW.y()+dz*AW.z())/2;
  
  double pu = Dot(p,U);
  double pv = Dot(p,V);
  double pw = Dot(p,W);

  double near = 1./(pw-lw);
  double far  = 1./(pw+lw);

  double q = pu-lu;
  left = (q* (q>0?far:near)+1)*(xres-1)/2;
  q = pu+lu;
  right =(q* (q<0?far:near)+1)*(xres)/2;
  q = pv-lv;
  bottom = (q* (q>0?far:near)+1)*(yres-1)/2;
  q = pv+lv;
  top = (q* (q<0?far:near)+1)*(yres)/2;


  double x = i*sx;
  double y = j*sy;
  double z = k*sz;

  double xd = dx*sx;
  double yd = dy*sy;
  double zd = dz*sz;

  double wx, wy, wz;

  gluProject(x+xd, y+yd, z+zd, 
	     (double *)model, (double *)proj, vp, 
	     &wx, &wy, &wz );
  
//   printf(" project: %5.1lf %5.1lf %5.1lf\n"
// 	 "  p/w     %5.1lf %5.1lf \n"
// 	 "  p/w*res %5.1lf %5.1lf \n"
// 	 "  glu     %5.1lf %5.1lf %5.1lf\n",
// 	 pu, pv,pw,
// 	 pu/pw, pv/pw,
// 	 (pu/pw+1)*(xres-1)/2, (pv/pw+1)*(yres-1)/2,
// 	 wx,wy,wz );
//   scanf("%*c");

//   glColor3f(0,1,0);
//   glBegin(GL_LINE_LOOP);
//   glVertex2i( left, bottom );
//   glVertex2i( right, bottom );
//   glVertex2i( right, top );
//   glVertex2i( left, top );
//   glEnd();

//   printf("green : %.1f %.1f %.1f %.1f\n", left,right,bottom,top);

  return pw;
}

double
Hase::bbox( int x, int y, int z, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  double hdx = dx*sx/2.;
  double hdy = dy*sy/2.;
  double hdz = dz*sz/2.;

  double p[4];

  p[0] = x*sx+hdx;
  p[1] = y*sy+hdy;
  p[2] = z*sz+hdz;
  p[3] = 1;

  double pp[4];
  
  for (int i=0; i<4; i++) {
    pp[i] = PM[3][i];
    for (int j=0; j<3; j++)
      pp[i] += PM[j][i]*p[j];
  }

#ifdef CHECK
  double wx, wy, wz,wx1, wy1, wz1;
  double I[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

  gluProject( p[0], p[1], p[2],
	      (double *)model, (double *)proj, vp, 
	      &wx, &wy, &wz );
	      
  gluProject( p[0], p[1], p[2],
	      (double *)PM, I, vp, 
	      &wx1, &wy1, &wz1 );

  printf("cmp:   %5.1lf %5.1lf %5.1lf %5.1lf\n"
	 "       %5.1lf %5.1lf %5.1lf \n" 
	 "       %5.1lf %5.1lf %5.1lf \n" 
	 "       %5.1lf %5.1lf %5.1lf \n",
	 pp[0], pp[1], pp[2], pp[3],
	 vp[0]+vp[2]*(pp[0]/pp[3]+1)/2, 
	 vp[1]+vp[3]*(pp[1]/pp[3]+1)/2,
	 (pp[2]/pp[3]+1)/2,

	 wx, wy, wz,
	 wx1, wy1, wz1);
#endif

  double lx = APM[0][0]*hdx + APM[1][0]*hdy + APM[2][0]*hdz;
  double ly = APM[0][1]*hdx + APM[1][1]*hdy + APM[2][1]*hdz;
  //  double lz = APM[0][2]*hdx + APM[1][2]*hdy + APM[2][2]*hdz;
  double lw = APM[0][3]*hdx + APM[1][3]*hdy + APM[2][3]*hdz;

 
  double near = 1./(pp[3]-lw);
  double far  = 1./(pp[3]+lw);

  double at = pp[0]-lx;
  left = vp[0] + vp[2]*(at* (at>0?far:near)+1)/2;
  at = pp[0]+lx;
  right =vp[0]+vp[2]*(at* (at<0?far:near)+1)/2;
  at = pp[1]-ly;
  bottom = vp[1]+vp[3]*(at* (at>0?far:near)+1)/2;
  at = pp[1]+ly;
  top = vp[1]+vp[3]*(at* (at<0?far:near)+1)/2;

  return pp[3];
}



double
Hase::hw_project( int i, int j, int k, int di, int dj, int dk,
		  double &left, double &right, double &top, double &bottom )
{
  iotimer_t  start = read_time();
  double x = i*sx;
  double y = j*sy;
  double z = k*sz;

  double dx = di*sx;
  double dy = dj*sy;
  double dz = dk*sz;

  double wx, wy, wz;

  gluProject(x, y, z, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  left = right = wx;
  top = bottom = wy;

  gluProject(x+dx, y, z, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x, y+dy, z, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x+dx, y+dy, z, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x, y, z+dz, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x+dx, y, z+dz, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x, y+dy, z+dz, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  gluProject(x+dx, y+dy, z+dz, (double *)model, (double *)proj, vp, &wx, &wy, &wz );
  if ( left > wx ) left = wx;
  else if ( right < wx ) right = wx;
  if ( bottom > wy ) bottom = wy;
  else if ( top < wy ) top = wy;

  iotimer_t end = read_time();
  project_timer += end-start;
  
  return wz;
}

void
Hase::reset_view( GeometryData *gd )
{
  static int init = 1;
  static int do_test = 0;

  if ( init ) {
    cerr << "init\n";
    // switch to the other window
    TCLTask::lock();
    clString myname(clString(".ui")+id+".gl.gl");
    char *name = strdup(myname());
    tkwin=Tk_NameToWindow(the_interp, name, Tk_MainWindow(the_interp));
    if(!tkwin){
      cerr << "Unable to locate window!\n";
      TCLTask::unlock();
      return;
    }
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(the_interp, name);
    if(!cx){
      cerr << "Unable to create OpenGL Context!\n";
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }
    if (!glXMakeCurrent(dpy, win, cx))
      cerr << "*glXMakeCurrent failed.\n";
    
    TCLTask::unlock();
    
    glLogicOp( GL_XOR );

    hist_size = 8;
    hist = new unsigned short[hist_size*4];
    
    hw_init = 1;
    init = 0;
  }
  
  if ( hw_init ) {
    if ( minmax ) { 
      cerr << "hw init minmax\n";
      glEnable(GL_MINMAX_EXT);
      glMinmaxEXT( GL_MINMAX_EXT, GL_LUMINANCE, GL_TRUE);
    }
    else {
      cerr << "hw init histogram\n";
      glHistogramEXT(GL_HISTOGRAM_EXT, 
		     hist_size, 
		     GL_RGBA, 
		     GL_TRUE);
    }
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR)
      cerr << "EXT GL Err: " << (char*)gluErrorString(errcode)
	   << endl;
    
    hw_init = 0;
  }
  // reset view
  View &view = *gd->view;
  xres = gd->xres;
  yres = gd->yres;
  double znear = gd->znear;
  double zfar = gd->zfar;
  
  double aspect=double(xres)/double(yres);
  double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov()/2.))));
  
  glViewport(0, 0, xres, yres);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy, aspect, znear, zfar);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  Point eyep(view.eyep());
  Point lookat(view.lookat());
  Vector up(view.up());
  gluLookAt(eyep.x(), eyep.y(), eyep.z(),
	    lookat.x(), lookat.y(), lookat.z(),
	    up.x(), up.y(), up.z());
  
  
  // get the transformation matrix
  
  glGetDoublev( GL_MODELVIEW_MATRIX, (double *)model);
  glGetDoublev( GL_PROJECTION_MATRIX, (double *)proj);
  glGetIntegerv( GL_VIEWPORT, vp );
  
  
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++ ) {
      PM[j][i] = 0;
      for (int k=0; k<4; k++)
	PM[j][i] += proj[k][i]*model[j][k];
      APM[j][i] = fabs(PM[j][i]);
    }
  }

  // pre-compute constants for projecting screen points back to object space
  Cx = PM[1][2]/PM[0][1];
  Cz = Cx * (PM[0][2] - PM[1][2]);
  Cw = Cx * (PM[0][3] - PM[1][3]);
  Cd = 1/(PM[1][0] - PM[1][2]*PM[0][0]);

  
  // clear screen
  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  // set state
  glDrawBuffer(GL_FRONT);
//   glColor3f(0.8,0.4,0.2);
  glColor3f(1,0,0);
  glDisable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glDisable(GL_CULL_FACE);
  glEnable(GL_HISTOGRAM_EXT);
  glDisable(GL_COLOR_MATERIAL);
  glReadBuffer(GL_FRONT);

  for (int i=0; i<256; i++) {
    collect_copy[i].time = 0;
    collect_copy[i].n = 0;
    collect_copy[i].vis = 0;
    collect_copy[i].area = 0;
    collect_copy[i].empty = 0;
  }

  if ( do_test ) {
    test();
    do_test = 0;
  }
}



int
Hase::hw_visible( double left, double bottom, double right, double top )
{
  if ( left >= xres || right < 0 || bottom >= yres || top < 0 )
    return 0;

  int l = trunc(left+0.5);
  int r = trunc(right+0.5);
  int b = trunc(bottom+0.5);
  int t = trunc(top+0.5);

  if ( l < 0 ) l = 0;
  if ( r >= xres ) r = xres-1;
  if ( b < 0 ) b = 0;
  if ( t >= yres ) r = yres-1;
  
  if ( l==r || b==t ) return 0;

  int area = (r-l)*(t-b);
  if ( area > max_area ) return 1;

  // get histogram of this bounding box
  if ( finish ) glFinish();

  iotimer_t start = read_time();
  glCopyPixels( l, b, r-l, t-b, GL_COLOR );
  if ( finish ) glFinish();
  iotimer_t mid = read_time();
  if ( minmax ) {
    glGetMinmaxEXT( GL_MINMAX_EXT,
		    GL_TRUE,
		    GL_RED,
		    GL_UNSIGNED_SHORT,
		    hist );
  }
  else {
    glGetHistogramEXT( GL_HISTOGRAM_EXT, 
		       GL_TRUE, 
		       GL_RGBA,
		       GL_UNSIGNED_SHORT, 
		       hist);
  }
  if ( finish ) glFinish();
  iotimer_t end = read_time();
  
  int vis = minmax ? hist[0] == 0 : hist[0] > 0;

  int a = int(sqrt(double(area))/2);
  collect_copy[a].time += mid-start;
  collect_copy[a].area += area;
  collect_copy[a].n++;
  if ( vis ) collect_copy[a].vis++;
//   if ( minmax ) cerr << "minmax = (" << area << ") " << hist[0] << " " << hist[1] << endl;
  if ( minmax && hist[1] == 0 )
    collect_copy[a].empty++;
  
  
  copy_timer+= mid-start;
  hist_timer+= end-mid;

  return vis;
}

void
Hase::test()
{
  FILE *file1 = fopen("test1", "w");
  FILE *file2 = fopen("test2", "w");
  FILE *file3 = fopen("test3", "w");

  cerr << "testing... " ;
  glFinish();
  for (int i=1; i<512; i++) {
    iotimer_t start = read_time();
    
    glCopyPixels( 0, 0, i, i, GL_COLOR );
    glFinish();
    
    iotimer_t mid = read_time();
    
    glBegin(GL_POLYGON);
    glVertex2i(0,0);
    glVertex2i(0,1);
    glVertex2i(1,1);
    glVertex2i(1,0);
    glEnd();
    glFinish();
    
    iotimer_t mm = read_time();
    
    glBegin(GL_POLYGON);
    glVertex2i(0,0);
    glVertex2i(0,1);
    glVertex2i(1,1);
    glVertex2i(1,0);
    glEnd();
    glCopyPixels( 0, 0, i, i, GL_COLOR );
    glFinish();
    
    iotimer_t end = read_time();
    
    fprintf( file1, "%4d %.3lf \n", i, (mid-start)*cycleval*1e-9);
    fprintf( file2, "%4d %.3lf \n", i, (mm - mid)*cycleval*1e-9);
    fprintf( file3, "%4d %.3lf \n", i, (end - mm)*cycleval*1e-9);
  }
  
  fclose(file1);
  fclose(file2);
  fclose(file3);
  
  //   cerr << "repeat.. ";
  //   file = fopen("test2", "w");
  //   for (int i=1, k=512*512; i<512; i*=2, k/=4) {
  //     iotimer_t start = read_time();
  //     for (int j=0; j<k; j++) {
  //       glCopyPixels( 0, 0, i, i, GL_COLOR );
  //       glFinish();
  //     }
  //     iotimer_t end = read_time();
  //     fprintf( file, "%4d %.3lf\n", 
  // 	     i, (end-start)*cycleval*1e-9);
  
  
  //   cerr << "repeat(2).. ";
  
  //   fprintf( file, "\n\n");
  //   for (int i=1, k=512*512; i<512; i*=2, k/=4) {
  //     iotimer_t start = read_time();
  //     for (int j=0; j<k; j++) {
  //       glCopyPixels( 0, 0, i, i, GL_COLOR );
  //     }
  //     glFinish();
  //     iotimer_t end = read_time();
  //     fprintf( file, "%4d %.3lf\n", 
  // 	     i, (end-start)*cycleval*1e-9);
  //   }
  //   fclose(file);
  cerr << "done" << endl;
}

void
Hase::test_project( int i, int j, int k, int dx, int dy, int dz )
{
  double cl, cr, cb, ct;
  double el, er, eb, et;

  bbox_projection( i, j, k, dx, dy, dz, el, er, et, eb );
  hw_project( i, j, k, dx, dy, dz, cl, cr, ct, cb );

  if ( cl < el || cr > er || cb < eb || ct > et ) {
    char *lm =  ( cl < el ) ? "*" : " ";
    char *rm =  ( cr > er ) ? "*" : " ";
    char *tm =  ( ct > et ) ? "*" : " ";
    char *bm =  ( cb < eb ) ? "*" : " ";
    printf("project bug: at [%d,%d,%d] size[%dx%dx%d]\n"
	   "              %sleft     %sright   %sbottom  %stop\n"
	   "   computed:   %5.2lf   %5.2lf   %5.2lf  %5.2lf \n"
	   "   estimated:  %5.2lf   %5.2lf   %5.2lf  %5.2lf \n",
	   i, j, k, dx, dy, dz,
	   lm, rm, bm, tm,
	   cl, cr, cb, ct,
	   el, er, eb, et);
    scanf("%*c");
  }
} // End namespace Yarden
}


























