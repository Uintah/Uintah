/*
 *  SageVFem.cc    View Depended Iso Surface Extraction
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

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Timer.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Geom/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/SurfacePort.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Datatypes/Color.h>
#include <Geom/GeomOpenGL.h>

#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <TCL/TCLTask.h>
#include <TCL/GuiVar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <Geom/Material.h>
#include <Geom/Triangles.h>
#include <Geom/View.h>
#include <Geom/Group.h>
#include <Geom/Geom.h>
#include <Geom/Tri.h>
#include <Geom/Line.h>
#include <Geom/Box.h>
#include <Geom/Pt.h>
#include <Geom/Transform.h>
#include <Geometry/Point.h>
#include <Geometry/Transform.h>
#include <Geom/BBoxCache.h>
#include <Malloc/Allocator.h>
#include <Geometry/Vector.h>
#include <TCL/GuiVar.h>
#include <Math/Trig.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>
#include <Packages/Yarden/Core/Datatypes/Screen.h>
#include <Packages/Yarden/Core/Datatypes/Clock.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/mcube_scan.h>


//#define DOUBLE
#define VIS_WOMAN

#ifdef VIS_WOMAN
  #define SHORT
#endif

#ifdef CHAR
  typedef unsigned char Value;
  typedef ScalarFieldRGchar FIELD_TYPE ;
  #define GET_FIELD(f) (f->getRGBase()->getRGChar())
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

int offset = 0;
extern int show;

iotimer_t rebuild_start, rebuild_make,
  rebuild_draw, rebuild_get,
  rebuild_build, rebuild_last, rebuild_end;
iotimer_t make_start, make_lock,make_info,make_get, make_make, make_mat,
  make_err, make_done;

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

iotimer_t extract_timer, vis_timer;
CPUTimer t_extract;
int bbox_count;
int debug = 1;
int scan_yes;
int scan_no;
int print_stat = 0;

int stopat = 10000000;
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
    int value, bbox_visible, bbox_not_visible, reduce,
      extract, visible, not_visible;
    value = visible = bbox_visible = bbox_not_visible = reduce =
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

struct SageVFemNode;

// Cell
struct SageVFemCell {
  SageVFemNode *node;
  int i, j, k;
  int dx, dy, dz;
  int mask;
};

// Stack
class SageVFemStack {
public:
  int size;
  int pos;
  int depth;
  int use;
  SageVFemCell *top;
  SageVFemCell *stack;

public:
  SageVFemStack() { size = 0; pos = 0; depth = 0; top=0; stack = 0;}
  ~SageVFemStack() { if ( stack ) delete stack; }

  void resize( int s );
  void push( SageVFemNode *, int, int, int, int, int, int, int );
  void pop( SageVFemNode *&, int &, int& , int &, int &, int &, int &, int &);
  int empty() { return pos==0; }
  void print() { printf("Stack max depth = %d / %d [%.2f]\n  # of op = %d\n",
			depth, size, 100.0*depth/size,use);}
  void reset() { top = stack; pos=0;}
};

void
SageVFemStack::resize( int s )
{
  if ( s > size ) {
    if ( stack ) delete stack;
    stack = scinew SageVFemCell[s];
    size = s;
  }
  pos = 0;
  depth = 0;
  use = 0;
  top = stack;
}

inline void
SageVFemStack::pop( SageVFemNode *&node, int &i, int &j, int &k, 
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
SageVFemStack::push( SageVFemNode *node, int i, int j, int k, int dx,
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

// SageVFemTree

struct SageVFemNode {
  SageVFemNode *child;
  Value min, max;
  unsigned char type;
};

class SageVFemTree {
public:
  FIELD_TYPE *field;
  SageVFemNode **tree;
  SageVFemNode **tree_next;
  int size; 
  int dx, dy, dz;
  int depth;
  int last;
  int mask;
  
public:
  SageVFemTree() {tree=0; size=0;}
  ~SageVFemTree() {if ( tree ) delete [] tree; }

  //SageVFemNode &operator[](int i) { return tree[i]; }
  void init( FIELD_TYPE *);
  void build( int level, SageVFemNode &, int mask, int x, int y, int z );
  void get_minmax( Value v, Value &min, Value &max );
  void child_minmax( SageVFemNode &child, int x, int y, int z, int dx, int dy, int dz,
		     int mask, Value &min, Value &max );
  void fill( SageVFemNode &node,
	     int x, int y, int  z, int dx, int dy, int dz, int mask,
	     Value &pmin, Value &pmax );
};

int ss[10];

void
SageVFemTree::init( FIELD_TYPE *f )
{
  field = f;
  dx = field->grid.dim1()-1; // number of cells = number of pts -1
  dy = field->grid.dim2()-1;
  dz = field->grid.dim3()-1;

  if (tree ) {
    for (int l=0; l<depth+1; l++ )
      delete tree[l];
    delete tree;
    delete tree_next;
  }
  int dim = dx;
  if ( dy > dim ) dim = dy;
  if ( dz > dim ) dim = dz;
  int mdim = dim-1;  //  = number of cells - 1 = number_of pts -2
  for (depth=1,mask=1; mdim > 1 ; mdim>>=1, depth++, mask<<=1);

  int new_size = 0;
  tree = new SageVFemNode*[depth+1];
  tree_next = new SageVFemNode*[depth+1];

  int s = 1;
  printf("depth: %d\n",depth);
  for (int i=depth; i>0; i--) {
    int n = ((dx>>i)+1)*((dy>>i)+1)*((dz>>i)+1);
    ss[depth-i] = s;
    tree_next[depth-i] = tree[depth-i] = scinew SageVFemNode[s];
    new_size += s;
    printf("SageVFemTree [%d]  %d %d   [%3dx%3dx%3d]\n", 
	   depth-i, s, new_size,
	   ((dx>>i)+1),((dy>>i)+1),((dz>>i)+1));
    s = n*8;;
  }

  printf( "SageVFemTree: %d / %ld\n", new_size,  long(dim+1)*(dim+1)*(dim+1)*8/7 );
  printf( "\tmemory  node: (2*%d + %d + %d) = %d bytes\n"
	  "\t  tree = %.0fMB   data = %.0fMB\n", 
	  sizeof(Value), sizeof(SageVFemNode*),sizeof(int),sizeof(SageVFemNode),
	  new_size*sizeof(SageVFemNode)/float(1024*1024),
	  (dx*dy*dz)*sizeof(Value)/float(1024*1024)) ;

  build( 1, tree[0][0], mask, dx-2, dy-2, dz-2 );

  cerr << "fill [" << dx << ", " << dy << ", " << dz << "]  depth= " 
       << depth << "\n" ;


  Value min, max;
  fill( tree[0][0], 0, 0, 0, dx-2, dy-2, dz-2, mask, min, max );
  cerr << "Min " << int(min) << "  Max " << int(max) << "\n";
}

void
SageVFemTree::build( int level, SageVFemNode &node, int mask, int dx, int dy, int dz )
{
  if ( !( dx>1 || dy>1 || dz>1 ) ) {
    node.child = 0;
    node.type = 7;
//     printf("build: end at level %d.   %dx%dx%d\n", level ,dx, dy,dz );
    return;
  }
  
  //assert (ss[level] >= 8 );
  
  node.child = tree_next[level];
  tree_next[level] += 8;
  ss[level]-=8;
  
  unsigned type = 0;
  int dx1, dy1, dz1;
  if ( (mask & dx) && (dx > 1) ) {
    dx1 = dx & ~mask;
    dx = mask-1;
    type = 1;
  }
  if ( (mask & dy) && (dy > 1) ) {
    dy1 = dy & ~mask;
    dy  = mask-1;
    type += 2;
  }
  if ( (mask & dz) &&(dz > 1) ) {
    dz1 = dz & ~mask;
    dz  = mask-1;
    type+=4;
  }

  type = (~type) & 0x7;
  node.type = type;

  for (int i=0; i<8; i++ )
    if ( ! (i & type) )
      build ( level+1, node.child[i], mask>>1,
	      i & 1 ? dx1 : dx,
	      i & 2 ? dy1 : dy, 
	      i & 4 ? dz1 : dz );
    else
      node.child[i].child = 0;
}

inline void
SageVFemTree::get_minmax( Value v, Value &min, Value &max )
{
  if ( v < min ) min = v;
  else if ( v > max ) max = v;
}
 
inline void
SageVFemTree::child_minmax( SageVFemNode &child, int x, int y, int z, int dx, int dy, int dz,
			int mask,  Value &min, Value &max )
{
  Value cmin = 0;
  Value cmax = 0;

  fill( child, x, y, z, dx, dy, dz, mask, cmin, cmax );

  if ( cmin < min ) min = cmin;
  if ( max < cmax ) max = cmax;
}

int iii = 0;
void
SageVFemTree::fill( SageVFemNode &node, int x, int y, int  z, int dx, int dy, int dz,
		int mask, Value &pmin, Value &pmax )
{
  SageVFemNode *child = node.child;
  int type = node.type;
  
  Value min, max;
  
  if ( !child ) {
    min = max = field->grid(x,y,z   );
    for (int i=0; i<3; i++ )
      for (int j=0; j<3; j++)
	for (int k=0; k<3; k++)
	  get_minmax( field->grid( x+i, y+j, z+k ), min, max );
  }
  else {
    int dx1, dy1, dz1;
    if ( (mask & dx) && dx>1) {
      dx1 = dx & ~mask;
      dx  = mask-1;
    }
    if ( (mask & dy) && dy>1) {
      dy1 = dy & ~mask;
      dy  = mask-1;
    }
    if ( (mask & dz) && dz>1) {
      dz1 = dz & ~mask;
      dz  = mask-1;
    }
    mask >>= 1;
    
    fill( child[0], x, y, z, dx, dy, dz, mask, min, max );
    if ( !(type & 0x1) ) child_minmax( child[1], x+dx+1, y,    z,
				       dx1, dy, dz, mask, min, max );
    if ( !(type & 0x2) ) child_minmax( child[2], x,    y+dy+1, z,
				       dx, dy1, dz, mask, min, max );
    if ( !(type & 0x3) ) child_minmax( child[3], x+dx+1, y+dy+1, z,
				       dx1, dy1, dz, mask, min, max );
    if ( !(type & 0x4) ) child_minmax( child[4], x,    y,    z+dz+1,
				       dx, dy, dz1, mask, min, max );
    if ( !(type & 0x5) ) child_minmax( child[5], x+dx+1, y,    z+dz+1,
				       dx1, dy, dz1, mask, min, max );
    if ( !(type & 0x6) ) child_minmax( child[6], x,    y+dy+1, z+dz+1,
				       dx, dy1, dz1, mask, min, max );
    if ( !(type & 0x7) ) child_minmax( child[7], x+dx+1, y+dy+1, z+dz+1,
				       dx1, dy1, dz1, mask, min, max );
  }
  
  node.min = min;
  node.max = max;
  pmin = min;
  pmax = max;
}

// Warp

struct Warp {
  double xscale;
  double yscale;
  double x;
  double y;
};


struct VFem {
  FIELD_TYPE *field;
  double scale;
  double offset_x, offset_y, offset_z;

  int dx, dy, dz;
  double gx, gy, gz;
  double sx, sy, sz;
  Value min, max;

  int mask;

  SageVFemTree tree;
};

// SageVFem

class SageVFem : public Module 
{
  ScalarFieldIPort* infield;  // input scalar fields (bricks)
  ScalarFieldIPort* incolorfield;
  ColorMapIPort* incolormap;

  GeometryOPort* ogeom;       // input from salmon - view point

  ScalarFieldHandle handle[7];
  VFem  vfem[7];

  Tk_Window tkwin;
  Window win;
  Display* dpy;
  GLXContext cx;

  CrowdMonitor widget_lock;

  //float* data;
  GLubyte *data;
  GuiDouble isoval;
  GuiDouble isoval_min, isoval_max;
  GuiInt tcl_value, tcl_bbox, tcl_visibility;
  GuiInt tcl_scan, tcl_depth, tcl_reduce, tcl_cover, tcl_init;
  GuiInt tcl_projection;
  
  int value, bbox_visibility, visibility, cutoff_depth;
  int scan, count_values, extract_all, projection;

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
  GeomPtsN* points;
  GeomGroup* tgroup;
  GeomObj* topobj;
  BBox box;

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

  SageVFemStack stack;
  SageVFemTree tree;
  Statistics statistics;
  int counter;

  CrowdMonitor lock;
  char *gl_name;
  int init_gl;

  Vector U,V,W;
  Vector AU, AV, AW;
  Vector X,Y,Z;
  int xres, yres;

  HScreen screen;
  int screen_id;
  
  double offset_x, offset_y, offset_z;
  double scale;

  double pixel_size[2000];

public:

  SageVFem( const clString& id);
  SageVFem( const SageVFem&, int deep );
  virtual ~SageVFem();

  virtual Module* clone(int deep);
  virtual void execute();

  void project( const Point &, Pt &);

  int  read_vfem_files();
  void new_field( int, FIELD_TYPE *field );
  void check( const Point &, double &, double &, double &, double & );
  void search();
  void search(int);
  void search( double );
  int  extract( double, int, int, int, int, int, int );
  int  make_current( int xres, int yres );
  void tcl_command(TCLArgs &, void *);
  void redraw( int xres, int yres);
  void compute_depth( double& znear, double& zfar);

  void display( Point [] );
  void redraw_done();
  double bbox_projection( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom );
  double bbox_projection1( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom );
  double bbox_projection2( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom );
  void dividing_cubes( double v, int x, int y, int z, int dx, int dy, int dz,
		       int px, int py );
  int adjust( double, double, int &);
};
  
extern "C" Module* make_SageVFem(const clString& id)
{
  printf("tri_case[136].vertex = {\n");
  for (int i=0; i<16; i++ )
    printf(" %d", tri_case[136].vertex[i] );
  printf("}\n");
  return scinew SageVFem(id);
}

static clString module_name("SageVFem");
static clString box_name("SageVFemBox");
static clString surface_name("SageVFem");
static clString surface_name1("SageVFemScreen");
static clString surface_name2("SageVFemBack");
static clString surface_name3("SageVFemPts");
static clString shadow_name("SageVFemShadow");

SageVFem::SageVFem(const clString& id)
  : Module("SageVFem", id, Filter ), isoval("isoval", id, this),
    isoval_min("isoval_min", id, this), isoval_max("isoval_max", id, this),
    tcl_bbox("bbox", id, this), 
    tcl_scan("scan", id, this),  tcl_value("value", id, this), 
    tcl_visibility("visibility", id, this),
    tcl_depth("cutoff_depth", id, this),
    tcl_reduce("reduce",id,this), tcl_cover("cover",id,this), 
    tcl_init("init",id,this),
    tcl_projection("projection",id,this)
{
  init_clock();
  printf( "SageVFem::SageVFem :: %d\n", tri_case[136].vertex[4]);
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
  
  int n =  4096;
  warp = 0;
  initialized = 0;

  screen.setup( 512, 512 );
  prev_value = -1;
  screen_id = 0;

  vfem[0].scale = .48828;
  vfem[1].scale = .72266;
  vfem[2].scale = .85937;
  vfem[3].scale = .9375;
  vfem[4].scale = .72266;
  vfem[5].scale = .9375;
  vfem[6].scale = .72266;

  vfem[0].offset_x = vfem[0].offset_y = 256*(1-vfem[0].scale);
  vfem[1].offset_x = vfem[1].offset_y = 256*(1-vfem[1].scale);
  vfem[2].offset_x = vfem[2].offset_y = 256*(1-vfem[2].scale);
  vfem[3].offset_x = vfem[3].offset_y = 256*(1-vfem[3].scale);
  vfem[4].offset_x = vfem[4].offset_y = 256*(1-vfem[4].scale);
  vfem[5].offset_x = vfem[5].offset_y = 256*(1-vfem[5].scale);
  vfem[6].offset_x = vfem[6].offset_y = 256*(1-vfem[6].scale);

  vfem[0].offset_z = 0;
  vfem[1].offset_z = 209;
  vfem[2].offset_z = 227;
  vfem[3].offset_z = 249;
  vfem[4].offset_z = 1106;
  vfem[5].offset_z = 1110;
  vfem[6].offset_z = 1117;

}
  
char *dir="/local/sci/raid0/yarden";
//char *dir="/scratch/yarden/";


int use_file_vfem[] = { 1, 1, 1, 1, 1, 1, 1 };

int 
SageVFem::read_vfem_files()
{
  char name[80];
  int first = -1;
 
  for ( int i=0; i<7; i++) {
    sprintf( name, "%s/vfem_%d.sff", dir, i );
    if ( !use_file_vfem[i] ) {
      cerr << "skip file " << name << endl;
      continue;
    }
    if ( first  == -1 ) first = i;

    cerr << "reading file " << name << endl;
    clString fn(name);
    Piostream* stream=auto_istream(fn);
    if(!stream){
      error(clString("Error reading file: ")+name);
      return 0;
    }
    // Read the file...
    //	stream->watch_progress(watcher, (void*)this);
    Pio(*stream, handle[i]);
    if(!handle[i].get_rep() || stream->error()){
      error("Error reading ScalarField from file");
      delete stream;
      return 0;
    }
    delete stream;
    new_field(i, GET_FIELD(handle[i]));
  }

  if ( first == -1 ) {
    cerr << "not field was read" << endl;
    return;
  }
  Value min = vfem[first].min;
  Value max = vfem[first].max;
  for ( i=0; i<7; i++) {
    if ( use_file_vfem[i] ) {
      if ( vfem[i].min < min ) min = vfem[i].min;
      if ( vfem[i].max > max ) max = vfem[i].max;
    }
  }

  cerr << "vfem read - done" << endl;
  isoval_min.set(min);
  isoval_max.set(max);
  isoval.set((min+max)/2);
  reset_vars();

  stack.resize( 1000 ); // more then enough
//   for (dim = 1; dim < mdim; dim <<=1);

  return 1;
}

SageVFem::SageVFem( const SageVFem& copy, int deep ) :
  Module(copy, deep), isoval("isoval", id, this ),
    isoval_min("isoval_min", id, this), isoval_max("isoval_max", id, this),
    tcl_bbox("bbox", id, this), 
    tcl_scan("scan", id, this),  tcl_value("value", id, this), 
    tcl_visibility("visibility", id, this),
    tcl_depth("cutoff_depth", id, this),
    tcl_reduce("reduce",id,this), tcl_cover("cover",id,this), 
    tcl_init("init",id,this),
    tcl_projection("projection",id,this)
{
    NOT_FINISHED("SageVFem::SageVFem");
}

SageVFem::~SageVFem()
{
}

Module* SageVFem::clone(int deep)
{
  return scinew SageVFem(*this, deep);
}


void
SageVFem::new_field( int f, FIELD_TYPE *field )
{
  vfem[f].field = field;

  dx = vfem[f].dx = field->grid.dim1()-1;
  dy = vfem[f].dy = field->grid.dim2()-1;
  dz = vfem[f].dz = field->grid.dim3()-1;
  
  int mdim = dx;
  if ( mdim < dy ) mdim = dy;
  if ( mdim < dz ) mdim = dz;

  field->get_bounds( bmin, bmax );

  box.extend(bmin);
  box.extend(bmax);

  cerr << "Field bounds: " << bmin << "   " << bmax << "\n";
  vfem[f].gx = bmax.x() - bmin.x();
  vfem[f].gy = bmax.y() - bmin.y();
  vfem[f].gz = bmax.z() - bmin.z();

  vfem[f].sx =  vfem[f].gx/ vfem[f].dx;
  vfem[f].sy =  vfem[f].gy/ vfem[f].dy;
  vfem[f].sz =  vfem[f].gz/ vfem[f].dz;

  double dmin, dmax;
  field->get_minmax( dmin, dmax );
  vfem[f].min = Value(dmin);
  vfem[f].max = Value(dmax);


  vfem[f].tree.init( field );
  vfem[f].mask = vfem[f].tree.mask;
}


inline void
SageVFem::project( const Point &p, Pt &q )
{
  Vector t = p - eye;
  double px = Dot(t, U );
  double py = Dot(t, V );
  double pz = Dot(t, W );
  q.x = (px/pz+1)*xres/2-0.5;
  q.y = (py/pz+1)*yres/2-0.5;
}

void SageVFem::execute()
{
  static int init = 1;

  if ( init ) {
    cerr << "vfem init" << endl;
    read_vfem_files();
    init = 0;
    GeomObj *bbox= scinew GeomMaterial( scinew GeomBox(box.min(), 
						       box.max(),
						       1),
					box_matl);
    box_id = ogeom->addObj( bbox, box_name );
    return;
  }

  cerr << "exec vfem " << endl;
  iotimer_t start = read_time();
  extract_timer = vis_timer = 0;

  gd = ogeom->getData(0, GEOM_VIEW);
  if ( gd == NULL ) {
    cerr << "using prev. view." << endl;
    gd = local_gd;
  }
  else {
    *local_gd->view = *gd->view;
    local_gd->znear = gd->znear;
    local_gd->xres = gd->xres;
    local_gd->yres = gd->yres;
  }
  
  view = gd->view;
  eye = view->eyep();
  xres = gd->xres;
  yres = gd->yres;

//   cerr << "View:\n"
//        << "\teye: " << eye << "  look at: " << view->lookat() << endl
//        << "\tup: " << view->up() << endl
//        << "\tfov: " << view->fov() << endl;

  Z = Vector(view->lookat()-eye);
  Z.normalize();
  X = Vector(Cross(Z, view->up()));
  X.normalize();
  Y = Vector(Cross(X, Z));
  double xviewsize= 1./Tan(DtoR(view->fov()/2.));
  double yviewsize=xviewsize*gd->yres/gd->xres;;
  U = X*xviewsize;
  V = Y*yviewsize;
  W = Z;

  X = X/xviewsize;
  Y = Y/yviewsize;

  search();
}

void
SageVFem::search( int f)
{ 
  if ( !use_file_vfem[f] ) 
    return;

  dx = vfem[f].dx;
  dy = vfem[f].dy;
  dz = vfem[f].dz;
  sx = vfem[f].sx;
  sy = vfem[f].sy;
  sz = vfem[f].sz;
  scale = vfem[f].scale;
  mask = vfem[f].mask;
  offset_x = vfem[f].offset_x;
  offset_y = vfem[f].offset_y;
  offset_z = vfem[f].offset_z;
  field = vfem[f].field;

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
 
  stack.reset();
  stack.push( vfem[f].tree.tree[0], 0, 0, 0, dx-1, dy-1, dz-1, mask );

  // SEARCH >>>

  if ( visibility ) {
    static double prev_iso_value = -1;
    redraw( xres, yres );
    if ( iso_value != prev_iso_value ) {
      glClearColor(1,1,1,0);
      glClear(GL_COLOR_BUFFER_BIT  );
      prev_iso_value = iso_value;
    }
//     show = visibility;
    search( iso_value );
    screen.display();
    redraw_done();
  }
  else 
    search(iso_value);

}





void
SageVFem::search()
{
  iotimer_t start = read_time();
  
  trid = 0;
  
  scan_yes = scan_no = 0;
  value = tcl_value.get();
  scan = tcl_scan.get();
  visibility = tcl_visibility.get();
  bbox_visibility = tcl_bbox.get();
  reduce =  tcl_reduce.get();
  screen.cover( tcl_cover.get() );
  extract_all = tcl_init.get();
  iso_value = isoval.get();
  projection = tcl_projection.get();
  statistics.reset();

  group = scinew GeomTrianglesP;
  points = scinew GeomPtsN(2000);
  tgroup=scinew GeomGroup;
  topobj=tgroup;
  
  
  ScalarFieldHandle colorfield;
  int have_colorfield=incolorfield->get(colorfield); ColorMapHandle cmap;
  int have_colormap=incolormap->get(cmap);
  
  if(have_colormap && !have_colorfield){
    // Paint entire surface based on colormap
    topobj=scinew GeomMaterial(tgroup, cmap->lookup(iso_value));
  } else if(have_colormap && have_colorfield){
      // Nothing - done per vertex
  } else {
    // Default material
    topobj=scinew GeomMaterial(tgroup, iso_value < 1000 ? flesh : bone);
  }

  
  iotimer_t start1 = read_time();

  screen.clear();

  int i=0;
  while ( i<7 && vfem[i].offset_z < eye.z() ) i++;
  if ( i == 7 ) i = 6;
  search( i );
  int f;
  for ( f=i-1; f>=0; f--) search(f);
  for ( f=i+1; f<7; f++) search(f);

  iotimer_t end1 = read_time();
  

  if ( points_id ) {
    ogeom->delObj(points_id);
    points_id = 0;
  }
  
  if( surface_id )
      ogeom->delObj(surface_id);
  
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
			      "Dividing Cubes");
  
  iotimer_t end = read_time();			
  printf("Scan: %d cells\n", statistics.extracted );
//   printf("Scan : %d %d\n", scan_yes, scan_no );	
  
  printf(" Search Timers: \n\tinit %.3lf  \n"
	 "\tsearch %.3lf (%.3lf  %.3lf) \n"
	 "\tall %.3lf\n ",
   	 (end-start -(end1-start1))*cycleval*1e-9,
   	 (end1-start1)*cycleval*1e-9, 
   	 vis_timer*cycleval*1e-9, extract_timer*cycleval*1e-9, 
   	 (end-start)*cycleval*1e-9);
  
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

int
SageVFem::adjust( double left, double right, int &x )
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


#define Deriv(u1,u2,u3,u4,d1,d2,d3,d4) (((val[u1]+val[u2]+val[u3]+val[u4])-\
					(val[d1]+val[d2]+val[d3]+val[d4]))/4.)
void
SageVFem::search( double v )
{
  tpos = 0;
  iotimer_t begin = read_time();
  iotimer_t start = read_time();
  int count=0;
  glLogicOp( GL_XOR );

  while ( !stack.empty() ) {

    if ( abort_flag )
      return;
      
    int i, j, k;
    int dx, dy, dz;
    int mask;
    SageVFemNode *node;
    //table[tpos].cycle = read_time() - start;
    iotimer_t end = read_time();
    
    //tpos++;
    stack.pop( node, i, j, k, dx, dy, dz, mask);
    
    //     cerr << "node: " 
    //  	 << " ["<<i<<","<<j<<","<<k<<"]  " 
    //  	 << " ("<<dx<<"x"<<dy<<"x"<<dz<<")  " 
    //  	 << node->min << " " << node->max << endl;
    // if ( tpos == table_size ) {
    //    cerr << "table too small!\n";
    //    exit(0);
    // }
    //table[tpos] = empty_table;
    if (  node && (v < node->min || node->max < v) ) {
      //table[tpos].value = (dx+1)*(dy+1)*(dz+1);
      statistics.value++;
      continue;
    }
    
    double left, right, top, bottom;
    left = bottom = 0;
    right = top = 10;

    if ( bbox_visibility  ) {
      double left, right, top, bottom;
      double l,r,t,b;
      
      double pw;
      
      if ( projection ) 
	pw = bbox_projection( i, j, k, dx+1, dy+1, dz+1,
			      left, right, top, bottom );
      else
	pw = bbox_projection1( i, j, k, dx+1, dy+1, dz+1,
			       left, right, top, bottom );
      
      
       if ( reduce ) {
	 if ( (right-left) <= 1 && (top-bottom) <= 1 ) {
	   int px,py;
	   top+= 0.5;
	   right += 0.5;
	   if ( adjust( left, right, px ) && adjust( bottom, top, py ) ) {
	     if ( screen.cover_pixel(px,py) ) {
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
	       points->add( Q, N );
	     }
	   }
	   continue;
	 }
       }
       iotimer_t vis_begin = read_time();
       int vis = screen.visible( left, bottom, right, top );
       iotimer_t vis_end = read_time();
       vis_timer += vis_end-vis_begin;
       
       if ( !vis ) {
	 statistics.bbox++;
	 continue;
       }
       //table[tpos].bbox_visible= (dx+1)*(dy+1)*(dz+1);
       
    }
    
    if ( !node ) {
      extract( v, i, j, k, 1,1,1);
      continue;
    }

    if ( !node->child ) {
      int start  = (eye.x() > offset_x+(i+2)*sx) ? 1 : 0; 
      if ( eye.y() > offset_y+(j+2)*sy ) start += 2;
      if ( eye.z() > offset_z+(k+2)*sz ) start += 4;
      
      int *order = permutation[start];

      if ( !extract_all && (right-left) < 2 && (top-bottom) < 2 ) {
	for (int o=7; o>=0; o--)
	  switch (order[o] ) {
	    case 0:
	      stack.push( 0, i,j,k, 1, 1, 1,0 );
	      break;
	    case 1:
	      stack.push( 0, i+1,j,k, 1, 1, 1, 0 );
	      break;
	    case 2:
	      stack.push( 0, i,j+1,k, 1, 1, 1, 0 );
	      break;
	    case 3:
	      stack.push( 0, i+1,j+1,k, 1, 1, 1, 0 );
	      break;
	    case 4:
	      stack.push( 0, i,j,k+1, 1, 1, 1, 0 );
	      break;
	    case 5:
	      stack.push( 0, i+1,j,k+1, 1, 1, 1, 0 );
	      break;
	    case 6:
	      stack.push( 0, i,j+1,k+1, 1, 1, 1, 0 );
	      break;
	    case 7:
	      stack.push( 0, i+1,j+1,k+1, 1, 1, 1, 0 );
	      break;
	  }
      }
      else {
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
    int start  = (eye.x() > offset_x+(i+dx+1)*sx) ? 1 : 0;
    if ( eye.y() > offset_y+(j+dy+1)*sy ) start += 2;
    if ( eye.z() > offset_z+(k+dz+1)*sz ) start += 4;
    
    int *order = permutation[start];
    
    int type = node->type;
    SageVFemNode *child = node->child;
    
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
  iotimer_t end = read_time();
  if ( visibility )  
    printf("Search time = %.3lf\n", (end - begin)*cycleval*1e-9);
  //shadow.stat();
}

//#include "mcube2.h"

int scan_type = 2;

int
SageVFem::extract( double iso, int i, int j, int k, int dx, int dy, int dz )
{
  //table[tpos].extract = 1;;
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
    //printf("Extract nothing:: [%d %d %d] \n", i,j,k );
    //t_extract.stop();
    extract_timer += read_time() - start;
    return 0;
  }

// #ifdef VIS_WOMAM
  //  double ps = pixel_size[offset+k];
  double ps = scale;

  double x0 = i * ps + offset_x;
  double y0 = j * ps + offset_y;
  double z0 = k + offset_z ;

  double x1 = x0 + dx*ps;
  double y1 = y0 + dy*ps;
  double z1 = z0+dz ;

// #else

//   double x0 = i*sx;
//   double x1 = (i+dx)*sx;
//   double y0 = j*sy;
//   double y1 = (j+dy)*sy;
//   double z0 = k*sz;
//   double z1 = (k+dz)*sz;
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

  for ( t=0; t<tcase->n; t++) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    int dir = 0;
    int e=2;
    
    scan_edges[0] = v0;
    scan_edges[1] = v1;
    double_edges[0] = double_edges[1] = 1;
    
    for (; v2 != -1; v1=v2,v2=vertex[v++]) {
      int l= (p[v1].y-p[v0].y)*(p[v2].x-p[v0].x)
	+ (p[v1].x-p[v0].x)*(p[v0].y-p[v2].y);
      int type = l <  0 ? 1 : -1;
      //      if ( l*dir < 0)
      //	double_edges[e-1] *=2;
      double_edges[e] = type;
      scan_edges[e] = v2;
      e++;
      dir = l;
      group->add(q[v0], q[v1], q[v2]);
    }
    scan_edges[e] = scan_edges[0];
    double_edges[e] = double_edges[0] = (double_edges[e-1] > 0) ? 1: -1;
    if ( double_edges[2] < 0 )
      double_edges[1] = -1;
    vis += screen.scan(p, e,  scan_edges, double_edges);
  }
  
  if ( vis ) {
    tgroup->add(tmp );
  }
  else
    delete tmp;

  statistics.extracted++;
  extract_timer += read_time() - start;
  return 1;
}

void
SageVFem::tcl_command(TCLArgs& args, void* userdata) {
  if (args[1] == "redraw") {
    reset_vars();
    //redraw();
  } else {
    Module::tcl_command(args, userdata);
  }
}

int
SageVFem::make_current( int xres, int yres) {
  make_start = read_time();
  TCLTask::lock();
  clString myname(clString(".ui")+id+".gl.gl");
  tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));
  if(!tkwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return 0;
  }
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  make_info = read_time();
  cx=OpenGLGetContext(the_interp, myname());
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return 0;
  }
  make_get = read_time();
  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
  make_make = read_time();

  // Clear the screen...
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,xres,0,yres,-1,1);
  glViewport(0, 0, xres, yres );
  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT  );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR)
    cerr << "1- GL Err: " << (char*)gluErrorString(errcode)
	 << endl;
  
  return 1;
}

void
SageVFem::redraw( int xres, int yres)
{
  int errcode;
  rebuild_start = read_time();
  int ok = make_current( xres, yres ) ;
  rebuild_make = read_time();

  if (!ok )
    return;
  
  glDrawBuffer(GL_FRONT);
  glColor3f(1,0,0);
  glDisable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glEnable(GL_LOGIC_OP);
  glEnable(GL_BLEND);
  glBlendEquationEXT( GL_LOGIC_OP);
}

void
SageVFem::redraw_done()
{
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
  rebuild_end = read_time();
  cerr <<"Redraw: done\n";
}

void
SageVFem::display( Point p[] )
{
  glColor3f(1,1,1);
  glBegin(GL_LINE_LOOP );
  glVertex2f( p[0].x(), p[0].y() );
  glVertex2f( p[1].x(), p[1].y() );
  glVertex2f( p[2].x(), p[2].y() );
  glEnd();
}


/*
void
SageVFem::bbox_projection( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  
  Vector p = Point((i+dx/2.)*sx,(j+dy/2.)*sy,(k+dz/2.)*sz)-eye;

  double lu = (dx*AU.x()+dy*AU.y()+dz*AU.z())/2;
  double lv = (dx*AV.x()+dy*AV.y()+dz*AV.z())/2;
  double lw = (dx*AW.x()+dy*AW.y()+dz*AW.z())/2;
  
  double pu = Dot(p,U);
  double pv = Dot(p,V);
  double pw = Dot(p,W);

  int su = (U.x()*pw > W.x()*pu)*2-1;
  int sv = (U.y()*pw > W.y()*pv)*2-1;
  int sw = (U.z()*pw > W.z()*)*2-1;
  double near = 1./(pw-lw);
  double far  = 1./(pw+lw);

  double q = pu-lu;
  left = (q* (q>0?far:near)+1)*xres/2;
  q = pu+lu;
  right =(q* (q<0?far:near)+1)*xres/2;
  q = pv-lv;
  bottom = (q* (q>0?far:near)+1)*yres/2;
  q = pv+lv;
  top = (q* (q<0?far:near)+1)*yres/2;

  glColor3f(0,1,0);
  glBegin(GL_LINE_LOOP);
  glVertex2i( left, bottom );
  glVertex2i( right, bottom );
  glVertex2i( right, top );
  glVertex2i( left, top );
  glEnd();

  printf("green : %.1f %.1f %.1f %.1f\n", left,right,bottom,top);
}
*/

double
SageVFem::bbox_projection( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  
  Vector p = Point(offset_x+(i+dx/2.)*sx,
		   offset_y+(j+dy/2.)*sy,
		   offset_z+(k+dz/2.)*sz) - eye;

  double lu = (dx*AU.x()+dy*AU.y()+dz*AU.z())/2;
  double lv = (dx*AV.x()+dy*AV.y()+dz*AV.z())/2;
  double lw = (dx*AW.x()+dy*AW.y()+dz*AW.z())/2;
  
  double pu = Dot(p,U);
  double pv = Dot(p,V);
  double pw = Dot(p,W);

  double near = 1./(pw-lw);
  double far  = 1./(pw+lw);

  double q = pu-lu;
  left = (q* (q>0?far:near)+1)*xres/2.-0.5;
  q = pu+lu;
  right =(q* (q<0?far:near)+1)*xres/2.-0.5;
  q = pv-lv;
  bottom = (q* (q>0?far:near)+1)*yres/2.-0.5;
  q = pv+lv;
  top = (q* (q<0?far:near)+1)*yres/2.-0.5;

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
SageVFem::bbox_projection2( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  
  Vector p = Point(offset_x+(i+dx/2.)*sx,
		   offset_y+(j+dy/2.)*sy,
		   offset_z+(k+dz/2.)*sz) - eye;
  
  double ds = dx*sx;

  double d = dy*sy;
  if ( d < ds ) ds = d;
  d = dz*sz;
  if ( d < ds ) ds = d;
  ds /= sqrt(2);

  Vector R = p+U*ds;
  
  double pu = Dot(p,U);
  double pv = Dot(p,V);
  double pw = Dot(p,W);

  double x = (pu/pw+1)*xres/2;
  double y = (pv/pw+1)*yres/2;
  
  double Ru = Dot(R,U);
  double Rw = Dot(R,W);

  right = (Ru/Rw+1)*xres/2;
  double len = right-x;
  left = x-len;
  top = y+len;
  bottom = y-len;

  glColor3f(1,0,0);
  glBegin(GL_LINE_LOOP);
  glVertex2i( left, bottom );
  glVertex2i( right, bottom );
  glVertex2i( right, top );
  glVertex2i( left, top );
  glEnd();
  printf("red   : %.1f %.1f %.1f %.1f\n", left,right,bottom,top);
  return pw;
}
  

double
SageVFem::bbox_projection1( int i, int j, int k, int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{

  Vector p1 = Point(offset_x+(i+dx/2.)*sx,
		   offset_y+(j+dy/2.)*sy,
		   offset_z+(k+dz/2.)*sz) - eye;

  Vector offset = Vector(offset_x, offset_y,offset_z);
  
  double pw = Dot(p1,W);


  Pt p;
  Pt q[8];
  
  project( Point(i*sx, j*sy, k*sz)+offset, p );
  left = right = p.x;
  top = bottom = p.y;
  q[0].x = p.x; q[0].y = p.y;
  
  project( Point(i*sx, j*sy, (k+dz)*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[1].x = p.x; q[1].y = p.y;

  project( Point(i*sx, (j+dy)*sy, k*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[2].x = p.x; q[2].y = p.y;

  project( Point(i*sx, (j+dy)*sy, (k+dz)*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[3].x = p.x; q[3].y = p.y;

  project( Point((i+dx)*sx, j*sy, k*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[4].x = p.x; q[4].y = p.y;

  project( Point((i+dx)*sx, j*sy, (k+dz)*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[5].x = p.x; q[5].y = p.y;

  project( Point((i+dx)*sx, (j+dy)*sy, k*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[6].x = p.x; q[6].y = p.y;

  project( Point((i+dx)*sx, (j+dy)*sy, (k+dz)*sz)+offset, p );
  if ( p.x < left ) left = p.x;
  else if ( p.x > right ) right = p.x;
  if ( p.y < bottom ) bottom = p.y;
  else if ( p.y > top ) top = p.y;
  q[7].x = p.x; q[7].y = p.y;


//   glColor3f(1,1,1);
//   glBegin(GL_LINE_LOOP);
//   glVertex2f(q[0].x,q[0].y);
//   glVertex2f(q[1].x,q[1].y);
//   glVertex2f(q[2].x,q[2].y);
//   glVertex2f(q[3].x,q[3].y);
//   glEnd();

//   glBegin(GL_LINE_LOOP);
//   glVertex2f(q[4].x,q[4].y);
//   glVertex2f(q[5].x,q[5].y);
//   glVertex2f(q[6].x,q[6].y);
//   glVertex2f(q[7].x,q[7].y);
//   glEnd();
  
//   glBegin(GL_LINE_LOOP);
//   glVertex2f(q[0].x,q[0].y);
//   glVertex2f(q[1].x,q[1].y);
//   glVertex2f(q[5].x,q[5].y);
//   glVertex2f(q[4].x,q[4].y);
//   glEnd();

//   glBegin(GL_LINE_LOOP);
//   glVertex2f(q[2].x,q[2].y);
//   glVertex2f(q[3].x,q[3].y);
//   glVertex2f(q[1].x,q[1].y);
//   glVertex2f(q[0].x,q[0].y);
//   glEnd();
  
//   glColor3f(1,1,0);
//   glBegin(GL_LINE_LOOP);
//   glVertex2i( left, bottom );
//   glVertex2i( right, bottom );
//   glVertex2i( right, top );
//   glVertex2i( left, top );
//   glEnd();

//   printf("yellow: %.1f %.1f %.1f %.1f\n", left,right,bottom,top);
  return pw;
}


#ifdef ONE


double pu, pv, pw;
double max_x, min_x, max_y, min_y;

void
fun( double u, double v, double w )
{
  pu += u;
  pv += v;
  pw += w;
  double z = 1./pw;
  double x = pu*z;
  double y = pv*z;
  if (x < min_x) min_x = x;
  else if ( x > max_x ) max_x = x;
  if (y < min_y) min_y = y;
  else if ( y > max_y ) max_y = y;
}

void
SageVFem::bbox_projection( int i, int j, int k int dx, int dy, int dz,
	      double &left, double &right, double &top, double &bottom )
{
  Vector p = Point(i*sx,j*sy,k*sz)-eye;
  pu = Dot(p,U);
  pv = Dot(p,V);
  pw = Dot(p,W);

  double d[3],du[3], dv[3], dw[3];
  d[0] = dx*sx;
  d[1] = dy*sy;
  d[2] = dz*sz;
  du[0] = d[0]*U.x();
  du[1] = d[1]*U.y();
  du[2] = d[2]*U.z();
  dv[0] = d[0]*V.x();
  dv[1] = d[1]*V.y();
  dv[2] = d[2]*V.z();
  dw[0] = d[0]*W.x();
  dw[1] = d[1]*W.y();
  dw[2] = d[2]*W.z();


  double z = 1./pw;
  min_x = max_x = pu*z;
  min_y = max_y = pv*z;
  
  fun(  du[0],  dv[0],  dw[0]);
  fun(  du[1],  dv[1],  dw[1]);
  fun( -du[0], -dv[0], -dw[0]);
  fun(  du[2],  dv[2],  dw[2]);
  fun(  du[0],  dv[0],  dw[0]);
  fun( -du[1], -dv[1], -dw[1]);
  fun( -du[0], -dv[0], -dw[0]);

  left = min_x; right = max_x;
  bottom = min_y; top = max_y;
}

#endif


#ifdef FULL  
  for (;;) {
    qw = pw+dw[i];
    if ((pw*du[i]-pu*dw[i])*pw*qw > 0 ) {
      // move in the di direction
      pos[i] = pos[i] ? 0 : 1;
      pu += du[i];
      pw = qw;
      du[i] = -du[i];
      dw[i] = -dw[i];
      if ( first == 1 ) first = i;
    }
    else {
      i = (i+1)%3;
      qw = pw+dw[i];
      if ((pw*du[i]-pu*dw[i])*pw*qw > 0 ) {
	// move in the di direction
	pos[i] = pos[i] ? 0 : 1;
	pu += du[i];
	pw = qw;
	du[i] = -du[i];
	dw[i] = -dw[i];
	if ( first == -1 ) first = i;
      }
      else {
	if ( first == -1 ) {
	  i = 2;
	  qw = pw+dw[i];
	  if ((pw*du[i]-pu*dw[i])*pw*qw > 0 ) {
	    // move in the di direction
	    pos[i] = pos[i] ? 0 : 1;
	    pu += du[i];
	    pw = qw;
	    du[i] = -du[i];
	    dw[i] = -dw[i];
	    first = i;
	  }
	}
	else
	  break;
      }
    }
  }
    
#endif

void
SageVFem::dividing_cubes( double v, int x, int y, int z, int dx, int dy, int dz,
		      int px, int py )
{
  if ( screen.cover_pixel(px,py) ) {
   printf( "Dividing Cubes [%d %d %d] [%dx%dx%d] scr[%d %d]\n",
 	  x,y,z,dx,dy,dz, px,py);
    //    printf("\t show\n");
    //points->add(Point((x+dx/2)*sz, (y+dy/2)*sy, (z+dz/2)*sz));
    //points->add(Point((x)*sz, (y)*sy, (z)*sz));
   // points->add(Point(x*sx, y*sy, z*sz), Point(px, 511-py, 0));
    printf("\t\t %d %d\n", px,511-py);
  }
}















