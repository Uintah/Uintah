/*
 *  Noise.cc:  The NOISE algorithm
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <stdio.h>
#include <time.h>
#include <Core/Persistent/Pstreams.h>          
#include <Core/Geometry/BBox.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <iostream>

#include <Packages/Yarden/Core/Datatypes/SpanTree.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/Noise.h>
//#include <Datatypes/Clock.h>


namespace Yarden {

using namespace SCIRun;
using std::cerr;

extern "C" Module* make_Noise(const clString& id)
{
    return scinew Noise(id);
}

const double epsilon = 1.e-8;

//static void u_interp( ScalarField *, GeomTrianglesPT1d *, int, double );
static void r_interp( ScalarField *, GeomTrianglesP *, int, double );

void mcube_init( int xdim, int ydim, Value *grid, int);
void mcube( GeomTrianglesP *, double, int );


static clString module_name("Noise");
static clString surface_name("NoiseSurface");
static clString transparent_name("NoiseSurfaceMultiTransParent");

static int my_number  = 0;

Noise::Noise(const clString& id) :
  Module("Noise", id, Filter),
  isoval("isoval", id, this ),
  isoval_min("isoval_min", id, this ),
  isoval_max("isoval_max", id, this ),
  tcl_bbox("bbox", id, this),
  tcl_alpha("alpha", id, this),
  tcl_trans("trans",id,this),
  tcl_np("np",id,this),
  tcl_map_type("map",id,this),
  lock("Noise search lock")
{
    // Create input ports
    inforest=scinew SpanForestIPort(this,
				  "Span Forest", SpanForestIPort::Atomic);
    add_iport(inforest);

    incolorfield=scinew ScalarFieldIPort(this, "Color Field", ScalarFieldIPort::Atomic);
    add_iport(incolorfield);

    incolormap=scinew ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
    add_iport(incolormap);


    // Create output port
    ogeom = scinew GeometryOPort( this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    //isoval.set(0);
    matl=scinew Material(Color(0,0,0), Color(0,.8,0), Color(.7,.7,.7), 20);

    old_min = old_max = 0;
    init = 0;
    surface_id = 0;
    forest_generation = -1;
    triangles = 0;
    //    init_clock();

    int n = my_number++;
    my_name = surface_name + to_string(n);
    cerr << "Noise: my name is " << my_name << "  " << n << " " << my_number << "\n";
}



Noise::~Noise()
{
}


void
Noise::execute()
{
  if ( !inforest->get(forest) ) {
    return;
  }
  map_type = tcl_map_type.get();

  if ( forest->generation != forest_generation ) {
    double min, max;
    forest->field->get_minmax(min, max);
    printf("Noise: min %lf  max %lf\n", min,max);
    isoval_min.set(min);
    isoval_max.set(max);
    //isoval.set((min+max)/2);
    reset_vars();    
    int nt = forest->tree.size();
    if ( triangles )
      delete triangles;
    triangles = scinew GeomTrianglesP*[nt];
    printf("Noise: %d trees\n", nt );
    forest_generation = forest->generation;
  }
  else {
    extract();
  }
}

void
Noise::extract()
{
  v = isoval.get();
  v += epsilon;

  alpha = tcl_alpha.get();
  trans = tcl_trans.get();

  GeomGroup *group = scinew GeomGroup;
  surface=group;

  ScalarFieldHandle colorfield;
  int have_colorfield=incolorfield->get(colorfield);
  ColorMapHandle cmap;
  int have_colormap=incolormap->get(cmap);
  printf(" %s colormap\n", have_colormap ? "have" : "does not have" );
  
//   if(have_colormap && !have_colorfield){
//     // Paint entire surface based on colormap
//     surface = scinew GeomMaterial( group, cmap->lookup(v));
//   } else if(have_colormap && have_colorfield){
//     // Nothing - done per vertex
//   } else {
//     // Default material
//     surface = scinew GeomMaterial( group, matl );
//   }

  field = forest->field.get_rep();
#ifdef FIELD_FLOAT
  ScalarFieldRGfloat *r_grid = forest->field->getRGBase()->getRGFloat();
#else
  ScalarFieldRGdouble *r_grid = forest->field->getRGBase()->getRGDouble();
#endif
  ScalarFieldUG *u_grid = forest->field->getUG();

  if ( u_grid ) {
    //    interp = u_interp;
  }
  else {
    interp = r_interp;
    mcube_init( r_grid->grid.dim3(), r_grid->grid.dim2(),
		&r_grid->grid(0,0,0), map_type );
  }
  

  //np = forest->tree.size();
  np = tcl_np.get();

  //  iotimer_t start = read_time();
  trees = n_trees = forest->tree.size();
  Thread::parallel(Parallel<Noise>(this, &Noise::do_search), np, true);
  //  Task::multiprocess(np, do_search_parallel, this);
  //  iotimer_t end = read_time();
  //  printf("Noise: elapase time: %.3lf sec\n",(end-start)*cycleval*1e-12);
  if ( surface_id ) {
    ogeom->delObj(surface_id);
  }

  TextPiostream stream("tmp.tri", Piostream::Write);
  //Pio(stream, sh);

  for (int i=0; i<n_trees; i++ ) {
    printf("Noise: set[%d] size = %d\n", i, triangles[i]->size());
    if ( triangles[i]->size() > 0 ) {
  //     triangles[i]->cmap = cmap->raw1d;
//       printf("cmap 0x%x\n", cmap->raw1d);
      group->add(triangles[i]);
      triangles[i]->io(stream);
    }
    else
      delete triangles[i];
  }
  if ( group->size() == 0 ) {
    delete group;
    surface_id = 0;
  }
  else {
    cerr << "BBox = "<< forest->bbox.min()<<" "<< forest->bbox.max() << "\n";
    surface_id = ogeom->addObj
      (  tcl_bbox.get() ? new GeomBBoxCache( surface, forest->bbox ) : surface,
	 trans ? transparent_name : my_name );
  }
  ogeom->flushViews();
}



void
Noise::do_search( int proc )
{
  while (1) {
    lock.lock();
    int tree = --trees;
    lock.unlock();
    if ( tree < 0 )
      return;
    if ( trans )
      triangles[tree] = 0; //scinew GeomTranspTrianglesPT( alpha );
    else
      triangles[tree] = scinew GeomTrianglesP;
    printf("Noise %d: size = %d\n",proc, forest->tree[proc].span.size() );
    int n = count( forest->tree[tree] );
    printf("count [%d] [size=%d] = %d\n",tree, forest->tree[tree].span.size(),n);
    triangles[tree]->reserve_clear( 3*n );
    
    search( forest->tree[tree], triangles[tree] );
    //printf("Noise %d: cells %d found %d\n",proc, n, triangles[tree]->size() );
  }
}
  
int
Noise::count( SpanTree &tree )
{
  int counter = 0;
  _count_min_max( &tree.span[0], tree.span.size(), counter );

  return counter;
}


void
Noise::search( SpanTree &tree, GeomTrianglesP *triangles)
{
  int n = count( tree );
  int m =  field->getUG() ? n*2 : n *4 ;

  _search_min_max( &tree.span[0], tree.span.size(), triangles );

}

void
Noise::_search_min_max( SpanPoint p[], int n, GeomTrianglesP *triangles )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p+n/2;

  if ( point->min < v ) {
    if ( point->max > v )
      (*interp)( field, triangles, point->index, v );

    _search_max_min( point+1, (n-1)/2, triangles );
    _search_max( p, n/2, triangles );
  }
  else
    _search_max_min( p, n/2, triangles );
}

void
Noise::_search_max_min( SpanPoint p[], int n, GeomTrianglesP *triangles )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;
    
  if ( point->max > v ) {
    if ( point->min < v )
      (*interp)( field, triangles, point->index, v );
	
    _search_min_max( p, n/2, triangles );
    _search_min( point+1, (n-1)/2, triangles );
  }
  else
    _search_min_max( point+1, (n-1)/2, triangles );

}
    
void
Noise::_search_min( SpanPoint p[], int n, GeomTrianglesP *triangles )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;

  if ( point->min < v ) {
    (*interp)( field, triangles, point->index, v );

    // Right Son.
    if ( n > 2) {	
      SpanPoint *child = point+1+(n-1)/4;

      if ( child->min < v )
	(*interp)( field, triangles, child->index, v );

      _search_min( child+1, ((n-1)/2 -1)/2, triangles);
      _search_min( point+1, (n-1)/4, triangles );
    }

    // Left son: collect all.
    _collect( p, n/2, triangles );
  }
  else {
    SpanPoint *child = p+n/4;
    if ( child->min < v )
      (*interp)( field, triangles, child->index, v );

    _search_min( p+n/4+1, (n/2-1)/2, triangles );
    _search_min( p, n/4, triangles );
  }

}

void
Noise::_search_max( SpanPoint p[], int n, GeomTrianglesP *triangles )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;
    
  if ( point->max > v ) {
    (*interp)( field, triangles, point->index, v );

    if ( n > 1 ) {
      SpanPoint *child = p+n/4;

      if ( child->max > v )
	(*interp)( field, triangles, child->index, v );

      _search_max( p, n/4, triangles );
      _search_max( child+1, (n-2)/4, triangles );
    }
    _collect( point+1, (n-1)/2, triangles );
  }
  else
    if ( n > 1 ) {
      SpanPoint *child = point+1+(n-1)/4;

      if ( child->max > v )
	(*interp)( field, triangles, child->index, v );

      _search_max( point+1, (n-1)/4, triangles );
      _search_max( child+1, (n-3)/4 , triangles);
    }

}
    

void
Noise::_collect( SpanPoint p[], int n, GeomTrianglesP *triangles )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p;
  for ( int i=0; i<n; i++, point++ )
    (*interp)( field, triangles, point->index, v );
}

/*
 * Count
 */

void
Noise::_count_min_max( SpanPoint p[], int n, int &counter )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p+n/2;

  if ( point->min <= v ) {
    if ( point->max > v )
      counter++;

    _count_max_min( point+1, (n-1)/2, counter );
    _count_max( p, n/2, counter );
  }
  else
    _count_max_min( p, n/2, counter );
}

void
Noise::_count_max_min( SpanPoint p[], int n, int &counter )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;
    
  if ( point->max >= v ) {
    if ( point->min < v )
      counter++;
	
    _count_min_max( p, n/2, counter );
    _count_min( point+1, (n-1)/2, counter );
  }
  else
    _count_min_max( point+1, (n-1)/2, counter );
}
    
void
Noise::_count_min( SpanPoint p[], int n, int &counter )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;

  if ( point->min <= v ) {
    counter++;

    // Right Son.
    if ( n > 2) {	
      SpanPoint *child = point+1+(n-1)/4;

      if ( child->min < v )
	counter++;

      _count_min( child+1, ((n-1)/2 -1)/2, counter);
      _count_min( point+1, (n-1)/4, counter );
    }

    // Left son: collect all.
    _count_collect( n/2, counter );
  }
  else {
    SpanPoint *child = p+n/4;
    if ( child->min < v )
      counter++;

    _count_min( p+n/4+1, (n/2-1)/2, counter );
    _count_min( p, n/4, counter );
  }
}

void
Noise::_count_max( SpanPoint p[], int n, int &counter )
{
  if ( n <= 0 )
    return;

  SpanPoint *point = p + n/2;
    
  if ( point->max >= v ) {
    counter++;

    if ( n > 1 ) {
      SpanPoint *child = p+n/4;

      if ( child->max > v )
	counter++;

      _count_max( p, n/4, counter );
      _count_max( child+1, (n-2)/4, counter );
    }
    _count_collect( (n-1)/2, counter );
  }
  else
    if ( n > 1 ) {
      SpanPoint *child = point+1+(n-1)/4;

      if ( child->max > v )
	counter++;

      _count_max( point+1, (n-1)/4, counter );
      _count_max( child+1, (n-3)/4, counter );
    }
}
    

void
Noise::_count_collect( int n, int &counter )
{
  if ( n <= 0 )
    return;

  counter += n;
}


static void
u_interp( ScalarField *field, GeomTrianglesP *triangles, int index, double v )
{
  ScalarFieldUG *u_field = (ScalarFieldUG *) field;

  int *n = u_field->mesh->elems[index]->n;

  
  Array1<NodeHandle> &nodes = u_field->mesh->nodes;
  Point &n0 = nodes[n[0]]->p;
  Point &n1 = nodes[n[1]]->p;
  Point &n2 = nodes[n[2]]->p;
  Point &n3 = nodes[n[3]]->p;

  double v0 = u_field->data[n[0]];
  double v1 = u_field->data[n[1]];
  double v2 = u_field->data[n[2]];
  double v3 = u_field->data[n[3]];

  
  if ( v < v1 ) {
    /* one triangle */
    Point p1(Interpolate( n0, n3, (v-v0)/(v3-v0)));
    Point p2(Interpolate( n0, n2, (v-v0)/(v2-v0)));
    Point p3(Interpolate( n0, n1, (v-v0)/(v1-v0)));

    triangles->add( p1, p2, p3 );
  }
  else if ( v < v2 ) {
    /* two triangle */
    Point p1(Interpolate( n0, n3, (v-v0)/(v3-v0)));
    Point p2(Interpolate( n0, n2, (v-v0)/(v2-v0)));
    Point p3(Interpolate( n1, n3, (v-v1)/(v3-v1)));

    triangles->add( p1, p2, p3 );
    
    Point p4(Interpolate( n1, n2, (v-v1)/(v2-v1)));
    triangles->add( p2, p3, p4 );

  }
  else {
    /* one triangle */
    
    Point p1 = Interpolate( n0, n3, (v-v0)/(v3-v0));
    Point p2 = Interpolate( n1, n3, (v-v1)/(v3-v1));
    Point p3 = Interpolate( n2, n3, (v-v2)/(v3-v2));

    triangles->add( p1, p2, p3 );
  }
}
 

static void
r_interp( ScalarField *, GeomTrianglesP *triangles, int index, double v )
{
  printf("interp\n");
  mcube( triangles, v, index );
}

} // End namespace Yarden
