/*
 *  span.cc:  Preprocess for the NOISE algorithm
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

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SpanPort.h>
#include <PSECore/Datatypes/SpanTree.h> 
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Thread.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <PSECommon/Modules/Visualization/Span.h>
//#include <Datatypes/Clock.h>



namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::Containers;

using namespace SCICore::Thread;

Module* make_Span(const clString& id)
 {
    return scinew Span(id);
}

static clString module_name("Span");

Span::Span(const clString& id) :
  Module("Span", id, Filter), 
  tcl_split("split", id, this ), tcl_maxsplit("max_split", id, this ),
  tcl_z_max("z_max", id, this ),
  tcl_z_from("z_from", id, this),  tcl_z_size("z_size", id, this )
{
    // Create the input port
    infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);

    //Create the output port
    ospan=scinew SpanForestOPort(this, "SpanForest", SpanForestIPort::Atomic);
    add_oport(ospan);
    
    field_generation = -1;
    //tcl_maxsplit.set(10);
    forest_id = 0;
    forest_generation = 0;
    z_from = 0;
    z_size = 64;;
}


Span::~Span()
{
}


void
Span::tcl_command(TCLArgs& args, void* userdata) {
  if (args[1] == "split") {
    reset_vars();
  } else {
    Module::tcl_command(args, userdata);
  }
}

// static void do_create_span_parallel(void* obj, int proc)
// {
//   Span* module=(Span*)obj;
//   module->create_span_parallel(proc);
// }

void
Span::execute()
{
  printf("Span: execute\n");
  // input
  nt = 0;
  if(!infield->get(field)) {
    printf("Span: no field\n");
    return;
  }
  execute_ext();
}

void
Span::execute_ext()
{
  printf("Span: execute_ext\n");
#ifdef FIELD_FLOAT
  ScalarFieldRGfloat * r_grid = field->getRGBase()->getRGFloat();
#else
  ScalarFieldRGdouble *r_grid = field->getRGBase()->getRGDouble();
#endif

  if ( field->generation !=  field_generation ) {
    printf("Span: new generation %d\n", field->generation );
    double min, max;
    field->get_minmax( min, max );
    printf("field minmax = %lf %lf\n", min, max);
    Array3<Value> &grid = r_grid->grid;

    field_generation =  field->generation;
    int z = r_grid->grid.dim3();
    tcl_z_max.set( z );
    if ( z_size > z ) 
      tcl_z_size.set(z);
    if ( z_from > z ) {
      z_from = 0;
      tcl_z_from.set( 0 );
    }
  }
  else {
    nt = tcl_split.get();
    z_from = tcl_z_from.get();
    z_size = tcl_z_size.get();
    forest = scinew SpanForest;
    forest->field = field;
    forest->generation = forest_generation++;
    forest->bbox.reset();
    forest->bbox.extend( Point(0, 0, z_from) );	
    forest->bbox.extend( Point(r_grid->grid.dim1(), r_grid->grid.dim2(),
			       z_from+z_size));
    forest->tree.setsize( nt );

    create_span();
  }
}


void
Span::create_span()
{
  Thread::parallel(Parallel<Span>(this, &Span::do_create), nt, true);
  // Task::multiprocess(nt, do_create_span_parallel, this);
  printf( "Span: %d tree(s) created.\n", forest->tree.size());
  ospan->send( SpanForestHandle( forest ) );
  printf("Span: done\n");
}


void
Span::do_create( int proc )
{
#ifdef FIELD_FLOAT
  ScalarFieldRGfloat * r_grid = field->getRGBase()->getRGFloat();
#else
  ScalarFieldRGdouble * r_grid = field->getRGBase()->getRGDouble();
#endif

  //  printf("span %d\n", proc);
  // iotimer_t start = read_time();
  Array3<Value> &grid = r_grid->grid;

  int dz = z_size/nt;
  int from = z_from+proc*dz;
  int to = from + dz;
  int pos = from*grid.dim1()*grid.dim2();

  if ( proc == nt-1 )
    to = z_from+z_size-1;

  forest->tree[proc].span.setsize( grid.dim1()*grid.dim2()*(to-from) );
  forest->tree[proc].span.remove_all();
  
  printf("Span: %d %d %d\n",grid.dim1(), grid.dim2(), to-from+1);
  int n = 0;
  for ( int k=from; k<to; k++, pos += grid.dim1() )
    for ( int j=0; j<grid.dim2()-1; j++, pos++ )
      for ( int i=0; i<grid.dim1()-1; i++, pos++  ) {

	Value min, max, v;
	min = max = grid(i,j,k);
	
	v = grid(i+1,j,k);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i,j+1,k);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i+1,j+1,k);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i,j,k+1);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i+1,j,k+1);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i,j+1,k+1);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	v = grid(i+1,j+1,k+1);
	if ( v < min ) min = v;
	else if ( v > max ) max = v;
	
	if ( min < max ) {
	  forest->tree[proc].span.add(SpanPoint( min, max, pos ));
	  if ( pos > 1000 && pos < 1010 )
	    printf("CELL %d at [%d,%d,%d] = [%lf %lf]\n", 
		   pos, i,j,k,min,max);
	  n++;
	}
      }

  printf("n=%d\n",n);
//   iotimer_t end = read_time();
//   fprintf( stderr, "Span %d create: %d  in %.3f sec\n",
// 	   proc, forest->tree[proc].span.size(), (end-start)*cycleval*1e-12);
//   start = read_time();
  select_min( &forest->tree[proc].span[0], forest->tree[proc].span.size() );
  //  end = read_time();
//   fprintf( stderr, "Span %d done  in %.3f sec\n",
//      proc, (end-start)*cycleval*1e-12);
}



    

inline void
swap( SpanPoint &a, SpanPoint &b )
{
  SpanPoint t = a;
  a = b;
  b = t;
}


void
Span::select_min( SpanPoint p[], int n )
{
  if ( n < 2 )
    return;
  
  int k = n/2;
  int l = 0;
  int r = n-1;
  while ( r > l ) {
    int mid = (l+r)/2;
    if ( p[l].min > p[mid].min ) swap( p[l], p[mid] );
    if ( p[l].min > p[r].min ) swap( p[l], p[r] );
    if ( p[mid].min > p[r].min ) swap( p[mid], p[r] );
    
    Value v = p[r].min;
    
    int i,j;
    for( i=l-1, j=r; ; ) {
      while ( p[++i].min < v );
      while ( p[--j].min > v );
      if ( i >= j ) 
	break;

      swap( p[i], p[j] );
    }
    swap( p[i], p[r]) ;

    if ( i >= k ) r = i-1;
    if ( i <= k ) l = i+1;
  }

  select_max( p, n/2 );
  select_max( p+n/2+1, (n-1)/2 );
}
      
    
void
Span::select_max( SpanPoint p[], int n )
{
  if ( n < 2 )
    return;

  int k = n/2;
  int l = 0;
  int r = n-1;
  while ( r > l ) {
    int mid = (l+r)/2;
    if ( p[l].max > p[mid].max ) swap( p[l], p[mid] );
    if ( p[l].max > p[r].max ) swap( p[l], p[r] );
    if ( p[mid].max > p[r].max ) swap( p[mid], p[r] );
    
    Value v = p[r].max;

    int i,j;
    for( i=l-1, j=r; ; ) {
      while ( p[++i].max < v );
      while ( p[--j].max > v );
      if ( i >= j ) 
	break;

      swap( p[i], p[j] );
    }
    swap( p[i], p[r]) ;

    if ( i >= k ) r = i-1;
    if ( i <= k ) l = i+1;
  }

  select_min( p, n/2 );
  select_min( p+n/2+1, (n-1)/2 );
}

} // End namespace Modules
} // End namespace PSECommon

