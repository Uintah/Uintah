// File: mcube
// Athour:  Yarden Livnat

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Containers/Array3.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/table.h>
#include <Packages/Yarden/Core/Datatypes/SpanTree.h>
using std::cerr;

namespace Yarden {

using namespace SCIRun;
  using std::cerr;

  //using namespace Dataflow::Dataflow;
  //using namespace Core::Datatypes;
  //using namespace Core::GuiInterface;
  //using namespace Core::Math;

static int i001, i010, i011, i100, i101, i110, i111;
static int x_dim, y_dim;
static Value *data;
static int map_type;
//static GeomTrianglesP *triangles;

const int hist_size = 100;
int hist[hist_size];

#define nl(a) transformation[a]
#define Depth(x,y) pt.z()
  
void
print_hist()
{
  int min,max,sum;

  min = max = hist[0];
  sum = hist[0];
  for (int i=1; i<hist_size; i++ ) {
    if ( hist[i] < min ) min = hist[i];
    else if ( hist[i] > max ) max = hist[i];
    sum += hist[i];
  }

  for (int  i=0; i<hist_size; i++) {
    printf(" %2d: [%7d]  %.2f%%\n", i, hist[i], hist[i]*100.0/sum);
  }
}
    
void
mcube_init( int xdim, int ydim, Value *grid, int map)
{
  x_dim = xdim;
  y_dim = ydim;

  data = grid;

  i001 = 1;
  i010 = xdim;
  i011 = i010 + i001;
  i100 = xdim*ydim;
  i101 = i100 + i001;
  i110 = i100 + i010;
  i111 = i110 + i001;

  map_type = map;
  printf("size of Value: %d\n", sizeof(Value) );
}

  

#define INTERP
#ifdef INTERP


/*inline*/ void
mcube_interp( int cell, int x, int y, int z, int *transformation,
	      double iso, int from, int to, Point &pt)
{
  int p = transformation[from];
  int q = transformation[to];
  
  switch ( p << 4 | q ) {
  case 0x10:
  case 0x01:
    pt.x(x+(iso - data[cell]) / ( data[cell+1] - data[cell]));
    pt.y(y);
    pt.z(z);
    break;
  case 0x32:
  case 0x23:
    pt.x( x + (iso - data[cell+i010]) / (data[cell+i011] - data[cell+i010]));
    pt.y(y+1);
    pt.z(z);
    break;
  case 0x54:
  case 0x45:
    pt.x(x + (iso - data[cell+i100]) /( data[cell+i101] - data[cell+i100]));
    pt.y(y);
    pt.z(z+1);
    break;
  case 0x76:
  case 0x67:
    pt.x(x + (iso - data[cell+i110]) /( data[cell+i111] - data[cell+i110]));
    pt.y(y+1);
    pt.z(z+1);
    break;

  case 0x20:
  case 0x02:
    pt.x(x);
    pt.y(y + (iso - data[cell]) /( data[cell+i010] - data[cell]));
    pt.z(z);
    break;
  case 0x31:
  case 0x13:
    pt.x(x+1);
    pt.y(y + (iso - data[cell+i001]) /( data[cell+i011] - data[cell+i001]));
    pt.z(z);
    break;
  case 0x64:
  case 0x46:
    pt.x(x);
    pt.y(y+(iso - data[cell+i100]) /( data[cell+i110] - data[cell+i100]));
    pt.z(z+1);
    break;
  case 0x75:
  case 0x57:
    pt.x(x+1);
    pt.y(y+(iso - data[cell+i101]) /( data[cell+i111] - data[cell+i101]));
    pt.z(z+1);
    break;

  case 0x40:
  case 0x04:
    pt.x(x);
    pt.y(y);
    pt.z(z + (iso - data[cell]) /( data[cell+i100] - data[cell]));
    break;
  case 0x51:
  case 0x15:
    pt.x(x+1);
    pt.y(y);
    pt.z(z + (iso - data[cell+i001]) / ( data[cell+i101] - data[cell+i001]));
    break;
  case 0x62:
  case 0x26:
    pt.x(x);
    pt.y(y+1);
    pt.z(z+(iso - data[cell+i010]) /( data[cell+i110] - data[cell+i010]));
    break;
  case 0x73:
  case 0x37:
    pt.x(x+1);
    pt.y(y+1);
    pt.z(z + (iso - data[cell+i011]) /( data[cell+i111] - data[cell+i011]));
    break;
  default:
    printf("mcube_interp: "
	   "Error. unexpected from-to combination: [%d %d] -> [%d %d]\n",
	   from, to, p, q );
    //exit(1);
  }
}

#else

void
mcube_interp( int cell, double iso, int from, int to, int v )
{
  int p = transformation[from];
  int q = transformation[to];

  register int id;
  int new_id;
  
  switch ( p << 4 | q ) {
  case 0x10:
  case 0x01:
    id = hash.search( cell<<2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x+(iso - data[cell]) /
	(float( data[cell+1]) - float(data[cell]));
      vertex[id].pos[1] = y;
      vertex[id].pos[2] = z;
      vert++;
      vertices++;
    }
    break;
  case 0x32:
  case 0x23:
    id = hash.search( (cell+i010)<<2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x + (iso - data[cell+i010]) /
	(float( data[cell+i011]) - float(data[cell+i010]));
      vertex[id].pos[1] = y+1;
      vertex[id].pos[2] = z;
      vert++;
      vertices++;
    }
    break;
  case 0x54:
  case 0x45:
    id = hash.search( (cell+i100)<<2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x + (iso - data[cell+i100]) /
	(float( data[cell+i101]) - float(data[cell+i100]));
      vertex[id].pos[1] = y;
      vertex[id].pos[2] = z+1;
      vert++;
      vertices++;
    }
    break;
  case 0x76:
  case 0x67:
    id = hash.search( (cell+i110)<<2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x + (iso - data[cell+i110]) /
	(float( data[cell+i111]) - float(data[cell+i110]));
      vertex[id].pos[1] = y+1;
      vertex[id].pos[2] = z+1;
      vert++;
      vertices++;
    }
    break;

  case 0x20:
  case 0x02:
    id = hash.search( (cell<<2)+1, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x;
      vertex[id].pos[1] = y + (iso - data[cell]) /
	(float( data[cell+i010]) - float(data[cell]));
      vertex[id].pos[2] = z;
      vert++;
      vertices++;
    }
    break;
  case 0x31:
  case 0x13:
    id = hash.search( ((cell+1)<<2)+1, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x+1;
      vertex[id].pos[1] = y + (iso - data[cell+i001]) /
	(float( data[cell+i011]) - float(data[cell+i001]));
      vertex[id].pos[2] = z;
      vert++;
      vertices++;
    }
    break;
  case 0x64:
  case 0x46:
    id = hash.search( ((cell+i100)<<2)+1, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x;
      vertex[id].pos[1] = y+(iso - data[cell+i100]) /
	(float( data[cell+i110]) - float(data[cell+i100]));
      vertex[id].pos[2] = z+1;
      vert++;
      vertices++;
    }
    break;
  case 0x75:
  case 0x57:
    id = hash.search( ((cell+i101)<<2)+1, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x+1;
      vertex[id].pos[1] = y+(iso - data[cell+i101]) /
	(float( data[cell+i111]) - float(data[cell+i101]));
      vertex[id].pos[2] = z+1;
      vert++;
      vertices++;
    }
    break;

  case 0x40:
  case 0x04:
    id = hash.search( (cell<<2)+2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x;
      vertex[id].pos[1] = y;
      vertex[id].pos[2] = z + (iso - data[cell]) /
	(float( data[cell+i100]) - float(data[cell]));
      vert++;
      vertices++;
    }
    break;
  case 0x51:
  case 0x15:
    id = hash.search( ((cell+1)<<2)+2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x+1;
      vertex[id].pos[1] = y;
      vertex[id].pos[2] = z + (iso - data[cell+i001]) /
	(float( data[cell+i101]) - float(data[cell+i001]));
      vert++;
      vertices++;
    }
    break;
  case 0x62:
  case 0x26:    id = hash.search( ((cell+i010)<<2)+2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x;
      vertex[id].pos[1] = y+1;
      vertex[id].pos[2] = z+(iso - data[cell+i010]) /
	(float( data[cell+i110]) - float(data[cell+i010]));
      vert++;
      vertices++;
    }
    break;
  case 0x73:
  case 0x37:
    id = hash.search( ((cell+i011)<<2)+2, new_id );
    if ( new_id ) {
      vertex[id].pos[0] = x+1;
      vertex[id].pos[1] = y+1;
      vertex[id].pos[2] = z + (iso - data[cell+i011]) /
	(float( data[cell+i111]) - float(data[cell+i011]));
      vert++;
      vertices++;
    }
    break;
  default:
    printf("mcube_interp: "
	   "Error. unexpected from-to combination: [%d %d] -> [%d %d]\n",
	   from, to, p, q );
    exit(1);
  }

  poly->vertex[v] = id;
}

#endif


void
mcube( GeomTrianglesP *triangles, double iso, int cell )
{
  Point p0,p1,p2,p3,p4,p5,p6,p7,p8;
  
  int x = cell % x_dim;
  int y = ((cell - x )/ x_dim) % y_dim;
  int z = cell/(x_dim*y_dim) ;

  if ( cell > 1000 && cell < 1010 ) {
    printf("cell %d at [%d,%d,%d]\n",cell,x,y,z);
    scanf("%*c");
  }
  int flags = 0;
  double isoTimes4 = 4*iso;

  //printf("\ncell %d: %d %d %d\n",cell,x,y,z);
  if ( data[cell]      >= iso ) flags |= 0x01;
  if ( data[cell+i001] >= iso ) flags |= 0x02;
  if ( data[cell+i010] >= iso ) flags |= 0x04;
  if ( data[cell+i011] >= iso ) flags |= 0x08;
  if ( data[cell+i100] >= iso ) flags |= 0x10;
  if ( data[cell+i101] >= iso ) flags |= 0x20;
  if ( data[cell+i110] >= iso ) flags |= 0x40;
  if ( data[cell+i111] >= iso ) flags |= 0x80;

  int *transformation = t[flags].transformationIndex;

  switch( t[flags].handler ) {
    
  case 0x00: 
    printf("mcube: empty cell !\n" );
    if ( cell > 1000 && cell < 1010 ) {
      printf("cell %d at [%d,%d,%d]\n",cell,x,y,z);
      printf("\t%lf %lf %lf %lf\n\t%lf %lf %lf %lf\n", 
	     data[cell] ,
	     data[cell+i001],
	     data[cell+i010],
	     data[cell+i011],
	     data[cell+i100],
	     data[cell+i101],
	     data[cell+i110],
	     data[cell+i111]);
      scanf("%*c");
    }

    break;
  
  case 0x01: 
    mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0 );
    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p1 );
    mcube_interp( cell, x, y, z, transformation, iso, 0, 1, p2 );
    //cerr << p0 << " " << p1 << " " << p2 << "\n";
    triangles->add(  p0, p1, p2 );
    break;

  case 0x03:
    mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0 );
    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p1 );
    mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p2 );
    //cerr << p0 << " " << p1 << " " << p2 << "\n";
    triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p3 );
    //cerr << p0 << " " << p2 << " " << p3 << "\n"; 
    triangles->add( p0,p2,p3);
    
    break;

  case 0x06: {

    /*
     *      Something of a special case here:  If the average of the vertices
     *      is greater than the iso, we create two separated polygons 
     *      at the vertices.  If it is less, then we make a little valley
     *      shape.
     */

    Value t[8];

    t[0] = data[cell];
    t[1] = data[cell+i001];
    t[2] = data[cell+i010];
    t[3] = data[cell+i011];
    t[4] = data[cell+i100];
    t[5] = data[cell+i101];
    t[6] = data[cell+i110];
    t[7] = data[cell+i111];
    if ( ( t[nl(0)] + t[nl(1)] + t[nl(2)] + t[nl(3)] ) > isoTimes4 ) {
      
      mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p0);
      mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p1);
      mcube_interp( cell, x, y, z, transformation, iso, 2, 3, p2);
      // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

      mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p3 );
      mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p4);
      // cerr << p2 << " " << p3 << " " << p4 << "\n"; 
	triangles->add( p2,p3,p4);

      mcube_interp( cell, x, y, z, transformation, iso, 1, 0, p5);
      // cerr << p4 << " " << p5 << " " << p0 << "\n"; 
	triangles->add( p4,p5,p0);

      // cerr << p4 << " " << p0 << " " << p2 << "\n"; 
	triangles->add( p4,p0,p2);

    } else {
      
      mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p0);
      mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p1);
      mcube_interp( cell, x, y, z, transformation, iso, 1, 0, p2);
      // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

      mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p3);
      mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p4);
      mcube_interp( cell, x, y, z, transformation, iso, 2, 3, p5);
      // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    }
    break;
    
  }
  
  case 0x07: 
    mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 3, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p3);
    // cerr << p2 << " " << p3 << " " << p0 << "\n"; 
	triangles->add( p2,p3,p0);
    
    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p4);
    // cerr << p3 << " " << p4 << " " << p0 << "\n"; 
	triangles->add( p3,p4,p0);
    break;

  case 0x0f: 
    mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p3);
    // cerr << p0 << " " << p2 << " " << p3 << "\n"; 
	triangles->add( p0,p2,p3);
    break;

  case 0x16: 

    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 0, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 1, 0, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);

    mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p6);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p7);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 3, p8);
    // cerr << p6 << " " << p7 << " " << p8 << "\n"; 
	triangles->add( p6,p7,p8);
    break;

  case 0x17: 

    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 2, 3, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 1, 3, p4);
    // cerr << p2 << " " << p3 << " " << p4 << "\n"; 
	triangles->add( p2,p3,p4);

    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p5);
    // cerr << p4 << " " << p5 << " " << p0 << "\n"; 
    triangles->add( p4,p5,p0);

    // cerr << p4 << " " << p0 << " " << p2 << "\n"; 
    triangles->add( p4,p0,p2);
    break;

  case 0x18: 

    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 0, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
    triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 1, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    break;

  case 0x19: 
    mcube_interp( cell, x, y, z, transformation, iso, 0, 1, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p3);
    // cerr << p0 << " " << p2 << " " << p3 << "\n"; 
	triangles->add( p0,p2,p3);
    
    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p5);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 1, p6);
    // cerr << p4 << " " << p5 << " " << p6 << "\n"; 
	triangles->add( p4,p5,p6);
    break;

  case 0x1b:
    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p4);
    // cerr << p2 << " " << p3 << " " << p4 << "\n"; 
	triangles->add( p2,p3,p4);
    
    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p5);
    // cerr << p4 << " " << p5 << " " << p0 << "\n"; 
	triangles->add( p4,p5,p0);
    
    // cerr << p4 << " " << p0 << " " << p2 << "\n"; 
	triangles->add( p4,p0,p2);
    break;

  case 0x1e: 

    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 0, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 1, 0, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p6);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p7);
    // cerr << p5 << " " << p6 << " " << p7 << "\n"; 
	triangles->add( p5,p6,p7);
    
    // cerr << p5 << " " << p7 << " " << p3 << "\n"; 
	triangles->add( p5,p7,p3);
    break;
    
  case 0x1f: 

    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p3);
    // cerr << p2 << " " << p3 << " " << p0 << "\n"; 
	triangles->add( p2,p3,p0);
    
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p4);
    // cerr << p3 << " " << p4 << " " << p0 << "\n"; 
	triangles->add( p3,p4,p0);
    break;

  case 0x27:

    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 5, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 1, 5, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p4);
    // cerr << p2 << " " << p3 << " " << p4 << "\n"; 
	triangles->add( p2,p3,p4);
    
    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p5);
    // cerr << p4 << " " << p5 << " " << p0 << "\n"; 
	triangles->add( p4,p5,p0);
    
    // cerr << p4 << " " << p0 << " " << p2 << "\n"; 
	triangles->add( p4,p0,p2);
    break;

  case 0x3c:
    
    mcube_interp( cell, x, y, z, transformation, iso, 4, 0, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 1, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p3);
    // cerr << p0 << " " << p2 << " " << p3 << "\n"; 
	triangles->add( p0,p2,p3);
    
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p5);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 1, p6);
    // cerr << p4 << " " << p5 << " " << p6 << "\n"; 
	triangles->add( p4,p5,p6);
    
    mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p7);
    // cerr << p4 << " " << p6 << " " << p7 << "\n"; 
	triangles->add( p4,p6,p7);
    break;

  case 0x3d: 

    mcube_interp( cell, x, y, z, transformation, iso, 0, 1, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 1, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 1, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    
    mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p6);
    // cerr << p3 << " " << p5 << " " << p6 << "\n"; 
	triangles->add( p3,p5,p6);
    break;

  case 0x3f: 
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 6, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
    mcube_interp( cell, x, y, z, transformation, iso, 2, 6, p3);
    // cerr << p0 << " " << p2 << " " << p3 << "\n"; 
	triangles->add( p0,p2,p3);
    break;

  case 0x6b: 

    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 6, 2, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 0, 2, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 3, 2, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 6, 7, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    
    mcube_interp( cell, x, y, z, transformation, iso, 6, 4, p6);
    // cerr << p5 << " " << p6 << " " << p3 << "\n"; 
	triangles->add( p5,p6,p3);
    
    mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p7);
    // cerr << p6 << " " << p7 << " " << p3 << "\n"; 
	triangles->add( p6,p7,p3);
    break;

  case 0x6f: {
    /*
     *      Something of a special case here:  If the average of the vertices
     *      is greater than the iso, we create two separated polygons 
     *      at the vertices.  If it is less, then we make a little valley
     *      shape.
     */
    
    Value    t[8];
    t[0] = data[cell];
    t[1] = data[cell+i001];
    t[2] = data[cell+i010];
    t[3] = data[cell+i011];
    t[4] = data[cell+i100];
    t[5] = data[cell+i101];
    t[6] = data[cell+i110];
    t[7] = data[cell+i111];

    if ( (t[nl(4)] + t[nl(5)] + t[nl(6)] + t[nl(7)]) < isoTimes4 ) {

      mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0);
      mcube_interp( cell, x, y, z, transformation, iso, 6, 4, p1);
      mcube_interp( cell, x, y, z, transformation, iso, 6, 7, p2);
      // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    
      mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p3);
      mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p4);
      // cerr << p2 << " " << p3 << " " << p4 << "\n"; 
	triangles->add( p2,p3,p4);
      
      mcube_interp( cell, x, y, z, transformation, iso, 5, 4, p5);
      // cerr << p4 << " " << p5 << " " << p0 << "\n"; 
	triangles->add( p4,p5,p0);
      
      // cerr << p4 << " " << p0 << " " << p2 << "\n"; 
	triangles->add( p4,p0,p2);
    } else {
      
      mcube_interp( cell, x, y, z, transformation, iso, 0, 4, p0);
      mcube_interp( cell, x, y, z, transformation, iso, 6, 4, p1);
      mcube_interp( cell, x, y, z, transformation, iso, 5, 4, p2);
      // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

      mcube_interp( cell, x, y, z, transformation, iso, 6, 7, p3);
      mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p4);
      mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p5);
      // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    }
    break;
  }
  
  case 0x7e: 
    
    mcube_interp( cell, x, y, z, transformation, iso, 1, 0, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 2, 0, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 4, 0, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);

    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p3);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p4);
    mcube_interp( cell, x, y, z, transformation, iso, 6, 7, p5);
    // cerr << p3 << " " << p4 << " " << p5 << "\n"; 
	triangles->add( p3,p4,p5);
    break;
    
  case 0x7f: 
    
    mcube_interp( cell, x, y, z, transformation, iso, 3, 7, p0);
    mcube_interp( cell, x, y, z, transformation, iso, 5, 7, p1);
    mcube_interp( cell, x, y, z, transformation, iso, 6, 7, p2);
    // cerr << p0 << " " << p1 << " " << p2 << "\n"; 
	triangles->add( p0,p1,p2);
    break;
    
  case 0xff: 
    printf("mcube: full cell !\n" );
    printf("\t%lf %lf %lf %lf\n\t%lf %lf %lf %lf\n", 
	   data[cell] ,
	   data[cell+i001],
	   data[cell+i010],
	   data[cell+i011],
	   data[cell+i100],
	   data[cell+i101],
	   data[cell+i110],
	   data[cell+i111]);
//     printf("mcube: v = %lf\n"
// 	   "%10lf %10lf %10lf %10lf\n%10lf %10lf %10lf %10lf\n",
// 	   iso,
// 	   data[cell],     
// 	   data[cell+i001],
// 	   data[cell+i010],
// 	   data[cell+i011],  
// 	   data[cell+i100],
// 	   data[cell+i101],
// 	   data[cell+i110],
// 	   data[cell+i111] );
    break;
  }
}



} // End namespace Yarden

