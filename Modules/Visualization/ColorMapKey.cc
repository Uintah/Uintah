/*
 *  ColorMapKey.cc: create a key for colormap
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Group.h>
#include <Geom/Text.h>
#include <Geom/TexSquare.h>
#include <Geom/Transform.h>
#include <Geometry/Transform.h>
#include <Malloc/Allocator.h>
#include <Geom/Sticky.h>
#include <values.h>
#include <stdio.h>

class ColorMapKey : public Module {
  ColorMapIPort *imap;
  ScalarFieldIPort *isf;
  GeometryOPort *ogeom;

  MaterialHandle white;
public:
  ColorMapKey(const clString &id);
  ColorMapKey(const ColorMapKey &copy, int deep);
  virtual ~ColorMapKey();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
Module *make_ColorMapKey(const clString &id) {
  return new ColorMapKey(id);
}
}

ColorMapKey::ColorMapKey(const clString &id)
  : Module("ColorMapKey", id, Filter)
{
  white = scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);
  
  // create the ports
  imap = new ColorMapIPort( this, "ColorMap", ColorMapIPort::Atomic );
  isf = new ScalarFieldIPort( this, "ScalarField", ScalarFieldIPort::Atomic );
  ogeom = new GeometryOPort( this, "Geometry", GeometryIPort::Atomic );

  add_iport( imap );
  add_iport( isf );
  add_oport( ogeom );
}

ColorMapKey::ColorMapKey(const ColorMapKey &copy, int deep)
  : Module( copy, deep )
{
  NOT_FINISHED("ColorMapKey::ColorMapKey");
}

ColorMapKey::~ColorMapKey()
{
}

Module* ColorMapKey::clone(int deep)
{
  return new ColorMapKey(*this, deep);
}

void ColorMapKey::execute() {
  // the colormap is essential - without it return
  ColorMapHandle map;
  if( imap->get(map) == 0) 
    return;

  // default billboard coordinates
  Vector bbvec(0,0,0);
  // default scaling
  Vector scale(1,1,1);

  GeomGroup *all = new GeomGroup();

  int i, j;
  int n = map->colors.size() / 10;
  int N;
  for( j = 1; pow(2,j) < n; j++ );
  N = pow(2,j);

  double xsize = 15. * 1.0/16.0;
  double ysize = 15. * 1.0/(double)N;
  TexSquare *sq = new TexSquare( Point( 0, -1, 0),
				 Point( xsize, -1, 0),
				 Point( xsize, -1 + ysize, 0 ),
				 Point( 0, -1 + ysize, 0 ) );

  unsigned char tex[ N * 16 * 3 ];
  for( i = 0; i < n-1; i++ ) {
    for( j = 0; j < 16; j++ ) {
      tex[ 3*(i*16 + j) ] = 255*map->colors[i*10]->diffuse.r();
      tex[ 3*(i*16 + j) + 1 ] = 255*map->colors[i*10]->diffuse.g();
      tex[ 3*(i*16 + j) + 2 ] = 255*map->colors[i*10]->diffuse.b();
    }
  }

  for( i = n-1; i < N; i++ ) {
    for( j = 0; j < 16; j++ ) {
      tex[ 3*(i*16 + j) ] = 0;
      tex[ 3*(i*16 + j) + 1 ] = 0;
      tex[ 3*(i*16 + j) + 2 ] = 0;
    }
  }

  sq->set_texture( tex, N, 16 );
  all->add( sq );
  
  // if the scalar field is present, we can add numbers and place the
  // billboard more intelligently.
  ScalarFieldHandle sf;
  if( isf->get( sf ) ) {

    ScalarFieldRG *grid = sf->getRG();
    // but we need it to be a regular grid.
    if( grid == 0 ) {
      cerr << "ColorMapKey wants ScalarFieldRG, didn't get it\n";
      return;
    }

    scale = Vector(grid->nx, grid->ny, 1 );
    
    // find min and max scalar values
    double max = -MAXDOUBLE;
    double min = MAXDOUBLE;
    int i, j;
    for( i = 0; i < grid->nx; i++ ) {
      for( j = 0; j < grid->ny; j++ ) {
	max = ( max < grid->grid(i,j,0) ) ? grid->grid(i,j,0) : max;
	min = ( min > grid->grid(i,j,0) ) ? grid->grid(i,j,0) : min;
      }
    }

    // some bases for positioning text
    double xloc = ((double)n / (double)N) * xsize;
    double yloc = -1 + 1.5 * ysize;

    // create min and max numbers at the ends
    char value[80];
    sprintf(value, "%g", min );
    all->add( new GeomMaterial( new GeomText(value, Point(xloc,yloc,0) ),
					     white) );
    sprintf(value, "%g", max );
    all->add( new GeomMaterial( new GeomText(value, Point(0,yloc,0)), white));

    // fill in 3 other places
    for( i = 1; i < 4; i++ ) {
      sprintf( value, "%g", max - i*(max-min)/4.0 );
      all->add( new GeomMaterial( new GeomText(value,Point(xloc*i/4.0,yloc,0)),
				  white) );
    }
    
    Point a, b;
    sf->get_bounds(a, b);
    bbvec = Vector(a.x() - 1, a.y() - 1, (a.z() + b.z()) / 2.0 );
  }

  GeomSticky *sticky = new GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ColorMapKey" );
}
