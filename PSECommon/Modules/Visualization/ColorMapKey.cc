//static char *id="@(#) $Id$";

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

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/CoreDatatypes/ColorMap.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRG.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomText.h>
#include <SCICore/Geom/TexSquare.h>
#include <SCICore/Geom/GeomTransform.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geom/Sticky.h>
#include <values.h>
#include <stdio.h>


namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

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

Module *make_ColorMapKey(const clString &id) {
  return new ColorMapKey(id);
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

  // this looks all general and everything, but when you look closely
  // at the TexSquare data structure, it turns out this is hard-coded
  // to have 64x64 entries.  So, we should make this TexSquare accordingly...

  double skip = (map->colors.size()+0.00001)/(64*64);
  double curr=0;

  double xsize = 15./16.0;
  double ysize = 0.1;
  TexSquare *sq = new TexSquare( Point( 0, -1, 0),
				 Point( xsize, -1, 0),
				 Point( xsize, -1 + ysize, 0 ),
				 Point( 0, -1 + ysize, 0 ) );

  unsigned char tex[ 64 * 64 * 3 ];
  for( int i = 0; curr < map->colors.size(); i++, curr+=skip ) {
      tex[ 3*i ] = 255*map->colors[(int)curr]->diffuse.r();
      tex[ 3*i + 1 ] = 255*map->colors[(int)curr]->diffuse.g();
      tex[ 3*i + 2 ] = 255*map->colors[(int)curr]->diffuse.b();
  }

  sq->set_texture( tex, 64, 64 );
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
    double xloc = xsize;
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:11  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:56  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
