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

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomText.h>
#include <SCICore/Geom/ColorMapTex.h>
#include <SCICore/Geom/GeomTransform.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geom/Sticky.h>
#include <values.h>
#include <stdio.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ColorMapKey : public Module {
  ColorMapIPort *imap;
  ScalarFieldIPort *isf;
  GeometryOPort *ogeom;

  MaterialHandle white;
public:
  ColorMapKey(const clString &id);
  virtual ~ColorMapKey();
  virtual void execute();
};
extern "C" Module *make_ColorMapKey(const clString &id) {
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

ColorMapKey::~ColorMapKey()
{
}

void ColorMapKey::execute() {
  // the colormap is essential - without it return
  ColorMapHandle map;
  if( imap->get(map) == 0) 
    return;

  GeomGroup *all = new GeomGroup();

  double xsize = 15./16.0;
  double ysize = 0.05;
  ColorMapTex *sq = new ColorMapTex( Point( 0, -1, 0),
				 Point( xsize, -1, 0),
				 Point( xsize, -1 + ysize, 0 ),
				 Point( 0, -1 + ysize, 0 ) );

  sq->set_texture( map->raw1d );
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
    double max = -1e30;
    double min = 1e30;

    int i,j,k;
    if( !map->IsScaled()) {
      for(i = 0; i < grid->nx; i++){
	for(j = 0; j < grid->ny; j++){
	  for(k = 0; k < grid->nz; k++){
	    max = ( grid->grid(i,j,k) > max) ? grid->grid(i,j,k) : max;
	    min = ( grid->grid(i,j,k) < min) ? grid->grid(i,j,k) : min;
	  }
	}
      }
      if (min == max) {
	min -= 0.001;
	max += 0.001;
      }
      map->Scale(min,max);
    } else {
      max = map->max;
      min = map->min;
    }
    // some bases for positioning text
    double xloc = xsize;
    double yloc = -1 + 1.1 * ysize;

    // create min and max numbers at the ends
    char value[80];
    sprintf(value, "%.2g", max );
    all->add( new GeomMaterial( new GeomText(value, Point(xloc,yloc,0) ),
					     white) );
    sprintf(value, "%.2g", min );
    all->add( new GeomMaterial( new GeomText(value, Point(0,yloc,0)), white));

    // fill in 3 other places
    for( i = 1; i < 4; i++ ) {
      sprintf( value, "%.2g", min + i*(max-min)/4.0 );
      all->add( new GeomMaterial( new GeomText(value,Point(xloc*i/4.0,yloc,0)),
				  white) );
    }    
  }

  GeomSticky *sticky = new GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ColorMapKey" );
}

} // End namespace Modules
} // End namespace PSECommon
