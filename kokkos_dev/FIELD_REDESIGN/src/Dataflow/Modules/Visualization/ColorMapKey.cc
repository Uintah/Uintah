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

//
// $Log$
// Revision 1.9.2.2  2000/10/26 10:03:46  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.9.2.1  2000/09/28 03:15:32  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.10  2000/06/13 20:31:19  kuzimmer
// Modified RescaleColorMap to set the scaled flag in the color map.
// Modified ColorMapKey so that it scales the colormap to the data if it
// wasn't previously scaled.
//
// Revision 1.9  2000/06/01 16:40:25  kuzimmer
// numbers were backwards on ColorMapKey
//
// Revision 1.8  2000/05/31 21:55:08  kuzimmer
// Modified ColorMapKey, it works!
//
// Revision 1.7  2000/03/17 09:27:30  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:07:05  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:48:05  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:55  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:03  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:11  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:56  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
