#include <Core/Datatypes/ColorMap.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Sticky.h>
#include <values.h>
#include <stdio.h>
#include <iostream>
using std::cerr;

#include "ParticleColorMapKey.h"

namespace Kurt {
using namespace SCIRun;


extern "C" Module *make_ParticleColorMapKey(const clString &id) {
  return new ParticleColorMapKey(id);
}

ParticleColorMapKey::ParticleColorMapKey(const clString &id)
  : Module("ParticleColorMapKey", id, Filter)
{
  white = scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);
  
  // create the ports
  imap = new ColorMapIPort( this, "ColorMap", ColorMapIPort::Atomic );
  iPort = new VisParticleSetIPort( this, "VisParticleSet", VisParticleSetIPort::Atomic );
  ogeom = new GeometryOPort( this, "Geometry", GeometryIPort::Atomic );

  add_iport( imap );
  add_iport( iPort );
  add_oport( ogeom );
}

ParticleColorMapKey::~ParticleColorMapKey()
{
}

void ParticleColorMapKey::execute() {
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
  
  VisParticleSetHandle part;
  if (iPort->get(part)){

    double max = -1e30;
    double min = 1e30;
    // All three particle variables use the same particle subset
    // so just grab one
    ParticleSubset *ps = part->getPositions().getParticleSubset();

  // default colormap--nobody has scaled it.
    if( !map->IsScaled()) {
      for(ParticleSubset::iterator iter = ps->begin();
	  iter != ps->end(); iter++){
	max = ( part->getScalars()[ *iter ] > max ) ?
	  part->getScalars()[ *iter ] : max;
	min = ( part->getScalars()[ *iter ] < min ) ?
	  part->getScalars()[ *iter ] : min;
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
    for(int i = 1; i < 4; i++ ) {
      sprintf( value, "%.2g", min + i*(max-min)/4.0 );
      all->add( new GeomMaterial( new GeomText(value,Point(xloc*i/4.0,yloc,0)),
				  white) );
    }
  }  
  GeomSticky *sticky = new GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ParticleColorMapKey" );
}
} // End namespace Kurt

