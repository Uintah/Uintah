#include <Packages/Uintah/Dataflow/Modules/Visualization/ParticleColorMapKey.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomSticky.h>
#include <values.h>
#include <stdio.h>
#include <iostream>
using std::cerr;


namespace Uintah {
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

DECLARE_MAKER(ParticleColorMapKey)

  ParticleColorMapKey::ParticleColorMapKey(GuiContext* ctx)
  : Module("ParticleColorMapKey", ctx, Filter, "Visualization", "Uintah")
{
  white = scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);
  
  // create the ports
}

ParticleColorMapKey::~ParticleColorMapKey()
{
}

void ParticleColorMapKey::execute() {
  // Create the  ports
  iPort= (ScalarParticlesIPort *) get_iport("Scaling Particles");
  imap= (ColorMapIPort *) get_iport("ColorMap"); 
  ogeom= (GeometryOPort *) get_oport("Geometry"); 
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

  sq->set_texture( map->rawRGBA_ );
  all->add( sq );
  
  ScalarParticlesHandle part;
  if (iPort->get(part)){

    double max = -1e30;
    double min = 1e30;
    
  // default colormap--nobody has scaled it.
    if( !map->IsScaled()) {
      part->get_minmax(min, max);

      if (min == max) {
	min -= 0.001;
	max += 0.001;
      }
      map->Scale(min,max);
    } else {
      max = map->getMax();
      min = map->getMin();
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

