//static char *id="@(#) $Id$";

/*
 *  ParticleVis.cc:  Convert a Particle Set into geoemtry
 *
 *  Written 
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Modified
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   Dec 1998
 *
 *  Copyright (C) 1994 SCI Group
 */
#include "ParticleVis.h"
#include <Kurt/DataArchive/VisParticleSet.h>
#include <Kurt/DataArchive/VisParticleSetPort.h>
#include <Kurt/Modules/Vis/VisControl.h>
#include <SCICore/Geom/GeomArrows.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Datatypes/ColorMap.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

ParticleVis::ParticleVis(const clString& id)
  : Module("ParticleVis", id, Filter), current_time("current_time", id, this),
    radius("radius", id, this), drawspheres("drawspheres", id, this),
    drawVectors("drawVectors",id,this), length_scale("length_scale", id, this),
    head_length("head_length", id, this), width_scale("width_scale",id,this),
    shaft_rad("shaft_rad", id,this), drawcylinders("drawcylinders", id, this),
    polygons("polygons", id, this),
    show_nth("show_nth", id, this),
    MIN_POLYS(8), MAX_POLYS(400),
    MIN_NU(4), MAX_NU(20), MIN_NV(2), MAX_NV(20)
{
  // Create the input port
  iPort=new VisParticleSetIPort(this, "Particles",
				VisParticleSetIPort::Atomic);
  add_iport(iPort);
  iCmap=new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
  add_iport(iCmap);
  ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  last_idx=-1;
  last_generation=-1;
  drawspheres.set(1);
/*   
  radius.set(0.05);
  polygons.set(100);
  drawVectors.set(0);
  drawcylinders.set(0);
  length_scale.set(0.1);
  width_scale.set(0.1);
  head_length.set(0.3);
  shaft_rad.set(0.1);
  show_nth.set(1); 
*/
  outcolor=new Material(Color(0.3,0.3,0.3), Color(0.3,0.3,0.3),
			   Color(0.3,0.3,0.3), 0);
    
}

ParticleVis::~ParticleVis()
{
}

void ParticleVis::execute()
{
  VisParticleSetHandle part;
    
  if (!iPort->get(part)){
    last_idx=-1;
    return;
  }

  cbClass = part->getCallbackClass();

  // grap the color map from the input port
  ColorMapHandle cmh;
  ColorMap *cmap;
  if( iCmap->get( cmh ) )
    cmap = cmh.get_rep();
  else {
    // create a default colormap
    Array1<Color> rgb;
    Array1<float> rgbT;
    Array1<float> alphas;
    Array1<float> alphaT;
    rgb.add( Color(1,0,0) );
    rgb.add( Color(0,0,1) );
    rgbT.add(0.0);
    rgbT.add(1.0);
    alphas.add(1.0);
    alphas.add(1.0);
    alphaT.add(1.0);
    alphaT.add(1.0);
      
    cmap = new ColorMap(rgb,rgbT,alphas,alphaT,16);
  }
  double max = -1e30;
  double min = 1e30;


  // All three particle variables use the same particle subset
  // so just grab one
  ParticleSubset *ps = part->getPositions().getParticleSubset();


  // default colormap--nobody has scaled it.
  if( !cmap->IsScaled()) {
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
    cmap->Scale(min,max);
    cerr << "min=" << min << ", max=" << max << '\n';
  }  

  //--------------------------------------
  cerr << "numParticles: " << ps->getParticleSet()->numParticles() << '\n';

  if( drawspheres.get() == 1 && ps->getParticleSet()->numParticles()) {
    float t = (polygons.get() - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
    int nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
    int nv = int(MIN_NV + t*(MAX_NV - MIN_NV));
    GeomGroup *obj = new GeomGroup;
    GeomArrows* arrows;
    if( drawVectors.get() == 1){
      arrows = new GeomArrows(width_scale.get(),
			      1.0 - head_length.get(),
			      drawcylinders.get(),
			      shaft_rad.get());
    }
    int count = 0;
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      count++;
      if (count == show_nth.get() ){ 
	GeomSphere *sp = new GeomSphere( part->getPositions()[*iter],
					 radius.get(), nu, nv, *iter);
	double value = part->getScalars()[*iter];
	obj->add( new GeomMaterial( sp,(cmap->lookup(value).get_rep())));
	count = 0;
      }
      if( drawVectors.get() == 1){
	Vector V = part->getVectors()[*iter];
	if(V.length2() * length_scale.get() > 1e-3 )
	  arrows->add( part->getPositions()[*iter],
		       V*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
    } 
    if( drawVectors.get() == 1){
      obj->add( arrows );
    }
    // Let's set it up so that we can pick the particle set -- Kurt Z. 12/18/98
    GeomPick *pick = new GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(pick, "Particles");      
  } else if( ps->getParticleSet()->numParticles() ) { // Particles
    GeomGroup *obj = new GeomGroup;
    GeomPts *pts = new GeomPts(ps->getParticleSet()->numParticles());
    pts->pickable = 1;
    int count = 0;
    GeomArrows* arrows;
    if( drawVectors.get() == 1){
      arrows = new GeomArrows(width_scale.get(),
			      1.0 - head_length.get(),
			      drawcylinders.get(),
			      shaft_rad.get());
    }
    
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){     
      count++;
      if (count == show_nth.get() ){ 
	double value = part->getScalars()[*iter];
	pts->add(part->getPositions()[*iter], cmap->lookup(value)->diffuse);
	count = 0;
      }
      if( drawVectors.get() == 1){
	Vector V = part->getVectors()[*iter];
	if(V.length2() * length_scale.get() > 1e-3 )
	  arrows->add( part->getPositions()[*iter],
		       V*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
    } 
    obj->add( pts );
    if( drawVectors.get() == 1){
      obj->add( arrows );
    }
    // GeomPick *pick = new GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(obj, "Particles");      
  }
//     GeomMaterial* matl=new GeomMaterial(obj,
//    					  new Material(Color(0,0,0),
//    							  Color(0,.6,0), 
//    							  Color(.5,.5,.5),20));
    
  //    ogeom->delAll();
  //    ogeom->addObj(matl, "Particles");
}


void ParticleVis::geom_pick(GeomPick* pick,void* userdata, GeomObj* picked_obj)
{
  cerr << "Caught stray pick event in ParticleVis!\n";
  cerr << "this = "<< this <<", pick = "<<pick<<endl;
  cerr << "User data = "<<userdata<<endl;
  //  cerr << "sphere index = "<<index<<endl<<endl;
  int id = 0;
  if ( ((GeomObj *)picked_obj)->getId( id ) )
    cerr<<"Id = "<< id <<endl;
  else
    cerr<<"Not getting the correct data\n";
  if( cbClass != 0 )
    ((VisControl *)cbClass)->callback( id );
  // Now modify so that points and spheres store index.
}
  
extern "C" Module* make_ParticleVis( const clString& id ) {
  return new ParticleVis( id );
}

} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.5  2000/08/11 16:11:09  bigler
// Replace int index parameter in geom_pick function with GeomObj*.
// Changed function to acces index from GeomObj*.
//
// Revision 1.4  2000/05/22 17:20:01  kuzimmer
// Updating new Viz tools
//
// Revision 1.3  2000/05/21 08:18:06  sparker
// Always compute min/max for particles
// Fill in grid data
// Be careful if a variable doesn't exist
//
// Revision 1.2  2000/05/20 08:03:56  sparker
// Got vis tools to work
//
// Revision 1.1  2000/05/20 02:30:07  kuzimmer
// Multiple changes for new vis tools
//
// Revision 1.12  2000/03/17 09:30:11  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.11  2000/01/27 04:52:17  kuzimmer
// changes necessary to make MaterialParticle files work with only Grids or only Particles
//
// Revision 1.10  1999/12/28 21:11:44  kuzimmer
// modified so that picking works again
//
// Revision 1.9  1999/12/09 21:29:54  kuzimmer
// Change necessary to fix picking
//
// Revision 1.8  1999/10/07 02:08:27  sparker
// use standard iostreams and complex type
//
// Revision 1.7  1999/09/21 21:22:05  kuzimmer
// added particle density control
//
// Revision 1.6  1999/09/21 16:12:24  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.5  1999/08/25 03:49:03  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:08  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:22  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:40:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 17:08:57  mcq
// Initial commit
//
// Revision 1.3  1999/06/09 23:23:43  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.2  1999/04/27 23:18:41  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
