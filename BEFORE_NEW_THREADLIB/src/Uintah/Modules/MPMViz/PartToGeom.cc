//static char *id="@(#) $Id$";

/*
 *  PartToGeom.cc:  Convert a Particle Set into geoemtry
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

#include <Uintah/Datatypes/Particles/ParticleSetPort.h>
#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>
#include "ParticleGridVisControl.h"

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

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;

class PartToGeom : public Module {
    ParticleSetIPort* iPort;
    ColorMapIPort *iCmap;
    GeometryOPort* ogeom;
    TCLdouble current_time;
    TCLdouble radius;
    TCLint drawcylinders;
    TCLdouble length_scale;
    TCLdouble head_length;
    TCLdouble width_scale;
    TCLdouble shaft_rad;
    TCLint drawVectors;
    TCLint drawspheres;
    TCLint polygons; // number of polygons used to represent
    // a sphere: [minPolys, MAX_POLYS]
    const int MIN_POLYS;    // polys, nu, and nv must correlate
    const int MAX_POLYS;  // MIN_NU*MIN_NV = MIN_POLYS
    const int MIN_NU;       // MAX_NU*MAX_NV = MAX_POLYS
    const int MAX_NU;
    const int MIN_NV;
    const int MAX_NV;
    MaterialHandle outcolor;
       
    int last_idx;
    int last_generation;
    void *cbClass;
 public:
    PartToGeom(const clString& id);
    virtual ~PartToGeom();
    virtual void geom_pick(GeomPick*, void*, int);

    virtual void execute();
};

PartToGeom::PartToGeom(const clString& id)
  : Module("PartToGeom", id, Filter), current_time("current_time", id, this),
    radius("radius", id, this), drawspheres("drawspheres", id, this),
    drawVectors("drawVectors",id,this), length_scale("length_scale", id, this),
    head_length("head_length", id, this), width_scale("width_scale",id,this),
    shaft_rad("shaft_rad", id,this), drawcylinders("drawcylinders", id, this),
    polygons("polygons", id, this), MIN_POLYS(8), MAX_POLYS(400),
    MIN_NU(4), MAX_NU(20), MIN_NV(2), MAX_NV(20)
{
  // Create the input port
  iPort=scinew ParticleSetIPort(this, "Particles", ParticleSetIPort::Atomic);
  add_iport(iPort);
  iCmap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
  add_iport(iCmap);
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  last_idx=-1;
  last_generation=-1;
  drawspheres.set(1);
  radius.set(0.05);
  polygons.set(100);
  drawVectors.set(0);
  drawcylinders.set(0);
  length_scale.set(0.1);
  width_scale.set(0.1);
  head_length.set(0.3);
  shaft_rad.set(0.1);
  outcolor=scinew Material(Color(0.5,0.5,0.5), Color(0.5,0.5,0.5),
			   Color(0.5,0.5,0.5), 0);
    
}

PartToGeom::~PartToGeom()
{
}

void PartToGeom::execute()
{
  ParticleSetHandle part;
    
  if (!iPort->get(part)){
    last_idx=-1;
    return;
  }

  //double time=current_time.get();
  Array1<double> timesteps;
  part->list_natural_times(timesteps);
  if(timesteps.size()==0){
    ogeom->delAll();
    last_idx=-1;
    return;
  }

  int timestep=0;
  /*    while(time>timesteps[timestep] && timestep<timesteps.size()-1)
	timestep++;

	if(timestep == last_idx && part->generation == last_generation)
	return;
    
	last_idx=timestep;
	last_generation=part->generation;
  */

  int posid = 0, sid, vid;
  Array1<Vector>pos;
  Array1<Vector>vectors;
  Array1<double>scalars;
  
  ParticleSet *ps = part.get_rep();
  if( MPVizParticleSet *mpvps = dynamic_cast <MPVizParticleSet *> (ps)){
    posid = part->position_vector();
    part->get(timestep, posid, pos);

    vid = part->find_vector( mpvps->getVectorId());
    part->get(timestep, vid, vectors);
    

    sid = part->find_scalar( mpvps->getScalarId());
    part->get(timestep, sid, scalars);
    
    cbClass = mpvps->getCallbackClass(); // hack need a better way.
  } else {
    cbClass = 0;
    posid=part->position_vector();
    part->get(timestep, posid, pos);
    part->get(timestep, 0, scalars);
  }
      
    
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


  // default colormap--nobody has scaled it.
  if( !cmap->IsScaled()) {
    int i;
    for( i = 0; i < scalars.size(); i++ ) {
      max = ( scalars[i] > max ) ? scalars[i] : max;
      min = ( scalars[i] < min ) ? scalars[i] : min;
    }
    if (min == max) {
      min -= 0.001;
      max += 0.001;
    }
    cmap->Scale(min,max);
  }   

  //--------------------------------------
  if( drawspheres.get() == 1 ) {
    float t = (polygons.get() - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
    int nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
    int nv = int(MIN_NV + t*(MAX_NV - MIN_NV));
    GeomGroup *obj = scinew GeomGroup;
    for (int i=0; i<pos.size();i++) {
      GeomSphere *sp = scinew GeomSphere(pos[i].asPoint(),radius.get(),
					 nu, nv);
      // Default ColorMap ie. unscaled n autoscale values.
      //      if( cmap->isScaled() ) {
	
//  	int index = cmap->rcolors.size() * ( (scalars[i] - min) /
// 					     (max - min + 1) ) ;
// 	obj->add( scinew GeomMaterial(sp,scinew Material( Color(1,1,1),
// 							  cmap->rcolors[index],
// 							  cmap->rcolors[index],
// 							  20 )));
//       } else { // ColorMap has been scaled just use it.
	obj->add( scinew GeomMaterial( sp,(cmap->lookup(scalars[i])).get_rep()));
//       }
    }
      
    if( drawVectors.get() == 1 && vectors.size() == pos.size()) {
      GeomArrows* arrows = new GeomArrows(width_scale.get(),
					  1.0 - head_length.get(),
					  drawcylinders.get(),
					  shaft_rad.get());
      //hard coded for now.
      for (int j = 0; j < pos.size(); j++){
	if(vectors[j].length2() * length_scale.get() > 1e-3 )
	  arrows->add( pos[j].asPoint(), vectors[j]*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
      obj->add( arrows );
    }
    // Let's set it up so that we can pick the particle set -- Kurt Z. 12/18/98
    GeomPick *pick = scinew GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(pick, "Particles");      
  } else { // Particles
    GeomGroup *obj = scinew GeomGroup;
    GeomPts *pts = scinew GeomPts(pos.size());
    pts->pickable = 1;
    for(int i=0;i<pos.size();i++) {
      pts->add(pos[i].asPoint(), (cmap->lookup(scalars[i]))->diffuse);
      //pts->add(pos[i].asPoint());
      //Color c = (cmap->lookup(scalars[i]))->diffuse;
      //cerr<< "diffuse component is ("<< c.r()<<", "<<c.g()<<", "<<c.b()<<")\n";
    }
    obj->add( pts );
    if( drawVectors.get() == 1 && vectors.size() == pos.size()) {
      GeomArrows* arrows = new GeomArrows(width_scale.get(),
					  1.0 - head_length.get(),
					  drawcylinders.get(),
					  shaft_rad.get());
      //hard coded for now.
      for (int j = 0; j < pos.size(); j++){
	if(vectors[j].length2() * length_scale.get() > 1e-3 )
	  arrows->add( pos[j].asPoint(), vectors[j]*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
      obj->add( arrows );
    }
    GeomPick *pick = scinew GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(pick, "Particles");      
  }
//     GeomMaterial* matl=scinew GeomMaterial(obj,
//    					  scinew Material(Color(0,0,0),
//    							  Color(0,.6,0), 
//    							  Color(.5,.5,.5),20));
    
  //    ogeom->delAll();
  //    ogeom->addObj(matl, "Particles");
}


void PartToGeom::geom_pick(GeomPick* pick, void* userdata, int index)
{
  cerr << "Caught stray pick event in PartToGeom!\n";
  cerr << "this = "<< this <<", pick = "<<pick<<endl;
  cerr << "User data = "<<userdata<<endl;
  cerr << "sphere index = "<<index<<endl<<endl;
  
  if( cbClass != 0 )
    ((ParticleGridVisControl *) cbClass)->callback( index );
  // Now modify so that points and spheres store index.
}
  
Module* make_PartToGeom( const clString& id ) {
  return new PartToGeom( id );
}

} // End namespace Modules
} // End namespace Uintah


//
// $Log$
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
