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
#include <Uintah/Modules/Visualization/ParticleFieldExtractor.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Datatypes/VectorParticlesPort.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Uintah {
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
  spin0=scinew ScalarParticlesIPort(this, "ScalarParticles",
				ScalarParticlesIPort::Atomic);
  spin1=scinew ScalarParticlesIPort(this, "ScaleScalarParticles",
				ScalarParticlesIPort::Atomic);
  vpin=scinew VectorParticlesIPort(this, "VectorParticles",
				VectorParticlesIPort::Atomic);
  add_iport(spin0);
  add_iport(spin1);
  add_iport(vpin);
  cin=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
  add_iport(cin);
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
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
  outcolor=scinew Material(Color(0.3,0.3,0.3), Color(0.3,0.3,0.3),
			   Color(0.3,0.3,0.3), 0);
    
}

ParticleVis::~ParticleVis()
{
}

void ParticleVis::execute()
{
  ScalarParticlesHandle part;
  ScalarParticlesHandle scaleSet;
  VectorParticlesHandle vect;
  bool hasScale = false, hasVectors = false;
  if (!spin0->get(part)){
    last_idx=-1;
    return;
  }

  if(spin1->get(scaleSet)){
    hasScale = true;
  }

  if(vpin->get(vect)){
    hasVectors = true;
  }

  cbClass = part->getCallbackClass();

  // grap the color map from the input port
  ColorMapHandle cmh;
  ColorMap *cmap;
  if( cin->get( cmh ) )
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
      
    cmap = scinew ColorMap(rgb,rgbT,alphas,alphaT,16);
  }
  double max = -1e30;
  double min = 1e30;


  // All three particle variables use the same particle subset
  // so just grab one
  ParticleSubset *ps = part->getPositions().getParticleSubset();


  // default colormap--nobody has scaled it.
  if( !cmap->IsScaled()) {
    part->get_minmax(min,max);
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
    GeomGroup *obj = scinew GeomGroup;
    GeomArrows* arrows;
    if( drawVectors.get() == 1){
      arrows = scinew GeomArrows(width_scale.get(),
			      1.0 - head_length.get(),
			      drawcylinders.get(),
			      shaft_rad.get());
    }
    int count = 0;
//     ParticleSubset::iterator iter;
//     ParticleSubset::iterator siter;
//     ParticleSubset *ss;
//     ParticleSubset::iterator psbeginaddr = ps->begin();
//     ParticleSubset::iterator psendaddr = ps->end();
//     ParticleSubset::iterator ssbeginaddr;
//     ParticleSubset::iterator ssendaddr;


//     if( hasScale ){
//       ss = scaleSet->getPositions().getParticleSubset();
//       siter = ss->begin();
//       ssbeginaddr = ss->begin();
//       ssendaddr = ss->end();
//     }
    
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      count++;
      if (count == show_nth.get() ){ 
	GeomSphere *sp = 0;
	if( hasScale ){
	  double smin = 0, smax = 0;
	  scaleSet->get_minmax(smin,smax);
	  double scalefactor =
	    (scaleSet->get()[*iter] - smin)/(smax - smin);
	  if( scalefactor >= 1e-6 )
	    sp = scinew GeomSphere( part->getPositions()[*iter],
				 scalefactor * radius.get(),
				 nu, nv, *iter);
	} else {
	  sp = scinew GeomSphere( part->getPositions()[*iter],
			       radius.get(), nu, nv, *iter);
	}
	double value = part->get()[*iter];
	if( sp != 0)
	  obj->add( scinew GeomMaterial( sp,(cmap->lookup(value).get_rep())));
	count = 0;
      }
      if( drawVectors.get() == 1 && hasVectors){
	Vector V = vect->get()[*iter];
	if(V.length2() * length_scale.get() > 1e-3 )
	  arrows->add( part->getPositions()[*iter],
		       V*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
    } 
    if( drawVectors.get() == 1 && hasVectors){
      obj->add( arrows );
    }
    // Let's set it up so that we can pick the particle set -- Kurt Z. 12/18/98
    GeomPick *pick = scinew GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(pick, "Particles");      
  } else if( ps->getParticleSet()->numParticles() ) { // Particles
    GeomGroup *obj = scinew GeomGroup;
    GeomPts *pts = scinew GeomPts(ps->getParticleSet()->numParticles());
    pts->pickable = 1;
    int count = 0;
    GeomArrows* arrows;
    if( drawVectors.get() == 1 && hasVectors){
      arrows = scinew GeomArrows(width_scale.get(),
			      1.0 - head_length.get(),
			      drawcylinders.get(),
			      shaft_rad.get());
    }
    
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){     
      count++;
      if (count == show_nth.get() ){ 
	double value = part->get()[*iter];
	pts->add(part->getPositions()[*iter], cmap->lookup(value)->diffuse);
	count = 0;
      }
      if( drawVectors.get() == 1 && hasVectors){
	Vector V = vect->get()[*iter];
	if(V.length2() * length_scale.get() > 1e-3 )
	  arrows->add( part->getPositions()[*iter],
		       V*length_scale.get(),
		       outcolor, outcolor, outcolor);
      }
    } 
    obj->add( pts );
    if( drawVectors.get() == 1 && hasVectors){
      obj->add( arrows );
    }
    // GeomPick *pick = scinew GeomPick( obj, this);
    ogeom->delAll();
    ogeom->addObj(obj, "Particles");      
  }
//     GeomMaterial* matl=new GeomMaterial(obj,
//    					  scinew Material(Color(0,0,0),
//    							  Color(0,.6,0), 
//    							  Color(.5,.5,.5),20));
    
  //    ogeom->delAll();
  //    ogeom->addObj(matl, "Particles");
}


void ParticleVis::geom_pick(GeomPick* pick, void* userdata, int index)
{
  cerr << "Caught stray pick event in ParticleVis!\n";
  cerr << "this = "<< this <<", pick = "<<pick<<endl;
  cerr << "User data = "<<userdata<<endl;
  cerr << "sphere index = "<<index<<endl<<endl;
  int id = 0;
  if ( ((GeomObj *)pick)->getId( id ) )
    cerr<<"Id = "<< id <<endl;
  else
    cerr<<"Not getting the correct data\n";
  if( cbClass != 0 )
    ((ParticleFieldExtractor *)cbClass)->callback( index );
  // Now modify so that points and spheres store index.
}
  
extern "C" Module* make_ParticleVis( const clString& id ) {
  return scinew ParticleVis( id );
}

} // End namespace Modules
} // End namespace Uintah



