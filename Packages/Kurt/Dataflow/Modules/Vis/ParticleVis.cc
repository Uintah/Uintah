
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
 *   Packages/Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   Dec 1998
 *
 *  Copyright (C) 1994 SCI Group
 */
#include "ParticleVis.h"
#include <Packages/Kurt/DataArchive/VisParticleSet.h>
#include <Packages/Kurt/DataArchive/VisParticleSetPort.h>
#include <Packages/Kurt/Dataflow/Modules/Vis/VisControl.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/ColorMap.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Kurt {
using namespace SCIRun;


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
    // Let's set it up so that we can pick the particle set -- Packages/Kurt Z. 12/18/98
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
} // End namespace Kurt



