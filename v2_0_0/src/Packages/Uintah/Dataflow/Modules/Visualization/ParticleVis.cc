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
#include <Core/Geom/GeomArrows.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomPoint.h> 
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomEllipsoid.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Datatypes/PropertyManager.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Dataflow/Modules/Selectors/ParticleFieldExtractor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Packages/Uintah/Core/Datatypes/PSet.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <stdio.h>
#include <iostream>
using std::cerr;

namespace Uintah {
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

ParticleVis::ParticleVis(GuiContext* ctx) :
  Module("ParticleVis", ctx, Filter, "Visualization", "Uintah"), 
  min_(ctx->subVar("min_")),  max_(ctx->subVar("max_")),
  isFixed(ctx->subVar("isFixed")),
  current_time(ctx->subVar("current_time")),
  radius(ctx->subVar("radius")), 
  auto_radius(ctx->subVar("auto_radius")), 
  drawcylinders(ctx->subVar("drawcylinders")),
  length_scale(ctx->subVar("length_scale")),
  auto_length_scale(ctx->subVar("auto_length_scale")),
  min_crop_length(ctx->subVar("min_crop_length")),
  max_crop_length(ctx->subVar("max_crop_length")),
  head_length(ctx->subVar("head_length")), 
  width_scale(ctx->subVar("width_scale")),
  shaft_rad(ctx->subVar("shaft_rad")),
  show_nth(ctx->subVar("show_nth")),
  drawVectors(ctx->subVar("drawVectors")),
  drawspheres(ctx->subVar("drawspheres")),
  polygons(ctx->subVar("polygons")),
  MIN_POLYS(8), MAX_POLYS(400),
  MIN_NU(4), MAX_NU(20), MIN_NV(2), MAX_NV(20)
{
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
  //outcolor=scinew Material(Color(0.3,0.3,0.3), Color(0.3,0.3,0.3),
//			   Color(0.3,0.3,0.0), 0);
  outcolor=scinew Material(Color(1.0,1.0,0.0), Color(1.0,1.0,0.0),
			   Color(1.0,1.0,0.0), 0);
    
}

ParticleVis::~ParticleVis()
{
}

void ParticleVis::execute()
{
  // Create the input port
  spin0= (ScalarParticlesIPort *) get_iport("Scalar Particles");
  spin1= (ScalarParticlesIPort *) get_iport("Scaling Particles");
  vpin= (VectorParticlesIPort *) get_iport("Vector Particles");
  tpin= (TensorParticlesIPort *) get_iport("Tensor Particles");
  cin= (ColorMapIPort *) get_iport("ColorMap"); 
  ogeom= (GeometryOPort *) get_oport("Geometry"); 
  ogeom->delAll();

  if(!spin0) {
    error("Unable to initialize iport 'Scalar Particles'.");
  }

  if(!ogeom) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }

  
  ScalarParticlesHandle part;
  ScalarParticlesHandle scaleSet;
  VectorParticlesHandle vect;
  TensorParticlesHandle tens;
  bool hasScale = false, 
     hasVectors = false, 
     hasTensors = false;
  if (!spin0->get(part)){
    last_idx=-1;
    return;
  } else if(!part.get_rep()) {
    last_idx=-1;
    return;
  }

  if( spin1 ) {
    if(spin1->get(scaleSet)){
      if( scaleSet.get_rep() ) {
	hasScale = true;
	gui->execute(id + " scalable 1");
      } else {
	gui->execute(id + " scalable 0");
      }
    } else {
      gui->execute(id + " scalable 0");
    }
  } else {
    gui->execute(id + " scalable 0");
  }
  
  if(vpin){
    if(vpin->get(vect)){
      if( vect.get_rep() != 0)
	hasVectors = true;
    }
  }
  if( tpin ) {
    if(tpin->get(tens)){
      if(tens.get_rep() != 0 )
	hasTensors = true;
    }
  }
  
  
  // grab the color map from the input port
  ColorMapHandle cmh;
  ColorMap *cmap;
  bool create_cmap = true;
  if (cin ) {
    if( cin->get( cmh ) ){
      if( cmh.get_rep() ){
	cmap = cmh.get_rep();
	create_cmap = false;
      }
    }
  }
  
  if( create_cmap ){
    // create a default colormap
    vector<Color> rgb;
    vector<float> rgbT;
    vector<float> alphas;
    vector<float> alphaT;
    rgb.push_back( Color(1,0,0) );
    rgb.push_back( Color(0,0,1) );
    rgbT.push_back(0.0);
    rgbT.push_back(1.0);
    alphas.push_back(1.0);
    alphas.push_back(1.0);
    alphaT.push_back(1.0);
    alphaT.push_back(1.0);
    
    cmap = scinew ColorMap(rgb,rgbT,alphas,alphaT);
  }
  double max = -1e30;
  double min = 1e30;
  

  // All three particle variables use the same particle subset
  // so just grab one
  PSet *pset = part->getParticleSet();
  cbClass = pset->getCallBackClass();
  vector<ShareAssignParticleVariable<Point> >& points = pset->getPositions();
  vector<ShareAssignParticleVariable<long64> >& ids = pset->getIDs();
  vector<ShareAssignParticleVariable<long64> >::iterator id_it = ids.begin();
  vector<ShareAssignParticleVariable<double> >& values = part->get();
  vector<ShareAssignParticleVariable<Point> >::iterator p_it = points.begin();
  vector<ShareAssignParticleVariable<Point> >::iterator p_it_end = points.end();
  vector<ShareAssignParticleVariable<double> >::iterator s_it = values.begin();
  vector<ShareAssignParticleVariable<double> >::iterator scale_it;
  vector<ShareAssignParticleVariable<Vector> >::iterator v_it;
  vector<ShareAssignParticleVariable<Matrix3> >::iterator t_it;

  if( hasScale ) scale_it = scaleSet->get().begin();
  if( hasVectors ) v_it = vect->get().begin();
  if( hasTensors ) t_it = tens->get().begin();
  
  
  bool have_particle = false;


  
  // Check to see if we are dealing with a new dataset
  // if so, check the generation number and update the particlesize
  // and vectorsize based on the cell size of the data.
  double cell_size = 0;  // may be used later
  double max_vlength = 0;  // may be used later
  if( auto_radius.get() || (hasVectors && auto_length_scale.get()) )
  {
    BBox spatial_box;
    IntVector low, hi, range;
    pset->getLevel()->findIndexRange( low, hi );
    range = hi -low;
    pset->getLevel()->getSpatialRange(spatial_box);
    Vector drange = spatial_box.max() - spatial_box.min();
    Vector crange(drange.x()/range.x(), drange.y()/range.y(),
		  drange.z()/range.z());
    // use the length of the domain diagonol as the domain size value
    double domain_size = sqrt(Dot(drange,drange));
    // use the length of a cell diagonol as the cell size value
    cell_size = sqrt(Dot(crange, crange));
    
    if( auto_radius.get() ){  // set particle radius to 1/4 cell size per
                              // request by container dynamics.
      double part_size = cell_size/4.0;
      if (part_size > 0)
	radius.set( part_size);
    }  
    if(hasVectors && auto_length_scale.get()) { 
      // lets mess with the vectors too...
      double max_length2 = 0;
      for(; p_it != p_it_end; v_it++, p_it++){
	ParticleSubset *ps = (*p_it).getParticleSubset();
	for(ParticleSubset::iterator iter = ps->begin();
	    iter != ps->end(); iter++){
	  max_length2 = (( max_length2 > ((*v_it)[*iter]).length2()) ?
			 max_length2 : ((*v_it)[*iter]).length2());
	}
      }
      // max_lenght2 is the length sqared of the longest vector.
      // Take the sqare root.
      max_vlength = sqrt(max_length2);
      // set the length scale so that the longest vector is scaled to
      // the size of the domain
      double len_scale_val = domain_size/(max_vlength); 
      length_scale.set( 0 );
      length_scale.set(len_scale_val );
      // manipulate head & width scale so that the head size is approx
      // cell size for a vector of max_vlength
      head_length.set( cell_size/(max_vlength*len_scale_val ));
      width_scale.set( head_length.get()/2.0);
      p_it = points.begin();
      v_it = vect->get().begin();
    }
  }
  
  for(; p_it != p_it_end; p_it++, s_it++, id_it++){
    ParticleSubset *ps = (*p_it).getParticleSubset();
    GeomGroup *obj = scinew GeomGroup;
    
    // default colormap--nobody has scaled it.
    if( !cmap->IsScaled()) {
      part->get_minmax(min,max);
      if (min == max) {
	min -= 0.001;
	max += 0.001;
      }
      cmap->Scale(min,max);
    }  
    
    //--------------------------------------
    if( drawspheres.get() == 1 && ps->getParticleSet()->numParticles()) {
      float t = (polygons.get() - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
      int nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
      int nv = int(MIN_NV + t*(MAX_NV - MIN_NV));
      GeomArrows* arrows;
      if( drawVectors.get() == 1){
	arrows = scinew GeomArrows(width_scale.get(),
				   1.0 - head_length.get(),
				   drawcylinders.get(),
				   shaft_rad.get());
      }
      int count = 0;
      for(ParticleSubset::iterator iter = ps->begin();
	  iter != ps->end(); iter++){
	count++;
	if (count == show_nth.get() ){ 
	  have_particle = true;
	  GeomObj *sp = 0;
	  
	  if( hasScale ) {
	    double smin = 1e30, smax = -1e30;
	    double sv = (*scale_it)[*iter];
	    if( isFixed.get() == 1 ){
	      smin = min_.get();
	      smax = max_.get();
	    } else {
	      scaleSet->get_minmax(smin,smax);
	      min_.set(smin);
	      max_.set(smax);
	    }
	    if( sv < smin ) sv = smin;
	    if( sv > smax ) sv = smax;

	    double scalefactor = 1;
	    // if smin = smax
	    // then set scale = 1 - Todd
	    // This was originally set to 0
	    if (smax-smin > 1e-10)
	      scalefactor = (sv - smin)/(smax - smin);

	    if( scalefactor >= 1e-6 ){
	      if(!hasTensors){
		//cout << "Particle ID for "<<*iter<<" = "<<(*id_it)[*iter]<<endl;
		sp = scinew GeomSphere( (*p_it)[*iter],
					scalefactor * radius.get(),
					nu, nv);
		sp->setId((long long)((*id_it)[*iter]));
	      } else { // make an ellips
		double matrix[16];
		Matrix3 M = (*t_it)[*iter];
		double e1,e2,e3;
		M.getEigenValues(e1,e2,e3);
		matrix[3] = matrix[7] = matrix[11] = matrix[12] =
		  matrix[13] = matrix[14] = 0;
		matrix[15] = 1;
		matrix[0] = M(0,0); matrix[1] = M(1,0); matrix[2] = M(2,0);
		matrix[4] = M(0,1); matrix[5] = M(1,1); matrix[6] = M(2,1);
		matrix[8] = M(0,2); matrix[9] = M(1,2); matrix[10] = M(2,2);
		
		//cout << "Particle ID for "<<*iter<<" = "<<(*id_it)[*iter]<<endl;
		sp = scinew GeomEllipsoid((*p_it)[*iter],
					  scalefactor * radius.get(),
					  nu, nv, &(matrix[0]), 2);
		sp->setId((long long)((*id_it)[*iter]));
	      }
	    }
	  } else {
	    if(!hasTensors){
	      //cout << "Particle ID for "<<*iter<<" = "<<(*id_it)[*iter]<<endl;
	      sp = scinew GeomSphere( (*p_it)[*iter], radius.get(), nu, nv);
	      sp->setId((long long)((*id_it)[*iter]));
	    } else {
	      double matrix[16];
	      Matrix3 M = (*t_it)[*iter];
	      if( M.Norm() > 1e-8){
		Vector v1(M(1,1),M(2,1),M(3,1));
		Vector v2(M(1,2),M(2,2),M(3,2));
		Vector v3(M(1,3),M(2,3),M(3,3));
		double norm = 1/Max(v1.length(), v2.length(), v3.length());
		matrix[3] = matrix[7] = matrix[11] = matrix[12] =
		  matrix[13] = matrix[14] = 0;
		matrix[15] = 1;
		matrix[0] = M(1,1)*norm; matrix[1] = M(2,1)*norm;
		matrix[2] = M(3,1)*norm; matrix[4] = M(1,2)*norm;
		matrix[5] = M(2,2)*norm; matrix[6] = M(3,2)*norm;
		matrix[8] = M(1,3)*norm; matrix[9] = M(2,3)*norm;
		matrix[10] = M(3,3)*norm;
		
		//cout << "Particle ID for "<<*iter<<" = "<<(*id_it)[*iter]<<endl;
		sp = scinew GeomEllipsoid((*p_it)[*iter],
					  radius.get(), nu, nv, &(matrix[0]),
					  2);
		sp->setId((long long)((*id_it)[*iter]));
	      }
	    }
	  }
	  double value = (*s_it)[*iter];
	  if( sp != 0) {
	    sp->setId((long long)((*id_it)[*iter]));
	    obj->add( scinew GeomMaterial( sp,(cmap->lookup(value).get_rep())));
	  }
	  if( drawVectors.get() == 1 && hasVectors){
	    Vector V = (*v_it)[*iter];
	    double len = V.length() * length_scale.get();
	    if(len > min_crop_length.get() ){
	      if( max_crop_length.get() == 0 ||
		  len < max_crop_length.get()){
		arrows->add( (*p_it)[*iter],
			     V*length_scale.get(),
			     outcolor, outcolor, outcolor);
	      }
	    }
	  }
	  count = 0;
	}
      }
      
      if(have_particle ){
	if( drawVectors.get() == 1 && hasVectors){
	  obj->add( arrows );
	}
	// Let's set it up so that we can pick 
	// the particle set -- Kurt Z. 12/18/98
	GeomPick *pick = scinew GeomPick( obj, this);
	ogeom->addObj(pick, "Particles");      
      }
    } else if( ps->getParticleSet()->numParticles() ) { // Particles
      GeomGroup *obj = scinew GeomGroup;
      GeomPoints *pts= scinew GeomPoints();
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
	  have_particle = true;
	  double value = (*s_it)[*iter];
	  pts->add((*p_it)[*iter], cmap->lookup(value));
	  count = 0;
	}
 	if( drawVectors.get() == 1 && hasVectors){
 	  Vector V = (*v_it)[*iter];
	  double len = V.length() * length_scale.get();
 	  if(len > min_crop_length.get() ){
	    if( max_crop_length.get() == 0 ||
		len < max_crop_length.get()){
	      arrows->add( (*p_it)[*iter],
			   V*length_scale.get(),
			   outcolor, outcolor, outcolor);
	    }
	  }
 	}
      }
      obj->add( pts );
       if( have_particle ){
	if( drawVectors.get() == 1 && hasVectors){
	  obj->add( arrows );
	}
      ogeom->addObj(obj, "Particles");      
      }
    }
    if(hasVectors) v_it++;
    if(hasTensors) t_it++;
    if(hasScale) scale_it++;
  }
}


void ParticleVis::geom_pick(GeomPickHandle pick,
			    void* userdata, GeomHandle picked_obj)
{
  cerr << "Caught stray pick event in ParticleVis!\n";
  cerr << "this = "<< this <<", pick = "<<pick.get_rep()<<endl;
  cerr << "User data = "<<userdata<<endl;
  //  cerr << "sphere index = "<<index<<endl<<endl;
  
  // This variable should not need to be initialized, because if the
  // getID fails then we don't use the value.  If it succeeds then we
  // initialize it.
  
  long long id;
  if ( picked_obj->getId(id)) {
    cerr<<"Id = "<< id <<endl;
    // Check to make sure that the ID is not bogus
    if( id != PARTICLE_FIELD_EXTRACTOR_BOGUS_PART_ID ) {
      // Check to see if we have a call back class
      if( cbClass != 0 ) {
	((ParticleFieldExtractor *)cbClass)->callback( id );
      } else {
	error("Cannot graph values, because no callback class was found.");
      } 
    } else {
      error("Particle has no id\nDoes the DataArchive have particleIDs saved?\n");
    }
  } else {
    error("Every particle should have some ID associated with it, but this one dies not.  Please save dataset and net and file a bug report to bugzilla.\n");
  }
}
  
DECLARE_MAKER(ParticleVis)
} // End namespace Uintah
