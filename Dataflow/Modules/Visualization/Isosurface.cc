//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : Isosurface.cc
//    Author : Yarden Livnat
//    Date   : Fri Jun 15 16:38:02 2001


#include <map>
#include <iostream>
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

#include <Dataflow/Modules/Visualization/Isosurface.h>

//#include <typeinfo>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/MaskedTetVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/TriSurfField.h>

#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>

#include <Dataflow/Network/Module.h>

namespace SCIRun {

DECLARE_MAKER(Isosurface)
//static string module_name("Isosurface");
static string surface_name("Isosurface");
static string widget_name("Isosurface");

Isosurface::Isosurface(GuiContext* ctx) : 
  Module("Isosurface", ctx, Filter, "Visualization", "SCIRun"), 
  gui_iso_value(ctx->subVar("isoval")),
  gui_iso_value_min(ctx->subVar("isoval-min")),
  gui_iso_value_max(ctx->subVar("isoval-max")),
  extract_from_new_field(ctx->subVar("extract-from-new-field")),
  use_algorithm(ctx->subVar("algorithm")),
  build_trisurf_(ctx->subVar("build_trisurf")),
  np_(ctx->subVar("np")),
  active_tab_(ctx->subVar("active_tab")),
  update_type_(ctx->subVar("update_type")),
  color_r_(ctx->subVar("color-r")),
  color_g_(ctx->subVar("color-g")),
  color_b_(ctx->subVar("color-b"))
{
  geom_id=0;
  
  prev_min=prev_max=0;
  last_generation = -1;
  mc_alg_ = 0;
  noise_alg_ = 0;
  sage_alg_ = 0;
}

Isosurface::~Isosurface()
{
}

void Isosurface::execute()
{
  update_state(NeedData);
  infield = (FieldIPort *)get_iport("Field");
  inColorMap = (ColorMapIPort *)get_iport("Color Map");
  osurf = (FieldOPort *)get_oport("Surface");
  ogeom = (GeometryOPort *)get_oport("Geometry");
  FieldHandle field;

  if (!infield) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!inColorMap) {
    error("Unable to initialize iport 'Color Map'.");
    return;
  }
  if (!osurf) {
    error("Unable to initialize oport 'Surface'.");
    return;
  }
  if (!ogeom) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  infield->get(field);
  if(!field.get_rep()){
    return;
  }

  update_state(JustStarted);

  if ( field->generation != last_generation ) {
    // new field
    new_field( field );
    last_generation = field->generation;
    if ( !extract_from_new_field.get() )
      return;

    // fall through and extract isosurface from the new field
  }

  // Color the surface
  //   have_colorfield=incolorfield->get(colorfield);
  have_ColorMap=inColorMap->get(cmap);
  
  iso_value = gui_iso_value.get();
  trisurf_mesh_ = 0;
  build_trisurf = build_trisurf_.get();
  const TypeDescription *td = field->get_type_description();
  switch (use_algorithm.get()) {
  case 0:  // Marching Cubes
    {
      if (! mc_alg_.get_rep()) {
	CompileInfo *ci = MarchingCubesAlg::get_compile_info(td);
	if (! DynamicLoader::scirun_loader().get(*ci, mc_alg_)) {
	  error( "Marching Cubes can not work with this field.");
	  return;
	}
      }
      // mc_alg_ should be set now
      MarchingCubesAlg *mc = dynamic_cast<MarchingCubesAlg*>(mc_alg_.get_rep());
      mc->set_np( np_.get() ); 
      if ( np_.get() > 1 )
	build_trisurf = false;
      mc->set_field( field.get_rep() );
      mc->search( iso_value, build_trisurf);
      surface = mc->get_geom();
      trisurf_mesh_ = mc->get_trisurf();
    }
    break;
  case 1:  // Noise
    {
      if (! noise_alg_.get_rep()) {
	CompileInfo *ci = NoiseAlg::get_compile_info(td);
	if (! DynamicLoader::scirun_loader().get(*ci, noise_alg_)) {
	  error( "NOISE can not work with this field.");
	  return;
	}
	NoiseAlg *noise = dynamic_cast<NoiseAlg*>(noise_alg_.get_rep());
	noise->set_field(field.get_rep());
      }
      NoiseAlg *noise = dynamic_cast<NoiseAlg*>(noise_alg_.get_rep());
      surface = noise->search(iso_value, build_trisurf);
      trisurf_mesh_ = noise->get_trisurf();
    }
    break;
  case 2:  // View Dependent
    {
      if (! sage_alg_.get_rep()) {
	CompileInfo *ci = SageAlg::get_compile_info(td);
	if (! DynamicLoader::scirun_loader().get(*ci, sage_alg_)) {
	  error( "SAGE can not work with this field.");
	  return;
	}
	SageAlg *sage = dynamic_cast<SageAlg*>(sage_alg_.get_rep());
	sage->set_field(field.get_rep());
      } 
      SageAlg *sage = dynamic_cast<SageAlg*>(sage_alg_.get_rep());
      GeomGroup *group = new GeomGroup;
      GeomPts *points = new GeomPts(1000);
      sage->search(iso_value, group, points);
      surface = group;
    }
    break;
  default: // Error
    error("Unknown Algorithm requested.");
    return;
  }
  send_results();
}

void
Isosurface::send_results()
{
  // stop showing the prev. surface
  if ( geom_id )
    ogeom->delObj( geom_id );

  // if no surface than we are done
  if( !surface ) {
    geom_id=0;
    
    return;
  }

  GeomObj *geom;
  
  if(have_ColorMap /* && !have_colorfield */){
    // Paint entire surface based on ColorMap
    geom=scinew GeomMaterial( surface , cmap->lookup(iso_value) );
//   } else if(have_ColorMap && have_colorfield){
//     geom = surface;     // Nothing - done per vertex
  } else {
    Color color(color_r_.get(), color_g_.get(), color_b_.get());
    MaterialHandle matl = scinew Material(color);
    geom=scinew GeomMaterial( surface, matl ); // Default material
  }

  // send to viewer
  geom_id=ogeom->addObj( geom, surface_name);

  // output surface
  if (build_trisurf && trisurf_mesh_.get_rep()) {
    TriSurfField<double> *ts = new TriSurfField<double>(trisurf_mesh_, Field::NODE);
    vector<double>::iterator iter = ts->fdata().begin();
    while (iter != ts->fdata().end()) { (*iter)=iso_value; ++iter; }
    FieldHandle fH(ts);
    osurf->send(fH);
  }
}

void
Isosurface::new_field( FieldHandle &field )
{
  const string type = field->get_type_description()->get_name();

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  if (! sfi) {
    error("Not a scalar input field.");
    return;
  }

  pair<double, double> minmax;
  if ( !field->get_property("minmax", minmax)) {
    sfi->compute_min_max(minmax.first, minmax.second);
    // cache this potentially expensive to compute value.
    field->set_property("minmax", minmax, true);
  }
  
  // reset the GUI

  // 1: field info
  ostringstream info;
  info << id << " set_info {" << type << "} " << field->generation;
  gui->execute(info.str().c_str());

  // 2: min/max
  if(minmax.first != prev_min || minmax.second != prev_max){
    ostringstream str;
    str << id << " set_minmax " << minmax.first << " " << minmax.second;
    gui->execute(str.str().c_str());
    prev_min = minmax.first;
    prev_max = minmax.second;
  }

  // delete any algorithms created for the previous field.
  if (mc_alg_.get_rep()) { 
    MarchingCubesAlg *mc = dynamic_cast<MarchingCubesAlg*>(mc_alg_.get_rep());
    mc->release(); 
    mc_alg_ = 0;
  }
  if (noise_alg_.get_rep()) { 
    NoiseAlg *noise = dynamic_cast<NoiseAlg*>(noise_alg_.get_rep());
    noise->release(); 
    noise_alg_ = 0;
  }
  if (sage_alg_.get_rep()) { 
    SageAlg *sage = dynamic_cast<SageAlg*>(sage_alg_.get_rep());
    sage->release(); 
    sage_alg_ = 0;
  }
}

} // End namespace SCIRun
