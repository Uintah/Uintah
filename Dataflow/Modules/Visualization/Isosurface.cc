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
  gui_iso_value_(ctx->subVar("isoval")),
  gui_iso_value_min_(ctx->subVar("isoval-min")),
  gui_iso_value_max_(ctx->subVar("isoval-max")),
  gui_iso_value_typed_(ctx->subVar("isoval-typed")),
  gui_iso_value_quantity_(ctx->subVar("isoval-quantity")),
  gui_extract_from_new_field_(ctx->subVar("extract-from-new-field")),
  gui_use_algorithm_(ctx->subVar("algorithm")),
  gui_build_trisurf_(ctx->subVar("build_trisurf")),
  gui_np_(ctx->subVar("np")),
  gui_active_isoval_selection_tab_(ctx->subVar("active-isoval-selection-tab")),
  gui_active_tab_(ctx->subVar("active_tab")),
  gui_update_type_(ctx->subVar("update_type")),
  gui_color_r_(ctx->subVar("color-r")),
  gui_color_g_(ctx->subVar("color-g")),
  gui_color_b_(ctx->subVar("color-b")),
  geom_id_(0),
  prev_min_(0),
  prev_max_(0),
  last_generation_(-1),
  mc_alg_(0),
  noise_alg_(0),
  sage_alg_(0)
{
}


Isosurface::~Isosurface()
{
}


void
Isosurface::execute()
{
  update_state(NeedData);
  FieldIPort *infield = (FieldIPort *)get_iport("Field");
  ColorMapIPort *inColorMap = (ColorMapIPort *)get_iport("Color Map");
  FieldHandle field;

  if (!infield)
  {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!inColorMap)
  {
    error("Unable to initialize iport 'Color Map'.");
    return;
  }

  infield->get(field);
  if(!field.get_rep())
  {
    return;
  }

  update_state(JustStarted);

  if ( field->generation != last_generation_ )
  {
    // new field
    new_field( field );
    last_generation_ = field->generation;
    if ( !gui_extract_from_new_field_.get() )
      return;

    // fall through and extract isosurface from the new field
  }

  // Color the surface
  have_ColorMap_ = inColorMap->get(cmap_);

  isovals_.resize(0);
  if (gui_active_isoval_selection_tab_.get() == "0")
  { // slider
    isovals_.push_back(gui_iso_value_.get());
  }
  else if (gui_active_isoval_selection_tab_.get() == "1")
  { // typed
    const double val = gui_iso_value_typed_.get();
    if (val < prev_min_ || val > prev_max_)
    {
      warning("Typed isovalue out of range -- skipping isosurfacing.");
      return;
    }
    isovals_.push_back(val);
  }
  else if (gui_active_isoval_selection_tab_.get() == "2")
  { // quantity
    int num=gui_iso_value_quantity_.get();
    if (num<1)
    {
      warning("Isosurface quantity must be at least one -- skipping isosurfacing.");
      return;
    }
    double di=(prev_max_ - prev_min_)/(num+1);
    for (int i=0; i<num; i++) 
      isovals_.push_back((i+1)*di+prev_min_);
  }
  else
  {
    error("Bad active_isoval_selection_tab value");
    return;
  }

  surface_.resize(0);
  trisurf_mesh_ = 0;
  build_trisurf_ = gui_build_trisurf_.get();
  const TypeDescription *td = field->get_type_description();
  switch (gui_use_algorithm_.get()) {
  case 0:  // Marching Cubes
    {
      if (! mc_alg_.get_rep())
      {
	CompileInfo *ci = MarchingCubesAlg::get_compile_info(td);
	if (!module_dynamic_compile(*ci, mc_alg_))
	{
	  error( "Marching Cubes can not work with this field.");
	  return;
	}
      }
      mc_alg_->set_np( gui_np_.get() ); 
      if ( gui_np_.get() > 1 )
      {
	build_trisurf_ = false;
      }
      mc_alg_->set_field( field.get_rep() );
      for (unsigned int iv=0; iv<isovals_.size(); iv++)
      {
	mc_alg_->search( isovals_[iv], build_trisurf_);
	surface_.push_back( mc_alg_->get_geom() );
      }
      // if multiple isosurfaces, just send last one for Field output
      trisurf_mesh_ = mc_alg_->get_field();
    }
    break;
  case 1:  // Noise
    {
      if (! noise_alg_.get_rep())
      {
	CompileInfo *ci = NoiseAlg::get_compile_info(td);
	if (! module_dynamic_compile(*ci, noise_alg_))
	{
	  error( "NOISE can not work with this field.");
	  return;
	}
	noise_alg_->set_field(field.get_rep());
      }
      for (unsigned int iv=0; iv<isovals_.size(); iv++)
      {
	surface_.push_back(noise_alg_->search(isovals_[iv], build_trisurf_));
      }
      // if multiple isosurfaces, just send last one for Field output
      trisurf_mesh_ = noise_alg_->get_field();
    }
    break;
  case 2:  // View Dependent
    {
      if (! sage_alg_.get_rep())
      {
	CompileInfo *ci = SageAlg::get_compile_info(td);
	if (! module_dynamic_compile(*ci, sage_alg_))
	{
	  error( "SAGE can not work with this field.");
	  return;
	}
	sage_alg_->set_field(field.get_rep());
      } 
      for (unsigned int iv=0; iv<isovals_.size(); iv++)
      {
	GeomGroup *group = new GeomGroup;
	GeomPts *points = new GeomPts(1000);
	sage_alg_->search(isovals_[0], group, points);
	surface_.push_back( group );
      }
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
  GeomGroup *geom = new GeomGroup;;
  
  for (unsigned int iv=0; iv<isovals_.size(); iv++)
  {
    MaterialHandle matl;
    if (have_ColorMap_) 
    {
      matl= cmap_->lookup(isovals_[iv]);
    }
    else
    {
      matl = scinew Material(Color(gui_color_r_.get(),
				   gui_color_g_.get(),
				   gui_color_b_.get()));
    }
    if (surface_[iv]) 
    {
      geom->add(scinew GeomMaterial( surface_[iv] , matl ));
    }
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  if (!ogeom)
  {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  // stop showing the prev. surface
  if ( geom_id_ )
  {
    ogeom->delObj( geom_id_ );
  }

  if (!geom->size())
  {
    geom_id_=0;
    return;
  }

  // send to viewer
  geom_id_ = ogeom->addObj( geom, surface_name);

  // output surface
  if (build_trisurf_ && trisurf_mesh_.get_rep())
  {
    FieldOPort *osurf = (FieldOPort *)get_oport("Surface");
    if (!osurf)
    {
      error("Unable to initialize oport 'Surface'.");
      return;
    }
    osurf->send(trisurf_mesh_);
  }
}


void
Isosurface::new_field( FieldHandle &field )
{
  const string type = field->get_type_description()->get_name();

  ScalarFieldInterface *sfi = field->query_scalar_interface(this);
  if (! sfi)
  {
    error("Not a scalar input field.");
    return;
  }

  pair<double, double> minmax;
  if ( !field->get_property("minmax", minmax))
  {
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
  if (minmax.first != prev_min_ || minmax.second != prev_max_)
  {
    ostringstream str;
    str << id << " set_minmax " << minmax.first << " " << minmax.second;
    gui->execute(str.str().c_str());
    prev_min_ = minmax.first;
    prev_max_ = minmax.second;
  }

  // delete any algorithms created for the previous field.
  if (mc_alg_.get_rep())
  { 
    MarchingCubesAlg *mc = dynamic_cast<MarchingCubesAlg*>(mc_alg_.get_rep());
    mc->release(); 
    mc_alg_ = 0;
  }
  if (noise_alg_.get_rep())
  { 
    NoiseAlg *noise = dynamic_cast<NoiseAlg*>(noise_alg_.get_rep());
    noise->release(); 
    noise_alg_ = 0;
  }
  if (sage_alg_.get_rep())
  { 
    SageAlg *sage = dynamic_cast<SageAlg*>(sage_alg_.get_rep());
    sage->release(); 
    sage_alg_ = 0;
  }
}

} // End namespace SCIRun
