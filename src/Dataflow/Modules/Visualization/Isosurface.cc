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

#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>

#include <Dataflow/Network/Module.h>

// Temporaries
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/CurveField.h>

namespace SCIRun {

DECLARE_MAKER(Isosurface)
static string surface_name("Isosurface");


Isosurface::Isosurface(GuiContext* ctx) : 
  Module("Isosurface", ctx, Filter, "Visualization", "SCIRun"), 
  gui_iso_value_(ctx->subVar("isoval")),
  gui_iso_value_min_(ctx->subVar("isoval-min")),
  gui_iso_value_max_(ctx->subVar("isoval-max")),
  gui_iso_value_typed_(ctx->subVar("isoval-typed")),
  gui_iso_value_quantity_(ctx->subVar("isoval-quantity")),
  gui_iso_quantity_range_(ctx->subVar("quantity-range")),
  gui_iso_quantity_min_(ctx->subVar("quantity-min")),
  gui_iso_quantity_max_(ctx->subVar("quantity-max")),
  gui_iso_value_list_(ctx->subVar("isoval-list")),
  gui_extract_from_new_field_(ctx->subVar("extract-from-new-field")),
  gui_use_algorithm_(ctx->subVar("algorithm")),
  gui_build_field_(ctx->subVar("build_trisurf")),
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
  last_generation_(-1)
{
}


Isosurface::~Isosurface()
{
  if (geom_id_)
  {
    GeometryOPort *ogport = (GeometryOPort*)get_oport("Geometry");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    ogport->delObj(geom_id_);
    ogport->flushViews();
    geom_id_ = 0;
  }
}


template <class FIELD>
FieldHandle append_fields(vector<FIELD *> fields)
{
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();

  unsigned int offset = 0;
  unsigned int i;
  for (i=0; i < fields.size(); i++)
  {
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    while (nitr != nitr_end)
    {
      Point p;
      imesh->get_center(p, *nitr);
      omesh->add_point(p);
      ++nitr;
    }

    typename FIELD::mesh_type::Elem::iterator eitr, eitr_end;
    imesh->begin(eitr);
    imesh->end(eitr_end);
    while (eitr != eitr_end)
    {
      typename FIELD::mesh_type::Node::array_type nodes;
      imesh->get_nodes(nodes, *eitr);
      unsigned int j;
      for (j = 0; j < nodes.size(); j++)
      {
	nodes[j] = ((unsigned int)nodes[j]) + offset;
      }
      omesh->add_elem(nodes);
      ++eitr;
    }
    
    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
    offset += (unsigned int)size;
  }

  FIELD *ofield = scinew FIELD(omesh, Field::NODE);
  offset = 0;
  for (i=0; i < fields.size(); i++)
  {
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    while (nitr != nitr_end)
    {
      double val;
      fields[i]->value(val, *nitr);
      typename FIELD::mesh_type::Node::index_type
	new_index(((unsigned int)(*nitr)) + offset);
      ofield->set_value(val, new_index);
      ++nitr;
    }

    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
    offset += (unsigned int)size;
  }

  return ofield;
}



void
Isosurface::execute()
{
  update_state(NeedData);
  FieldIPort *infield = (FieldIPort *)get_iport("Field");
  FieldHandle field;

  if (!infield)
  {
    error("Unable to initialize iport 'Field'.");
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
    if (!new_field( field )) return;
    last_generation_ = field->generation;
    if ( !gui_extract_from_new_field_.get() )
    {
      return;
    }
    // fall through and extract isosurface from the new field
  }

  // Color the surface.
  ColorMapIPort *inColorMap = (ColorMapIPort *)get_iport("Color Map");
  if (!inColorMap)
  {
    error("Unable to initialize iport 'Color Map'.");
    return;
  }
  ColorMapHandle cmap;
  const bool have_ColorMap = inColorMap->get(cmap);
  
  vector<double> isovals;
  if (gui_active_isoval_selection_tab_.get() == "0")
  { // slider / typed
    const double val = gui_iso_value_.get();
    if (val < prev_min_ || val > prev_max_)
    {
      warning("Typed isovalue out of range -- skipping isosurfacing.");
      return;
    }
    isovals.push_back(val);
  }
  else if (gui_active_isoval_selection_tab_.get() == "1")
  { // quantity
    int num=gui_iso_value_quantity_.get();
    if (num<1)
    {
      warning("Isosurface quantity must be at least one -- skipping isosurfacing.");
      return;
    }

    string range = gui_iso_quantity_range_.get();
    double qmax=prev_max_;
    double qmin=prev_min_;
    if (range == "colormap") {
      if (!have_ColorMap) {
	error("Error - I don't have a colormap");
	return;
      }
      qmin=cmap->getMin();
      qmax=cmap->getMax();
    } else if (range == "manual") {
      qmin=gui_iso_quantity_min_.get();
      qmax=gui_iso_quantity_max_.get();
    } // else we're using "field" and qmax and qmin were set above
    
    if (qmin>=qmax) {
      error("Can't use quantity tab if Min == Max");
      return;
    }
    double di=(qmax - qmin)/(num+1);
    for (int i=0; i<num; i++) 
      isovals.push_back((i+1)*di+qmin);
  }
  else if (gui_active_isoval_selection_tab_.get() == "2")
  { // list
    istringstream vlist(gui_iso_value_list_.get());
    double val;
    while(!vlist.eof())
    {
      vlist >> val;
      if (vlist.fail()) { break; }
      isovals.push_back(val);
    }
  }
  else
  {
    error("Bad active_isoval_selection_tab value");
    return;
  }

  bool build_field = gui_build_field_.get();
  vector<GeomObj *> geometries;
  vector<FieldHandle> fields;
  const TypeDescription *td = field->get_type_description();
  switch (gui_use_algorithm_.get()) {
  case 0:  // Marching Cubes
    {
      LockingHandle<MarchingCubesAlg> mc_alg;
      CompileInfoHandle ci = MarchingCubesAlg::get_compile_info(td);
      if (!module_dynamic_compile(ci, mc_alg))
      {
	error( "Marching Cubes can not work with this field.");
	return;
      }
      mc_alg->set_np( gui_np_.get() ); 
      if ( gui_np_.get() > 1 )
      {
	build_field = false;
      }
      mc_alg->set_field( field.get_rep() );
      for (unsigned int iv=0; iv<isovals.size(); iv++)
      {
	mc_alg->search( isovals[iv], build_field);
	geometries.push_back( mc_alg->get_geom() );
	fields.push_back( mc_alg->get_field() );
      }
    }
    break;
  case 1:  // Noise
    {
      LockingHandle<NoiseAlg> noise_alg;
      if (! noise_alg.get_rep())
      {
	CompileInfoHandle ci =
	  NoiseAlg::get_compile_info(td,
				     field->data_at() == Field::CELL,
				     field->data_at() == Field::FACE);
	if (! module_dynamic_compile(ci, noise_alg))
	{
	  error( "NOISE can not work with this field.");
	  return;
	}
	noise_alg->set_field(field.get_rep());
      }
      for (unsigned int iv=0; iv<isovals.size(); iv++)
      {
	geometries.push_back(noise_alg->search(isovals[iv], build_field));
	fields.push_back(noise_alg->get_field());
      }
    }
    break;

  case 2:  // View Dependent
    {
      LockingHandle<SageAlg> sage_alg;
      {
	CompileInfoHandle ci = SageAlg::get_compile_info(td);
	if (! module_dynamic_compile(ci, sage_alg))
	{
	  error( "SAGE can not work with this field.");
	  return;
	}
	sage_alg->set_field(field.get_rep());
      } 
      for (unsigned int iv=0; iv<isovals.size(); iv++)
      {
	GeomGroup *group = new GeomGroup;
	GeomPoints *points = new GeomPoints();
	sage_alg->search(isovals[0], group, points);
	geometries.push_back( group );
      }
    }
    break;
  default: // Error
    error("Unknown Algorithm requested.");
    return;
  }

  // Merged send_results.
  GeomGroup *geom = new GeomGroup;;

  for (unsigned int iv=0; iv<isovals.size(); iv++)
  {
    MaterialHandle matl;

    if (have_ColorMap)
    {
      matl= cmap->lookup(isovals[iv]);
    }
    else
    {
      matl = scinew Material(Color(gui_color_r_.get(),
				   gui_color_g_.get(),
				   gui_color_b_.get()));
    }
    if (geometries[iv]) 
    {
      geom->add(scinew GeomMaterial( geometries[iv] , matl ));
    }
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  if (!ogeom)
  {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  // Stop showing the previous surface.
  if ( geom_id_ )
  {
    ogeom->delObj( geom_id_ );
  }

  if (!geom->size())
  {
    geom_id_=0;
    return;
  }

  // Send to viewer.
  geom_id_ = ogeom->addObj( geom, surface_name);

  // Output surface.
  if (build_field && fields.size() && fields[0].get_rep())
  {
    FieldOPort *osurf = (FieldOPort *)get_oport("Surface");
    if (!osurf)
    {
      error("Unable to initialize oport 'Surface'.");
      return;
    }

    if (fields.size() == 1)
    {
      osurf->send(fields[0]);
    }
    else
    {
      vector<TriSurfField<double> *> tfields(fields.size());
      unsigned int i;
      for (i=0; i < fields.size(); i++)
      {
	tfields[i] = dynamic_cast<TriSurfField<double> *>(fields[i].get_rep());
      }

      vector<CurveField<double> *> cfields(fields.size());
      for (i=0; i < fields.size(); i++)
      {
	cfields[i] = dynamic_cast<CurveField<double> *>(fields[i].get_rep());
      }

      if (tfields[0])
      {
	osurf->send(append_fields(tfields));
      }
      else if (cfields[0])
      {
	osurf->send(append_fields(cfields));
      }
      else
      {
	osurf->send(fields[0]);
      }
    }
  }
}


bool
Isosurface::new_field( FieldHandle field )
{
  const string type = field->get_type_description()->get_name();

  ScalarFieldInterfaceHandle sfi = field->query_scalar_interface(this);
  if (!sfi.get_rep())
  {
    error("Input field does not contain scalar data.");
    return false;
  }

  // Set min/max
  pair<double, double> minmax;
  sfi->compute_min_max(minmax.first, minmax.second);
  if (minmax.first != prev_min_ || minmax.second != prev_max_)
  {
    gui_iso_value_min_.set(minmax.first);
    gui_iso_value_max_.set(minmax.second);
    prev_min_ = minmax.first;
    prev_max_ = minmax.second;
  }
  return true;
}

} // End namespace SCIRun
