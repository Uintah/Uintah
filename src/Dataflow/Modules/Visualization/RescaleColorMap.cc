/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  RescaleColorMap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Visualization/RescaleColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

DECLARE_MAKER(RescaleColorMap)
RescaleColorMap::RescaleColorMap(GuiContext* ctx)
  : Module("RescaleColorMap", ctx, Filter, "Visualization", "SCIRun"),
    isFixed(ctx->subVar("isFixed")),
    min(ctx->subVar("min")),
    max(ctx->subVar("max"))
{
}

RescaleColorMap::~RescaleColorMap()
{
}

void
RescaleColorMap::execute()
{
  ColorMapHandle cmap;
  ColorMapIPort *imap = (ColorMapIPort *)get_iport("ColorMap");
  ColorMapOPort *omap = (ColorMapOPort *)get_oport("ColorMap");
  if (!imap) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!omap) {
    error("Unable to initialize oport 'ColorMap'.");
    return;
  }
  if(!imap->get(cmap)) {
    return;
  }
  cmap = new ColorMap(*cmap.get_rep());
  if( isFixed.get() ){
    cmap->Scale(min.get(), max.get());
    port_range_type range = get_iports("Field");
    if (range.first == range.second)
      return;
    port_map_type::iterator pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      if (!ifield) {
	error("Unable to initialize iport '" + to_string(pi->second) + "'.");
	return;
      }
      FieldHandle field;
      if (ifield->get(field) && field.get_rep()) {

	//ScalarFieldInterface *sfi = field->query_scalar_interface();
	//VectorFieldInterface *vfi = field->query_vector_interface();
	string units;
	if (field->get_property("units", units))
	  cmap->units=units;
      }
      ++pi;
    }
  } else {
    port_range_type range = get_iports("Field");
    if (range.first == range.second)
      return;
    port_map_type::iterator pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      if (!ifield) {
	error("Unable to initialize iport '" + to_string(pi->second) + "'.");
	return;
      }
      FieldHandle field;
      if (ifield->get(field) && field.get_rep()) {

	ScalarFieldInterface *sfi;
	VectorFieldInterface *vfi;
	string units;
	if (field->get_property("units", units))
	  cmap->units=units;
	if ((sfi = field->query_scalar_interface(this)))
	{
	  sfi->compute_min_max(minmax_.first, minmax_.second);
	}
	else if ((vfi = field->query_vector_interface(this)))
	{
	  // get minmax of the vector field.
	  static pair<Vector, Vector> minmax;
	  if ( !field->get_property("minmax", minmax)) {
	    vfi->compute_min_max(minmax.first, minmax.second);
	    // cache this potentially expensive to compute value.
	    field->set_property("minmax", minmax, true);
	  }
	  minmax_.first = 0.0;
	  minmax_.second = minmax.second.length();
	} else {
          error("RescaleColorMap::Not a scalar or vector input field.");
          return;
	}
	cmap->Scale( minmax_.first, minmax_.second);
	min.set( minmax_.first );
	max.set( minmax_.second );
      }
      ++pi;
    }
  }
  omap->send(cmap);
}
} // End namespace SCIRun
