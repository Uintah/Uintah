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
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

extern "C" Module* make_RescaleColorMap(const string& id) {
  return new RescaleColorMap(id);
}

RescaleColorMap::RescaleColorMap(const string& id)
  : Module("RescaleColorMap", id, Filter, "Visualization", "SCIRun"),
    isFixed("isFixed", id, this),
    min("min", id, this ),
    max("max", id, this)
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
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omap) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if(!imap->get(cmap)) {
    return;
  }
  cmap = new ColorMap(*cmap.get_rep());
  if( isFixed.get() ){
    cmap->Scale(min.get(), max.get());
  } else {
    dynamic_port_range range = get_iports("Field");
    if (range.first == range.second)
      return;
    port_iter pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      if (!ifield) {
	postMessage("Unable to initialize "+name+"'s iport\n");
	return;
      }
      FieldHandle field;
      if (ifield->get(field)) {

	ScalarFieldInterface *sfi = field->query_scalar_interface();
	VectorFieldInterface *vfi = field->query_vector_interface();
	if (sfi) {
	  sfi->compute_min_max(minmax_.first, minmax_.second);
	} else if (vfi) {
	  // get minmax of the vector field.
	  static pair<Vector, Vector> minmax;
	  if ( !field->get("minmax", minmax)) {
	    vfi->compute_min_max(minmax.first, minmax.second);
	    // cache this potentially expensive to compute value.
	    field->store("minmax", minmax, true);
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
