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
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/DispatchScalar1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

/**************************************
CLASS
   RescaleColorMap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION
   RescaleColorMap.h
   Written by:

     Kurt Zimmerman<br>
     Department of Computer Science<br>
     University of Utah<br>
     June 1999

     Copyright (C) 1998 SCI Group

KEYWORDS
   ColorMap, Transfer Function

DESCRIPTION
   This module takes a color map and some data or vector field
   and scales the map to fit that field.

****************************************/

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


void RescaleColorMap::get_minmax(FieldHandle field) {
  dispatch_scalar1(field, dispatch_minmax);
}

void
RescaleColorMap::execute()
{
  ColorMapHandle cmap;
  ColorMapIPort *imap = (ColorMapIPort *)get_iport("ColorMap");
  ColorMapOPort *omap = (ColorMapOPort *)get_oport("ColorMap");
  if(!imap->get(cmap)) {
    return;
  }
  cmap = new ColorMap(*cmap.get_rep());
  if( isFixed.get() ){
    cmap->Scale(min.get(), max.get());
  } else {
    dynamic_port_range range = get_iports("Field");
    port_iter pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      FieldHandle field;
      if (ifield->get(field)) {
        if ( !field->is_scalar() ) {
          error("Not a scalar input field.");
          return;
        }
	get_minmax(field);
	if (!success_) {
	  error("Can not compute minmax for input field.");
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
