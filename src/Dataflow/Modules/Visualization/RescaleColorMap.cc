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
#include <Dataflow/Modules/Visualization/RescaleColorMap.h>
#include <iostream>
using std::cerr;
using std::endl;

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

extern "C" Module* make_RescaleColorMap(const clString& id) {
  return new RescaleColorMap(id);
}

RescaleColorMap::RescaleColorMap(const clString& id)
  : Module("RescaleColorMap", id, Filter, "Visualization", "SCIRun"),
    isFixed("isFixed", id, this),
    min("min", id, this ),
    max("max", id, this)
{
    // Create the output port
  omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
  add_oport(omap);

    // Create the input ports
  imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
  add_iport(imap);
  FieldIPort* ifield=scinew FieldIPort(this, "ScalarField",
						     FieldIPort::Atomic);
  add_iport(ifield);
  fieldports.add(ifield);
}

RescaleColorMap::~RescaleColorMap()
{
}

template<class F>
void RescaleColorMap::dispatch_minmax(F *f) {
  success_ = field_minmax(*f, minmax_);
}

void
RescaleColorMap::execute()
{
  ColorMapHandle cmap;
  if(!imap->get(cmap)) {
    return;
  }
  cmap = new ColorMap(*cmap.get_rep());
  if( isFixed.get() ){
    cmap->Scale(min.get(), max.get());
  } else {
    for(int i=0;i<fieldports.size()-1;i++){
      FieldHandle field;
      if(fieldports[i]->get(field)){
	string type = field->get_type_name();
	cerr << "field type = " << type << endl;

	if ( !field->is_scalar() ) {
	  cerr << "rescale colormap: not a scalar field\n";
	  return;
	}

	dispatch_scalar1(field, dispatch_minmax);

	if (!success_) {
	  cerr << "rescale colormap: can not compute minmax for field\n";
	  return;
	}

	cmap->Scale( minmax_.first, minmax_.second);
	min.set( minmax_.first );
	max.set( minmax_.second );
      }
    }
  }
  omap->send(cmap);
}

void 
RescaleColorMap::connection(ConnectionMode mode, int which_port, int)
{
    if(which_port > 0){
        if(mode==Disconnected){
	    remove_iport(which_port);
	    fieldports.remove(which_port-1);
	} else {
	    FieldIPort* p=scinew FieldIPort(this, "Field",
							FieldIPort::Atomic);
	    fieldports.add(p);
	    add_iport(p);
	}
    }
}

} // End namespace SCIRun

