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
#include <Core/Datatypes/Field.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
using std::endl;

#include <Dataflow/Modules/Visualization/RescaleColorMap.h>

namespace SCIRun {



extern "C" Module* make_RescaleColorMap(const clString& id) {
  return new RescaleColorMap(id);
}

RescaleColorMap::RescaleColorMap(const clString& id)
: Module("RescaleColorMap", id, Filter),
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

void
RescaleColorMap::execute()
{
#if 0 //FIX_ME with new fields.
    ColorMapHandle cmap;
    if(!imap->get(cmap)) {
	return;
    }
    if( isFixed.get() ){
      cmap->Scale(min.get(), max.get());
    } else {
      for(int i=0;i<fieldports.size()-1;i++){
        FieldHandle sfield;
        if(fieldports[i]->get(sfield)){
	  double min;
	  double max;
	  sfield->get_minmax(min, max);
	  cmap->Scale( min, max);
	  this->min.set( min );
	  this->max.set( max );
	}
      }
    }
    omap->send(cmap);
#endif
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

