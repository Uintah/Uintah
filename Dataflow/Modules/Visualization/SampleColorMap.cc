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
 *  SampleColorMap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Visualization/SampleColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

DECLARE_MAKER(SampleColorMap)
SampleColorMap::SampleColorMap(GuiContext* ctx)
  : Module("SampleColorMap", ctx, Filter, "Visualization", "SCIRun"),
    min(ctx->subVar("min")),
    max(ctx->subVar("max"))
{
}

SampleColorMap::~SampleColorMap()
{
}

void
SampleColorMap::execute()
{
  ColorMapHandle cmap;
  ColorMapIPort *imap = (ColorMapIPort *)get_iport("ColorMap");
  if (!imap) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }

  if(!imap->get(cmap)) {
    return;
  }

  cmap = new ColorMap(*cmap.get_rep());

  ostringstream ostr;
  for(int i =0; i < cmap->size(); i++){
    Color col = cmap->rawRampColor[i];
    ostr << col.r()<<" ";
    ostr << col.g()<<" ";
    ostr << col.b()<<" ";
  }
  min.set(cmap->getMin());
  max.set(cmap->getMax());
  
  gui->execute( id + " setColorMap " + ostr.str() );
  gui->execute( id + " redraw");

}
} // End namespace SCIRun
