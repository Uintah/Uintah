/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sci_values.h>

namespace SCIRun {

DECLARE_MAKER(RescaleColorMap)
RescaleColorMap::RescaleColorMap(GuiContext* ctx)
  : Module("RescaleColorMap", ctx, Filter, "Visualization", "SCIRun"),
    isFixed(ctx->subVar("isFixed")),
    min(ctx->subVar("min")),
    max(ctx->subVar("max")),
    makeSymmetric(ctx->subVar("makeSymmetric"))
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
  cmap.detach();
  if( isFixed.get() ){
    cmap->Scale(min.get(), max.get());
    port_range_type range = get_iports("Field");
    if (range.first == range.second) {
      omap->send(cmap);
      return;
    }
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
	string units;
	if (field->get_property("units", units))
	  cmap->set_units(units);
      }
      ++pi;
    }
    omap->send(cmap);
  } else {
    port_range_type range = get_iports("Field");
    if (range.first == range.second)
      return;
    port_map_type::iterator pi = range.first;
    // initialize the following so that the compiler will stop warning us about
    // possibly using unitialized variables
    double minv = MAXDOUBLE, maxv = -MAXDOUBLE;
    int have_some=0;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      if (!ifield) {
	error("Unable to initialize iport '" + to_string(pi->second) + "'.");
	return;
      }
      FieldHandle field;
      if (ifield->get(field) && field.get_rep()) {

	ScalarFieldInterfaceHandle sfi;
	string units;
	if (field->get_property("units", units))
	  cmap->set_units(units);
	if ((sfi = field->query_scalar_interface(this)).get_rep())
	{
	  sfi->compute_min_max(minmax_.first, minmax_.second);
	} else {
          error("RescaleColorMap::Not a scalar input field.");
          return;
	}
	if (!have_some || (minmax_.first < minv)) {
	  have_some=1;
	  minv=minmax_.first;
	}
	if (!have_some || (minmax_.second > maxv)) {
	  have_some=1;
	  maxv=minmax_.second;
	}
      }
      ++pi;
    }
    if (!have_some) {
      warning("No field provided! -- Color map can not be rescaled.\n"
	      "Make sure you connect a field to the field input port.");
      omap->send(cmap);
      return;
    }
    minmax_.first=minv;
    minmax_.second=maxv;
    if ( makeSymmetric.get() ) {
      float biggest = Max(Abs(minmax_.first), Abs(minmax_.second));
      minmax_.first=-biggest;
      minmax_.second=biggest;
    }
    cmap->Scale( minmax_.first, minmax_.second);
    min.set( minmax_.first );
    max.set( minmax_.second );
    omap->send(cmap);
  }
}
} // End namespace SCIRun
