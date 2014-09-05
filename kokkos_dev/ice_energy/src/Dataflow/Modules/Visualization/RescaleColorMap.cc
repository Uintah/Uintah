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
    gFrame_(ctx->subVar("main_frame")),
    gIsFixed_(ctx->subVar("isFixed")),
    gMin_(ctx->subVar("min")),
    gMax_(ctx->subVar("max")),
    gMakeSymmetric_(ctx->subVar("makeSymmetric")),
    cGeneration_(-1),
    error_(false)
{
}

RescaleColorMap::~RescaleColorMap()
{
}

void
RescaleColorMap::execute()
{
  ColorMapHandle cHandle;
  ColorMapIPort *cmap_port = (ColorMapIPort *)get_iport("ColorMap");
 
  // The colormap input is required.
  if (!cmap_port->get(cHandle) || !(cHandle.get_rep())) {
    error( "No colormap handle or representation" );
    return;
  }

  bool update = false;

  // Check to see if the input colormap has changed.
  if( cGeneration_ != cHandle->generation )
  {
    cGeneration_ = cHandle->generation;
    update = true;
  }
 
  string units;
  unsigned int nFields = 0;
  std::vector<FieldHandle> fHandles;

  port_range_type range = get_iports("Field");
  if (range.first != range.second) {
    port_map_type::iterator pi = range.first;
    
    while (pi != range.second) {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      
      // Increment here!  We do this because last one is always
      // empty so we can test for it before issuing empty warning.
      ++pi;

      FieldHandle fHandle;
      if (ifield->get(fHandle) && fHandle.get_rep()) {

	fHandles.push_back(fHandle);
	fHandle->get_property("units", units);

	if( nFields == fGeneration_.size() ) {
	  fGeneration_.push_back( fHandle->generation );
	  update = true;
	} else if ( fGeneration_[nFields] != fHandle->generation ) {
	  fGeneration_[nFields] = fHandle->generation;
	  update = true;
	}

	nFields++;
      } else if (pi != range.second) {
	warning("Input port " + to_string(nFields) + " contained no data.");
	return;
      }
    }
  }

  while( fGeneration_.size() > nFields ) {
    update = true;
    fGeneration_.pop_back();
  }

  int isFixed = gIsFixed_.get();
  double min = gMin_.get();
  double max = gMax_.get();
  int makeSymmetric = gMakeSymmetric_.get();

  if( isFixed_ != isFixed ||
      min_  != min  ||
      max_  != max  ||
      makeSymmetric_ != makeSymmetric ) {

    isFixed_ = isFixed;
    min_ = min;
    max_ = max;
    makeSymmetric_ = makeSymmetric;

    update = true;
  }

  if( !cHandle_.get_rep() ||
      update ||
      error_ ) {

    error_ = false;
    cHandle_ = cHandle;

    cHandle_.detach();

    if( units.length() != 0 )
      cHandle_->set_units(units);

    if( isFixed ){
      cHandle_->Scale(min, max);

    } else {

      if (fHandles.size() == 0) {
	error("No field(s) provided -- Color map can not be rescaled.");
	error_ = true;
	return;
      }

      // initialize the following so that the compiler will stop
      // warning us about possibly using unitialized variables
      double minv = MAXDOUBLE, maxv = -MAXDOUBLE;

      for( unsigned int i=0; i<fHandles.size(); i++ ) {
	FieldHandle fHandle = fHandles[i];

	ScalarFieldInterfaceHandle sfi;

	if ((sfi = fHandle->query_scalar_interface(this)).get_rep()) {
	  sfi->compute_min_max(minmax_.first, minmax_.second);
	} else {
	  error("An input field is not a scalar field.");
	  error_ = true;
	  return;
	}

	if ( minv > minmax_.first)
	  minv=minmax_.first;

	if ( maxv < minmax_.second)
	  maxv=minmax_.second;
      }

      minmax_.first  = minv;
      minmax_.second = maxv;

      if ( makeSymmetric_ ) {
	float biggest = Max(Abs(minmax_.first), Abs(minmax_.second));
	minmax_.first  = -biggest;
	minmax_.second =  biggest;
      }

      cHandle_->Scale( minmax_.first, minmax_.second);
      gMin_.set( minmax_.first );
      gMax_.set( minmax_.second );
    }
  }

  // Send the data downstream
  if( cHandle_.get_rep() )
  {
    ColorMapOPort *ocolormap_port = (ColorMapOPort *) get_oport("ColorMap");
    ocolormap_port->send( cHandle_ );
  }
}
} // End namespace SCIRun
