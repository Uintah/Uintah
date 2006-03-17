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


class RescaleColorMap : public Module {
public:

  //! Constructor taking [in] id as an identifier
  RescaleColorMap(GuiContext* ctx);

  virtual ~RescaleColorMap();
  virtual void execute();

protected:
  bool success_;

private:
  GuiString gFrame_;
  GuiInt    gIsFixed_;
  GuiDouble gMin_;
  GuiDouble gMax_;
  GuiInt    gMakeSymmetric_;

  ColorMapHandle cHandle_;

  pair<double,double> minmax_;
};


DECLARE_MAKER(RescaleColorMap)
RescaleColorMap::RescaleColorMap(GuiContext* ctx)
  : Module("RescaleColorMap", ctx, Filter, "Visualization", "SCIRun"),
    gFrame_(ctx->subVar("main_frame"), ""),
    gIsFixed_(ctx->subVar("isFixed"), 0),
    gMin_(ctx->subVar("min"), 0),
    gMax_(ctx->subVar("max"), 1),
    gMakeSymmetric_(ctx->subVar("makeSymmetric"), 0)
{
}

RescaleColorMap::~RescaleColorMap()
{
}

void
RescaleColorMap::execute()
{
  ColorMapHandle cHandle;
  std::vector<FieldHandle> fHandles;

  // Do this first so the ports are optional if a fixed scale is used.
  if( gIsFixed_.changed( true ) )
    inputs_changed_ = true;

  if( !getIHandle( "ColorMap",     cHandle,  true ) ) return;
  if( !getDynamicIHandle( "Field", fHandles, !gIsFixed_.get()  ) ) return;

  // Check to see if any values have changed.
  if( !cHandle_.get_rep() ||
      (gIsFixed_.get() == 0 && gMakeSymmetric_.changed( true )) ||
      (gIsFixed_.get() == 0 && (gMin_.changed( true ) ||
				gMax_.changed( true ))) ||
      inputs_changed_ ||
      execute_error_ ) {

    execute_error_ = false;

    cHandle_ = cHandle;
    cHandle_.detach();

    if( gIsFixed_.get() ) {
      cHandle_->Scale( gMin_.get(), gMax_.get());

    } else {

      // initialize the following so that the compiler will stop
      // warning us about possibly using unitialized variables
      double minv = DBL_MAX, maxv = -DBL_MAX;

      for( unsigned int i=0; i<fHandles.size(); i++ ) {
	FieldHandle fHandle = fHandles[i];

	string units;
	if( fHandle->get_property("units", units) )
	  cHandle_->set_units(units);
	  
	ScalarFieldInterfaceHandle sfi;
	VectorFieldInterfaceHandle vfi;

	if ((sfi = fHandle->query_scalar_interface(this)).get_rep()) {
	  sfi->compute_min_max(minmax_.first, minmax_.second);
	} else if ((vfi = fHandle->query_vector_interface(this)).get_rep()) {
	  vfi->compute_length_min_max(minmax_.first, minmax_.second);
	} else {
	  error("An input field is not a scalar or vector field.");
	  execute_error_ = true;
	  return;
	}

	if ( minv > minmax_.first)
	  minv = minmax_.first;

	if ( maxv < minmax_.second)
	  maxv = minmax_.second;
      }

      minmax_.first  = minv;
      minmax_.second = maxv;

      if ( gMakeSymmetric_.get() ) {
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
  setOHandle( "ColorMap",  cHandle_, true );
}

} // End namespace SCIRun
