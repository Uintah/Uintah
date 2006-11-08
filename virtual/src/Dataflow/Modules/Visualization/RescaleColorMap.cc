/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sci_values.h>

namespace SCIRun {


class RescaleColorMap : public Module {
public:

  //! Constructor taking [in] get_id() as an identifier
  RescaleColorMap(GuiContext* ctx);

  virtual ~RescaleColorMap();
  virtual void execute();

protected:
  bool success_;

private:
  GuiString gui_frame_;
  GuiInt    gui_is_fixed_;
  GuiDouble gui_min_;
  GuiDouble gui_max_;
  GuiInt    gui_make_symmetric_;

  ColorMapHandle colormap_output_handle_;

  pair<double,double> minmax_;
};


DECLARE_MAKER(RescaleColorMap)
RescaleColorMap::RescaleColorMap(GuiContext* ctx)
  : Module("RescaleColorMap", ctx, Filter, "Visualization", "SCIRun"),
    gui_frame_(get_ctx()->subVar("main_frame"), ""),
    gui_is_fixed_(get_ctx()->subVar("isFixed"), 0),
    gui_min_(get_ctx()->subVar("min"), 0),
    gui_max_(get_ctx()->subVar("max"), 1),
    gui_make_symmetric_(get_ctx()->subVar("makeSymmetric"), 0)
{
}

RescaleColorMap::~RescaleColorMap()
{
}

void
RescaleColorMap::execute()
{
  ColorMapHandle colormap_input_handle;
  std::vector<FieldHandle> field_input_handles;

  // Do this first so the ports are optional if a fixed scale is used.
  if( gui_is_fixed_.changed( true ) )
    inputs_changed_ = true;

  if( !get_input_handle( "ColorMap", colormap_input_handle, true ) ) return;

  if (!get_dynamic_input_handles("Field", field_input_handles,
				 !gui_is_fixed_.get() ) ) 
  {
    if (!gui_is_fixed_.get()) return;
  }
  // Check to see if any values have changed.
  if( inputs_changed_ || !colormap_output_handle_.get_rep() ||
      (gui_is_fixed_.get() == 0 && gui_make_symmetric_.changed( true )) ||
      (gui_is_fixed_.get() == 1 && (gui_min_.changed( true ) ||
      gui_max_.changed( true ))) ||execute_error_ ) 
  {

    update_state(Executing);

    execute_error_ = false;

    colormap_output_handle_ = colormap_input_handle;
    colormap_output_handle_.detach();

    if( gui_is_fixed_.get() ) 
    {
      colormap_output_handle_->Scale( gui_min_.get(), gui_max_.get());
    } 
    else 
    {

      // initialize the following so that the compiler will stop
      // warning us about possibly using unitialized variables
      double minv = DBL_MAX, maxv = -DBL_MAX;

      for( unsigned int i=0; i<field_input_handles.size(); i++ ) 
      {
        FieldHandle fHandle = field_input_handles[i];

        string units;
        if( fHandle->get_property("units", units) )
          colormap_output_handle_->set_units(units);
        
        if (!(fHandle->minmax(minmax_.first,minmax_.second)))
        {
          error("An input field is not a scalar or vector field.");
          execute_error_ = true;  
        } 

        if ( minv > minmax_.first) minv = minmax_.first;

        if ( maxv < minmax_.second) maxv = minmax_.second;
      }

      minmax_.first  = minv;
      minmax_.second = maxv;

      if ( gui_make_symmetric_.get() ) 
      {
        float biggest = Max(Abs(minmax_.first), Abs(minmax_.second));
        minmax_.first  = -biggest;
        minmax_.second =  biggest;
      }

      colormap_output_handle_->Scale( minmax_.first, minmax_.second);
      gui_min_.set( minmax_.first );
      gui_max_.set( minmax_.second );
    }
  }

  // Send the data downstream
  send_output_handle( "ColorMap",  colormap_output_handle_, true );
}

} // End namespace SCIRun
