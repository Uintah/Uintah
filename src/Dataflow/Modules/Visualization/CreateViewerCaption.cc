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
 *  CreateViewerCaption.cc: Choose one input field to be passed downstream
 *
 *  Written by:
 *   Allen R. Sanderson
 *   SCI Institute
 *   University of Utah
 *   January 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */

#include <stdio.h>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/GeomDisc.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSticky.h>

namespace SCIRun {

class CreateViewerCaption : public Module {
public:
  CreateViewerCaption(GuiContext *context);

  virtual ~CreateViewerCaption();

  virtual void execute();

protected:
  GeomHandle generate();
  GeomHandle generateTitle( int &nchars );

  virtual void tcl_command(GuiArgs& args, void* userdata);

protected:

  GeomHandle geometry_out_handle_;

  GuiInt    gui_show_value_;
  GuiString gui_value_;
  GuiString gui_format_;
  GuiInt    gui_size_;
  GuiDouble gui_location_x_;
  GuiDouble gui_location_y_;
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;

  MaterialHandle material_handle_;

  bool color_changed_;
};


DECLARE_MAKER(CreateViewerCaption)


CreateViewerCaption::CreateViewerCaption(GuiContext *context)
  : Module("CreateViewerCaption", context, Source, "Visualization", "SCIRun"),
    geometry_out_handle_(0),
    gui_show_value_(context->subVar("showValue"), 0),
    gui_value_(context->subVar("value"), ""),
    gui_format_(context->subVar("format"), "My Title"),
    gui_size_(context->subVar("size"), 100),
    gui_location_x_(context->subVar("location-x"), -31.0/32.0),
    gui_location_y_(context->subVar("location-y"),  31.0/32.0),
    gui_color_r_(context->subVar("color-r"), 1.0),
    gui_color_g_(context->subVar("color-g"), 1.0),
    gui_color_b_(context->subVar("color-b"), 1.0),
    material_handle_(scinew Material(Color(1.0, 1.0, 1.0))),
    color_changed_(false)
{
}

CreateViewerCaption::~CreateViewerCaption(){
}

void CreateViewerCaption::execute(){

  // Get the value via a matrix - Optional
  MatrixHandle matrix_in_handle;
  if( !get_input_handle( "Time Matrix",
			 matrix_in_handle, gui_show_value_.get() ) )
    return;

  if( matrix_in_handle.get_rep() ) {
    if( gui_value_.get() != to_string( matrix_in_handle->get(0, 0) ) ) {
      gui_value_.set( to_string( matrix_in_handle->get(0, 0) ) ); 
      gui_value_.reset();

      inputs_changed_  = true;
    }
  } else {
    if( gui_value_.get() != "" ) {
      gui_value_.set( "" );
      gui_value_.reset();
      
      inputs_changed_  = true;
    }
  }

  if( !geometry_out_handle_.get_rep() ||
      
      inputs_changed_ == true ||

      (gui_show_value_.get() && gui_value_.changed( true )) ||
      (gui_show_value_.get() && gui_format_.changed( true )) ||

      gui_size_.changed( true ) ||
      gui_location_x_.changed( true ) ||
      gui_location_y_.changed( true ) ||

      color_changed_  == true ) {

    update_state(Executing);

    color_changed_ = false;

    material_handle_->diffuse =
      Color(gui_color_r_.get(), gui_color_g_.get(), gui_color_b_.get());

    geometry_out_handle_ = generate();

    send_output_handle( string("Title"),
			geometry_out_handle_,
			string("Title Sticky") );
  }
}


GeomHandle CreateViewerCaption::generate()
{
  GeomGroup *group = scinew GeomGroup();

  int nchars = 0;

  group->add( generateTitle( nchars ) );

  double border = 0.025;    
  double scale = .00225;
    
  double dx =
    (nchars * 14.0 * gui_size_.get()/100.0 * scale + 2.0 * border) / 2;
  double dy =
     1.0    * 15.0 * gui_size_.get()/100.0 * scale + 2.0 * border;

  // Use an offset so the edge of the title is visable.
  Vector refVec = 31.0/32.0 *
    Vector( gui_location_x_.get(), gui_location_y_.get(), 0.0 );

  if( gui_location_x_.get() <= 0 &&
      gui_location_y_.get() >= 0 )       // Top Left
    refVec += Vector(   0, -dy, 0 );
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() >= 0 )  // Top Right
    refVec += Vector( -dx, -dy, 0 );
  else if( gui_location_x_.get() <= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Left
    refVec += Vector(   0,  0, 0 ) ;
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Right
    refVec += Vector( -dx,  0, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}


GeomHandle CreateViewerCaption::generateTitle( int &nchars )
{
  char fontstr[8];

  int fontsize = (int) (12.0 * gui_size_.get()/100.0);

  if( fontsize <  6 ) fontsize = 6;
  if( fontsize > 16 ) fontsize = 16;

  sprintf( fontstr, "%d", fontsize );

  char titlestr[64];

  string format = gui_format_.get();

  if( gui_show_value_.get() ) {
    if( format.find("%") == std::string::npos ||
	format.find("%") != format.find_last_of("%") ) {
      error( "Bad C Style format for the title.");
      error( "The format should be of the form: 'Blah blah %7.4f blah blah'");
      sprintf( titlestr, "Bad Format" );
    } else {
      double value;
      string_to_double(gui_value_.get(), value );
      sprintf( titlestr, format.c_str(), value );
    }
  } else {
    if( format.find("%") != std::string::npos ) {
      error( "Bad C Style format for the title.");
      error( "The format should be of the form: 'Blah blah blah blah'");
      sprintf( titlestr, "Bad Format" );
    } else
      sprintf( titlestr, format.c_str() );
  }

  nchars = strlen( titlestr );

  return scinew GeomText( titlestr,
			  Point(0,0,0),
			  material_handle_->diffuse,
			  string( fontstr ) );
}


void 
CreateViewerCaption::tcl_command(GuiArgs& args, void* userdata) {

  if(args.count() < 2) {
    args.error("CreateViewerCaption needs a minor command");
    return;
  }

  if (args[1] == "color_change") {
    color_changed_ = true;

    // The below works for only the geometry not for the text so update.
    /*
    // Get a handle to the output geom port.
    GeometryOPort *ogeom_port = (GeometryOPort *) get_oport("Title");
    
    gui_color_r_.reset();
    gui_color_g_.reset();
    gui_color_b_.reset();

    material_handle_->diffuse = Color(gui_color_r_.get(),
                                      gui_color_g_.get(),
                                      gui_color_b_.get());
    
    if (ogeom_port) ogeom_port->flushViews();
    */
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace SCIRun

