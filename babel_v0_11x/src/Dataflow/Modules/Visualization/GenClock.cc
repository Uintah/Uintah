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
 *  GenClock.cc: Choose one input field to be passed downstream
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

class GenClock : public Module {
public:
  GenClock(GuiContext *context);

  virtual ~GenClock();

  virtual void execute();

protected:
  GeomHandle generateAnalog ();
  GeomHandle generateDigital();
  GeomHandle generateTime( int &nchars );

  virtual void tcl_command(GuiArgs& args, void* userdata);

private:
  GeomHandle geometry_out_handle_;

  GuiInt    gui_type_;
  GuiInt    gui_bbox_;
  GuiString gui_format_;
  GuiDouble gui_min_;
  GuiDouble gui_max_;
  GuiString gui_current_;
  GuiInt    gui_size_;
  GuiDouble gui_location_x_;
  GuiDouble gui_location_y_;
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;

  MaterialHandle material_handle_;

  bool color_changed_;
};


DECLARE_MAKER(GenClock)


GenClock::GenClock(GuiContext *context)
  : Module("GenClock", context, Source, "Visualization", "SCIRun"),
    geometry_out_handle_(0),
    gui_type_(context->subVar("type"), 0),
    gui_bbox_(context->subVar("bbox"), 1),
    gui_format_(context->subVar("format"), "%8.3f seconds"),
    gui_min_(context->subVar("min"), 0),
    gui_max_(context->subVar("max"), 1),
    gui_current_(context->subVar("current"), "0"),
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

GenClock::~GenClock(){
}

void GenClock::execute(){

  // Get the time via a matrix
  MatrixHandle matrix_in_handle;
  if( !get_input_handle( "Time Matrix", matrix_in_handle, true ) ) return;
  
  if( gui_current_.get() != to_string( matrix_in_handle->get(0, 0) ) ) {
    gui_current_.set( to_string( matrix_in_handle->get(0, 0) ) ); 
    gui_current_.reset();

    inputs_changed_  = true;
  }

  if( inputs_changed_ ||

      !geometry_out_handle_.get_rep() ||

      gui_current_.changed( true ) ||
      gui_type_.changed( true ) ||
      gui_bbox_.changed( true ) ||
      gui_min_.changed( true ) ||
      gui_max_.changed( true ) ||
      gui_format_.changed( true ) ||
      gui_size_.changed( true ) ||
      gui_location_x_.changed( true ) ||
      gui_location_y_.changed( true ) ||
      
      color_changed_  ) {

    update_state(Executing);

    color_changed_  = false;

    material_handle_->diffuse =
      Color(gui_color_r_.get(), gui_color_g_.get(), gui_color_b_.get());

    if( gui_type_.get() % 2 == 0 )       // Analog or Analog/Digial
      geometry_out_handle_ = generateAnalog();

    else if( gui_type_.get() == 1 )      // Digital
      geometry_out_handle_ = generateDigital();
    
    send_output_handle( string("Clock"),
			geometry_out_handle_,
			string("Clock Sticky") );
  }
}

GeomHandle GenClock::generateAnalog()
{
  GeomGroup *group = scinew GeomGroup();

  GeomGroup *circle = scinew GeomGroup();

  double radius = 0.075 * gui_size_.get() / 100.0;

  Point last( 0, radius, 0 );

  for (int i=1; i<=36; i++) {
    double cx = sin( 2.0 * M_PI * i / 36.0 ) * radius;
    double cy = cos( 2.0 * M_PI * i / 36.0 ) * radius;
    
    Point next( cx, cy, 0 );

    circle->add( new GeomLine( last, next ) );

    last = next;
  }

  double value;
  string_to_double(gui_current_.get(), value );
 
  double t =
    2.0 * M_PI * (value-gui_min_.get()) / (gui_max_.get()-gui_min_.get());
  double cx = sin( t ) * radius;
  double cy = cos( t ) * radius;

  circle->add( new GeomLine( Point(0,0,0), Point( cx, cy, 0 ) ) );
  
  group->add( scinew GeomMaterial(circle, material_handle_) );
  
  double border = 0.025;    
  double scale = .00225;

  cx = radius;
  cy = radius;

  if( gui_type_.get() == 2 ) {    
    int nchars = 0;

    GeomHandle time = generateTime( nchars );

    double dx = nchars * 14.0 * gui_size_.get()/100.0 * scale;
    double dy = 1.0    * 15.0 * gui_size_.get()/100.0 * scale + 2.0 * border;

    Vector refVec(-dx/2.0, -radius-dy, 0 );
    Transform trans;
    trans.pre_translate( refVec );

    group->add( scinew GeomTransform( time, trans ) );

    if( cx < dx/2.0 )
      cx = dx/2.0;

    cy = radius+dy;
  }

  if( gui_bbox_.get() ) {
    GeomGroup *box = scinew GeomGroup();
    
    box->add( new GeomLine( Point(-cx-border,   -cy-border,0),
			    Point(+cx+border,   -cy-border,0) ) );

    box->add( new GeomLine( Point(+cx+border,   -cy-border,0),
			    Point(+cx+border,radius+border,0) ) );

    box->add( new GeomLine( Point(+cx+border,radius+border,0),
			    Point(-cx-border,radius+border,0) ) );
    
    box->add( new GeomLine( Point(-cx-border,radius+border,0),
			    Point(-cx-border,   -cy-border,0) ) );
    
    group->add( scinew GeomMaterial(box, material_handle_) );
  }

  // Use an offset so the border of the clock is visable.
  Vector refVec = 31.0/32.0 *
    Vector( gui_location_x_.get(), gui_location_y_.get(), 0.0 );

  if( gui_location_x_.get() <= 0 &&
      gui_location_y_.get() >= 0 )       // Top Left
    refVec += Vector(  cx, -radius, 0 );
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() >= 0 )  // Top Right
    refVec += Vector( -cx, -radius, 0 );
  else if( gui_location_x_.get() <= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Left
    refVec += Vector(  cx, cy, 0 ) ;
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Right
    refVec += Vector( -cx, cy, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}

GeomHandle GenClock::generateDigital()
{
  GeomGroup *group = scinew GeomGroup();

  int nchars = 0;

  group->add( generateTime( nchars ) );

  double border = 0.025;    
  double scale = .00225;
    
  double dx = nchars * 14.0 * gui_size_.get()/100.0 * scale + 2.0 * border;
  double dy = 1.0    * 15.0 * gui_size_.get()/100.0 * scale + 2.0 * border;

  if( gui_bbox_.get() ) {
    GeomGroup *box = scinew GeomGroup();
    
    box->add( new GeomLine( Point(-border,-border,0),
			    Point(dx,-border,0) ) );
    
    box->add( new GeomLine( Point(dx,-border,0),
			    Point(dx,dy,0) ) );

    box->add( new GeomLine( Point(dx,dy,0),
			    Point(-border,dy,0) ) );
    
    box->add( new GeomLine( Point(-border,dy,0),
			    Point(-border,-border,0) ) );
    
    group->add( scinew GeomMaterial(box, material_handle_) );
  }

  // Use an offset so the border of the clock is visable.
  Vector refVec = 31.0/32.0 *
    Vector( gui_location_x_.get(), gui_location_y_.get(), 0.0 );

  if( gui_location_x_.get() <= 0 &&
      gui_location_y_.get() >= 0 )       // Top Left
    refVec += Vector(  0, -dy, 0 );
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() >= 0 )  // Top Right
    refVec += Vector( -dx, -dy, 0 );
  else if( gui_location_x_.get() <= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Left
    {}
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Right
    refVec += Vector( -dx, 0, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}

GeomHandle GenClock::generateTime( int &nchars )
{
  char fontstr[8];

  int fontsize = (int) (12.0 * gui_size_.get()/100.0);

  if( fontsize <  6 ) fontsize = 6;
  if( fontsize > 16 ) fontsize = 16;

  sprintf( fontstr, "%d", fontsize );

  char timestr[64];

  string format = gui_format_.get();

  if( format.find("%") == std::string::npos ||
      format.find("%") != format.find_last_of("%") ) {
    error("Bad C Style format for the clock.");
    error("The format should be of the form: '%7.4f seconds'");
    sprintf( timestr, "Bad Format" );
  } else {
    double value;
    string_to_double(gui_current_.get(), value );
    sprintf( timestr, format.c_str(), value );
  }

  nchars = strlen( timestr );

  return scinew GeomText( timestr,
			  Point(0,0,0),
			  material_handle_->diffuse,
			  string( fontstr ) );
}

void 
GenClock::tcl_command(GuiArgs& args, void* userdata) {

  if(args.count() < 2) {
    args.error("GenClock needs a minor command");
    return;
  }

  if (args[1] == "color_change") {
    color_changed_ = true;

    // The below works for only the geometry not for the text so update.
    /*
    // Get a handle to the output geom port.
    GeometryOPort *ogeom_port = (GeometryOPort *) get_oport("Clock");
    
    gui_color_r_.reset();
    gui_color_g_.reset();
    gui_color_b_.reset();

    material_->diffuse = Color(gui_color_r_.get(),
                               gui_color_g_.get(),
                               gui_color_b_.get());
    
    if (ogeom_port) ogeom_port->flushViews();
    */
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace SCIRun

