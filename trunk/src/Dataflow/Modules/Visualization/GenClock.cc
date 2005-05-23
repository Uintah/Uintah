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
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/GeometryPort.h>
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

  GeomHandle generateAnalog ();
  GeomHandle generateDigital();
  GeomHandle generateTime   ( int &nchars );

  virtual void tcl_command(GuiArgs& args, void* userdata);

protected:

  GuiInt    iType_;
  GuiInt    iShowTime_;
  GuiInt    iBbox_;
  GuiString sFormat_;
  GuiDouble dMin_;
  GuiDouble dMax_;
  GuiDouble dCurrent_;
  GuiInt    dSize_;
  GuiString sLocation_;
  GuiDouble color_r_;
  GuiDouble color_g_;
  GuiDouble color_b_;

  int    type_;
  int    showTime_;
  int    bbox_;
  string format_;
  double min_;
  double max_;
  double current_;
  int    size_;
  string location_;

  MaterialHandle material_;

  bool update_;
};


DECLARE_MAKER(GenClock)


GenClock::GenClock(GuiContext *context)
  : Module("GenClock", context, Source, "Visualization", "SCIRun"),
    iType_(context->subVar("type")),
    iShowTime_(context->subVar("showtime")),
    iBbox_(context->subVar("bbox")),
    sFormat_(context->subVar("format")),
    dMin_(context->subVar("min")),
    dMax_(context->subVar("max")),
    dCurrent_(context->subVar("current")),
    dSize_(context->subVar("size")),
    sLocation_(context->subVar("location")),
    color_r_(ctx->subVar("color-r")),
    color_g_(ctx->subVar("color-g")),
    color_b_(ctx->subVar("color-b")),

    material_(scinew Material(Color(1., 1., 1.))),

    update_(false)
{
}

GenClock::~GenClock(){
}

void GenClock::execute(){

  // Get a handle to the output geom port.
  GeometryOPort *ogeom_port = (GeometryOPort *) get_oport("Clock");
  
  // Get the time via a matrix
  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Time Matrix");
  
  MatrixHandle mHandle;
  if (imatrix_port->get(mHandle) && mHandle.get_rep()) {
    current_ = (double) (mHandle->get(0, 0));
  } else {
    error( "No matrix handle or representation." );
    return;
  }

  if( update_   == true ||

      current_  != dCurrent_.get() ||
      type_     != iType_.get() ||
      showTime_ != iShowTime_.get() ||
      bbox_     != iBbox_.get() ||
      min_      != dMin_.get() ||
      max_      != dMax_.get() ||
      format_   != sFormat_.get() ||
      size_     != dSize_.get() ||
      location_ != sLocation_.get() ) {

    update_ = false;

    dCurrent_.set(current_);
    dCurrent_.reset();

    type_     = iType_.get();
    showTime_ = iShowTime_.get();
    bbox_     = iBbox_.get();
    min_      = dMin_.get();
    max_      = dMax_.get();
    format_   = sFormat_.get();
    size_     = dSize_.get();
    location_ = sLocation_.get();

    material_->diffuse = Color(color_r_.get(), color_g_.get(), color_b_.get());

    ogeom_port->delAll();

    if( type_ == 0 )           // Analog
      ogeom_port->addObj(generateAnalog(), string("Clock"));

    else if( type_ == 1 )      // Digital
      ogeom_port->addObj(generateDigital(), string("Clock"));
  }

  // Send the data downstream
  ogeom_port->flushViews();
}

GeomHandle GenClock::generateAnalog()
{
  GeomGroup *group = scinew GeomGroup();

  GeomGroup *circle = scinew GeomGroup();

  double radius = 0.075 * size_ / 100.0;

  Point last( 0, radius, 0 );

  for (int i=1; i<=36; i++) {
    double cx = sin( 2.0 * M_PI * i / 36.0 ) * radius;
    double cy = cos( 2.0 * M_PI * i / 36.0 ) * radius;
    
    Point next( cx, cy, 0 );

    circle->add( new GeomLine( last, next ) );

    last = next;
  }

  double cx = sin( 2.0 * M_PI * (current_-min_)/(max_-min_) ) * radius;
  double cy = cos( 2.0 * M_PI * (current_-min_)/(max_-min_) ) * radius;
    
  circle->add( new GeomLine( Point(0,0,0), Point( cx, cy, 0 ) ) );
  
  group->add( scinew GeomMaterial(circle, material_) );
  
  double border = 0.025;    
  double scale = .00225;

  cx = radius;
  cy = radius;

  if( showTime_ ) {    
    int nchars = 0;

    GeomHandle time = generateTime( nchars );

    double dx = nchars * 14.0 * size_/100.0 * scale;
    double dy = 1.0    * 15.0 * size_/100.0 * scale + 2.0 * border;

    Vector refVec(-dx/2.0, -radius-dy, 0 );
    Transform trans;
    trans.pre_translate( refVec );

    group->add( scinew GeomTransform( time, trans ) );

    if( cx < dx/2.0 )
      cx = dx/2.0;

    cy = radius+dy;
  }

  if( bbox_ ) {
    GeomGroup *box = scinew GeomGroup();
    
    box->add( new GeomLine( Point(-cx-border,   -cy-border,0),
			    Point(+cx+border,   -cy-border,0) ) );

    box->add( new GeomLine( Point(+cx+border,   -cy-border,0),
			    Point(+cx+border,radius+border,0) ) );

    box->add( new GeomLine( Point(+cx+border,radius+border,0),
			    Point(-cx-border,radius+border,0) ) );
    
    box->add( new GeomLine( Point(-cx-border,radius+border,0),
			    Point(-cx-border,   -cy-border,0) ) );
    
    group->add( scinew GeomMaterial(box, material_) );
  }

  Vector refVec;

  if( location_ == "Top Left" )
    refVec = Vector(-31.0/32.0, 31.0/32.0, 0 ) + Vector(  cx, -radius, 0 );
  else if( location_ == "Top Right" )
    refVec = Vector( 31.0/32.0, 31.0/32.0, 0 ) + Vector( -cx, -radius, 0 );
  else if( location_ == "Bottom Left" )
    refVec = Vector(-31.0/32.0,-31.0/32.0, 0 ) + Vector(  cx, cy, 0 ) ;
  else if( location_ == "Bottom Right" )
    refVec = Vector( 31.0/32.0,-31.0/32.0, 0 ) + Vector( -cx, cy, 0 );

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
    
  double dx = nchars * 14.0 * size_/100.0 * scale + 2.0 * border;
  double dy = 1.0    * 15.0 * size_/100.0 * scale + 2.0 * border;

  if( bbox_ ) {
    GeomGroup *box = scinew GeomGroup();
    
    box->add( new GeomLine( Point(-border,-border,0),
			    Point(dx,-border,0) ) );
    
    box->add( new GeomLine( Point(dx,-border,0),
			    Point(dx,dy,0) ) );

    box->add( new GeomLine( Point(dx,dy,0),
			    Point(-border,dy,0) ) );
    
    box->add( new GeomLine( Point(-border,dy,0),
			    Point(-border,-border,0) ) );
    
    group->add( scinew GeomMaterial(box, material_) );
  }

  Vector refVec;

  if( location_ == "Top Left" )
    refVec = Vector(-31.0/32.0, 31.0/32.0, 0 ) - Vector( 0, dy, 0 );
  else if( location_ == "Top Right" )
    refVec = Vector( 31.0/32.0, 31.0/32.0, 0 ) - Vector( dx, dy, 0 );
  else if( location_ == "Bottom Left" )
    refVec = Vector(-31.0/32.0,-31.0/32.0, 0 );
  else if( location_ == "Bottom Right" )
    refVec = Vector( 31.0/32.0,-31.0/32.0, 0 ) - Vector( dx, 0, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}

GeomHandle GenClock::generateTime( int &nchars )
{
  char fontstr[8];

  int fontsize = (int) (12.0 * size_/100.0);

  if( fontsize <  6 ) fontsize = 6;
  if( fontsize > 16 ) fontsize = 16;

  sprintf( fontstr, "%d", fontsize );

  char timestr[64];

  if( format_.find("%") == std::string::npos ||
      format_.find("%") != format_.find_last_of("%") ) {
    error("Bad C Style format for the clock.");
    error("The format should be of the form: '%7.4f seconds'");
    sprintf( timestr, "Bad Format" );
  } else {
    sprintf( timestr, format_.c_str(), current_ );
  }

  nchars = strlen( timestr );

  return scinew GeomText( timestr,
			  Point(0,0,0),
			  material_->diffuse,
			  string( fontstr ) );
}

void 
GenClock::tcl_command(GuiArgs& args, void* userdata) {

  if(args.count() < 2) {
    args.error("GenClock needs a minor command");
    return;
  }

  if (args[1] == "color_change") {
    update_ = true;

    // The below works for only the geometry not for the text so update.
    /*
    // Get a handle to the output geom port.
    GeometryOPort *ogeom_port = (GeometryOPort *) get_oport("Clock");
    
    color_r_.reset();
    color_g_.reset();
    color_b_.reset();

    material_->diffuse = Color(color_r_.get(), color_g_.get(), color_b_.get());
    
    if (ogeom_port) ogeom_port->flushViews();
    */
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace SCIRun

