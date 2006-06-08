//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  

#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

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

using namespace SCIRun;

class ShowString : public Module {
public:
  ShowString(GuiContext*);

  virtual ~ShowString();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

protected:
  GeomHandle generate( string str );
  GeomHandle generateTitle( string str, int &nchars, int &nlines );

private:
  GeomHandle geometry_out_handle_;

  GuiInt    gui_bbox_;
  GuiInt    gui_size_;
  GuiDouble gui_location_x_;
  GuiDouble gui_location_y_;
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;

  MaterialHandle material_handle_;

  bool color_changed_;
};


DECLARE_MAKER(ShowString)
ShowString::ShowString(GuiContext* context)
  : Module("ShowString", context, Source, "Visualization", "SCIRun"),
    gui_bbox_(context->subVar("bbox"), 1),
    gui_size_(context->subVar("size"), 1),
    gui_location_x_(context->subVar("location-x"), -31.0/32.0),
    gui_location_y_(context->subVar("location-y"),  31.0/32.0),
    gui_color_r_(context->subVar("color-r"), 1.0),
    gui_color_g_(context->subVar("color-g"), 1.0),
    gui_color_b_(context->subVar("color-b"), 1.0),
    material_handle_(scinew Material(Color(1.0, 1.0, 1.0))),
    color_changed_(false)
{
}

ShowString::~ShowString()
{
}

void ShowString::execute()
{
  StringHandle string_input_handle;
  if( !get_input_handle( "Format String", string_input_handle, true ) ) return;

  if( !geometry_out_handle_.get_rep() ||
      
      inputs_changed_ == true ||

      gui_bbox_.changed( true ) ||
      gui_size_.changed( true ) ||
      gui_location_x_.changed( true ) ||
      gui_location_y_.changed( true ) ||

      color_changed_  == true ) {

    update_state(Executing);

    color_changed_ = false;

    material_handle_->diffuse =
      Color(gui_color_r_.get(), gui_color_g_.get(), gui_color_b_.get());

    geometry_out_handle_ = generate( string_input_handle->get() );

    send_output_handle( string("Title"),
			geometry_out_handle_,
			string("Title Sticky") );
  }
}


GeomHandle ShowString::generate( string str )
{
  GeomGroup *group = scinew GeomGroup();

  int nchars = 0;
  int nlines = 1;

  double border = 0.025;    
  double scale = .000225;
  double fsizes[5] = { 60,100,140,180,240 };  

  group->add( generateTitle( str, nchars, nlines ) );
    
  double dx = (nchars * 0.75 * fsizes[gui_size_.get()] * scale + 2.0 * border);
  double dy = (nlines * 1.80 * fsizes[gui_size_.get()] * scale + 2.0 * border);

/*  if( gui_bbox_.get() ) 
  {
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
*/

  // Use an offset so the edge of the title is visable.
  Vector refVec = 31.0/32.0 *
    Vector( gui_location_x_.get(), gui_location_y_.get(), 0.0 );

  if( gui_location_x_.get() <= 0 &&
      gui_location_y_.get() >= 0 )       // Top Left
    refVec += Vector(   0, -dy, 0 );
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() >= 0 )  // Top Right
    refVec += Vector( -dx/2, -dy, 0 );
  else if( gui_location_x_.get() <= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Left
    refVec += Vector(   0,  0, 0 ) ;
  else if( gui_location_x_.get() >= 0 &&
	   gui_location_y_.get() <= 0 )  // Bottom Right
    refVec += Vector( -dx/2,  0, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}


GeomHandle ShowString::generateTitle( string istr, int &nchars, int &nlines )
{
  nchars = 0;
  nlines = 0;
    
  GeomTexts* texts = scinew GeomTexts();
  texts->set_font_index(gui_size_.get());
 
  std::string ostr;
  double scale = .000225;  
  double fsizes[5] = { 60,100,140,180,240 };  
  
  double dy =  (1.8 * fsizes[gui_size_.get()] * scale);
  double offset = 0.025;

  size_t ind = 0;
  size_t oldind = 0;
  while( (nlines < 10)&&(ind < istr.size()))
  {
    oldind =ind;
    for (;ind<istr.size();ind++) { if (istr[ind] == '\n') break; }
    if (istr[ind] == '\n') 
    {
        ostr = istr.substr(oldind,ind-oldind);
        ind++;
    }
    else
    {
        ostr = istr.substr(oldind);
        ind++;      
    }
    if (ostr.size() > 0)
    {
      if (nchars < (int) ostr.size()) nchars = ostr.size();
      nlines++;
    }
  }

  int totlines = nlines;
  
  nlines = 0;
  ind = 0;
  oldind = 0;
  while( (nlines < 10)&&(ind < istr.size()))
  {
    oldind =ind;
    for (;ind<istr.size();ind++) { if (istr[ind] == '\n') break; }
    if (istr[ind] == '\n') 
    {
        ostr = istr.substr(oldind,ind-oldind);
        ind++;
    }
    else
    {
        ostr = istr.substr(oldind);
        ind++;      
    }
    if (ostr.size() > 0)
    {
      nlines++;
      texts->add(ostr,Point(0,(totlines-nlines)*dy+offset,0),
		 material_handle_->diffuse);
    }
  }
  
  
  GeomHandle handle = dynamic_cast<GeomObj *>(texts);
  return(handle);
}


void ShowString::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) 
  {
    args.error("ShowString needs a minor command");
    return;
  }

  if (args[1] == "color_change") 
  {
    color_changed_ = true;
  } 
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun


