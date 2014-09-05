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
#include <Dataflow/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

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

using namespace SCIRun;

class ShowString : public Module {
public:
  ShowString(GuiContext*);

  virtual ~ShowString();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

  GeomHandle generate();
  GeomHandle generateTitle( int &nchars, int &nlines );
  
private:

  GuiInt    iBbox_;
  GuiInt    dSize_;
  GuiString sLocation_;
  GuiDouble color_r_;
  GuiDouble color_g_;
  GuiDouble color_b_;
  
  
  int    bbox_;
  int    size_;
  string location_;
  Color  color_;
  MaterialHandle material_;
  
  int    generation_;
  bool   update_;

  std::string str_;  

};


DECLARE_MAKER(ShowString)
ShowString::ShowString(GuiContext* ctx)
  : Module("ShowString", ctx, Source, "Visualization", "SCIRun"),
    iBbox_(ctx->subVar("bbox")),
    dSize_(ctx->subVar("size")),
    sLocation_(ctx->subVar("location")),
    color_r_(ctx->subVar("color-r")),
    color_g_(ctx->subVar("color-g")),
    color_b_(ctx->subVar("color-b")),
    color_(1., 1., 1.),
    generation_(0),
    update_(false)  
{
    material_ = scinew Material;
    material_->diffuse = Color(color_r_.get(), color_g_.get(), color_b_.get());
}

ShowString::~ShowString()
{
}

void ShowString::execute()
{
  // Get a handle to the output geom port.
  GeometryOPort *ogeom_port = (GeometryOPort *) get_oport(0);

  // Get the time via a matrix
  StringIPort *istring_port = (StringIPort *)get_iport(0);
  
  StringHandle stringH;

  if( istring_port->get(stringH) && stringH.get_rep()) 
  {
    str_ = stringH->get();
  } 
  else 
  {
    error( "No string on input port" );
    return;
  }



  if( update_   == true ||
      bbox_      != iBbox_.get() ||
      size_      != dSize_.get() ||
      location_  != sLocation_.get() ||
      generation_ != stringH->generation) 
  {
    generation_ = stringH->generation;

    update_ = false;

    bbox_      = iBbox_.get();
    size_      = dSize_.get();
    location_  = sLocation_.get();
    color_ = Color(color_r_.get(), color_g_.get(), color_b_.get());

    material_->diffuse = Color(color_r_.get(), color_g_.get(), color_b_.get());
    ogeom_port->delAll();
    ogeom_port->addObj(generate(), string("Title"));
  }

  // Send the data downstream
  ogeom_port->flushViews();

}


GeomHandle ShowString::generate()
{
  GeomGroup *group = scinew GeomGroup();

  int nchars = 0;
  int nlines = 1;

  double border = 0.025;    
  double scale = .000225;
  double fsizes[5] = { 60,100,140,180,240 };  

  group->add( generateTitle( nchars, nlines ) );
    
  double dx = (nchars * 0.75 * fsizes[size_] * scale + 2.0 * border);
  double dy = (nlines * 1.8 * fsizes[size_] * scale + 2.0 * border);

  if( bbox_ ) 
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
    
    group->add( scinew GeomMaterial(box, material_) );
  }

  Vector refVec;

  if( location_ == "Top Left" )
    refVec = Vector(-31.0/32.0, 31.0/32.0, 0 ) - Vector(  0, dy, 0 );
  else if( location_ == "Top Center" )
    refVec = Vector(    0/32.0, 31.0/32.0, 0 ) - Vector( dx/2, dy, 0 );
  else if( location_ == "Bottom Left" )
    refVec = Vector(-31.0/32.0,-31.0/32.0, 0 ) - Vector( 0,  0, 0 );
  else if( location_ == "Bottom Center" )
    refVec = Vector(    0/32.0,-31.0/32.0, 0 ) - Vector( dx/2,  0, 0 );

  Transform trans;
  trans.pre_translate( refVec );

  return scinew GeomSticky( scinew GeomTransform( group, trans ) );
}


GeomHandle ShowString::generateTitle( int &nchars, int &nlines )
{
  nchars = 0;
  nlines = 0;
  size_ = dSize_.get();  
    
  GeomTexts* texts = scinew GeomTexts();
  texts->set_font_index(size_);
 
  std::string str;
  double scale = .000225;  
  double fsizes[5] = { 60,100,140,180,240 };  
  
  double dy =  (1.8 * fsizes[size_] * scale);
  double offset = 0.025;

  size_t ind = 0;
  size_t oldind = 0;
  while( (nlines < 10)&&(ind < str_.size()))
  {
    oldind =ind;
    for (;ind<str_.size();ind++) { if (str_[ind] == '\n') break; }
    if (str_[ind] == '\n') 
    {
        str = str_.substr(oldind,ind-oldind);
        ind++;
    }
    else
    {
        str = str_.substr(oldind);
        ind++;      
    }
    if (str.size() > 0)
    {
      if (nchars < str.size()) nchars = str.size();
      nlines++;
    }
  }

  int totlines = nlines;
  
  nlines = 0;
  ind = 0;
  oldind = 0;
  while( (nlines < 10)&&(ind < str_.size()))
  {
    oldind =ind;
    for (;ind<str_.size();ind++) { if (str_[ind] == '\n') break; }
    if (str_[ind] == '\n') 
    {
        str = str_.substr(oldind,ind-oldind);
        ind++;
    }
    else
    {
        str = str_.substr(oldind);
        ind++;      
    }
    if (str.size() > 0)
    {
      nlines++;
      texts->add(str,Point(0,(totlines-nlines)*dy+offset,0),color_);
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
    update_ = true;
  } 
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun


