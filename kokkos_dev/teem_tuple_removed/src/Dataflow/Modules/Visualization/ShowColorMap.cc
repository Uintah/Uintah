/*
 *  ColorMapKey.cc: create a key for colormap
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Updated by:
 *   Michael Callahan
 *   January 2001
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomSticky.h>

#include <stdio.h>
#include <string.h>
#include <iostream>

namespace SCIRun {

class ShowColorMap : public Module {
  GuiString gui_length_;
  GuiString gui_side_;
  GuiInt gui_numlabels_;
  GuiDouble gui_scale_;
  GuiInt gui_num_sig_digits_;
  GuiString gui_units_;
  GuiInt gui_text_color_;
  GuiInt gui_text_fontsize_;
  GuiInt gui_extra_padding_;

public:
  ShowColorMap(GuiContext*);
  virtual ~ShowColorMap();
  virtual void execute();
};

  DECLARE_MAKER(ShowColorMap)

ShowColorMap::ShowColorMap(GuiContext* ctx)
  : Module("ShowColorMap", ctx, Filter, "Visualization", "SCIRun"),
    gui_length_(ctx->subVar("length")),
    gui_side_(ctx->subVar("side")),
    gui_numlabels_(ctx->subVar("numlabels")),
    gui_scale_(ctx->subVar("scale")),
    gui_num_sig_digits_(ctx->subVar("numsigdigits")),
    gui_units_(ctx->subVar("units")),
    gui_text_color_(ctx->subVar("text_color")),
    gui_text_fontsize_(ctx->subVar("text-fontsize")),
    gui_extra_padding_(ctx->subVar("extra-padding"))
{
}


ShowColorMap::~ShowColorMap()
{
}


void
ShowColorMap::execute()
{
  // the colormap is essential - without it return
  ColorMapIPort *imap = (ColorMapIPort *)get_iport("ColorMap");
  ColorMapHandle cmap;
  if (!imap->get(cmap))
  { 
    warning("No input color map.");
    return;
  }

  GeomGroup *all = scinew GeomGroup();

  Point  ref1;
  Vector out;
  Vector along;

  Color text_color;
  if( gui_text_color_.get() == 0 ){
    text_color = Color(0,0,0);
  }else{
    text_color = Color(1,1,1);
  }
  MaterialHandle text_material = scinew Material(text_color);
  if (gui_side_.get() == "left")
  {
    out = Vector(-0.05, 0.0, 0.0);
    if (gui_length_.get() == "full" || gui_length_.get() == "half2")
    {
      if (gui_extra_padding_.get())
      {
	ref1 = Point(-1.0, -14.0/16.0, 0.0);
      }
      else
      {
	ref1 = Point(-1.0, -15.0/16.0, 0.0);
      }
    }
    else
    {
      ref1 = Point(-1.0, 1.0/16.0, 1.0);
    }
    if (gui_length_.get() == "full")
    {
      if (gui_extra_padding_.get())
      {
	along = Vector(0.0, 29.0/16.0, 0.0);
      }
      else
      {
	along = Vector(0.0, 30.0/16.0, 0.0);
      }
    }
    else
    {
      along = Vector(0.0, 14.0/16.0, 0.0);
    }
  }
  else if (gui_side_.get() == "bottom")
  {
    out = Vector(0.0, -0.05, 0.0);
    if (gui_length_.get() == "full" || gui_length_.get() == "half1")
    {
      ref1 = Point(-15.0/16.0, -1.0, 0.0);
    }
    else
    {
      ref1 = Point(1.0/16.0, -1.0, 0.0);
    }
    if (gui_length_.get() == "full")
    {
      if (gui_extra_padding_.get())
      {
	along = Vector(29.0/16.0, 0.0, 0.0);
      }
      else
      {
	along = Vector(30.0/16.0, 0.0, 0.0);
      }
    }
    else
    {
      if (gui_extra_padding_.get())
      {
	along = Vector(13.0/16.0, 0.0, 0.0);
      }
      else
      {
	along = Vector(14.0/16.0, 0.0, 0.0);
      }
    }
  }

  const Point  ref0(ref1 - out);
  ColorMapTex *sq = scinew ColorMapTex(ref0,
				       ref0 + along,
				       ref0 + along + out,
				       ref0 + out);
  double scale = gui_scale_.get();
  string str = gui_units_.get();
  if (str == "") str = cmap->units;
  // So if the maximum number of digits the number will take up is
  // at most 25 then the length of str better be less than 80-25-1.
  // See size of value and num_sig_digits below.
  if (str.length() > 50) {
    error("Length of units string is too long.  Make it smaller than 50 characters please.");
    return;
  }

  sq->set_texture( cmap->rawRGBA_ );
  all->add( sq );
  const int numlabels = gui_numlabels_.get();
  if (numlabels > 1 && numlabels < 50)
  {
    // Fill in the text.
    const double minval = cmap->getMin()*scale;
    const double maxval = cmap->getMax()*scale;

    Point p0  = ref0 - out * 0.02; 
    char value[80];
    GeomLines *lines = scinew GeomLines();
    GeomTexts *texts = scinew GeomTexts();
    texts->set_font_index(gui_text_fontsize_.get());
    int num_sig_digits = gui_num_sig_digits_.get();
    if (num_sig_digits < 1) {
      warning("Number of significant digits needs to be at least 1.  Setting the number of significant digits to 1.");
      gui_num_sig_digits_.set(1);
      num_sig_digits = 1;
    }
    if (num_sig_digits > 20) {
      warning("Number of significant digits needs to be less than or equal to 20.  Setting the number of significant digits to 20");
      gui_num_sig_digits_.set(20);
      num_sig_digits = 20;
    }
    for(int i = 0; i < numlabels; i++ )
    {
      sprintf(value, "%.*g %s", num_sig_digits,
	      minval + (maxval-minval)*(i/(numlabels-1.0)),
	      str.c_str());
      const Point loc = p0 + along * (i/(numlabels-1.0));
      texts->add(value, loc, text_color);
      lines->add(loc, text_material, loc + out * 0.5, text_material);
    }    
    all->add(texts);
    all->add(lines);
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  GeomSticky *sticky = scinew GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ShowColorMap Sticky" );
  ogeom->flushViews();
}

} // End namespace SCIRun
