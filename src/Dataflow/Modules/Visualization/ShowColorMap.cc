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
  GuiString gui_units_;
  GuiInt gui_text_color_;
  MaterialHandle text_color_;

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
    gui_units_(ctx->subVar("units")),
    gui_text_color_(ctx->subVar("text_color"))
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

  if( gui_text_color_.get() == 0 ){
    text_color_ = new Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 1);
  }else{
    text_color_ = new Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 1);
  }
  if (gui_side_.get() == "left")
  {
    out = Vector(-0.05, 0.0, 0.0);
    if (gui_length_.get() == "full" || gui_length_.get() == "half2")
    {
      ref1 = Point(-1.0, -15.0/16.0, 0.0);
    }
    else
    {
      ref1 = Point(-1.0, 1.0/16.0, 1.0);
    }
    if (gui_length_.get() == "full")
    {
      along = Vector(0.0, 30.0/16.0, 0.0);
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
      along = Vector(30.0/16.0, 0.0, 0.0);
    }
    else
    {
      along = Vector(14.0/16.0, 0.0, 0.0);
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
    GeomGroup *labels = scinew GeomGroup();
    for(int i = 0; i < numlabels; i++ )
    {
      sprintf(value, "%.2g %s", minval + (maxval-minval)*(i/(numlabels-1.0)),
	      str.c_str());
      labels->add(scinew GeomText(value, p0 + along * (i/(numlabels-1.0))));
      labels->add(new GeomLine(p0 + along * (i/(numlabels-1.0)),
			       p0 + along * (i/(numlabels-1.0)) + out * 0.5));
    }    
    all->add(scinew GeomMaterial(labels, text_color_));
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  GeomSticky *sticky = scinew GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ShowColorMap Sticky" );
  ogeom->flushViews();
}

} // End namespace SCIRun
