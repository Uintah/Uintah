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
#include <Core/Geom/Sticky.h>

#include <stdio.h>
#include <string.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

class ShowColorMap : public Module {
  GuiString gui_length_;
  GuiString gui_side_;
  GuiInt gui_numlabels_;

  MaterialHandle white_;

public:
  ShowColorMap(const string &id);
  virtual ~ShowColorMap();
  virtual void execute();
};

extern "C" Module *make_ShowColorMap(const string &id)
{
  return new ShowColorMap(id);
}


ShowColorMap::ShowColorMap(const string &id)
  : Module("ShowColorMap", id, Filter, "Visualization", "SCIRun"),
    gui_length_("length", id, this),
    gui_side_("side", id, this),
    gui_numlabels_("numlabels", id, this)
{
  white_ = scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);
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
  if (gui_side_.get() == "left")
  {
    out = Vector(-0.05, 0.0, 0.0);
    if (gui_length_.get() == "full" || gui_length_.get() == "half1")
    {
      ref1 = Point(-1.0, 15.0/16.0, 0.0);
    }
    else
    {
      ref1 = Point(-1.0, -1.0/16.0, 1.0);
    }
    if (gui_length_.get() == "full")
    {
      along = Vector(0.0, -30.0/16.0, 0.0);
    }
    else
    {
      along = Vector(0.0, -14.0/16.0, 0.0);
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

  sq->set_texture( cmap->raw1d );
  all->add( sq );
  
  double min, max;
  if (cmap->IsScaled())
  {
    min = cmap->min;
    max = cmap->max;
  }
  else
  {
    // If the scalar field is present, we can add numbers and place the
    // billboard more intelligently.
    bool computed_min_max_p = false;
    port_range_type range = get_iports("ScalarField");
    port_map_type::iterator pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *isf = (FieldIPort *)get_iport(pi->second);
      ++pi;
      FieldHandle sf;
      ScalarFieldInterface *sfi;
      if (isf->get(sf) && (sfi = sf->query_scalar_interface()))
      {
	double minval, maxval;
	sfi->compute_min_max(minval, maxval);
	
	if (computed_min_max_p)
	{
	  min = Min(min, minval);
	  max = Max(max, maxval);
	}
	else
	{ 
	  min = minval;
	  max = maxval;
	  computed_min_max_p = true;
	}
      }
    }
    if (computed_min_max_p)
    {
      cmap->Scale(min, max);
    }
    else
    {
      min = cmap->min;
      max = cmap->max;
    }
  }

  const int numlabels = gui_numlabels_.get();
  if (numlabels > 1 && numlabels < 50)
  {
    // Fill in the text.
    Point p0  = ref0 - out * 0.02; 
    char value[80];
    GeomGroup *labels = scinew GeomGroup();
    for(int i = 0; i < numlabels; i++ )
    {
      sprintf(value, "%.2g", min + (max-min)*(i/(numlabels-1.0)));
      labels->add(scinew GeomText(value, p0 + along * (i/(numlabels-1.0))));
      labels->add(new GeomLine(p0 + along * (i/(numlabels-1.0)),
			       p0 + along * (i/(numlabels-1.0)) + out * 0.5));
    }    
    all->add(scinew GeomMaterial(labels, white_));
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  GeomSticky *sticky = scinew GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ShowColorMap" );
  ogeom->flushViews();
}

} // End namespace SCIRun
