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
  : Module("ShowColorMap", id, Filter, "Visualization", "SCIRun")
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

  double xsize = 15./16.0;
  double ysize = 0.05;
  ColorMapTex *sq = scinew ColorMapTex( Point( 0, -1, 0),
					Point( xsize, -1, 0),
					Point( xsize, -1 + ysize, 0 ),
					Point( 0, -1 + ysize, 0 ) );

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

  // Some bases for positioning text.
  double xloc = xsize;
  double yloc = -1 + 1.1 * ysize;

  // Create min and max numbers at the ends.
  char value[80];
  sprintf(value, "%.2g", max );
  all->add( scinew GeomMaterial( scinew GeomText(value, Point(xloc,yloc,0) ),
				 white_) );
  sprintf(value, "%.2g", min );
  all->add( scinew GeomMaterial( scinew GeomText(value,
						 Point(0,yloc,0)), white_));

  // Fill in 3 other places.
  for(int i = 1; i < 4; i++ )
  {
    sprintf( value, "%.2g", min + i*(max-min)/4.0 );
    all->add( scinew GeomMaterial( scinew GeomText(value,
						   Point(xloc*i/4.0,yloc,0)),
				   white_) );
  }    

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  GeomSticky *sticky = scinew GeomSticky(all);
  ogeom->delAll();
  ogeom->addObj( sticky, "ShowColorMap" );
  ogeom->flushViews();
}

} // End namespace SCIRun
