//static char *id="@(#) $Id$";

/*
 *  Grid.cc: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Geom/TimeGrid.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Cross;
using SCICore::Math::Max;

Persistent* make_TimeGrid()
{
    return scinew TimeGrid(0,0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1));
}

PersistentTypeID TimeGrid::type_id("TimeGrid", "GeomObj", make_TimeGrid);

TimeGrid::TimeGrid(int nf,int nu, int nv, const Point& corner,
		   const Vector& u, const Vector& v)
: corner(corner), u(u), v(v), dimU(nu), dimV(nv), tmap_size(0), bmap(0)
{
  adjust();
  
  tmap.resize(nf);
  time.resize(nf);
  tmap_dlist.resize(nf);

  for(int i=0;i<nf;i++) {
    tmap[i] = 0;
    time[i] = 0.0;
    tmap_dlist[i] = 0;
  }

}

TimeGrid::TimeGrid(const TimeGrid& copy)
: GeomObj(copy)
{
}

void TimeGrid::adjust()
{
    w=Cross(u, v);
    w.normalize();
}

void TimeGrid::set_active(int which, double t)
{
  if (which >= time.size())
    cerr << "To big!\n";
  time[which] = t;
  active = which; // this is the important part...

  cerr << t << " Set Active " << which << "\n";
  
}

void TimeGrid::set(int i, int j, const MaterialHandle& /*matl*/,
		   const double &alpha)
{
    if(!tmap[active]){
      if (!tmap_size) {
	// allocate texture inteligently...
	int mdim = Max(dimU,dimV);
	int pwr2 = 1;
	while (mdim>=pwr2) {
	  pwr2 *= 2;
	}
	tmap_size = pwr2;
      }
//      tmap[active] = scinew float[tmap_size*tmap_size*4]; 
      tmap[active] = scinew float[tmap_size*tmap_size]; 

      if (!active)
	bmap = scinew float[tmap_size*tmap_size*4];
    }
#if 0
    const Color& dif = matl.get_rep()->diffuse;

    tmap[active][index] = dif.r();
    tmap[active][index+1] = dif.g();
    tmap[active][index+2] = dif.b();
    tmap[active][index+3] = alpha;
#else
    int index=(j*tmap_size) + i;
    tmap[active][index] = alpha;
#endif
}

void TimeGrid::get_bounds(BBox& bb)
{
  bb.extend(corner);
  bb.extend(corner+u);
  bb.extend(corner+v);
  bb.extend(corner+u+v);
}

TimeGrid::~TimeGrid()
{
  // run through and delete all of the textures...

  for(int i=0;i<time.size();i++)
    if (tmap[i])
      delete tmap[i];
}

GeomObj* TimeGrid::clone()
{
    return scinew TimeGrid(*this);
}

#define TimeGrid_VERSION 1

void TimeGrid::io(Piostream& stream)
{
    stream.begin_class("TimeGrid", TimeGrid_VERSION);
    GeomObj::io(stream);
    if(stream.reading())
	adjust();
    stream.end_class();
}    

bool TimeGrid::saveobj(ostream&, const clString&, GeomSave*)
{
  return 0;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/09/04 06:01:50  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.4  1999/08/19 23:18:07  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/17 23:50:34  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:53  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

