/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Geom/TimeGrid.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {


Persistent* make_TimeGrid()
{
    return scinew TimeGrid(0,0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1));
}

PersistentTypeID TimeGrid::type_id("TimeGrid", "GeomObj", make_TimeGrid);

TimeGrid::TimeGrid(int nf,int nu, int nv, const Point& corner,
		   const Vector& u, const Vector& v)
  : bmap(0),
    dimU(nu),
    dimV(nv),
    tmap_size(0),
    corner(corner),
    u(u),
    v(v)
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

} // End namespace SCIRun


