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

#include <SCICore/Geom/tGrid.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#ifdef _WIN32
#include <string.h>
#include <memory.h>
#else
#include <strings.h>
#endif

#if defined(__sun)||defined(_WIN32)
#define bcopy(src,dest,n) memcpy(dest,src,n)
#endif

namespace SCICore {
namespace GeomSpace {

Persistent* make_TexGeomGrid()
{
    return scinew TexGeomGrid(0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1));
}

PersistentTypeID TexGeomGrid::type_id("TexGeomGrid", "GeomObj", make_TexGeomGrid);

TexGeomGrid::TexGeomGrid(int nu, int nv, const Point& corner,
			 const Vector& u, const Vector& v, int nchan)
  : corner(corner), u(u), v(v), 
  tmap_dlist(-1),dimU(nu),dimV(nv),num_chan(nchan),convolve(0),conv_dim(0),
  kernal_change(0)
{
  using SCICore::Math::Max;

  adjust();

  int delt = 2*convolve*(conv_dim/2);
  int mdim = Max(nu,nv);
  int pwr2 = 1;
  while (mdim>=pwr2) {
    pwr2 *= 2;
  }
  tmap_size = pwr2+delt;
  tmapdata = scinew unsigned short[(pwr2+delt)*(pwr2+delt)*num_chan];


  if (delt)
    cerr << "Got a problem...\n";
}

TexGeomGrid::TexGeomGrid(const TexGeomGrid& copy)
: GeomObj(copy)
{
}

void TexGeomGrid::do_convolve(int dim, float* data)
{
  conv_dim = dim;
  for(int i=0;i<dim*dim;conv_data[i++] = *data++)
    ;
  kernal_change=1;
  convolve=1;
}

void TexGeomGrid::adjust()
{
  using namespace Geometry;

    w=Cross(u, v);
    w.normalize();
}

void TexGeomGrid::set(unsigned short* buf, int datadim)
{
  cerr << "Initing texture...";
  
  unsigned short* curpos=buf;
  unsigned short* datapos=tmapdata;

  int delt = convolve*(conv_dim/2);

  if (!delt) {
    for(int y=0;y<dimV;y++) {
      bcopy(curpos,datapos,num_chan*dimU*2);
    /*  for (int x=0;x<dimU;x++) {
        datapos[x*dimU+y] = curpos[x*dimU+y]; */
        curpos += num_chan*dimU;
        datapos += num_chan*tmap_size;    
    }
  } else { // have to add boundary for convolution..
    cerr << "We shouldn't be here...\n";
    datapos += delt*tmap_size*num_chan;
    for(int y=0;y<dimV;y++) {
      bcopy(curpos,datapos+delt,num_chan*dimU);
      curpos += num_chan*dimU;
      datapos += num_chan*tmap_size;
    }
    
  }
    cerr << ". done!\n";
  
  MemDim = datadim;
}

void TexGeomGrid::get_bounds(BBox& bb)
{
  bb.extend(corner);
  bb.extend(corner+u);
  bb.extend(corner+v);
  bb.extend(corner+u+v);
}

GeomObj* TexGeomGrid::clone()
{
    return scinew TexGeomGrid(*this);
}

#define TexGeomGrid_VERSION 1

void TexGeomGrid::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("TexGeomGrid", TexGeomGrid_VERSION);
    GeomObj::io(stream);
    SCICore::Geometry::Pio(stream, corner);
    SCICore::Geometry::Pio(stream, u);
    SCICore::Geometry::Pio(stream, v);
    if(stream.reading())
	adjust();
    stream.end_class();
}    

bool TexGeomGrid::saveobj(ostream&, const clString&, GeomSave*)
{
  return 0;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/08/29 00:46:58  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 17:54:45  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/19 23:18:07  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/17 23:50:35  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:54  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:57  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//
