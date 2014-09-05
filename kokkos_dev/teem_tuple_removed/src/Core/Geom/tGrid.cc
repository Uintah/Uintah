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

#include <Core/Geom/tGrid.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
#ifdef _WIN32
#include <string.h>
#include <memory.h>
#else
#include <strings.h>
#endif

#if defined(__sun)||defined(_WIN32)
#define bcopy(src,dest,n) memcpy(dest,src,n)
#endif

namespace SCIRun {

Persistent* make_TexGeomGrid()
{
    return scinew TexGeomGrid(0,0,Point(0,0,0), Vector(1,0,0), Vector(0,0,1));
}

PersistentTypeID TexGeomGrid::type_id("TexGeomGrid", "GeomObj", make_TexGeomGrid);

TexGeomGrid::TexGeomGrid(int nu, int nv, const Point& corner,
			 const Vector& u, const Vector& v, int nchan)
  : tmap_dlist(-1), corner(corner), u(u), v(v), 
  dimU(nu), dimV(nv), num_chan(nchan), convolve(0), conv_dim(0),
  kernal_change(0)
{

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

    stream.begin_class("TexGeomGrid", TexGeomGrid_VERSION);
    GeomObj::io(stream);
    Pio(stream, corner);
    Pio(stream, u);
    Pio(stream, v);
    if(stream.reading())
	adjust();
    stream.end_class();
}    

} // End namespace SCIRun

