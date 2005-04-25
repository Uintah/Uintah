/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

