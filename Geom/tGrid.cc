
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

#include <Geom/tGrid.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>
#include <strings.h>

#ifdef __sun
#define bcopy(src,dest,n) memcpy(dest,src,n)
#endif

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

void TexGeomGrid::get_bounds(BSphere& bs)
{
  bs.extend(corner);
  bs.extend(corner+u);
  bs.extend(corner+v);
  bs.extend(corner+u+v);
}

void TexGeomGrid::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("TexGeomGrid::make_prims");
}

GeomObj* TexGeomGrid::clone()
{
    return scinew TexGeomGrid(*this);
}

void TexGeomGrid::preprocess()
{
    NOT_FINISHED("TexGeomGrid::preprocess");
}

void TexGeomGrid::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("TexGeomGrid::intersect");
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

bool TexGeomGrid::saveobj(ostream&, const clString&, GeomSave*)
{
  return 0;
}

#ifdef __GNUG__
#include <Classlib/Array2.cc>

template class Array2<double>;
template class Array2<MaterialHandle>;
template class Array2<Vector>;

template void Pio(Piostream&, Array2<double>&);
template void Pio(Piostream&, Array2<MaterialHandle>&);
template void Pio(Piostream&, Array2<Vector>&);

#endif


#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array2.cc>

static void _dummy_(Piostream& p1, Array2<MaterialHandle>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array2<Vector>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array2<double>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

