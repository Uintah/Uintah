
/*
 * PadField.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include "PadField.h"

namespace Kurt {

using namespace SCIRun;
using std::cerr;


extern "C" Module* make_PadField( const clString& id) {
  return new PadField(id);
}


PadField::PadField(const clString& id)
  : Module("PadField", id, Filter), pad_mode("pad_mode", id, this),
  xpad("xpad", id, this), ypad("ypad", id, this), zpad("zpad", id, this)
{
  // Create the input ports
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					   ScalarFieldIPort::Atomic);

  outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					   ScalarFieldIPort::Atomic);

  add_iport(inscalarfield);
  // Create the output port
  add_oport(outscalarfield);

}

PadField::~PadField()
{

}

void PadField::execute(void)
{
  ScalarFieldHandle sfield;

  if (!inscalarfield->get(sfield)) {
    return;
  }
  else if (!sfield.get_rep()) {
    return;
  }
  
  if (!sfield->getRGBase())
    return;

      
  ScalarFieldRGdouble *rgdouble = sfield->getRGBase()->getRGDouble();

  int xsize, ysize, zsize;
  int xst = 0, yst = 0, zst = 0;

  if (!rgdouble) {
    cerr << "Not a double field!\n";
    return;
  } else {
    if(pad_mode.get() == 0){
      xsize = rgdouble->nx + xpad.get();
      ysize = rgdouble->ny + ypad.get();
      zsize = rgdouble->nz + zpad.get();
    } else {
      xsize = rgdouble->nx + 2 * xpad.get();
      xst = xpad.get();
      ysize = rgdouble->ny + 2 * ypad.get();
      yst = ypad.get();
      zsize = rgdouble->nz + 2 * zpad.get();
      zst = zpad.get();
    }
  }
  Point min, max;
  rgdouble->get_bounds(min, max);
  ScalarFieldRGdouble *rg = new ScalarFieldRGdouble();
  cerr<<"xsize, ysize, zsize = "<< xsize<<", "<<ysize<<", "<<zsize<<endl;
  rg->resize(xsize, ysize, zsize);
  double xscale = xpad.get()/double(rgdouble->nx);
  double yscale = ypad.get()/double(rgdouble->ny);
  double zscale = zpad.get()/double(rgdouble->nz);
  Vector v( (max.x() - min.x())* xscale,
	    (max.y() - min.y())* yscale,
	    (max.z() - min.z())* zscale);
  if( pad_mode.get() )
    rg->set_bounds( min - v, max + v);
  else
    rg->set_bounds(min, max + v);
    
  int i, j, k, ii, jj, kk;
  
  for(i = 0, ii = xst; i < rgdouble->nx; i++, ii++){
    for(j = 0, jj = yst; j < rgdouble->ny; j++, jj++){
      for(k = 0, kk = zst; k < rgdouble->nz; k++, kk++){
	rg->grid(ii,jj,kk) = rgdouble->grid(i,j,k);
      }
    }
  }
  
  outscalarfield->send(ScalarFieldHandle(rg));

}

} // End namespace Kurt
