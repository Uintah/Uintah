/*
 *  VolRendNoTex.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geom/TexSlices.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

#include <iostream.h>

#include <stdio.h>

class VolRendTexSlices : public Module {
   ScalarFieldIPort *ifield;
   GeometryOPort* ogeom;
   
   int init;
   MaterialHandle outcolor;
   int geom_id;

   int field_id; // id for the scalar field...

public:
   VolRendTexSlices(const clString& id);
   VolRendTexSlices(const VolRendTexSlices&, int deep);
   virtual ~VolRendTexSlices();
   virtual Module* clone(int deep);
   virtual void execute();
};

extern "C" {
Module* make_VolRendTexSlices(const clString& id)
{
   return scinew VolRendTexSlices(id);
}
};

static clString module_name("VolRendTexSlices");

VolRendTexSlices::VolRendTexSlices(const clString& id)
: Module("VolRendTexSlices", id, Filter)
{
    // Create the input ports
    ifield = scinew ScalarFieldIPort( this, "Scalar Field",
					    ScalarFieldIPort::Atomic);
    add_iport(ifield);

    // Create the output port
    ogeom = scinew GeometryOPort(this, "Geometry", 
				 GeometryIPort::Atomic);
    add_oport(ogeom);
    geom_id=0;
    init=0;
}

VolRendTexSlices::VolRendTexSlices(const VolRendTexSlices& copy, int deep)
: Module(copy, deep)
{
   NOT_FINISHED("VolRendTexSlices::VolRendTexSlices");
}

VolRendTexSlices::~VolRendTexSlices()
{
}

Module* VolRendTexSlices::clone(int deep)
{
   return scinew VolRendTexSlices(*this, deep);
}

void VolRendTexSlices::execute()
{
    // get the scalar field and colormap...if you can
    ScalarFieldHandle sfh;
    if (!ifield->get(sfh))
	return;
    if (!sfh.get_rep()) return;

    ScalarFieldRG *sfield = sfh->getRG();

    double minv,maxv,mmax;

    sfield->get_minmax(minv,maxv);

    mmax = 255.0/(maxv-minv);

    if (!sfield)
	return;

    if (!init || sfield->generation != field_id) {  // setup scalar field...
	field_id = sfield->generation;
      init=1;
      Point min, max;
      sfield->get_bounds(min, max);
      GeomTexSlices *ts = scinew GeomTexSlices(sfield->nx, sfield->ny, 
					       sfield->nz, min, max);
      for (int i=0; i<sfield->nx; i++)
	  for (int j=0; j<sfield->ny; j++)
	      for (int k=0; k<sfield->nz; k++) {
		  unsigned char val = (sfield->grid(i,j,k)-minv)*mmax;
		  ts->Xmajor(i,j,k)=ts->Ymajor(j,i,k)=ts->Zmajor(k,i,j)=val;
	      }

      if (geom_id) {
	  ogeom->delObj(geom_id);
	  geom_id=0;
      }
	if (!geom_id) {
	    geom_id = ogeom->addObj(ts,"VolRendTexSlices TransParent"); // no bbox...
	}
    }
}
