//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGchar.h>
#include <SCICore/Geom/GeomTexSlices.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream.h>

#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;

class VolRendTexSlices : public Module {
   ScalarFieldIPort *ifield;
   GeometryOPort* ogeom;
   
   int init;
   MaterialHandle outcolor;
   int geom_id;

   int field_id; // id for the scalar field...
   TCLdouble accum;
   TCLdouble bright;

   double ac;
   double br;
public:
   VolRendTexSlices(const clString& id);
   VolRendTexSlices(const VolRendTexSlices&, int deep);
   virtual ~VolRendTexSlices();
   virtual Module* clone(int deep);
   virtual void execute();
};

Module* make_VolRendTexSlices(const clString& id) {
  return new VolRendTexSlices(id);
}

static clString module_name("VolRendTexSlices");

VolRendTexSlices::VolRendTexSlices(const clString& id)
: Module("VolRendTexSlices", id, Filter), accum("accum", id, this),
  bright("bright", id, this), ac(.1), br(.6)
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
: Module(copy, deep), accum("accum", id, this), bright("bright", id, this)
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
    
    ScalarFieldRGBase *sfrgb = sfh->getRGBase();
    if (!sfrgb) {
	error("Needs to be a ScalarFieldRGchar!");
	return;
    }

    ScalarFieldRGchar *sfield = sfrgb->getRGChar();
    if (!sfield) {
	error("Needs to be a ScalarFieldRGchar!");
	return;
    }

    double minv,maxv,mmax;
    sfield->get_minmax(minv,maxv);
    cerr <<"minv="<<minv<<" maxv="<<maxv<<"\n";

    mmax = 255.0/(maxv-minv);

    if (!sfield)
	return;

    if (!init || sfield->generation != field_id ||
	ac != accum.get() || br != bright.get()) {  // setup scalar field...
      field_id = sfield->generation;
      init=1;
      Point min, max;
      sfield->get_bounds(min, max);
      GeomTexSlices *ts = scinew GeomTexSlices(sfield->nx, sfield->ny, 
					       sfield->nz, min, max);
      ts->accum=ac=accum.get();
      ts->bright=br=bright.get();
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:54  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:17  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:58:02  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
