//static char *id="@(#) $Id$";


/*
 *  RescaleSegFld.cc:  Rescale a SegFld
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/SegFldPort.h>
#include <DaveW/Datatypes/General/SegFld.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class RescaleSegFld : public Module {
    SegFldIPort* iSegFld;
    SegFldOPort* oSegFld;
    TCLint coreg;
    TCLdouble scale;
    SegFldHandle osh;
    double last_scale;
    int generation;
public:
    RescaleSegFld(const clString& id);
    virtual ~RescaleSegFld();
    virtual void execute();
};

Module* make_RescaleSegFld(const clString& id)
{
    return new RescaleSegFld(id);
}

static clString module_name("RescaleSegFld");

RescaleSegFld::RescaleSegFld(const clString& id)
: Module("RescaleSegFld", id, Filter),
  scale("scale", id, this), coreg("coreg", id, this),
  generation(-1), last_scale(0)
{
    iSegFld=scinew SegFldIPort(this, "SegFld", SegFldIPort::Atomic);
    add_iport(iSegFld);
    // Create the output port
    oSegFld=scinew SegFldOPort(this, "SegFld", SegFldIPort::Atomic);
    add_oport(oSegFld);
}

RescaleSegFld::~RescaleSegFld()
{
}

void RescaleSegFld::execute()
{
    SegFldHandle isurf;
    if(!iSegFld->get(isurf))
	return;

    double new_scale=scale.get();
    double s=pow(10.,new_scale);
	
    Point min, max;
    isurf->get_bounds(min, max);
    if (coreg.get()) {
	BBox b;
	b.extend(min);
	b.extend(max);
	s=1./b.longest_edge()*1.7;
	cerr << "b.min="<<b.min()<<" b.max="<<b.max()<<"   s="<<s<<"\n";
	scale.set(log10(s));
	reset_vars();
	new_scale=scale.get();
    }

    if (generation == isurf->generation && new_scale == last_scale) {
	oSegFld->send(isurf);
	return;
    }
    
    last_scale = new_scale;
    max.x(max.x()*s);
    max.y(max.y()*s);
    max.z(max.z()*s);
    isurf->set_bounds(min, max);
    oSegFld->send(isurf);
    return;
}


} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.4  1999/10/07 02:06:28  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:23  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:38  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:02  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:13  dmw
// Added and updated DaveW Datatypes/Modules
//
//
