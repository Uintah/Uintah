//static char *id="@(#) $Id$";

/*
 *  LabelSurface.cc:  Label a specific surf in a surftree
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1997
 *
 *  Copyright (C) 1997 SCI Group
 *
 */

#include <config.h>
#include <Containers/String.h>
#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CoreDatatypes/BasicSurfaces.h>
#include <CommonDatatypes/SurfacePort.h>
#include <CoreDatatypes/SurfTree.h>
#include <TclInterface/TCLvar.h>
#include <iostream.h>
#include <Malloc/Allocator.h>
#include <Geometry/BBox.h>

namespace PSECommon {
namespace Modules {

using PSECommon::Dataflow::Module;
using PSECommon::CommonDatatypes::SurfTree;
using PSECommon::CommonDatatypes::SurfaceIPort;
using PSECommon::CommonDatatypes::SurfaceOPort;
using PSECommon::CommonDatatypes::SurfaceHandle;
using PSECommon::CommonDatatypes::TriSurface;

using namespace SCICore::TclInterface;
using SCICore::Containers::Array1;

class LabelSurface : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
    TCLint numberf;
    TCLstring namef;
    int generation;
    Array1<clString> origNames;
public:
    LabelSurface(const clString& id);
    LabelSurface(const LabelSurface&, int deep);
    virtual ~LabelSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_LabelSurface(const clString& id) {
  return new LabelSurface(id);
}

//static clString module_name("LabelSurface");

LabelSurface::LabelSurface(const clString& id)
: Module("LabelSurface", id, Filter), generation(-1),
  numberf("numberf", id, this), namef("namef", id, this)
{
    // Create the input ports
    iport=new SurfaceIPort(this, "In Surf", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);
}

LabelSurface::LabelSurface(const LabelSurface& copy, int deep)
: Module(copy, deep), generation(-1),
  numberf("numberf", id, this), namef("namef", id, this)
{
    NOT_FINISHED("LabelSurface::LabelSurface");
}

LabelSurface::~LabelSurface()
{
}

Module* LabelSurface::clone(int deep)
{
    return new LabelSurface(*this, deep);
}

void LabelSurface::execute()
{
    SurfaceHandle iSurf;
    int i;
    if(!iport->get(iSurf))
	return;
    SurfTree* st=iSurf->getSurfTree();
    TriSurface *ts=iSurf->getTriSurface();
    if (st) {
      if (st->generation != generation) {
	  origNames.resize(0);
	  for (i=0; i<st->surfI.size(); i++) origNames.add(st->surfI[i].name);
      }
      int fnum=numberf.get();
      clString fname=namef.get();
      
      for (i=0; i<st->surfI.size(); i++)
	  st->surfI[i].name=origNames[i];
      
      if (fnum>0 && fnum<st->surfI.size() && 
	  st->surfI[fnum].name != fname) {
	cerr << "Added label: "<<fname<<" to surface number: "<<fnum<<"\n";
	st->surfI[fnum].name=fname;
      }
    } else if (ts) {
      ts->name = namef.get();
    }
    oport->send(iSurf);
}	

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:57  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:27  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
