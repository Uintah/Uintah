
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

#include <sci_config.h>
#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/BasicSurfaces.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <Core/Geometry/BBox.h>

namespace SCIRun {



class LabelSurface : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
    TCLint numberf;
    TCLstring namef;
    int generation;
    Array1<clString> origNames;
public:
    LabelSurface(const clString& id);
    virtual ~LabelSurface();
    virtual void execute();
};

extern "C" Module* make_LabelSurface(const clString& id) {
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

LabelSurface::~LabelSurface()
{
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

} // End namespace SCIRun

