
/*
 *  CStoGeom.cc:  Convert a Mesh into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/ContourSet.h>
#include <DaveW/Datatypes/General/ContourSetPort.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomPolyline.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::Thread;
using std::cerr;

class CStoGeom : public Module {
    ContourSetIPort* ics;
    GeometryOPort* ogeom;

    void cs_to_geom(const ContourSetHandle&, GeomGroup*);
    TCLint showLines;
    TCLint showPoints;
public:
    CStoGeom(const clString& id);
    virtual ~CStoGeom();
    virtual void execute();
};

Module* make_CStoGeom(const clString& id)
{
    return scinew CStoGeom(id);
}

CStoGeom::CStoGeom(const clString& id)
: Module("CStoGeom", id, Filter), showPoints("showPoints", id, this),
  showLines("showLines", id, this)
{
    // Create the input port
    ics=scinew ContourSetIPort(this, "ContourSet", ContourSetIPort::Atomic);
    add_iport(ics);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

CStoGeom::~CStoGeom()
{
}

void CStoGeom::execute()
{
    ContourSetHandle cs;
    update_state(NeedData);
    if (!ics->get(cs))
	return;

    update_state(JustStarted);

    int min_matl=8000;
    int max_matl=-8000;
    
    int i,j;
    for (i=0; i<cs->level_map.size(); i++) {
	for (j=0; j<cs->level_map[i].size(); j++) {
	    if (cs->level_map[i][j]<min_matl) min_matl=cs->level_map[i][j];
	    if (cs->level_map[i][j]>max_matl) max_matl=cs->level_map[i][j];
	}
    }

    int ngroups=7;
    if (7>(max_matl-min_matl+1)) ngroups=max_matl-min_matl+1;

    Array1<GeomGroup* > lines(ngroups);
    Array1<GeomPts* > points(ngroups);
    Array1<GeomMaterial* > pmatl(ngroups);
    Array1<GeomMaterial* > lmatl(ngroups);

    for(i=0;i<ngroups;i++) lines[i] = scinew GeomGroup;
    for(i=0;i<ngroups;i++) points[i] = scinew GeomPts(0);
    
    for (i=0; i<cs->levels.size(); i++) {
	if (i%500 == 0) update_progress(i, cs->levels.size());
	for (j=0; j<cs->levels[i].size(); j++) {
	    GeomPolyline* lgroup = scinew GeomPolyline;
	    int matl=(cs->level_map[i][j]-min_matl)%7;
	    for (int k=0; k<cs->levels[i][j].size(); k++) {
		points[matl]->add(cs->levels[i][j][k]);
		lgroup->add(cs->levels[i][j][k]);
	    }
	    if (cs->levels[i][j].size()>1)
		lgroup->add(cs->levels[i][j][0]);
	    lines[matl]->add(lgroup);
	}
    }

    ogeom->delAll();
	
    MaterialHandle c[7];
    c[0]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);

    for(i=0;i<ngroups;i++) {
	pmatl[i] = scinew GeomMaterial(points[i], c[i]);
	lmatl[i] = scinew GeomMaterial(lines[i], c[i]);

	clString tmpl("ContourLine ");
	tmpl += (char) ('0' + (i+min_matl));

	clString tmpp("ContourPoints ");
	tmpp += (char) ('0' + (i+min_matl));

	if (showPoints.get())
	    ogeom->addObj(pmatl[i], tmpp());
	else
	    delete pmatl[i];
	if (showLines.get())
	    ogeom->addObj(lmatl[i], tmpl());
	else
	    delete lmatl[i];
    }
}

} // End namespace Modules
} // End namespace DaveW
