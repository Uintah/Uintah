//static char *id="@(#) $Id$";

/*
 *  AddWells.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/GeometryPort.h>
#include <Geom/GeomCylinder.h>
#include <Geom/GeomGroup.h>
#include <Geom/GeomLine.h>
#include <Geom/Material.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLvar.h>
#include <fstream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class AddWells : public Module {
    TCLstring filename;
    TCLdouble radius;
    MaterialHandle lt_red;
    MaterialHandle lt_green;
    MaterialHandle lt_blue;
    MaterialHandle lt_purple;
    MaterialHandle lt_yellow;
    GeometryOPort* ogeom;
    int ndeep;
    double* dp;
    double* ms;
    double toms(double depth);
public:
    AddWells(const clString& id);
    AddWells(const AddWells&, int deep);
    virtual ~AddWells();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_AddWells(const clString& id) {
  return new AddWells(id);
}

static clString module_name("AddWells");

AddWells::AddWells(const clString& id)
: Module("AddWells", id, Source),
  filename("filename", id, this), radius("radius", id, this)
{
   // Create the output port
   ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);
   lt_red = scinew Material(Color(.2,0,0), Color(.8,0,0),
			 Color(.5,.5,.5), 20);
   lt_green = scinew Material(Color(0,.2,0), Color(0,.4,0),
			 Color(.5,.5,.5), 20);
   lt_blue = scinew Material(Color(0,0,.2), Color(0,0,.8),
			 Color(.5,.5,.5), 20);
   lt_purple = scinew Material(Color(.2,0,.2), Color(.8,0,.8),
			 Color(.5,.5,.5), 20);
   lt_yellow = scinew Material(Color(.2,.2,0), Color(.8,.8,0),
			 Color(.5,.5,.5), 20);
}

AddWells::AddWells(const AddWells& copy, int deep)
: Module(copy, deep),
  filename("filename", id, this), radius("radius", id, this)
{
   NOT_FINISHED("AddWells::AddWells");
}

AddWells::~AddWells()
{
}

Module* AddWells::clone(int deep)
{
   return scinew AddWells(*this, deep);
}

double AddWells::toms(double depth) {
    int i=0;
    while(i<ndeep){
	if(dp[i] > depth)
	    break;
	i++;
    }
    i--;
    double frac=(depth-dp[i])/(dp[i+1]-dp[i]);
    double m=frac*(ms[i+1]-ms[i])+ms[i];
    return m;
}

void AddWells::execute()
{
    clString fn(filename.get());
    ifstream in(fn());
    double rad=radius.get();
    GeomGroup* all=new GeomGroup;
    in >> ndeep;
    dp=new double[ndeep+1];
    ms=new double[ndeep+1];
    dp[0]=ms[0]=0;
    for(int i=0;i<ndeep;i++){
	in >> dp[i+1] >> ms[i+1];
    }
    while(in){
	double x, y, depth;
	int id;
	in >> id >> y >> x >> depth;
	if(!in)break;
	cerr << "y=" << y << ", x=" << x << ": ";
	y-=-97.89384;
	y*=133/(-97.94162- -97.89384);

	x-=33.17863;
	x*=97/(33.20800-33.17863);
	cerr << "y=" << y << ", x=" << x << '\n';
	all->add(new GeomLine(Point(x,y,toms(0)), Point(x,y,toms(depth))));
	int count;
	in >> count;
	if(!in)break;
	for(int i=0;i<count;i++){
	    double top, bottom;
	    char t[100];
	    in >> top >> bottom >> t;
	    clString type(t);
	    GeomCylinder* cyl=new GeomCylinder(Point(x,y,toms(top)),
					       Point(x,y,toms(bottom)),
					       rad, 12, 2);
	    MaterialHandle matl(lt_red);
	    if(type=="Oil"){
		matl=lt_blue;
	    } else if(type=="Oil/Gas"){
		matl=lt_purple;
	    }
	    all->add(new GeomMaterial(cyl, matl));
	}
    }
    delete[] ms;
    delete[] dp;
    ogeom->delAll();
    ogeom->addObj(new GeomMaterial(all, lt_green), "Wells");
    ogeom->flushViews();
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:58:10  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
