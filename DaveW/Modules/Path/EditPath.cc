
/*
 *  EditPath.cc:  Convert a Mesh into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/GeometryPort.h>
#include <DaveW/Datatypes/General/PathPort.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace DaveW::Datatypes;

class EditPath : public Module {
    PathIPort* ipath;
    PathOPort* opath;
    GeometryOPort* ogeom;

    MusilRNG mr;
    TCLint showElems;
    TCLint showNodes;
public:
    EditPath(const clString& id);
    virtual ~EditPath();
    virtual void execute();
};

Module* make_EditPath(const clString& id)
{
    return scinew EditPath(id);
}

EditPath::EditPath(const clString& id)
: Module("EditPath", id, Filter), showNodes("showNodes", id, this),
  showElems("showElems", id, this)
{
    // Create the input port
    ipath=scinew PathIPort(this, "Path", PathIPort::Atomic);
    add_iport(ipath);
    opath=scinew PathOPort(this, "Path", PathIPort::Atomic);
    add_oport(opath);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

EditPath::~EditPath()
{
}

void EditPath::execute()
{
    PathHandle path;
    update_state(NeedData);
    ipath->get(path);
    update_state(JustStarted);
    View v(Point(mr()*2-1, mr()*2-1, mr()*2-1), Point(0,0,0), Vector(0,0,1), 10);
    
    ogeom->setView(0,v);
}

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.2  1999/12/03 00:29:23  dmw
// improved the module for John Day...
//
// Revision 1.1  1999/12/02 21:57:33  dmw
// new camera path datatypes and modules
//
