
/*
 *  Surface.h: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Surface_h
#define SCI_project_Surface_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/HashTable.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Thread/CrowdMonitor.h>

namespace SCICore {

namespace Geometry {
  class Grid;
}
namespace Geom {
  class GeomObj;
}

namespace Datatypes {

using Containers::LockingHandle;
using Geometry::Point;
using Geometry::Grid;
using GeomSpace::GeomObj;
using Containers::HashTable;
using Containers::Array1;
using Containers::clString;

class  Surface;
struct Node;
class TriSurface;
class PointsSurface;
class SurfTree;

typedef LockingHandle<Surface> SurfaceHandle;
typedef LockingHandle<Node> NodeHandle;

class SCICORESHARE Surface : public Datatype {
protected:
    enum Representation {
	TriSurf,
	PointsSurf,
	Unused,
	STree,
	RepOther
    };
    Surface(Representation, int closed);
private:
    Representation rep;
public:
    SCICore::Thread::CrowdMonitor monitor;
    int hash_x;
    int hash_y;
    Point hash_min;
    double resolution;
    int closed;
    clString name;
    Grid *grid;
    HashTable<int, int> *pntHash;

    // Boundary conditions...
    enum BoundaryType {
	DirichletExpression,
	DirichletData,
	BdryNone
    };
    clString boundary_expr;
    BoundaryType boundary_type;
    void set_bc(const clString& expr);

    Surface(const Surface& copy);
    virtual ~Surface();
    virtual Surface* clone()=0;
    virtual int inside(const Point& p)=0;
    virtual void construct_grid(int, int, int, const Point &, double)=0;
    virtual void construct_grid()=0;
    virtual void destroy_grid();
    virtual void destroy_hash();
    SurfTree* getSurfTree();
    TriSurface* getTriSurface();
    PointsSurface* getPointsSurface();
    virtual void get_surfnodes(Array1<NodeHandle>&)=0;
    virtual void set_surfnodes(const Array1<NodeHandle>&)=0;
    virtual GeomObj* get_obj(const ColorMapHandle&)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/09/05 05:32:28  dmw
// updated and added Modules from old tree to new
//
// Revision 1.4  1999/08/28 17:54:36  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/25 03:48:42  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:55  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:29  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:56  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:47  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:29  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:44  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_Surface_h */
