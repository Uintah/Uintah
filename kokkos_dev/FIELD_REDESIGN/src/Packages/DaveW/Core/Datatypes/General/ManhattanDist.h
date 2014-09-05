
/*
 *  ManhattanDist.h:  For coregistering a set of points to a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_ManhattanDist_h
#define SCI_DaveW_Datatypes_ManhattanDist_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Geometry/Point.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Datatypes::ScalarFieldRGint;
using SCICore::Datatypes::ScalarField;
using SCICore::Geometry::Point;
using SCICore::Containers::Array1;
using SCICore::Containers::Array3;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ManhattanDist : public ScalarFieldRGint {
    Array1<Point> pts;
    int fullyInitialized;
public:
    Array3<Array1<int> > closestNodeIdx;
    ManhattanDist(const Array1<Point>&pts, int n=100, int init=0,
		  double minX=0, double minY=0, double minZ=0,
		  double maxX=1, double maxY=1, double maxZ=1);
    ManhattanDist(const ManhattanDist& copy);
    ManhattanDist();
    virtual ~ManhattanDist();
    virtual ScalarField* clone();
    double dist(const Point& p);
    double dist(const Point& p, int &idx);
    double dist2(const Point& p);
    double dist2(const Point& p, int &idx);
    int distFast(const Point& p);
    void partial_initialize();
    void computeCellDistance(int i, int j, int k);
    void initialize();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:47:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:52:59  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:01  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif

