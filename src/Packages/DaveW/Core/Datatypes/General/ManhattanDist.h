
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

#ifndef SCI_Packages_DaveW_Datatypes_ManhattanDist_h
#define SCI_Packages_DaveW_Datatypes_ManhattanDist_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Geometry/Point.h>

namespace DaveW {
using namespace SCIRun;

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
} // End namespace DaveW



#endif

