
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

#ifndef SCI_project_ManhattanDist_h
#define SCI_project_ManhattanDist_h 1

#include <Classlib/Array1.h>
#include <Classlib/Array3.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Geometry/Point.h>

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
    double dist2(const Point& p);
    int distFast(const Point& p);
    void partial_initialize();
    void computeCellDistance(int i, int j, int k);
    void initialize();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

