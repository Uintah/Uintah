
/*
 *  ContourSet.h: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ContourSet_h
#define SCI_project_ContourSet_h 1

#include <Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/String.h>

class ContourSet;
typedef LockingHandle<ContourSet> ContourSetHandle;

class ContourSet : public Datatype {
public:
    Array1<Array1<Point> > contours;
    Vector basis[3];
    Vector origin;
    double space;
    clString name;

    ContourSet();
    ContourSet(const ContourSet &copy);
    virtual ~ContourSet();
    void translate(const Vector &v);
    void scale(double sc);
    void rotate(const Vector &rot);
    // Persistent representation...
    virtual void io(Piostream&);
};

#endif /* SCI_project_ContourSet_h */
