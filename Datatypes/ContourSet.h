
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

#include <Datatypes/Datatype.h>

#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class ContourSet;
typedef LockingHandle<ContourSet> ContourSetHandle;

class ContourSet : public Datatype {
public:
    Array1<Array1<Point> > contours;
    Vector basis[3];
    Vector origin;
    BBox bbox;
    double space;
    clString name;

    ContourSet();
    ContourSet(const ContourSet &copy);
    virtual ~ContourSet();
    virtual ContourSet* clone();
    void translate(const Vector &v);
    void scale(double sc);
    void rotate(const Vector &rot);
    void build_bbox();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_ContourSet_h */
