
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

#include <Datatypes/Datatype.h>

#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>

class Point;
class TriSurface;

class Surface;
typedef LockingHandle<Surface> SurfaceHandle;

class Surface : public Datatype {
protected:
    enum Representation {
	TriSurf,
    };
    Surface(Representation);
private:
    Representation rep;
public:
    clString name;
    Surface(const Surface& copy);
    virtual ~Surface();
    virtual Surface* clone()=0;
    virtual int inside(const Point& p)=0;

    TriSurface* getTriSurface();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_Surface_h */
