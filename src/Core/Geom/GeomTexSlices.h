
/*
 *  GeomTexSlices.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_GeomTexSlices_h
#define SCI_Geom_GeomTexSlices_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array3.h>

namespace SCIRun {


class SCICORESHARE GeomTexSlices : public GeomObj {
    Point min, max;
    int nx, ny, nz;
    unsigned int texids[3];
    int have_drawn;
public:
    double accum;
    double bright;
    Array3<char> Xmajor;
    Array3<char> Ymajor;
    Array3<char> Zmajor;

    GeomTexSlices(int, int, int, const Point&, const Point&);
    GeomTexSlices(const GeomTexSlices&);
    virtual ~GeomTexSlices();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif /* SCI_Geom_Grid_h */
