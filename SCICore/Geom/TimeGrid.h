
/*
 *  TimeGrid.h: ?
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_TimeGeom_Grid_h
#define SCI_TimeGeom_Grid_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array2.h>

class GeomColormapInterface;

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE TimeGrid : public GeomObj {
    Array1<float *> tmap;

    float *bmap; // blend map...

    int dimU,dimV;
    int tmap_size; // all the same size!!!

    Array1<int>    tmap_dlist;
    Array1<double> time;

    Point corner;
    Vector u, v, w; // all the same!!!
    int active;
    void adjust();
public:
    TimeGrid(int,int, int, const Point&, const Vector&, const Vector&);
    TimeGrid(const TimeGrid&);
    virtual ~TimeGrid();

    virtual GeomObj* clone();

    // methor for textur mapping...

    GeomColormapInterface* map; // color map to be used...

    void set_active(int, double);

    void set(int,int, const MaterialHandle&, const double&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:34  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:14  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:14  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Grid_h */
