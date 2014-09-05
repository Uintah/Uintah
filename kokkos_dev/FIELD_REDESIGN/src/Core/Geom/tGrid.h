
/*
 *  tGrid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_tGrid_h
#define SCI_Geom_tGrid_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array2.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE TexGeomGrid : public GeomObj {
    int tmap_size;
    int tmap_dlist;
    Point corner;
    Vector u, v, w;
    void adjust();

    unsigned short* tmapdata; // texture map
    int MemDim;
    int dimU,dimV;

    int num_chan;
    int convolve;
    int conv_dim;

    int kernal_change;

    float conv_data[25];
public:
    TexGeomGrid(int, int, const Point&, const Vector&, const Vector&,
		int chanels=3);
    TexGeomGrid(const TexGeomGrid&);
    virtual ~TexGeomGrid();

    virtual GeomObj* clone();

    void set(unsigned short* data,int datadim);

    void do_convolve(int dim, float* data);
  
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:51  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:35  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:54  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:15  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:15  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Grid_h */
