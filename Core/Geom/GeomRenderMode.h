
/*
 * GeomRenderMode.h:  Object to switch rendering mode
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_RenderMode_h
#define SCI_Geom_RenderMode_h 1

#include <SCICore/Geom/GeomContainer.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomRenderMode : public GeomContainer {
public:
    enum DrawType {
        WireFrame,
        Flat,
        Gouraud
    };
private:
    DrawType drawtype;
public:
    GeomRenderMode(DrawType, GeomObj* child);
    GeomRenderMode(const GeomRenderMode&);
    virtual ~GeomRenderMode();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/10/07 02:07:44  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/28 17:54:42  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:24  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:43  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:07  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:00  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//


#endif /* SCI_Geom_RenderMode_h */
