
/*
 *  HeadLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_HeadLight_h
#define SCI_Geom_HeadLight_h 1

#include <SCICore/share/share.h>

#include <SCICore/Geom/Light.h>
#include <SCICore/Geom/Color.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::PersistentSpace::Persistent;

class SCICORESHARE HeadLight : public Light {
    Color c;
public:
    HeadLight(const clString& name, const Color&);
    virtual ~HeadLight();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:31  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:18  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:48  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:11  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:08  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_HeadLight_h */
