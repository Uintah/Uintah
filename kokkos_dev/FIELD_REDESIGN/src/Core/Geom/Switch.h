
/*
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Switch_h
#define SCI_Geom_Switch_h 1

#include <SCICore/Geom/GeomContainer.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomSwitch : public GeomContainer {
    int state;
    GeomSwitch(const GeomSwitch&);
public:
    GeomSwitch(GeomObj*, int state=1);
    virtual ~GeomSwitch();
    virtual GeomObj* clone();
    void set_state(int st);
    int get_state();
    virtual void get_bounds(BBox&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomTimeSwitch : public GeomContainer {
    double tbeg;
    double tend;
    GeomTimeSwitch(const GeomTimeSwitch&);
public:
    GeomTimeSwitch(GeomObj*, double tbeg, double tend);
    virtual ~GeomTimeSwitch();
    virtual GeomObj* clone();

    // For OpenGL
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
// Revision 1.4  1999/10/07 02:07:50  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:34  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:23  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:52  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:13  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:13  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Switch_h */
