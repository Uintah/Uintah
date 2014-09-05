
/*
 *  TriStrip.h: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_TriStrip_h
#define SCI_Geom_TriStrip_h 1

#include <SCICore/Geom/GeomVertexPrim.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomTriStrip : public GeomVertexPrim {
public:
    GeomTriStrip();
    GeomTriStrip(const GeomTriStrip&);
    virtual ~GeomTriStrip();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    int size(void);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomTriStripList : public GeomObj {
    int n_strips;
    Array1<float> pts;
    Array1<float> nrmls;
    Array1<int>   strips;
public:
    GeomTriStripList();
    virtual ~GeomTriStripList();

    virtual GeomObj* clone();

    void add(const Point&);
    void add(const Point&, const Vector&);
    
    void end_strip(void); // ends a tri-strip

    Point get_pm1(void);
    Point get_pm2(void);

    void permute(int,int,int);
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void get_bounds(BBox&);

   int size(void);
   int num_since(void);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:47  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:28  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:16  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:47  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:09  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:05  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_TriStrip_h */
