
/*
 *  GeomText.h:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#ifndef SCI_Geom_Text_h
#define SCI_Geom_Text_h 1

#include <SCICore/Geom/GeomObj.h>
#ifdef _WIN32
#define WINGDIAPI __declspec(dllimport)
#define APIENTRY __stdcall
#define CALLBACK APIENTRY
#endif
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomText : public GeomObj {
  static int init;
  static GLuint fontbase;
public:
  clString text;
  Point at;
  Color c;
public:
  GeomText();
  GeomText( const clString &, const Point &, const Color &c = Color(1,1,1));
  GeomText(const GeomText&);
  virtual ~GeomText();
  virtual GeomObj* clone();

  virtual void reset_bbox();
  virtual void get_bounds(BBox&);

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
// Revision 1.4  1999/10/07 02:07:46  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:26  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:45  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:08  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:03  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//

#endif /* SCI_Geom_Text_h */

