
/*
 *  GeomScene.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef GeomScene_h
#define GeomScene_h 1

#include <SCICore/share/share.h>

#include <SCICore/Persistent/Persistent.h>

#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/View.h>

#include <iosfwd>

namespace SCICore {
namespace GeomSpace {

using SCICore::PersistentSpace::Persistent;
using SCICore::Containers::clString;

class Lighting;
class GeomObj;

struct SCICORESHARE GeomScene : public Persistent {
    GeomScene();
    GeomScene(const Color& bgcolor, const View& view, Lighting* lighting,
	     GeomObj* topobj);
    Color bgcolor;
    View view;
    Lighting* lighting;
    GeomObj* top;
    virtual void io(Piostream&);
    bool save(const clString& filename, const clString& format);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/10/07 02:07:45  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:39:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:07  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:01  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//

#endif // ifndef GeomScene_h

