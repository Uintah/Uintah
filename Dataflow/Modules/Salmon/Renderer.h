
/*
 *  Renderer.h: Abstract interface to a renderer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Renderer_h
#define SCI_project_Renderer_h

#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
  namespace GeomSpace {
    class Color;
    class GeomObj;
    class GeomPick;
    class View;
  }
  namespace TclInterface {
    class TCLArgs;
  }
  namespace Multitask {
    template<class T> class AsyncReply;
    class Semaphore;
  }
}

namespace PSECore {
  namespace Datatypes {
    struct GeometryData;
  }
}

namespace PSECommon {
namespace Modules {

using PSECore::Datatypes::GeometryData;

using SCICore::Containers::clString;
using SCICore::Containers::AVLTree;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::GeomPick;
using SCICore::GeomSpace::Color;
using SCICore::GeomSpace::View;
using SCICore::TclInterface::TCLArgs;
using SCICore::Multitask::AsyncReply;

class Roe;
class Salmon;
class Renderer;

typedef Renderer* (*make_Renderer)();
typedef int (*query_Renderer)();
class RegisterRenderer;

class Renderer {
public:
    static Renderer* create(const clString& type);
    static AVLTree<clString, RegisterRenderer*>* get_db();

    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height)=0;
    virtual void old_redraw(Salmon*, Roe*);
    virtual void redraw(Salmon*, Roe*, double tbeg, double tend,
			int nframes, double framerate);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&, int&)=0;
    virtual void hide()=0;
    virtual void dump_image(const clString&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat=1)=0;
    virtual void listvisuals(TCLArgs&);
    virtual void setvisual(const clString& wname, int i, int width, int height);

    int compute_depth(Roe* roe, const View& view, double& near, double& far);

    int xres, yres;
    virtual void getData(int datamask, AsyncReply<GeometryData*>* result);
};

class RegisterRenderer {
public:
    clString name;
    query_Renderer query;
    make_Renderer maker;
    RegisterRenderer(const clString& name, query_Renderer tester,
		     make_Renderer maker);
    ~RegisterRenderer();
};

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/25 03:47:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:37:38  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:52  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:10  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//

#endif
