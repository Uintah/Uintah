
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

#include <Core/share/share.h>

#include <Core/Persistent/Persistent.h>

#include <Core/Geom/Color.h>
#include <Core/Geom/View.h>

#include <iosfwd>

namespace SCIRun {


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

} // End namespace SCIRun


#endif // ifndef GeomScene_h

