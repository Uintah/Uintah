

#ifndef GeomScene_h
#define GeomScene_h 1

#include <Classlib/Persistent.h>
#include <Geom/Color.h>
#include <Geom/View.h>
#include <Classlib/Boolean.h>

class Lighting;
class GeomObj;
class ostream;

struct GeomScene : public Persistent {
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

#endif

