
/*
 *  Pick.h: Picking information for Geometry objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Pick_h
#define SCI_Geom_Pick_h 1

#include <Geom/Container.h>
#include <Geom/Material.h>
#include <Classlib/Array1.h>
#include <Geometry/Vector.h>
#include <Multitask/ITC.h>

class Roe;
class BaseWidget;
class MessageBase;
class Module;

struct BState {
   unsigned int control:1;
   unsigned int alt:1;
   unsigned int shift:1;
   unsigned int btn:2;
};


class GeomPick : public GeomContainer {
    Module* module;
    void* cbdata;
    Array1<Vector> directions;
    BaseWidget* widget;
    int widget_data;
    int selected;
    int ignore;
    MaterialHandle highlight;

    GeomPick(const GeomPick&);
public:
    GeomPick(GeomObj*, Module* module);
    GeomPick(GeomObj*, Module* module, BaseWidget*, int widget_data);
    GeomPick(GeomObj*, Module* module,
	     const Vector&);
    GeomPick(GeomObj*, Module* module,
	     const Vector&, const Vector&);
    GeomPick(GeomObj*, Module* module,
	     const Vector&, const Vector&, const Vector&);
    GeomPick(GeomObj*, Module* module, const Array1<Vector>&);
    virtual ~GeomPick();
    virtual GeomObj* clone();
    int nprincipal();
    Vector principal(int i);
    void set_principal(const Vector&);
    void set_principal(const Vector&, const Vector&);
    void set_principal(const Vector&, const Vector&, const Vector&);
    void set_highlight(const MaterialHandle& matl);
    void set_module_data(void*);
    void set_widget_data(int);

    void pick(Roe* roe, const BState& bs);
    void moved(int axis, double distance, const Vector& delta, const BState& bs);
    void release(const BState& bs);

    void ignore_until_release();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_Pick_h */
