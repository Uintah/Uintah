
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

#include <Classlib/Array1.h>
#include <Geom/Geom.h>
#include <Geometry/Vector.h>
#include <Multitask/ITC.h>

class MessageBase;
class Module;

class GeomPick {
    Array1<Vector> directions;
    MaterialHandle hightlight;
    Mailbox<MessageBase*>* mailbox;
    void* cbdata;
    Module* module;
    GeomPick(const GeomPick&);
public:
    GeomPick(Module*);
    GeomPick(Module* module, const Vector&);
    GeomPick(Module* module, const Vector&, const Vector&);
    GeomPick(Module* module, const Vector&, const Vector&, const Vector&);
    GeomPick(Module* module, const Array1<Vector>&);
    ~GeomPick();
    int nprincipal();
    Vector principal(int i);
    void set_principal(const Vector&);
    void set_principal(const Vector&, const Vector&);
    void set_principal(const Vector&, const Vector&, const Vector&);
    void set_highlight(const MaterialHandle& matl);
    void set_reply(Mailbox<MessageBase*>*);
    void set_cbdata(void*);

    void pick();
    void moved(int axis, double distance,
	       const Vector& delta);
    void release();
};

#endif /* SCI_Geom_Pick_h */
