
/*
 *  PickMessage.h: Messages back to Modules about pick info
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_PickMessage_h
#define SCI_Geom_PickMessage_h 1

#include <Comm/MessageBase.h>
#include <Geometry/Vector.h>
class Module;

class GeomPickMessage : public MessageBase {
public:
    Module* module;
    int axis;
    double distance;
    Vector delta;
    void* cbdata;
    GeomPickMessage(Module*, void*);
    GeomPickMessage(Module*, void*, int);
    GeomPickMessage(Module*, int, double, const Vector&, void*);
    virtual ~GeomPickMessage();
};

#endif /* SCI_Geom_Pick_h */
