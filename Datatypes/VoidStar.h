
/*
 *  VoidStar.h: Just has a rep member -- other trivial classes can inherit
 *		from this, rather than having a full-blown datatype and data-
 *		port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Datatypes_VoidStar_h
#define SCI_Datatypes_VoidStar_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>

class Phantoms;
class Pulses;

class VoidStar;
typedef LockingHandle<VoidStar> VoidStarHandle;
class VoidStar : public Datatype {
protected:
    enum Representation {
	PhantomsType,
        PulsesType,
        Other
    };
    VoidStar(Representation);
private:
    Representation rep;
public:
    VoidStar(const VoidStar& copy);
    virtual ~VoidStar();
    virtual VoidStar* clone()=0;
    Pulses* getPulses();
    Phantoms* getPhantoms();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


// Phantoms class definition

typedef struct _Phantom {
    clString type;
    Point min;
    Point max;
    double T1;
    double T2;
} Phantom;
void Pio(Piostream&, Phantom&);

class Phantoms : public VoidStar {
public:
    Array1<Phantom> objs;
public:
    Phantoms(Representation r=PhantomsType);
    Phantoms(const Phantoms& copy);
    virtual ~Phantoms();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};



// Pulses class definition

typedef struct _Pulse {
    clString name;
    double start;
    double stop;
    double amplitude;
    char direction;
    int samples;
} Pulse;
void Pio(Piostream&, Pulse&);

class Pulses : public VoidStar {
public:
    Array1<Pulse> objs;
public:
    Pulses(Representation r=PulsesType);
    Pulses(const Pulses& copy);
    virtual ~Pulses();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
