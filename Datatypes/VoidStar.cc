
/*
 *  VoidStar.cc:Just has a rep member -- other trivial classes can inherit
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

#include <Datatypes/VoidStar.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

PersistentTypeID VoidStar::type_id("VoidStar", "Datatype", 0);

VoidStar::VoidStar(Representation rep)
: rep(rep)
{
}

VoidStar::VoidStar(const VoidStar& copy)
: rep(copy.rep)
{
    NOT_FINISHED("VoidStar::VoidStar");
}

VoidStar::~VoidStar()
{
}

#define VoidStar_VERSION 1
void VoidStar::io(Piostream& stream) {
    stream.begin_class("VoidStar", VoidStar_VERSION);
    int* repp=(int*)&rep;
    Pio(stream, *repp);
    stream.end_class();
}




// Here's the code for the Phantoms

Phantoms* VoidStar::getPhantoms() {
    if (rep==PhantomsType) {
	return (Phantoms*)this;
    } else
	return 0;
}

static Persistent* make_Phantoms()
{
    return scinew Phantoms;
}
PersistentTypeID Phantoms::type_id("Phantoms", "VoidStar", make_Phantoms);

Phantoms::Phantoms(Representation r)
: VoidStar(r)
{
}

Phantoms::Phantoms(const Phantoms& copy)
: VoidStar(copy)
{
    NOT_FINISHED("Phantoms::Phantoms");
}

Phantoms::~Phantoms() {
}

VoidStar* Phantoms::clone()
{
    return scinew Phantoms(*this);
}

#define Phantoms_VERSION 1
void Phantoms::io(Piostream& stream) {
    /* int version=*/stream.begin_class("Phantoms", Phantoms_VERSION);
    VoidStar::io(stream);
    Pio(stream, objs);
    stream.end_class();
}
void Pio(Piostream& stream, Phantom& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p.type);
    Pio(stream, p.min);
    Pio(stream, p.max);
    Pio(stream, p.T1);
    Pio(stream, p.T2);
    stream.end_cheap_delim();
}



// Here's the code for the pulses

Pulses* VoidStar::getPulses() {
    if (rep==PulsesType) {
	return (Pulses*)this;
    } else
	return 0;
}

static Persistent* make_Pulses()
{
    return scinew Pulses;
}
PersistentTypeID Pulses::type_id("Pulses", "VoidStar", make_Pulses);

Pulses::Pulses(Representation r)
: VoidStar(r)
{
}

Pulses::Pulses(const Pulses& copy)
: VoidStar(copy)
{
    NOT_FINISHED("Pulses::Pulses");
}

Pulses::~Pulses() {
}

VoidStar* Pulses::clone()
{
    return scinew Pulses(*this);
}

#define Pulses_VERSION 1
void Pulses::io(Piostream& stream) {
    /* int version=*/stream.begin_class("Pulses", Pulses_VERSION);
    VoidStar::io(stream);
    Pio(stream, objs);
    stream.end_class();
}

void Pio(Piostream& stream, Pulse& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p.name);
    Pio(stream, p.start);
    Pio(stream, p.stop);
    Pio(stream, p.amplitude);
    Pio(stream, p.direction);
    Pio(stream, p.samples);
    stream.end_cheap_delim();
}


#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<VoidStar>;

#include <Classlib/Array1.cc>
template class Array1<Phantom>
template class Array1<Pulse>
template void Pio(Piostream&, Array1<Phantom>&);
template void Pio(Piostream&, Array1<Pulse>&);
#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<Phantom>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array1<Pulse>& p2)
{
    Pio(p1, p2);
}

#endif
#endif
