/*
 * PhantomData.cc : Contains the PhantomXYZ and PhantomUVW classes inherited
 *		    from the VoidStar class.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Modules/Haptics/PhantomData.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

// PHANTOM XYZ DEFNS

static Persistent* make_PhantomXYZ()
{
    return scinew PhantomXYZ;
}
PersistentTypeID PhantomXYZ::type_id("PhantomXYZ", "VoidStar", make_PhantomXYZ);

PhantomXYZ::PhantomXYZ()
: VoidStar(), sem(0), Esem(0)
{
   
}

PhantomXYZ::PhantomXYZ(const Vector& XYZ) : VoidStar(),
sem(0), Esem(0)
{
    position=XYZ;
}

PhantomXYZ::PhantomXYZ(const PhantomXYZ& copy)
: position(copy.position), VoidStar(copy), sem(0), Esem(0)
{
    NOT_FINISHED("PhantomXYZ::PhantomXYZ");
}

PhantomXYZ::~PhantomXYZ() {
}

VoidStar* PhantomXYZ::clone()
{
    return scinew PhantomXYZ(*this);
}

#define PhantomXYZ_VERSION 1
void PhantomXYZ::io(Piostream& stream) {
    /* int version=*/stream.begin_class("PhantomXYZ", PhantomXYZ_VERSION);
    VoidStar::io(stream);
    Pio(stream, position);
    stream.end_class();
}

// PHANTOM UVW DEFINITIONS
static Persistent* make_PhantomUVW()
{
    return scinew PhantomUVW;
}
PersistentTypeID PhantomUVW::type_id("PhantomUVW", "VoidStar", make_PhantomUVW);

PhantomUVW::PhantomUVW()
: VoidStar(), sem(0)
{
}

PhantomUVW::PhantomUVW(const Vector& UVW) : VoidStar(), sem(0)
{
    force = UVW;
}

PhantomUVW::PhantomUVW(const PhantomUVW& copy)
: force(copy.force), VoidStar(copy), sem(0)
{
    NOT_FINISHED("PhantomUVW::PhantomUVW");
}

PhantomUVW::~PhantomUVW() {
}

VoidStar* PhantomUVW::clone()
{
    return scinew PhantomUVW(*this);
}

#define PhantomUVW_VERSION 1
void PhantomUVW::io(Piostream& stream) {
    /* int version=*/stream.begin_class("PhantomUVW", PhantomUVW_VERSION);
    VoidStar::io(stream);
    Pio(stream, force);
    stream.end_class();
}
