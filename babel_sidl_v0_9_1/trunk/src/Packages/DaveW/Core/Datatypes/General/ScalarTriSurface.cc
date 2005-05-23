
/*
 *  ScalarTriSurfFieldace.cc: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ScalarTriSurfFieldace.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Grid.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

namespace DaveW {
static Persistent* make_ScalarTriSurfFieldace()
{
    return scinew ScalarTriSurfFieldace;
}

PersistentTypeID ScalarTriSurfFieldace::type_id("ScalarTriSurfFieldace", "Surface", make_ScalarTriSurfFieldace);

ScalarTriSurfFieldace::ScalarTriSurfFieldace()
: TriSurfFieldace(Unused)
{
}

ScalarTriSurfFieldace::ScalarTriSurfFieldace(const TriSurfFieldace& ts, const Array1<double>& d)
: TriSurfFieldace(ts, Unused), data(d)
{
}

ScalarTriSurfFieldace::ScalarTriSurfFieldace(const TriSurfFieldace& ts)
: TriSurfFieldace(ts, Unused)
{
}

ScalarTriSurfFieldace::ScalarTriSurfFieldace(const ScalarTriSurfFieldace& copy)
: TriSurfFieldace(copy, Unused), data(copy.data)
{
    NOT_FINISHED("ScalarTriSurfFieldace::ScalarTriSurfFieldace");
}

ScalarTriSurfFieldace::~ScalarTriSurfFieldace() {
}
#define ScalarTriSurfFieldace_VERSION 1

void ScalarTriSurfFieldace::io(Piostream& stream) {
using namespace SCIRun;

    /*int version=*/
    stream.begin_class("ScalarTriSurfFieldace", ScalarTriSurfFieldace_VERSION);
    TriSurfFieldace::io(stream);
    Pio(stream, data);
    stream.end_class();
}

Surface* ScalarTriSurfFieldace::clone()
{
    return scinew ScalarTriSurfFieldace(*this);
}

GeomObj* ScalarTriSurfFieldace::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("ScalarTriSurfFieldace::get_obj");
    return 0;
}
} // End namespace DaveW


