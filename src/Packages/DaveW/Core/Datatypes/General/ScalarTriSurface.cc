
/*
 *  ScalarTriSurface.cc: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ScalarTriSurface.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Grid.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

namespace DaveW {
static Persistent* make_ScalarTriSurface()
{
    return scinew ScalarTriSurface;
}

PersistentTypeID ScalarTriSurface::type_id("ScalarTriSurface", "Surface", make_ScalarTriSurface);

ScalarTriSurface::ScalarTriSurface()
: TriSurface(Unused)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts, const Array1<double>& d)
: TriSurface(ts, Unused), data(d)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts)
: TriSurface(ts, Unused)
{
}

ScalarTriSurface::ScalarTriSurface(const ScalarTriSurface& copy)
: TriSurface(copy, Unused), data(copy.data)
{
    NOT_FINISHED("ScalarTriSurface::ScalarTriSurface");
}

ScalarTriSurface::~ScalarTriSurface() {
}
#define ScalarTriSurface_VERSION 1

void ScalarTriSurface::io(Piostream& stream) {
using namespace SCIRun;

    /*int version=*/
    stream.begin_class("ScalarTriSurface", ScalarTriSurface_VERSION);
    TriSurface::io(stream);
    Pio(stream, data);
    stream.end_class();
}

Surface* ScalarTriSurface::clone()
{
    return scinew ScalarTriSurface(*this);
}

GeomObj* ScalarTriSurface::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("ScalarTriSurface::get_obj");
    return 0;
}
} // End namespace DaveW


