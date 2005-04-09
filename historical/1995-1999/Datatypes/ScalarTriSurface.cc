
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
#include <iostream.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/TrivialAllocator.h>
#include <Classlib/Queue.h>
#include <Datatypes/ScalarTriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Grid.h>
#include <Math/MiscMath.h>
#include <Malloc/Allocator.h>

static Persistent* make_ScalarTriSurface()
{
    return scinew ScalarTriSurface;
}

PersistentTypeID ScalarTriSurface::type_id("ScalarTriSurface", "Surface", make_ScalarTriSurface);

ScalarTriSurface::ScalarTriSurface()
: TriSurface(ScalarTriSurf)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts, const Array1<double>& d)
: TriSurface(ts, ScalarTriSurf), data(d)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts)
: TriSurface(ts, ScalarTriSurf)
{
}

ScalarTriSurface::ScalarTriSurface(const ScalarTriSurface& copy)
: TriSurface(copy, ScalarTriSurf), data(copy.data)
{
    NOT_FINISHED("ScalarTriSurface::ScalarTriSurface");
}

ScalarTriSurface::~ScalarTriSurface() {
}
#define ScalarTriSurface_VERSION 1

void ScalarTriSurface::io(Piostream& stream) {
    /*int version=*/stream.begin_class("ScalarTriSurface", ScalarTriSurface_VERSION);
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
