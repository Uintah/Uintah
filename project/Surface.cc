/*
 *  Surface.cc: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Surface.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>

PersistentTypeID Surface::typeid("Surface", "Datatype", 0);
static Persistent* make_TriSurface()
{
    return new TriSurface;
}
PersistentTypeID TriSurface::typeid("TriSurface", "Surface", make_TriSurface);

Surface::Surface() {
}

Surface::~Surface() {
}

Surface::Surface(const Surface& copy) {
}

#define SURFACE_VERSION 1

void Surface::io(Piostream& stream) {
    int version=stream.begin_class("Surface", SURFACE_VERSION);
    // Nothing to store...
    stream.end_class();
}

TriSurface::TriSurface() {
}

TriSurface::TriSurface(const TriSurface& t) {
}

TriSurface::~TriSurface() {
}

int TriSurface::inside(const Point& p) {
    return 1;
}

ObjGroup* TriSurface::getGeomFromSurface() {
    ObjGroup* group = new ObjGroup;
    for (int i=0; i<elements.size(); i++) {
	group->add(new Triangle(points[elements[i]->i1], 
				points[elements[i]->i2],
				points[elements[i]->i3]));
    }
    return group;
}

void TriSurface::add_point(const Point& p) {
    points.add(p);
}

void TriSurface::add_triangle(int i1, int i2, int i3) {
    elements.add(new TSElement(i1, i2, i3));
}

#define TRISURFACE_VERSION 1

void TriSurface::io(Piostream& stream) {
    int version=stream.begin_class("TriSurface", TRISURFACE_VERSION);
    Pio(stream, points);
    Pio(stream, elements);
    stream.end_class();
}

void Pio(Piostream& stream, TSElement*& data)
{
    if(stream.reading())
	data=new TSElement(0,0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->i1);
    Pio(stream, data->i2);
    Pio(stream, data->i3);
    stream.end_cheap_delim();
}
