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
#include <Geometry/Point.h>

Surface::Surface() {
}

Surface::~Surface() {
}

Surface::Surface(const Surface& copy) {
}

void Surface::io(Piostream& s) {
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

void TriSurface::add_point(const Point& p) {
    points.add(p);
}

void TriSurface::add_triangle(int i1, int i2, int i3) {
    elements.add(new TSElement(i1, i2, i3));
}

