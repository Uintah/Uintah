
/*
 *  TopoSurfTree.cc: Tree of non-manifold bounding surfaces
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
  *
 *  Copyright (C) 1997 SCI Group
 */
#include <iostream.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/TopoSurfTree.h>
#include <Malloc/Allocator.h>

static Persistent* make_TopoSurfTree()
{
    return scinew TopoSurfTree;
}

PersistentTypeID TopoSurfTree::type_id("TopoSurfTree", "Surface", make_TopoSurfTree);

TopoSurfTree::TopoSurfTree(Representation r)
: SurfTree(r)
{
}

TopoSurfTree::TopoSurfTree(const TopoSurfTree& copy, Representation)
: SurfTree(copy)
{
    NOT_FINISHED("TopoSurfTree::TopoSurfTree");
}

TopoSurfTree::~TopoSurfTree() {
}	

Surface* TopoSurfTree::clone()
{
    return scinew TopoSurfTree(*this);
}

GeomObj* TopoSurfTree::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("TopoSurfTree::get_obj");
    return 0;
}

#define TopoSurfTree_VERSION 1

void TopoSurfTree::io(Piostream& stream) {
    int version=stream.begin_class("TopoSurfTree", TopoSurfTree_VERSION);
    SurfTree::io(stream);		    
    Pio(stream, patches);
    Pio(stream, wires);
    Pio(stream, junctions);
    stream.end_class();
}
