/*
 *  MultiMesh.h: Unstructured MultiMeshes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/MultiMesh.h>
#include <Classlib/String.h>
#include <iostream.h>

static Persistent* make_MultiMesh()
{
    return new MultiMesh;
}

PersistentTypeID MultiMesh::type_id("MultiMesh", "Datatype", make_MultiMesh);

MultiMesh::MultiMesh()
{
}

MultiMesh::MultiMesh(const MultiMesh& copy)
: meshes(copy.meshes)
{
}

MultiMesh::~MultiMesh() {
}

void MultiMesh::clean_up() {

    for (int i=0; i<meshes.size(); i++) {
	meshes[i]->compute_neighbors();
	meshes[i]->remove_delaunay(0,0);
	meshes[i]->remove_delaunay(1,0);
	meshes[i]->remove_delaunay(2,0);
	meshes[i]->remove_delaunay(3,0);
	meshes[i]->pack_all();
    }
}
    

MultiMesh* MultiMesh::clone()
{
    return new MultiMesh(*this);
}

#define MultiMesh_VERSION 1

void MultiMesh::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("MultiMesh", MultiMesh_VERSION);
    Pio(stream, meshes);
    stream.end_class();
}
