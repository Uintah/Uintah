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
    return new Mesh;
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
    int nnodes=meshes[meshes.size()-1]->nodes.size();
    for (int i=0; i<meshes.size(); i++) {
	for (int j=0; j<meshes[i]->elems.size(); j++) {
	    if ((meshes[i]->elems[j]) && 
		(meshes[i]->elems[j]->n[0] >= nnodes ||
		 meshes[i]->elems[j]->n[1] >= nnodes ||
		 meshes[i]->elems[j]->n[2] >= nnodes ||
		 meshes[i]->elems[j]->n[3] >= nnodes))
		meshes[i]->elems[j]=0;
	}
	meshes[i]->pack_elems();
	meshes[i]->compute_neighbors();
    }
}
    
MultiMesh* MultiMesh::clone()
{
    return new MultiMesh(*this);
}

#define MultiMesh_VERSION 1

void MultiMesh::io(Piostream& stream)
{
    int version=stream.begin_class("MultiMesh", MultiMesh_VERSION);
    Pio(stream, meshes);
    stream.end_class();
}
