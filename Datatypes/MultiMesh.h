
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

#ifndef SCI_project_MultiMesh_h
#define SCI_project_MultiMesh_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/Mesh.h>

class MultiMesh;
typedef LockingHandle<MultiMesh> MultiMeshHandle;

class MultiMesh : public Datatype {
public:
    Array1<MeshHandle> meshes;

    MultiMesh();
    MultiMesh(const MultiMesh&);
    virtual MultiMesh *clone();
    virtual ~MultiMesh();
    inline void add_mesh(const MeshHandle &m, int i) {
	meshes[i]=m->clone();
    }
    void clean_up();
    void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
