/*
 *  SelectMesh.cc:  SelectMesh from a MultiMesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MultiMesh.h>
#include <Datatypes/MultiMeshPort.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>

class SelectMesh : public Module {
    MultiMeshHandle mmesh_handle;
    MultiMeshIPort* immesh;
    MeshOPort* omesh;
    TCLint total_levels;
    TCLint selected_level;
public:
    SelectMesh(const clString& id);
    SelectMesh(const SelectMesh&, int deep);
    virtual ~SelectMesh();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_SelectMesh(const clString& id)
{
    return new SelectMesh(id);
}

static RegisterModule db1("Mesh", "SelectMesh", make_SelectMesh);
static RegisterModule db2("Dave", "SelectMesh", make_SelectMesh);

SelectMesh::SelectMesh(const clString& id)
: Module("SelectMesh", id, Filter), 
  total_levels("total_levels", id, this), 
  selected_level("selected_level", id, this)
{
    immesh=new MultiMeshIPort(this, "Input MultiMesh",MultiMeshIPort::Atomic);
    add_iport(immesh);
    omesh=new MeshOPort(this, "Output Mesh", MeshIPort::Atomic);
    add_oport(omesh);
}

SelectMesh::SelectMesh(const SelectMesh& copy, int deep)
: Module(copy, deep),
  total_levels("total_levels", id, this), 
  selected_level("selected_level", id, this)
{
    NOT_FINISHED("SelectMesh::SelectMesh");
}

SelectMesh::~SelectMesh()
{
}

Module* SelectMesh::clone(int deep)
{
    return new SelectMesh(*this, deep);
}

void SelectMesh::execute()
{
//    cerr << "In SelectMesh...\n";
//    if (mmesh_handle.get_rep()) {
//	for (int i=0; i<mmesh_handle->meshes.size(); i++) {
//	    if (mmesh_handle->meshes[i].get_rep()) {
//		cerr << "   LastMesh " << i << " had " << mmesh_handle->meshes[i]->elems.size() << " elements.\n";
//	    } else {
//		cerr << "   LastMesh " << i << " was an empty handle.\n";
//	    }
//	}
//    } else {
//	cerr << "   No Meshes in Last Mesh\n";
//    }

    MultiMeshHandle new_mmesh_handle;
    if (immesh->get(new_mmesh_handle)) {
//	cerr << "Going with the old one...\n";
	mmesh_handle=new_mmesh_handle;
    }
    if (!mmesh_handle.get_rep()) {
//	cerr << "No MultiMesh -- returning...\n";
	return;
    }
//    for (int i=0; i<mmesh_handle->meshes.size(); i++) {
//	if (mmesh_handle->meshes[i].get_rep()) {
//	    cerr << "   NewMesh " << i << " had " << mmesh_handle->meshes[i]->elems.size() << " elements.\n";
//	} else {
//	    cerr << "   NewMesh " << i << " was an empty handle.\n";
//	}
//    }
    if (total_levels.get() != mmesh_handle->meshes.size()) {
	total_levels.set(mmesh_handle->meshes.size());
    }	
//    cerr << "Sending Mesh: " << selected_level.get()-1 << "\n";
    omesh->send(mmesh_handle->meshes[selected_level.get()-1]);
}
