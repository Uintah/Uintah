/*
 *  MMeshToSFUG.cc:  Takes in a MultiMesh and a SFUG, and output a SFUG
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
#include <Datatypes/MultiMesh.h>
#include <Datatypes/MultiMeshPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>

class MMeshToSFUG : public Module {
    MultiMeshHandle mmesh_handle;
    ScalarFieldHandle sf_handle;
    MultiMeshIPort* immesh;
    ScalarFieldIPort* isf;
    ScalarFieldOPort* osf;
    TCLint total_levels;
    TCLint selected_level;
public:
    MMeshToSFUG(const clString& id);
    MMeshToSFUG(const MMeshToSFUG&, int deep);
    virtual ~MMeshToSFUG();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MMeshToSFUG(const clString& id)
{
    return new MMeshToSFUG(id);
}
};

MMeshToSFUG::MMeshToSFUG(const clString& id)
: Module("MMeshToSFUG", id, Filter), 
  total_levels("total_levels", id, this), 
  selected_level("selected_level", id, this)
{
    immesh=new MultiMeshIPort(this, "Input MultiMesh",MultiMeshIPort::Atomic);
    add_iport(immesh);
    isf=new ScalarFieldIPort(this, "Input ScalarField",MultiMeshIPort::Atomic);
    add_iport(isf);
    osf=new ScalarFieldOPort(this,"Output ScalarField",MultiMeshIPort::Atomic);
    add_oport(osf);
}

MMeshToSFUG::MMeshToSFUG(const MMeshToSFUG& copy, int deep)
: Module(copy, deep),
  total_levels("total_levels", id, this), 
  selected_level("selected_level", id, this)
{
    NOT_FINISHED("MMeshToSFUG::MMeshToSFUG");
}

MMeshToSFUG::~MMeshToSFUG()
{
}

Module* MMeshToSFUG::clone(int deep)
{
    return new MMeshToSFUG(*this, deep);
}

void MMeshToSFUG::execute()
{
    MultiMeshHandle new_mmesh_handle;
    ScalarFieldHandle new_sf_handle;
    if (immesh->get(new_mmesh_handle)) {
	mmesh_handle=new_mmesh_handle;
    }
    if (isf->get(new_sf_handle)) {
	sf_handle=new_sf_handle;
    }
    if (!mmesh_handle.get_rep()) {
	return;
    }
    if (!sf_handle.get_rep()) {
	return;
    }
    ScalarFieldUG* sfug;
    if (!(sfug=sf_handle->getUG())) return;
    if (total_levels.get() != mmesh_handle->meshes.size()) {
	total_levels.set(mmesh_handle->meshes.size());
    }	
    ScalarFieldUG *sfout = new ScalarFieldUG(sfug->typ);
    sfout->mesh = mmesh_handle->meshes[selected_level.get()-1];
    sfout->mesh.detach();
    sfout->mesh->detach_nodes();
    sfout->mesh->compute_neighbors();
    sfout->data = sfug->data;
    
    osf->send(sfout);
}
