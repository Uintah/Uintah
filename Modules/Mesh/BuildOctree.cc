/*
 *  BuildOctree.cc:  Build an octree from a Scalar{/Vector}FieldRG
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
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/Octree.h>
#include <Datatypes/OctreePort.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Malloc/Allocator.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>
#include <Widgets/CrosshairWidget.h>

class BuildOctree : public Module {
    ScalarFieldIPort* isf;
    VectorFieldIPort* ivf;
    OctreeOPort* otree;
    GeometryOPort* owidget;
    
    clString waiting_command;

    OctreeTopHandle treeHandle;    
    CrosshairWidget* widget;
    CrowdMonitor widget_lock;
    TCLint same_input;
public:
    BuildOctree(const clString& id);
    BuildOctree(const BuildOctree&, int deep);
    virtual ~BuildOctree();
    virtual Module* clone(int deep);
    void partial_execute();
    virtual void geom_moved(GeomPick*, int, double, const Vector& delta, void*);
    virtual void geom_release(GeomPick*, void *);
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_BuildOctree(const clString& id)
{
    return scinew BuildOctree(id);
}
};

BuildOctree::BuildOctree(const clString& id)
: Module("BuildOctree", id, Filter), same_input("same_input", id, this)
{
    waiting_command = "";
    isf=scinew ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_iport(isf);
    ivf=scinew VectorFieldIPort(this, "VectorField", VectorFieldIPort::Atomic);
    add_iport(ivf);
    otree=scinew OctreeOPort(this, "Output Octree", OctreeIPort::Atomic);
    add_oport(otree);
    owidget=scinew GeometryOPort(this, "Widget Geometry", GeometryIPort::Atomic);
    add_oport(owidget);
    widget = scinew CrosshairWidget(this, &widget_lock, .1);
    owidget->addObj(widget->GetWidget(), "CrosshairWidget", &widget_lock);
    owidget->flushViews();
}

BuildOctree::BuildOctree(const BuildOctree& copy, int deep)
: Module(copy, deep), same_input("same_input", id, this)
{
    NOT_FINISHED("BuildOctree::BuildOctree");
}

BuildOctree::~BuildOctree()
{
}

Module* BuildOctree::clone(int deep)
{
    return scinew BuildOctree(*this, deep);
}

void BuildOctree::geom_moved(GeomPick*, int, double, const Vector&,
				void*)
{    
}

void BuildOctree::geom_release(GeomPick*, void*)
{    
}

void BuildOctree::execute()
{
    ScalarFieldHandle sfHandle;
    ScalarFieldRG* sfrg;
    VectorFieldHandle vfHandle;
    VectorFieldRG* vfrg;

    sfrg=0;
    vfrg=0;

    // handle tcl commands
    if (waiting_command != "") {
	if (treeHandle.get_rep() && treeHandle->tree) {
	    if (waiting_command == "push_level") {
		treeHandle->tree->push_level(widget->GetPosition());
	    } else if (waiting_command == "pop_level") {
		treeHandle->tree->pop_level(widget->GetPosition());
	    } else if (waiting_command == "top_level") {
		treeHandle->tree->top_level();
	    } else if (waiting_command == "bottom_level") {
		treeHandle->tree->bottom_level();
	    } else if (waiting_command == "push_all_levels") {
		treeHandle->tree->push_all_levels();
	    } else if (waiting_command == "pop_all_levels") {
		treeHandle->tree->pop_all_levels();
	    }
	}
	waiting_command = "";
    }

    // if no input
    if ((!isf->get(sfHandle)) && (!ivf->get(vfHandle))) return;

    // no unstructured fields!
    if ((sfHandle.get_rep()) && (!(sfrg=sfHandle->getRG()))) {
	cerr << "Can't build an octree with an unstructured scalar field!\n";
	sfrg=0;
    }
    if ((vfHandle.get_rep()) && (!(vfrg=vfHandle->getRG()))) {
	cerr << "Can't build an octree with an unstructured vector field!\n";
	vfrg=0;
    }

    // "Same Input" button pushed and no new connections
    if (treeHandle.get_rep() && treeHandle->tree && same_input.get()) {
	otree->send(treeHandle);
	return;
    }

    // if both are fields are empty
    if (!sfrg && !vfrg) return;
	
    // "Same Input" button is pushed, we already have a tree, and there's
    // a new scalar field that's just been attached
    // Or not same input and there's a scalar field waiting
    if ((same_input.get() && (!(treeHandle.get_rep() && treeHandle->tree) || 
			      !treeHandle->scalars) && sfrg) ||
	(!same_input.get() && sfrg)) {

	// if not "same input" and we have a tree and the scalar field
	// is the same size as our tree we'll reuse it.
	if (!same_input.get() && treeHandle.get_rep() && treeHandle->tree && 
	    treeHandle->scalars &&
	    (treeHandle->nx == sfrg->nx) &&
	    (treeHandle->ny == sfrg->ny) &&
	    (treeHandle->nz == sfrg->nz)) {
	    treeHandle->scalars=1;
	    treeHandle->tree->erase_last_level_scalars();
	    treeHandle->tree->insert_scalar_field(sfrg);
	    treeHandle->tree->build_scalar_tree();

	// if we have a tree and vectors in it    
	} else if (treeHandle.get_rep() && treeHandle->tree &&
		   treeHandle->vectors) {

	    // if same size add scalars
	    if ((treeHandle->nx == sfrg->nx) &&
		(treeHandle->ny == sfrg->ny) &&
		(treeHandle->nz == sfrg->nz)) {
		if (treeHandle->scalars) {
		    treeHandle->tree->erase_last_level_scalars();
		} else {
		    treeHandle->scalars=1;
		}
		treeHandle->tree->insert_scalar_field(sfrg);
		treeHandle->tree->build_scalar_tree();

	    // can't add scalar field if we already had vectors of a diff size
	    } else {
		cerr << "Scalar field is wrong size\n";
	    }

	// build a new scalar tree
	} else {
	    treeHandle=0;  
	    BBox bb;
	    Point pmin, pmax;
	    sfrg->get_bounds(pmin, pmax);
	    bb.extend(pmin);
	    bb.extend(pmax);
	    treeHandle = scinew OctreeTop(sfrg->nx, sfrg->ny, sfrg->nz, bb);
	    treeHandle->scalars=1;
	    treeHandle->tree->insert_scalar_field(sfrg);
	    treeHandle->tree->build_scalar_tree();
	}

    // "Same Input" button is pushed, we already have a tree, and there's
    // a new vector field that's just been attached
    // Or not same input and there's a vector field waiting
    } else if ((same_input.get() && 
		(!(treeHandle.get_rep() && treeHandle->tree) || 
			      !treeHandle->vectors) && vfrg) ||
	(!same_input.get() && vfrg)) {

	// if not "same input" and we have a tree and the vector field
	// is the same size as our tree we'll reuse it.
	if (!same_input.get() && treeHandle.get_rep() && treeHandle->tree && 
	    treeHandle->vectors &&
	    (treeHandle->nx == vfrg->nx) &&
	    (treeHandle->ny == vfrg->ny) &&
	    (treeHandle->nz == vfrg->nz)) {
	    treeHandle->vectors=1;
	    treeHandle->tree->erase_last_level_vectors();
	    treeHandle->tree->insert_vector_field(vfrg);
	    treeHandle->tree->build_vector_tree();

	// if we have a tree and scalars in it    
	} else if (treeHandle.get_rep() && treeHandle->tree &&
		   treeHandle->scalars) {

	    // if same size add vectors
	    if ((treeHandle->nx == vfrg->nx) &&
		(treeHandle->ny == vfrg->ny) &&
		(treeHandle->nz == vfrg->nz)) {
		if (treeHandle->vectors) {
		    treeHandle->tree->erase_last_level_vectors();
		} else {
		    treeHandle->vectors=1;
		}
		treeHandle->tree->insert_vector_field(vfrg);
		treeHandle->tree->build_vector_tree();

	    // can't add vector field if we already had scalars of a diff size
	    } else {
		cerr << "Vector field is wrong size\n";
	    }

	// build a new vector tree
	} else {
	    treeHandle=0;  
	    BBox bb;
	    Point pmin, pmax;
		vfrg->get_bounds(pmin, pmax);
	    bb.extend(pmin);
	    bb.extend(pmax);
	    treeHandle = scinew OctreeTop(vfrg->nx, vfrg->ny, vfrg->nz, bb);
	    treeHandle->vectors=1;
	    treeHandle->tree->insert_vector_field(vfrg);
	    treeHandle->tree->build_vector_tree();
	}
    }
    otree->send(treeHandle);
}

void BuildOctree::tcl_command(TCLArgs& args, void* userdata)
{
   if(args.count() < 2){
      args.error("BuildOctree needs a minor command");
      return;
   }
   if (args[1] == "push_level") {
       waiting_command = "push_level";
       want_to_execute();
   } else if (args[1] == "pop_level") {
       waiting_command = "pop_level";
       want_to_execute();
   } else if (args[1] == "top_level") {
       waiting_command = "top_level";
       want_to_execute();
   } else if (args[1] == "bottom_level") {
       waiting_command = "bottom_level";
       want_to_execute();
   } else if (args[1] == "push_all_levels") {
       waiting_command = "push_all_levels";
       want_to_execute();
   } else if (args[1] == "pop_all_levels") {
       waiting_command = "pop_all_levels";
       want_to_execute();
   } else {
       Module::tcl_command(args, userdata);
   }
}
