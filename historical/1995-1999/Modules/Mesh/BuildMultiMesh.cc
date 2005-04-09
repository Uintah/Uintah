/*
 *  BuildMultiMesh.cc:  BuildMultiMesh Triangulation in 3D
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
#include <Classlib/Pstreams.h>		// for writing temp meshes out
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/MultiMeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MultiMesh.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Pt.h>
#include <Geom/Switch.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>
#include <Widgets/PointWidget.h>

#include <stdio.h>

using sci::Element;
using sci::Node;
using sci::Mesh;
using sci::MeshHandle;

class BuildMultiMesh : public Module {
    ScalarFieldIPort* isf;
    ColorMapIPort* icmap;
    MultiMeshOPort* ommesh;
    ScalarFieldOPort* osf;
    GeometryOPort* opoints;
    Array1<GeometryOPort* > owidgets;

    MeshHandle mesh_handle;
    ScalarFieldHandle sf_handle;
    ScalarFieldUG* sfug;
    ScalarFieldUG* sfout;
    Array1<int> point_ids;
    CrowdMonitor widget_lock;
    Array1<PointWidget *> widgets;
    Array1<GeomSwitch *> geom_switches;

    Array1<double> weightings;
    Array1<Array1<int> > level_sets;
    Array1<MaterialHandle> level_matl;

    Array1<int> node_map;
    Array1<int> last_source_sel;
    Array1<double> last_charge;
    Array1<double> last_falloff;
    Array1<double> last_wsize;
    Array1<TCLint* > source_sel;
    Array1<TCLdouble* > charge;
    Array1<TCLdouble* > falloff;
    Array1<TCLdouble* > wsize;
    double last_bias;
    int last_levels;
    int last_gl_falloff;
    int widget_changed;
    int numNodes;
    int need_to_addObj_widget;
    TCLint PE;
    TCLint numSources;
    TCLint sameInput;
    TCLint gl_falloff;
    TCLint levels;
    TCLdouble bias;
    TCLdouble min_weight;
    TCLdouble max_weight;

    clString myid;
    int have_ever_executed;
    int need_full_execute;
    MultiMeshHandle mmeshHndl;
    MultiMesh *mmesh;
    int want_partial_execute_only;

    int read1;
    int read2;
    int write;

public:
    BuildMultiMesh(const clString& id);
    BuildMultiMesh(const BuildMultiMesh&, int deep);
    virtual ~BuildMultiMesh();
    virtual Module* clone(int deep);
    void partial_execute();
    virtual void geom_moved(GeomPick*, int, double, const Vector& delta, void*);
    virtual void geom_release(GeomPick*, void *);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_BuildMultiMesh(const clString& id)
{
    return scinew BuildMultiMesh(id);
}
}

BuildMultiMesh::BuildMultiMesh(const clString& id)
: Module("BuildMultiMesh", id, Filter), numSources("numSources", id, this),
  have_ever_executed(0), sameInput("sameInput", id, this), 
  levels("levels", id, this), gl_falloff("gl_falloff", id, this),
  min_weight("min_weight", id, this), max_weight("max_weight", id, this),
  PE("PE", id, this), need_to_addObj_widget(0), want_partial_execute_only(0), 
  bias("bias", id, this), read1(0), read2(0), write(0)
{
    myid=id;
    widget_changed=0;
    isf=scinew ScalarFieldIPort(this, "Input Field", ScalarFieldIPort::Atomic);
    add_iport(isf);
    icmap=scinew ColorMapIPort(this, "Input ColorMap", ColorMapIPort::Atomic);
    add_iport(icmap);
    ommesh=scinew MultiMeshOPort(this, "Output MultiMesh",MultiMeshIPort::Atomic);
    add_oport(ommesh);
    opoints=scinew GeometryOPort(this, "Node Geometry", GeometryIPort::Atomic);
    add_oport(opoints);
    osf=scinew ScalarFieldOPort(this, "Ouput Field", ScalarFieldIPort::Atomic);
    add_oport(osf);
    owidgets.add(scinew GeometryOPort(this, "Node Geometry", GeometryIPort::Atomic));
    add_oport(owidgets[0]);
}

BuildMultiMesh::BuildMultiMesh(const BuildMultiMesh& copy, int deep)
: Module(copy, deep), numSources("numSources", id, this),
  have_ever_executed(0), sameInput("sameInput", id, this),
  levels("levels", id, this), gl_falloff("gl_falloff", id, this),
  min_weight("min_weight", id, this), max_weight("max_weight", id, this),
  PE("PE", id, this), need_to_addObj_widget(0), want_partial_execute_only(0),
  bias("bias", id, this), read1(0), read2(0), write(0)
{
    NOT_FINISHED("BuildMultiMesh::BuildMultiMesh");
}

BuildMultiMesh::~BuildMultiMesh()
{
}

Module* BuildMultiMesh::clone(int deep)
{
    return scinew BuildMultiMesh(*this, deep);
}

void BuildMultiMesh::connection(ConnectionMode mode, int which_port, 
				int output) {
    if (!output) return;
    if (which_port >= 3) {
	if (mode==Disconnected) {
	    numSources.set(numSources.get()-1);
	    remove_oport(which_port);
	    delete owidgets[which_port];
	    owidgets.remove(which_port);
	    widget_changed=1;
	} else {
	    numSources.set(numSources.get()+1);
	    GeometryOPort* g=scinew 
		GeometryOPort(this, "Attractor Widget", GeometryIPort::Atomic);
	    add_oport(g);
	    owidgets.add(g);
	    PointWidget *new_pw=scinew PointWidget(this, &widget_lock, 1);
	    widgets.add(new_pw);
	    GeomSwitch *gs=scinew GeomSwitch(new_pw->GetWidget(),1);
	    geom_switches.add(gs);
	    need_to_addObj_widget++;
	    clString srcName;
	    srcName = "s" + to_string(widgets.size());
	    source_sel.add(scinew TCLint(srcName, myid, this));
	    source_sel[source_sel.size()-1]->set(1);
	    last_source_sel.add(1);
	    clString chrName;
	    chrName = "ch" + to_string(widgets.size());
	    charge.add(scinew TCLdouble(chrName, myid, this));
	    charge[charge.size()-1]->set(1);
	    last_charge.add(1);
	    clString faName;
	    faName = "fa" + to_string(widgets.size());
	    falloff.add(scinew TCLdouble(faName, myid, this));
	    falloff[falloff.size()-1]->set(.01);
	    last_falloff.add(.01);
	    clString szName;
	    szName = "sz" + to_string(widgets.size());
	    wsize.add(scinew TCLdouble(szName, myid, this));
	    wsize[wsize.size()-1]->set(.1);
	    last_wsize.add(.1);
	    widget_changed=1;
	    want_to_execute();
	}
    }
}

void BuildMultiMesh::geom_release(GeomPick*, void *) {
    if (PE.get()) {
	want_partial_execute_only=1;
	widget_changed=1;
	want_to_execute();
    }
}

void BuildMultiMesh::geom_moved(GeomPick*, int, double, const Vector&,
				void*)
{    
}

// Called when node weightings change, and as a precursor to full execution
// Handles all of the "first time through" allocations, and calculating
//	the new node weightings.  The actual multimesh construction is
//	handled in execute().
// Handles outputing the widget and the mesh points through the ogeom and
//	owidget ports.
// Ouputs the multimesh if nothing changed. (need_full_execute won't have to)
// Orders and groups nodes by weightings (matl indicates node level)
void BuildMultiMesh::partial_execute() {
    int data_changed=widget_changed;
    if (read2) {
	data_changed=1;
	widgets[0]->SetPosition(Point(0.576, 0.464, 0.7295));
cerr << "Widgets[0] is at: "<<widgets[0]->GetPosition()<<"\n";
    }
    widget_changed=0;
    if(!isf->get(sf_handle))
	return;
    if (!(sfug=sf_handle->getUG())) {
	cerr << "Can't handle Regular Grids in BuildMultiMesh";
	return;
    }
    mesh_handle=sfug->mesh;
    Mesh *mesh=mesh_handle.get_rep();
    if (!have_ever_executed || !sameInput.get()) {
	if (!have_ever_executed) {
	    level_sets.resize(levels.get());
	    level_matl.resize(levels.get());
	    have_ever_executed = 1;
	}
	data_changed=1;
    }
    
    // now we have to compare old and new values
    if (levels.get() != last_levels) {
	data_changed=1;
	level_sets.resize(levels.get());
	level_matl.resize(levels.get());
    }
    if (gl_falloff.get() != last_gl_falloff) data_changed=1;
    if (bias.get() != last_bias) data_changed=1;
    if (!data_changed) {	
	for (int i=0; i<numSources.get(); i++) {
	    if (source_sel[i]->get() != last_source_sel[i]) data_changed=1;
	    if (charge[i]->get() != last_charge[i]) data_changed=1;
	    if (falloff[i]->get() != last_falloff[i]) data_changed=1;
	    if (wsize[i]->get() != last_wsize[i]) data_changed=1;
	}
    }

    // update "last" values
    last_levels=levels.get();
    last_gl_falloff=gl_falloff.get();
    last_bias=bias.get();
    for (int i=0; i<numSources.get(); i++) {
	last_source_sel[i]=source_sel[i]->get();
	widget_lock.write_lock();
	geom_switches[i]->set_state(last_source_sel[i]);
	widget_lock.write_unlock();
	last_charge[i]=charge[i]->get();
	last_falloff[i]=falloff[i]->get();
	last_wsize[i]=wsize[i]->get();
    }
    if (data_changed) {
	numNodes=mesh->nodes.size();
	weightings.resize(numNodes);
	Array1<double> rand_list(numNodes);
	MusilRNG rng(numNodes);
	double min=10000000;
	double max=-10000000;
	double total=0;
	// compute new node weights
	int i;
cerr << "Widgets[0] is at: "<<widgets[0]->GetPosition()<<"\n";
	for (i=0; i<numNodes; i++) {
//	    rand_list[i]=.5;
	    rand_list[i]=rng();
	    weightings[i]=0;
	    for (int src=0; src<numSources.get(); src++) {
		if (last_source_sel[src]) {
		    double dist=(mesh->nodes[i]->p -
				  widgets[src]->GetPosition()).length();
		    weightings[i]+=last_charge[src]*
			Exp(-dist*last_falloff[src]);
		}
	    }
	    if (weightings[i] > max) max=weightings[i];
	    if (weightings[i] < min) min=weightings[i];
	    total+=weightings[i];
	}
	for (i=0; i<numNodes; i++) {
	    weightings[i]-=min;
	}
	total-=min*numNodes;
	double recip_avg=numNodes/total;
	for (i=0; i<numSources.get(); i++) {
	    if (last_charge[i] > max) max=last_charge[i];
	    if (last_charge[i] < min) min=last_charge[i];
	}
	// update min and max values of ColorMap -- build cmap if necessary
	min_weight.set(min);
	max_weight.set(max);
	ColorMapHandle cmap;
	int have_cmap=icmap->get(cmap);
	if (!have_cmap) {
	    cmap=scinew ColorMap(30, min, max);
	    cmap->build_default();
	}
	// set source widget sizes and colors, and on/off switches

	for (i=0; i<numSources.get(); i++) {
	    widgets[i]->SetMaterial(PointWidget::PointMatl, cmap->lookup(last_charge[i]));
	    widgets[i]->SetScale(last_wsize[i]);
	    widget_lock.write_lock();
	    geom_switches[i]->set_state(last_source_sel[i]);
	    widget_lock.write_unlock();
	}
	while (need_to_addObj_widget) {
	    clString name = "Source " + 
		to_string(widgets.size()+1-need_to_addObj_widget);
	    owidgets[owidgets.size()-1-need_to_addObj_widget]->
		addObj(geom_switches[geom_switches.size()-
				     need_to_addObj_widget], name, 
		       &widget_lock);
	    need_to_addObj_widget--;
	}
	owidgets[0]->flushViews();

	// calculate cut-off percentages for each level
	Array1<double> cut_offs(last_levels);
	for (i=0; i<last_levels; i++) {
	    level_sets[i].remove_all();
	    if (i==0) {
		cut_offs[i]=last_gl_falloff/100.;;
	    } else {
		cut_offs[i]=cut_offs[i-1]*last_gl_falloff/100.;
	    }
	}

	// add nodes to level_sets
	for (i=0; i<numNodes; i++) {
            int done=0;
	    double sc_wtng=weightings[i]*recip_avg;
	    double bias_wtng=sc_wtng*last_bias+rand_list[i]*(1-last_bias);
	    for (int lvl=0; !done && lvl<last_levels-1; lvl++) {
		if (bias_wtng > cut_offs[lvl]) {
		    level_sets[lvl].add(i);
		    done=1;
		}
	    }
	    if (!done) {
		level_sets[last_levels-1].add(i);
	    }
	}

	node_map.resize(numNodes);
	for (int lvl=0, cc=0; lvl<level_sets.size(); lvl++) {
	    for (int k=0; k<level_sets[lvl].size(); k++, cc++) {
		for (i=0; i<numNodes; i++) {
		    if ((level_sets[lvl])[k] == i) {
			node_map[cc]=i;
		    }
		}
	    }
	}

	// set level_materials
	for (i=0; i<last_levels; i++) {
	    if (level_sets[i].size()) {
		level_matl[i]=cmap->lookup(min+(max-min)*i/(last_levels-1.));
	    } else {
		level_matl[i]=scinew Material(Color(0,0,0), Color(.7,.7,.7),
					   Color(.5,.5,.5), 20);
	    }
	}

	// delete old node geometry
	for (i=0; i<point_ids.size(); i++) {
	    if (point_ids[i]) {
		opoints->delObj(point_ids[i]);
		point_ids[i]=0;
	    }
	}

	// create and send new node geometry
	point_ids.resize(last_levels);
	for (i=0; i<last_levels; i++) {
	    GeomPts *geomPts=scinew GeomPts(level_sets[i].size());
	    for (int j=0; j<level_sets[i].size(); j++) {
		geomPts->add(mesh->nodes[(level_sets[i])[j]]->p);
	    }
	    GeomMaterial *geomMat=scinew GeomMaterial(geomPts, level_matl[i]);
	    clString pntName="Nodes Level " + to_string(i+1);
	    point_ids[i]=opoints->addObj(geomMat, pntName);
	}
	opoints->flushViews();
	need_full_execute=1;
cerr << "Recomputing and resending the nodes w/ new materials!\n";
    }
    if (write) {
	FILE *f;
	char nm[250];
	for (i=0; i<last_levels; i++) {
	    sprintf(nm,"/home/sci/u2/dweinste/mydata/mm/dump/%d/a.pts", i);
	    f=fopen(nm,"wt");
	    for (int k=0; k<=i; k++) {
		for(int j=0; j<level_sets[k].size(); j++) {
		    Point p(mesh->nodes[(level_sets[k])[j]]->p);
		    fprintf(f, "%lf %lf %lf\n", p.x(), p.y(), p.z());
		}
	    }
	    fclose(f);
	}
	write=0;
    }
}

#if 0

void BuildMultiMesh::partial_execute() {
    int data_changed=widget_changed;
    if(!isf->get(sf_handle))
	return;
    if (!(sfug=sf_handle->getUG())) {
	cerr << "Can't handle Regular Grids in BuildMultiMesh";
	return;
    }
    mesh_handle=sfug->mesh;
    Mesh *mesh=mesh_handle.get_rep();
    if (!have_ever_executed) {
	level_sets.resize(levels.get());
	level_matl.resize(levels.get());
	have_ever_executed = 1;
	data_changed=1;
    }
    
    // now we have to compare old and new values
    if (levels.get() != last_levels) {
	data_changed=1;
	level_sets.resize(levels.get());
	level_matl.resize(levels.get());
    }

    // update "last" values
    last_levels=levels.get();
    if (data_changed) {
	numNodes=mesh->nodes.size();
	int numPer=numNodes/level_sets.size();
	for (int i=0; i<level_sets.size(); i++) {
	    level_matl[i]=new Material(Color(0,0,0), Color(.7,.7,.7),
				       Color(.5,.5,.5), 20);
	    GeomPts *geomPts=new GeomPts(numPer);
	    for (int j=i*numPer; j<(i+1)*numPer; j++) {
		geomPts->add(mesh->nodes[j]->p);
	    }
	    GeomMaterial *geomMat=new GeomMaterial(geomPts, level_matl[i]);
	    clString pntName="Nodes Level "+ to_string(i+1);
	    opoints->addObj(geomMat, pntName);
	}
	opoints->flushViews();
	need_full_execute=1;
    }
}

#endif
// Handles multimesh construction.  partial_execute() checks changed status,
//	and handles first time allocations/namings.  This code builds the
//	multimesh from the weightings through an incremental Delaunay
//	triangulation.
// Ouputs the multimesh.
void BuildMultiMesh::execute()
{
    partial_execute();

    if (want_partial_execute_only) {
	return;
    } else {
	want_partial_execute_only=1;
    }


    if (!sfug || !mesh_handle.get_rep() || !need_full_execute) return;
    need_full_execute=0;

    mmeshHndl=mmesh=0;
    mmeshHndl=mmesh=scinew MultiMesh;
    mmesh->meshes.resize(last_levels);
    MeshHandle mesh;

    if (!read1 && !read2) {
    mesh = scinew Mesh;

    BBox bbox;
    int nn=mesh_handle->nodes.size();
    int i;
    for(i=0;i<nn;i++)
        bbox.extend(mesh_handle->nodes[i]->p);

    double epsilon=1.e-4;

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    bbox.extend(bbox.max()-Vector(epsilon, epsilon, epsilon));
    bbox.extend(bbox.min()+Vector(epsilon, epsilon, epsilon));

    // Make the bbox square...
    Point center(bbox.center());
    double le=1.0001*bbox.longest_edge();
    Vector diag(le, le, le);
    Point bmin(center-diag/2.);

    // Make the initial mesh with a tetra which encloses the bounding
    // box.  The first point is at the minimum point.  The other 3
    // have one of the coordinates at bmin+diagonal*5??

    mesh->nodes.add(new Node(bmin-Vector(le, le, le)));
    mesh->nodes.add(new Node(bmin+Vector(le*5, 0, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, le*5, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, 0, le*5)));

    Element* el=new Element(mesh.get_rep(), 0, 1, 2, 3);
    el->orient();
    el->faces[0]=el->faces[1]=el->faces[2]=el->faces[3]=-1;
    mesh->elems.add(el);
    }

    sfout = new ScalarFieldUG(ScalarFieldUG::NodalValues);

    int count=0;
    for (int i=0; i<level_sets.size(); i++) {

	if (read1 || read2) {
	    int r;
	    if (read1) r=1; else r=2;
	    char fn[250];
	    sprintf(fn, "/home/sci/u2/dweinste/mydata/mm/w%d/%d/a.mesh", r, i);
	    Piostream* stream=auto_istream(fn);
	    Pio(*stream, mesh);
	}

	for (int j=0; j<level_sets[i].size(); j++) {

	    if (!read1 && !read2 && count<100)
		mesh->insert_delaunay(mesh_handle->nodes[(level_sets[i])[j]]->p);

	    sfout->data.add(sfug->data[node_map[count]]);
	    if (!((count++)%50)) {	//update progress every fifty nodes
		update_progress(count, mesh_handle->nodes.size());
	    }
	}
	if (!read1 && !read2)
	mesh->pack_elems();
	mmesh->add_mesh(mesh, i);
    }
    if (!read1 && !read2) 
	mmesh->clean_up();
    read1=read2=0;
    ommesh->send(mmeshHndl);
    osf->send(sfout);
}

void BuildMultiMesh::tcl_command(TCLArgs& args, void* userdata)
{
   if(args.count() < 2){
      args.error("BuildMultiMesh needs a minor command");
      return;
   }
   if (args[1] == "partial_execute") {
       widget_changed=1;
       want_to_execute();
   } else if (args[1] == "read1") {
       need_full_execute=1;
       read1=1;
       want_to_execute();
   } else if (args[1] == "read2") {
       need_full_execute=1;
       read2=1;
       want_to_execute();
   } else if (args[1] == "write") {
       want_partial_execute_only=1;
       write=1;
       want_to_execute();
   } else if (args[1] == "needexecute") {
       cerr << "In needexecute...\n";
       want_partial_execute_only=0;
       Module::tcl_command(args, userdata);
   } else {
       Module::tcl_command(args, userdata);
   }
}

