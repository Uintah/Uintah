/*
 *  Delaunay.cc:  Delaunay Triangulation in 3D
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

class Delaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
public:
    Delaunay(const clString& id);
    Delaunay(const Delaunay&, int deep);
    virtual ~Delaunay();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_Delaunay(const clString& id)
{
    return new Delaunay(id);
}

static RegisterModule db1("Mesh", "Delaunay", make_Delaunay);

Delaunay::Delaunay(const clString& id)
: Module("Delaunay", id, Filter)
{
    iport=new MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new MeshOPort(this, "Delaunay Mesh", MeshIPort::Atomic);
    add_oport(oport);
}

Delaunay::Delaunay(const Delaunay& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Delaunay::Delaunay");
}

Delaunay::~Delaunay()
{
}

Module* Delaunay::clone(int deep)
{
    return new Delaunay(*this, deep);
}

static int face_idx(Mesh* mesh, int p, int f)
{
    Element* e=mesh->elems[p];
    int n=e->faces[f];
    if(n==-1)
	return -1;
    Element* ne=mesh->elems[n];
    for(int i=0;i<4;i++){
	if(ne->faces[i]==p)
	    return (n<<2)|i;
    }
    cerr << "face_idx confused!\n";
    return 0;
}

void Delaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;

    // Get our own copy of the mesh...
    mesh_handle.detach();
    Mesh* mesh=mesh_handle.get_rep();
    mesh->elems.remove_all();

    int nnodes=mesh->nodes.size();
    BBox bbox;
    for(int i=0;i<nnodes;i++)
	bbox.extend(mesh->nodes[i]->p);

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
    Point bmax(center+diag/2.);

    // Make the initial mesh with a tetra which encloses the bounding
    // box.  The first point is at the minimum point.  The other 3
    // have one of the coordinates at bmin+diagonal*3.
    mesh->nodes.add(new Node(bmin));
    mesh->nodes.add(new Node(bmin+Vector(le*3, 0, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, le*3, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, 0, le*3)));

    Element* el=new Element(mesh, nnodes+0, nnodes+1, nnodes+2, nnodes+3);
    el->orient();
    el->faces[0]=el->faces[1]=el->faces[2]=el->faces[3]=-1;
    mesh->elems.add(el);

    for(int node=0;node<nnodes;node++){
	// Add this node...
	update_progress(node, nnodes);
	Point p(mesh->nodes[node]->p);

	// Find which element this node is in
	int in_element;
	if(!mesh->locate(p, in_element)){
	    error("Mesher upset - point outside of domain...");
	    return;
	}

	Array1<int> to_remove;
	to_remove.add(in_element);
	Array1<int> done;
	done.add(in_element);
	HashTable<Face, int> face_table;

	// Find it's neighbors...
	// We might be able to fix this loop to make it
	// O(N) instead of O(n^2) - use a Queue
	int i=0;
	while(i<to_remove.size()){
	    // See if the neighbor should also be removed...
	    int tr=to_remove[i];
	    Element* e=mesh->elems[tr];
	    // Add these faces to the list of exposed faces...
	    Face f1(e->n[1], e->n[2], e->n[3]);
	    Face f2(e->n[2], e->n[3], e->n[0]);
	    Face f3(e->n[3], e->n[0], e->n[1]);
	    Face f4(e->n[0], e->n[1], e->n[2]);

	    // If the face is in the list, remove it.
	    // Otherwise, add it.
	    int dummy;
	    if(face_table.lookup(f1, dummy))
		face_table.remove(f1);
	    else
		face_table.insert(f1, face_idx(mesh, tr, 0));

	    if(face_table.lookup(f2, dummy))
		face_table.remove(f2);
	    else
		face_table.insert(f2, face_idx(mesh, tr, 1));

	    if(face_table.lookup(f3, dummy))
		face_table.remove(f3);
	    else
		face_table.insert(f3, face_idx(mesh, tr, 2));

	    if(face_table.lookup(f4, dummy))
		face_table.remove(f4);
	    else
		face_table.insert(f4, face_idx(mesh, tr, 3));

	    for(int j=0;j<4;j++){
		int skip=0;
		int neighbor=e->face(j);
		for(int ii=0;ii<done.size();ii++){
		    if(neighbor==done[ii]){
			skip=1;
			break;
		    }
		}
		if(neighbor==-1 || neighbor==-2)
		    skip=1;
		if(!skip){
		    // Process this neighbor
		    if(!skip){
			// See if this element is deleted by this point
			Element* ne=mesh->elems[neighbor];
			Point cen;
			double rad2;
			ne->get_sphere2(cen, rad2);
			double ndist2=(p-cen).length2();
			if(ndist2 < rad2){
			    // This one must go...
			    to_remove.add(neighbor);
			}
		    }
		    done.add(neighbor);
		}
	    }
	    i++;
	}
	// Remove the to_remove elements...
	for(i=0;i<to_remove.size();i++){
	    int idx=to_remove[i];
	    delete mesh->elems[idx];
	    mesh->elems[idx]=0;
	}

	// Add the new elements from the faces...
	HashTableIter<Face, int> fiter(&face_table);

	// Make a copy of the face table.  We use the faces in there
	// To compute the new neighborhood information
	HashTable<Face, int> new_faces(face_table);
	for(fiter.first();fiter.ok();++fiter){
	    Face f(fiter.get_key());
	    Element* ne=new Element(mesh, node, f.n[0], f.n[1], f.n[2]);

	    // If the new element is not degenerate, add it to the mix...
	    if(ne->orient()){
		int nen=mesh->elems.size();

		// The face neighbor is in the Face data item
		Face f1(ne->n[1], ne->n[2], ne->n[3]);
		Face f2(ne->n[2], ne->n[3], ne->n[0]);
		Face f3(ne->n[3], ne->n[0], ne->n[1]);
		Face f4(ne->n[0], ne->n[1], ne->n[2]);
		int ef;
		if(new_faces.lookup(f1, ef)){
		    // We have this face...
		    if(ef==-1){
			ne->faces[0]=-1; // Boundary
		    } else {
			int which_face=ef%4;
			int which_elem=ef/4;
			ne->faces[0]=which_elem;
			mesh->elems[which_elem]->faces[which_face]=nen;
		    }
		    new_faces.remove(f1);
		} else {
		    new_faces.insert(f1, (nen<<2)|0);
		    ne->faces[0]=-3;
		}
		if(new_faces.lookup(f2, ef)){
		    // We have this face...
		    if(ef==-1){
			ne->faces[1]=-1; // Boundary;
		    } else {
			int which_face=ef%4;
			int which_elem=ef/4;
			ne->faces[1]=which_elem;
			mesh->elems[which_elem]->faces[which_face]=nen;
		    }
		    new_faces.remove(f2);
		} else {
		    new_faces.insert(f2, (nen<<2)|1);
		    ne->faces[1]=-3;
		}
		if(new_faces.lookup(f3, ef)){
		    // We have this face...
		    if(ef==-1){
			ne->faces[2]=-1; // Boundary
		    } else {
			int which_face=ef%4;
			int which_elem=ef/4;
			ne->faces[2]=which_elem;
			mesh->elems[which_elem]->faces[which_face]=nen;
		    }
		    new_faces.remove(f3);
		} else {
		    new_faces.insert(f3, (nen<<2)|2);
		    ne->faces[2]=-3;
		}
		if(new_faces.lookup(f4, ef)){
		    // We have this face...
		    if(ef==-1){
			ne->faces[3]=-1;
		    } else {
			int which_face=ef%4;
			int which_elem=ef/4;
			ne->faces[3]=which_elem;
			mesh->elems[which_elem]->faces[which_face]=nen;
		    }
		    new_faces.remove(f4);
		} else {
		    new_faces.insert(f4, (nen<<2)|3);
		    ne->faces[3]=-3;
		}
		mesh->elems.add(ne);
	    } else {
		cerr << "Degenerate element (node=" << node << ")\n";
		// Temporary...
#if 0
		for(int i=0;i<mesh->elems.size();i++){
		    Element* ne=mesh->elems[i];
		    if(ne){
			Point cen;
			double rad2;
			ne->get_sphere2(cen, rad2);
			double ndist2=(p-cen).length2();
			if(ndist2 < rad2){
			    int found=0;
			    for(int j=0;j<to_remove.size();j++){
				if(to_remove[j]==i){
				    found=1;
				    break;
				}
			    }
			    ASSERT(found);
			}
		    }
		}
#endif
	    }
	}
	if(new_faces.size() != 0)
	    cerr << "There are " << new_faces.size() << " unresolved faces (node=" << node << ")\n";
	// Temporary...
#if 0
	for(i=0;i<mesh->elems.size();i++){
	    if(mesh->elems[i]){
		for(int j=0;j<4;j++){
		    ASSERT(mesh->elems[i]->faces[j] != -3);
		}
		Element* ne=mesh->elems[i];
		Point cen;
		double rad2;
		ne->get_sphere2(cen, rad2);
		for(int ip=0;ip<=node;ip++){
		    Point pt(mesh->nodes[ip]->p);
		    double ndist2=(pt-cen).length2();
		    if(ndist2*1.000001 < rad2){
			cerr << "Not delaunay!\n";
			cerr << "ndist2=" << ndist2 << endl;
			cerr << "rad2=" << rad2 << endl;
		    }
		}
	    }
	}
#endif
#if 0
	mesh->compute_neighbors();
	int nn=0;
	for(i=0;i<mesh->elems.size();i++){
	    if(mesh->elems[i]){
		for(int j=0;j<4;j++){
		    nn+=mesh->elems[i]->face(j);
		}
	    }
	}
#endif
    }
    // Pack the elements...
    Array1<Element*> new_elems;
    int nelems=mesh->elems.size();
    for(i=0;i<nelems;i++){
	Element* e=mesh->elems[i];
	if(e){
	    new_elems.add(e);
	}
    }
    mesh->elems=new_elems;
    mesh->compute_neighbors();
    oport->send(mesh);
}
