
/*
 *  MeshView.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Tetra.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <strstream.h>


class MeshView : public Module {
    MeshIPort* inport;
    GeometryOPort* ogeom;
    
    TCLint numLevels, seedTet;
    int oldLev, oldSeed;
    TCLdouble clipX, clipY, clipZ;
    double oldClipX, oldClipY, oldClipZ;
    int deep;
    TCLint allLevels;
//    int numShare, oldShare;
    Array1<int> levels;

    MaterialHandle mat1;
    MaterialHandle mat2;
public:
    MeshView(const clString& id);
    MeshView(const MeshView&, int deep);
    virtual ~MeshView();
    virtual Module* clone(int deep);
    virtual void execute();
    void initList();
    void addTet(int row, int ind);
    void makeLevels(const MeshHandle&);
};

static Module* make_MeshView(const clString& id)
{
    return new MeshView(id);
}

static RegisterModule db1("Fields", "MeshView", make_MeshView);
static RegisterModule db2("Visualization", "MeshView", make_MeshView);
static clString mesh_name("Mesh");

MeshView::MeshView(const clString& id)
: Module("MeshView", id, Filter), numLevels("numLevels", id, this),
  seedTet("seedTet", id, this), clipX("clipX", id, this), 
  clipY("clipY", id, this), clipZ("clipZ", id, this),
  allLevels("allLevels", id, this)
{

    inport=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

	oldSeed = -1; 
	oldClipY = 10000; oldClipX = 10000; oldClipZ = 10000;

    // Set up Material Properties
    mat1=new Material(Color(.5, .5, .5), Color(.5, .5, .5),
		      Color(.1, .1, .1), 10);
    mat2=new Material(Color(1, 0, 0), Color(1, 0, 0),
		      Color(.1, .1, .1), 10);
}	

MeshView::MeshView(const MeshView& copy, int deep)
: Module(copy, deep), numLevels("numLevels", id, this),
  seedTet("seedTet", id, this), clipX("clipX", id, this), 
  clipY("clipY", id, this), clipZ("clipZ", id, this),
  allLevels("allLevels", id, this)
{
    NOT_FINISHED("MeshView::MeshView");
}

MeshView::~MeshView()
{
}

Module* MeshView::clone(int deep)
{
    return new MeshView(*this, deep);
}

void MeshView::execute()
{
    MeshHandle mesh;
    if(!inport->get(mesh))
	return;
    Point bmin, bmax;
    mesh->get_bounds(bmin, bmax);
    char buf[1000];
    ostrstream str(buf, 1000);
    str << "MeshView_set_bounds " << id << " " << bmin.x() << " " << bmax.x() << " " << bmin.y() << " " << bmax.y() << " " << bmin.z() << " " << bmax.z() << '\0';

    TCL::execute(str.str());
    
    ogeom->delAll();

    if (oldSeed != seedTet.get()) 
    {
	makeLevels(mesh);
	ostrstream str3(buf, 1000);
	str3 << "MeshView_set_minmax_nl " << id << " " << 0 << " " << deep << '\0';
	TCL::execute(str3.str());
	oldSeed = seedTet.get();
    }

    oldLev = numLevels.get();
    oldClipX = clipX.get();
    oldClipY = clipY.get();
    oldClipZ = clipZ.get();
    GeomGroup *othGroup = new GeomGroup;
    GeomGroup *levGroup = new GeomGroup;
    GeomGroup *group = new GeomGroup;
    int numTetra=mesh->elems.size();

    ostrstream str2(buf, 1000);
    str2 << "MeshView_set_minmax_numTet " << id << " " << 0 << " " << numTetra - 1 << '\0';
    TCL::execute(str2.str());

    int aL, nL;
    double cX, cY, cZ;
    aL = allLevels.get();
    nL = numLevels.get() + 1;
    cX = clipX.get();
    cY = clipY.get();
    cZ = clipZ.get();

    for (int i = 0; i < numTetra; i++){
	if (((aL == 0) && (levels[i] == nL)) ||
	    ((aL == 1) && (levels[i] <= nL))) 
	{
	    Element* e=mesh->elems[i];
	    Point p1(mesh->nodes[e->n[0]]->p);
	    Point p2(mesh->nodes[e->n[1]]->p);
	    Point p3(mesh->nodes[e->n[2]]->p);
	    Point p4(mesh->nodes[e->n[3]]->p);

	    if (((p1.x() >= cX) && (p2.x() >= cX) && (p3.x() >= cX) &&
		 (p4.x() >= cX)) &&
		((p1.y() >= cY) && (p2.y() >= cY) && (p3.y() >= cY) &&
		 (p4.y() >= cY)) &&
		((p1.z() >= cZ) && (p2.z() >= cZ) && (p3.z() >= cZ) &&
		 (p4.z() >= cZ))) 
		{
		GeomTetra *nTet = new GeomTetra(p1, p2, p3, p4);
		if (levels[i] == nL)
		{
		    levGroup -> add(nTet);
			cerr << "Adding " << i << "to levGroup\n";
		}
		else	
		{
		    othGroup -> add(nTet);
			cerr << "Adding " << i << "to objGroup\n";
		}
	    }
	}
    }

    levGroup -> set_matl(mat1);
    othGroup -> set_matl(mat2);

    group -> add(othGroup);
    group -> add(levGroup);
    ogeom -> addObj(group, mesh_name);
}

void MeshView::makeLevels(const MeshHandle& mesh)
{
    int counter = 0;
    
    int numTetra=mesh->elems.size();
    levels.remove_all();
    levels.grow(numTetra);
    for (int i = 0; i < numTetra; i++)
	levels[i] = -1;

    Queue<int> q;
    q.append(seedTet.get() + 1);
    q.append(-2);
	
    deep = 0;
    while(counter < numTetra)
    {
	int x = q.pop();
	if (x == -2) 
	{
	    deep++;
	    q.append(-2);
	} 
	else if (levels[x] == -1)
	{
	    levels[x] = deep;
	    counter++;
	    Element* e=mesh->elems[x];
	    for(int i = 0; i < 4; i++)
	    {
		int neighbor=e->face(i);
		if(neighbor !=-1 && levels[neighbor] == -1)
		    q.append(neighbor);
	    }
	}
    }
}
