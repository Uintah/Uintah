
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

typedef struct _list
{
        int tetra;
        struct _list * next;
} LIST;

typedef LIST * LPTR;

typedef LIST QUEUE;
typedef QUEUE * QPTR;

class MeshView : public Module {
    MeshIPort* inport;
    GeometryOPort* ogeom;
    
    int numLevels, oldLev;
    int seedTet, oldSeed;
    double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    double clipX, oldClipX, clipY, oldClipY, clipZ, oldClipZ;
    int deep;
    int allLevels;
    int numShare, oldShare;
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
: Module("MeshView", id, Filter)
{
#ifdef OLDUI
    numLevels=0;
    levSlide = new MUI_slider_int("Number of Levels",
				  &numLevels,
				  MUI_widget::Immediate, 1);
    add_ui(levSlide);
    deep = 4;
    levSlide -> set_minmax(0, deep);

    oldSeed = -1;
    seedTet = 0;
    seedSlide = new MUI_slider_int("Starting Tetra", &seedTet,
				   MUI_widget::Immediate, 1);
    add_ui(seedSlide);	

    oldShare = 1;
    numShare = 3;
    MUI_slider_int *tmp = new MUI_slider_int("Shared Verts", &numShare,
					     MUI_widget::Immediate, 1);
    add_ui(tmp);
    tmp -> set_minmax(1,3);

    oldClipX = Xmax;
    clipX = Xmin;
    MUI_slider_real *XclipSlide = new MUI_slider_real("X Clipping plane", 
						     &clipX,
						     MUI_widget::Immediate, 1);
    add_ui(XclipSlide);
    XclipSlide -> set_minmax(Xmin, Xmax);

    oldClipY = Ymax;
    clipY = Ymin;
    MUI_slider_real *YclipSlide = new MUI_slider_real("Y Clipping plane", 
						     &clipY,
						     MUI_widget::Immediate, 1);
    add_ui(YclipSlide);
    YclipSlide -> set_minmax(Ymin, Ymax);

    oldClipZ = Zmax;
    clipZ = Zmin;
    MUI_slider_real *ZclipSlide = new MUI_slider_real("Z Clipping plane", 
						     &clipZ,
						     MUI_widget::Immediate, 1);
    add_ui(ZclipSlide);
    ZclipSlide -> set_minmax(Zmin, Zmax);

    allLevels = 1;
    add_ui(new MUI_onoff_switch("Show all levels", &allLevels,
				MUI_widget::Immediate));
#endif

    inport=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    // Set up Material Properties
    mat1=new Material(Color(.5, .5, .5), Color(.5, .5, .5),
		      Color(.1, .1, .1), 10);
    mat2=new Material(Color(1, 0, 0), Color(1, 0, 0),
		      Color(.1, .1, .1), 10);
}	

MeshView::MeshView(const MeshView& copy, int deep)
: Module(copy, deep)
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
    
    ogeom->delAll();

    if ((oldSeed != seedTet) || (numShare != oldShare)){
	makeLevels(mesh);
#ifdef OLDUI
	levSlide -> set_minmax(0, deep);
#endif
	oldSeed = seedTet;
	oldShare = numShare;
    }

    oldLev = numLevels;
    oldClipX = clipX;
    oldClipY = clipY;
    oldClipZ = clipZ;
    GeomGroup *othGroup = new GeomGroup;
    GeomGroup *levGroup = new GeomGroup;
    GeomGroup *group = new GeomGroup;
    int numTetra=mesh->elems.size();
#ifdef OLDUI
    seedSlide -> set_minmax(0, numTetra - 1);
#endif
    for (int i = 0; i < numTetra; i++){
	if (((allLevels == 0) && (levels[i] == numLevels)) ||
	    ((allLevels == 1) && (levels[i] <= numLevels))) {
	    Element* e=mesh->elems[i];
	    Point p1(mesh->nodes[e->n[0]]->p);
	    Point p2(mesh->nodes[e->n[1]]->p);
	    Point p3(mesh->nodes[e->n[2]]->p);
	    Point p4(mesh->nodes[e->n[3]]->p);

	    if (((p1.x() >= clipX) && (p2.x() >= clipX) && (p3.x() >= clipX) &&
		 (p4.x() >= clipX)) &&
		((p1.y() >= clipY) && (p2.y() >= clipY) && (p3.y() >= clipY) &&
		 (p4.y() >= clipY)) &&
		((p1.z() >= clipZ) && (p2.z() >= clipZ) && (p3.z() >= clipZ) &&
		 (p4.z() >= clipZ))) {
		GeomTetra *nTet = new GeomTetra(p1, p2, p3, p4);
		if (levels[i] == numLevels)
		    levGroup -> add(nTet);
		else	
		    othGroup -> add(nTet);
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
    q.append(seedTet);
    q.append(-2);
	
    deep = 0;
    while(counter < numTetra){
	int x = q.pop();
	if (x == -2) {
	    deep++;
	    q.append(-2);
	} else {
	    levels[x] = deep;
	    counter++;
	    Element* e=mesh->elems[x];
	    for(int i=0;i<4;i++){
		int neighbor=e->face(i);
		if(neighbor !=-1 && levels[neighbor] == -1)
		    q.append(neighbor);
	    }
	}
    }
}
