
/*
 *  MeshView.cc:  This module provides various tools for aiding in
 *  visualization of 3-D unstructured meshes
 *
 *  Written by:
 *   Carole Gitlin and Steven G. Parker
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
#include <Geom/Material.h>
#include <Geom/Tetra.h>
#include <Geom/Line.h>
#include <Geom/Switch.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <strstream.h>
#include <Math/Mat.h>
#include <math.h>
#include <limits.h>

#define ALL 0
#define OUTER 1

struct MVEdge {
    int n[2];
    MVEdge(int, int);
    int hash(int hash_size) const;
    int operator==(const MVEdge&) const;
};

class MeshView : public Module {
    MeshIPort* inport;
    GeometryOPort* ogeom;

    int haveVol, haveAsp, haveSize;
    
    TCLint numLevels,               //  The number of levels being shown
           seedTet;                 //  The seed element to build out from
    int oldLev, oldSeed;            //  Previous values of the above two
    TCLdouble clipX, clipY, clipZ;  //  Positive clipping planes
    TCLdouble clipNX, clipNY, clipNZ;  // Negative clipping planes
    double oldClipX, oldClipY, oldClipZ;    // Previous values of the
    double oldClipNX, oldClipNY, oldClipNZ; // clipping planes
	Point oldMin, oldMax;           //  Previous bounding box values
    int deep,                       //  How many levels out there are from a
                                    //  given seed
        oldNumTet;                  //  The previous number of elements
    TCLint allLevels, elmMeas;	    //  Flag of whether to show all levels
                                    //    or only the current one

    TCLint elmSwitch;

    Array1<int> levels;
    Array1< Array1<int> > levStor;
    Array1<GeomSwitch*> levSwitch;
    Array1<GeomSwitch*> auxSwitch;
    Array1<GeomGroup*> levTetra;
    Array1<GeomGroup*> auxTetra;
    Array1<GeomGroup*> levEdges;
    Array1<GeomMaterial*> levMatl;
    Array1<GeomMaterial*> auxMatl;

    Array1<double> volMeas, aspMeas, sizeMeas;
    CrowdMonitor geom_lock;
    GeomGroup* eGroup;

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
    void doClip(int ind, double cX, double cNX, double cY,
		double cNY, double cZ, double cNZ, const MeshHandle& mesh); 
    void makeEdges(const MeshHandle& mesh);
    void calcMeasures(const MeshHandle& mesh, double *min, double *max);
    double volume(Point p1, Point p2, Point p3, Point p4);
    double aspect_ratio(Point p1, Point p2, Point p3, Point p4);
    double calcSize(const MeshHandle& mesh, int ind);
    void get_sphere(Point p1, Point p2, Point p3, Point p4, double& rad);
    double getDistance(Point p0, Point p1, Point p2, Point p3);
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
  clipNX("clipNX", id, this), clipNY("clipNY", id, this), 
  clipNZ("clipNZ", id, this), allLevels("allLevels", id, this),
  elmMeas("elmMeas", id, this), elmSwitch("elmSwitch", id, this)
{

    // Create an input port, of type Mesh
    inport=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    // Initialize the 'old' values
    oldSeed = -1; 
    oldClipY = 10000; oldClipX = 10000; oldClipZ = 10000;
    oldClipNY = 10000; oldClipNX = 10000; oldClipNZ = 10000;

    // Set up Material Properties
    mat1=new Material(Color(.5, .5, .5), Color(.5, .5, .5),
		      Color(.1, .1, .1), 10);
    mat2=new Material(Color(1, 0, 0), Color(1, 0, 0),
		      Color(.1, .1, .1), 10);
    oldMin = Point(0.0, 0.0, 0.0);
    oldMax = Point(0.0, 0.0, 0.0);
    oldNumTet = 0;
}	

MeshView::MeshView(const MeshView& copy, int deep)
: Module(copy, deep), numLevels("numLevels", id, this),
  seedTet("seedTet", id, this), clipX("clipX", id, this),
  clipY("clipY", id, this), clipZ("clipZ", id, this),
  clipNX("clipNX", id, this), clipNY("clipNY", id, this), 
  clipNZ("clipNZ", id, this), allLevels("allLevels", id, this),
  elmMeas("elmMeas", id, this), elmSwitch("elmSwitch", id, this)
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
	char buf[1000];
    if(!inport->get(mesh))
	return;
    Point bmin, bmax;
    mesh->get_bounds(bmin, bmax);

    haveVol = haveAsp = haveSize = 0;

    // If the new bounding box values aren't equal to the old ones, reset
    //  the values on the slider
    if ((bmin != oldMin) || (bmax != oldMax))
    {
        ostrstream str(buf, 1000);
        str << id << " set_bounds " << bmin.x() << " " << bmax.x() << " " << bmin.y() << " " << bmax.y() << " " << bmin.z() << " " << bmax.z() << '\0';

        TCL::execute(str.str());
        oldMin = bmin;
        oldMax = bmax;
        clipX.set(bmin.x()); clipNX.set(bmax.x());
        clipY.set(bmin.y()); clipNY.set(bmax.y());
        clipZ.set(bmin.z()); clipNZ.set(bmax.z());
    }
    

    if (oldSeed != seedTet.get()) 
    {
        makeLevels(mesh);
        ostrstream str3(buf, 1000);
        str3 << id << " set_minmax_nl " << " " << 0 << " " << deep-1 << '\0';
        TCL::execute(str3.str());
        oldSeed = seedTet.get();

    }

    oldLev = numLevels.get();
    oldClipX = clipX.get();
    oldClipY = clipY.get();
    oldClipZ = clipZ.get();
    int numTetra=mesh->elems.size();
    volMeas.grow(numTetra);
    aspMeas.grow(numTetra);
    sizeMeas.grow(numTetra);

    if (oldNumTet != numTetra)
    {
        ostrstream str2(buf, 1000);
        str2 << id << " set_minmax_numTet " << " " << 0 << " " << numTetra - 1 << '\0';
        TCL::execute(str2.str());
        oldNumTet = numTetra;
    }
    double cX, cY, cZ, cNX, cNY, cNZ;

    int nL = numLevels.get();
    int aL = allLevels.get();

    cX = clipX.get(); cNX = clipNX.get();
    cY = clipY.get(); cNY = clipNY.get();
    cZ = clipZ.get(); cNZ = clipNZ.get();


    BBox tb;

//    geom_lock.write_lock();
    GeomGroup *group = new GeomGroup;
    int needAux = 0;
    int start, finish;

    if ((allLevels.get()) == 0)
    {
	start = 0;
	finish = nL;
    }
    else
    {
	start = nL;
	finish = nL;
    }

/*    if (elmSwitch.get())
    {
	double measMin, measMax;
	for (int j = 0; j < deep; j++)
	{
	    levSwitch[j] -> set_state(0);
	    auxSwitch[j] -> set_state(0);
	    calcMeasures(mesh, &measMin, &measMax);
	}
//        ostrstream str4(buf, 1000);
//        str4 << id << " do_measure " << " " << measMin << " " << measMax << '\0';
//        TCL::execute(str4.str());
	
    }
    else	
    {
*/
	for( int i = 0; i < deep; i++)
	{
	    if (((aL == OUTER) && (i != nL)) || (i > nL))
	    {	
		levSwitch[i] -> set_state(0);
		auxSwitch[i] -> set_state(0);	
	    }	
	    else
	    {
		levMatl[i] -> get_bounds(tb);

		needAux = 0;
		if ((tb.min().x() < cX) || (tb.max().x() > cNX) ||
		    (tb.min().y() < cY) || (tb.max().y() > cNY) ||
		    (tb.min().z() < cZ) || (tb.max().z() > cNZ))
		{
		    doClip(i, cX, cNX, cY, cNY, cZ, cNZ, mesh);
		    needAux = 1;
		}

		if (needAux)
		{
		    levSwitch[i] -> set_state(0);
		    auxSwitch[i] -> set_state(1);
		}
		else
		{
		    levSwitch[i] -> set_state(1);
		    auxSwitch[i] -> set_state(0);
		}
	    }
	    if (i < nL)
	    {
		levMatl[i]->set_matl(mat2);
	    }
	    else
	    {
		levMatl[i]->set_matl(mat1);
	    }
	}
//    }
    geom_lock.write_unlock();
    ogeom->flushViews();
}


void MeshView::makeLevels(const MeshHandle& mesh)
{
    int counter = 0;

    GeomGroup *group = new GeomGroup;
    int numTetra=mesh->elems.size();
    levels.remove_all();
    levels.grow(numTetra);
    for (int i = 0; i < numTetra; i++)
        levels[i] = -1;

    Queue<int> q;
    q.append(seedTet.get());
    q.append(-2);
	
    levStor.remove_all();
    levStor.grow(1);
    levTetra.remove_all();
    levTetra.grow(1);
    levTetra[0] = new GeomGroup;
    levSwitch.remove_all();
    deep = 0;
    while(counter < numTetra)
    {
        int x = q.pop();
        if (x == -2) 
        {
            deep++;
            q.append(-2);
	    levStor.grow(1);
	    levTetra.grow(1);
	    levTetra[deep] = new GeomGroup;
        } 
        else if (levels[x] == -1)
        {
            levStor[deep].add(x);
	    levels[x] = deep;
            counter++;
            Element* e=mesh->elems[x];
	    Point p1(mesh->nodes[e->n[0]]->p);
	    Point p2(mesh->nodes[e->n[1]]->p);
	    Point p3(mesh->nodes[e->n[2]]->p);
	    Point p4(mesh->nodes[e->n[3]]->p);
	    
	    GeomTetra *nTet = new GeomTetra(p1, p2, p3, p4);                
	    levTetra[deep] -> add(nTet);
	    for(int i = 0; i < 4; i++)
            {
        	int neighbor=e->face(i);
        	if(neighbor !=-1 && levels[neighbor] == -1)
        	    q.append(neighbor);
            }
        }
    }
    deep++;
    levMatl.remove_all();
    levMatl.grow(deep);
    auxMatl.remove_all();
    auxMatl.grow(deep);
    auxSwitch.remove_all();
    levSwitch.grow(deep);
    auxSwitch.grow(deep);

    auxTetra.remove_all();
    auxTetra.grow(deep);

    int EdgeOnly = 0;
    if (EdgeOnly)
	makeEdges(mesh);
 
    for(i = 0;i < deep;i++)
    {
	if (!EdgeOnly)
	{
	    GeomTetra *dumT = new GeomTetra(Point(0,0,0), Point(0, 0, 0.01),
					    Point(0,0.01,0), Point(0.1,0,0));
	    auxTetra[i] = new GeomGroup;
	    auxTetra[i] -> add(dumT);
	    levMatl[i]=new GeomMaterial(levTetra[i], mat1);
	    auxMatl[i]= new GeomMaterial(auxTetra[i], mat1);
	    levSwitch[i] = new GeomSwitch(levMatl[i], 0);
	    auxSwitch[i] = new GeomSwitch(auxMatl[i], 0);
	    group -> add(levSwitch[i]);
	    group -> add(auxSwitch[i]);
	}
	else
	{
	    levMatl[i]=new GeomMaterial(levEdges[i], mat1);
	}
    }

    ogeom -> delAll();
    ogeom -> addObj(group, mesh_name, &geom_lock);    
}


void MeshView::doClip(int ind, double cX, double cNX, double cY,
double cNY, double cZ, double cNZ, const MeshHandle& mesh) 
{

    auxTetra[ind] -> remove_all();

    for (int j = 0; j < levStor[ind].size(); j++)
    {
	int l = levStor[ind][j];
	Element* e=mesh->elems[l];
	Point p1(mesh->nodes[e->n[0]]->p);
	Point p2(mesh->nodes[e->n[1]]->p);
	Point p3(mesh->nodes[e->n[2]]->p);
	Point p4(mesh->nodes[e->n[3]]->p);
	
	if (((p1.x() >= cX) && (p2.x() >= cX) && (p3.x() >= cX) && 
          (p4.x() >= cX)) &&
 	    ((p1.x() <= cNX) && (p2.x() <= cNX) && (p3.x() <= cNX) && 
          (p4.x() <= cNX)) &&
 	    ((p1.y() >= cY) && (p2.y() >= cY) && (p3.y() >= cY) && 
          (p4.y() >= cY)) &&
 	    ((p1.y() <= cNY) && (p2.y() <= cNY) && (p3.y() <= cNY) && 
          (p4.y() <= cNY)) &&
  	    ((p1.z() >= cZ) && (p2.z() >= cZ) && (p3.z() >= cZ) && 
          (p4.z() >= cZ)) &&
 	    ((p1.z() <= cNZ) && (p2.z() <= cNZ) && (p3.z() <= cNZ) && 
         (p4.z() <= cNZ)))
	{		
	    GeomTetra *t = new GeomTetra(p1, p2, p3, p4);
	    auxTetra[ind] ->add(t);
	
	}
    }
    auxMatl[ind] = new GeomMaterial(auxTetra[ind], mat1);

}

void MeshView::calcMeasures(const MeshHandle& mesh, double *min, double *max)
{
    int i, e = elmMeas.get();
    int numTetra=mesh->elems.size();

    *min = DBL_MAX;
    *max = DBL_MIN;
    if ((e == 1) && !haveVol)
    {
	for (i = 0; i < numTetra; i++)
	{
	    Element* e=mesh->elems[i];
	    volMeas[i] = volume(mesh->nodes[e->n[0]]->p,
				mesh->nodes[e->n[1]]->p,
				mesh->nodes[e->n[2]]->p,
				mesh->nodes[e->n[3]]->p);
	    if (*min > volMeas[i])
		*min = volMeas[i];
	    if (*max < volMeas[i])
		*max = volMeas[i];
	}
	haveVol = 1; 
    }
    else if ((e == 2) && !haveAsp)
    {
	for (i = 0; i < numTetra; i++)
	{
	    Element* e=mesh->elems[i];
	    aspMeas[i] = aspect_ratio(mesh->nodes[e->n[0]]->p,
				      mesh->nodes[e->n[1]]->p,
				      mesh->nodes[e->n[2]]->p,
				      mesh->nodes[e->n[3]]->p);
	    if (*min > aspMeas[i])
		*min = aspMeas[i];
	    if (*max < aspMeas[i])
		*max = aspMeas[i];
	}
	haveAsp = 1; 
    }
    else if ((e == 3) && !haveSize)
    {
	if (!haveVol)
	{
	    for (i = 0; i < numTetra; i++)
	    {
		Element* e=mesh->elems[i];
		volMeas[i] = volume(mesh->nodes[e->n[0]]->p,
				    mesh->nodes[e->n[1]]->p,
				    mesh->nodes[e->n[2]]->p,
				    mesh->nodes[e->n[3]]->p);
	    }
		haveVol = 1; 
	}
	for (i = 0; i < numTetra; i++)
	{
	    sizeMeas[i] = calcSize(mesh, i);
	    if (*min > sizeMeas[i])
		*min = sizeMeas[i];
	    if (*max < sizeMeas[i])
		*max = sizeMeas[i];
	}
	haveSize = 1;
    }

}

double MeshView::volume(Point p1, Point p2, Point p3, Point p4)
{
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();

    double a1 = x2*(y3*z4 - y4*z3) + x3*(y4*z2 - y2*z4) + x4*(y2*z3 - y3*z2);
    double a2 =-x3*(y4*z1 - y1*z3) - x4*(y1*z3 - y3*z1) - x1*(y3*z4 - y4*z3);
    double a3 = x4*(y1*z2 - y2*z1) + x1*(y2*z4 - y4*z2) + x2*(y4*z1 - y1*z4);
    double a4 =-x1*(y2*z3 - y3*z2) - x2*(y3*z1 - y1*z3) - x3*(y1*z2 - y2*z1);

    return(a1 + a2 + a3 + a4) / 6.;
}

double MeshView::aspect_ratio(Point p0, Point p1, Point p2, Point p3)
{
    double rad, len;
    get_sphere(p0, p1, p2, p3, rad);
    rad = sqrt(rad);
    len = getDistance(p0, p1, p2, p3);
    
    double ar = 4 * sqrt(1.5 * (rad / len));

    return ar;
}

double MeshView::calcSize(const MeshHandle& mesh, int ind)
{
    Element* e=mesh->elems[ind];
    double a=volMeas[e->face(0)];
    double b=volMeas[e->face(1)];
    double c=volMeas[e->face(2)];
    double d=volMeas[e->face(3)];

    double m1 = Min(Min(Min(a, b), c), d);
    double m2 = Max(Max(Max(a, b), c), d);

    double m3 = Max(volMeas[ind] / m2, m1 / volMeas[ind]);

    return m3;
}
    
double MeshView::getDistance(Point p0, Point p1, Point p2, Point p3)
{
    double d1, d2, d3, d4, d5, d6;

    d1 = (p0 - p1).length2();
    d2 = (p0 - p2).length2();
    d3 = (p0 - p3).length2();
    d4 = (p1 - p2).length2();
    d5 = (p1 - p3).length2();
    d6 = (p2 - p3).length2();

    double m1 = Max(d1, d2);
    double m2 = Max(d3, d4);
    double m3 = Max(d5, d6);

    double dis = Max(Max(m1, m2), m3);
    return sqrt(dis);

}

void MeshView::get_sphere(Point p0, Point p1, Point p2, Point p3, double& rad)
{
    Vector v1(p1 - p0);
    Vector v2(p2 - p0);
    Vector v3(p3 - p0);

    Point cen;

    double c0=(p0 - Point(0,0,0)).length2();
    double c1=(p1 - Point(0,0,0)).length2();
    double c2=(p2 - Point(0,0,0)).length2();
    double c3=(p3 - Point(0,0,0)).length2();

    double mat[3][3];
    mat[0][0]=v1.x();
    mat[0][1]=v1.y();
    mat[0][2]=v1.z();
    mat[1][0]=v2.x();
    mat[1][1]=v2.y();
    mat[1][2]=v2.z();
    mat[2][0]=v3.x();
    mat[2][1]=v3.y();
    mat[2][2]=v3.z();
    double rhs[3];
    rhs[0]=(c1-c0)*0.5;
    rhs[1]=(c2-c0)*0.5;
    rhs[2]=(c3-c0)*0.5;
    matsolve3by3(mat, rhs);
    cen=Point(rhs[0], rhs[1], rhs[2]);
    rad=(p0-cen).length2();
}

MVEdge::MVEdge(int n0, int n1)
{
    if (n0 < n1)
    {
	n[0] = n0;
	n[1] = n1;
    }
    else
    {
	n[0] = n1;
	n[1] = n0;
    }
}

int MVEdge::hash(int hash_size) const
{
    return (((n[0]*7+5)^(n[1]*5+3))^(3*hash_size+1))%hash_size;
}

int MVEdge::operator==(const MVEdge& e) const
{
    return n[0]==e.n[0] && n[1]==e.n[1];
}

void MeshView::makeEdges(const MeshHandle& mesh)
{
    Array1< HashTable<MVEdge, int> > edge_table;

    edge_table.grow(deep);
    levEdges.remove_all();

    levEdges.grow(deep);

    for (int i = 0; i < mesh->elems.size(); i++)
    {
	Element* e=mesh->elems[i];
	
	MVEdge e1(e->n[0], e->n[1]);
	MVEdge e2(e->n[0], e->n[2]);
	MVEdge e3(e->n[0], e->n[3]);
	MVEdge e4(e->n[1], e->n[2]);
	MVEdge e5(e->n[1], e->n[3]);
	MVEdge e6(e->n[2], e->n[3]);

	int l=levels[i];

	int dummy=0;
	int aL = allLevels.get();
	
	int lu = edge_table[l].lookup(e1, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e1, dummy))))))
	    edge_table[l].insert(e1, 0);

	lu = edge_table[l].lookup(e2, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e2, dummy))))))
	    edge_table[l].insert(e2, 0);

	lu = edge_table[l].lookup(e3, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e3, dummy))))))
	    edge_table[l].insert(e3, 0);

	lu = edge_table[l].lookup(e4, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e4, dummy))))))
	    edge_table[l].insert(e4, 0);

	lu = edge_table[l].lookup(e5, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e5, dummy))))))
	    edge_table[l].insert(e5, 0);

	lu = edge_table[l].lookup(e6, dummy);
	if (((aL == OUTER) && !lu) ||
	    ((aL == ALL) && (!lu || ((l > 0) &&
	     !(edge_table[l-1].lookup(e6, dummy))))))
	    edge_table[l].insert(e6, 0);

    }

    for (i = 0; i < deep; i++)
    {
	HashTableIter<MVEdge, int> eiter(&edge_table[i]);
	levEdges[i] = new GeomGroup;
	for(eiter.first(); eiter.ok(); ++eiter)
	{	
	    MVEdge e(eiter.get_key());
	    Point p1(mesh->nodes[e.n[0]]->p);
	    Point p2(mesh->nodes[e.n[1]]->p);
	    GeomLine* gline = new GeomLine(p1, p2);
	    levEdges[i] -> add(gline);
	}
    }
}
