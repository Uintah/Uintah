
/*
 *  MeshView.cc:  This module provides various tools for aiding in
 *  visualization of 3-D unstructured meshes
 *
 *  Written by:
 *   Carole Gitlin
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Tetra.h>
#include <Geom/Line.h>
#include <Geom/Switch.h>
#include <Geom/Tri.h>
#include <Geom/Sphere.h>
#include <Geom/Cylinder.h>
#include <Geom/Pick.h>
#include <TCL/TCLvar.h>
#include <TCL/Histogram.h>
#include <Widgets/CrosshairWidget.h>
#include <iostream.h>
#include <strstream.h>
#include <Math/Mat.h>
#include <math.h>
#include <limits.h>

#define ALL 0
#define OUTER 1

#define SURFACE 0
#define VOLUME 1

#define ADD 1
#define DELETE 2

class MeshView : public Module {
    MeshIPort* inport;
    ColorMapIPort* colorPort;
    GeometryOPort* ogeom;
    MeshOPort* oport;

    CrowdMonitor widget_lock;

    int haveVol, haveAsp, haveSize,
         showing, oldTech, toEdit;
    
    TCLint numLevels,               //  The number of levels being shown
           editMode,
           display,
           select,
           render;
    int oldLev, oldSeed, Seed;	    //  Previous values of the above two
    TCLdouble clipX, clipY, clipZ;  //  Positive clipping planes
    TCLdouble clipNX, clipNY, clipNZ;  // Negative clipping planes
    TCLdouble radius;
    double oldClipX, oldClipY, oldClipZ;    // Previous values of the
    double oldClipNX, oldClipNY, oldClipNZ; // clipping planes
    Point oldMin, oldMax;           //  Previous bounding box values
    int deep,                       //  How many levels out there are from a
                                    //  given seed
        oldNumTet,                  //  The previous number of elements
        oldInside, oldRadius,
        lastTech,
        oldEdit,
        oldDisplay;
    TCLint allLevels, 		    //  Flag of whether to show all levels
                                    //    or only the current one
           elmMeas,    		    //  Which measure we are looking at
           elmSwitch,	      	    //  Whether only showing elements in
                                    //    range, or showing them hilited in
                                    //    mesh
           inside,		    //  elements are either inside the
                                    //    specified range or outside
           tech;		    //  Switch between quantitative and
                                    //    manipulative techniques	   
    CrosshairWidget *startingTet;   //  Selecting Starting tetrahedron
    Histogram histo;                //  Histogram for showing measures

    Point oldPoint, editPoint;
    double oldmMin, oldmMax;
    int oldMeas, switchSet, oldElm, 
        changed, oldNL, oldALL, oldSurf, oldRender;

    Array1<int> levels, toDrawTet, toDrawMeas;
    Array1< Array1<int> > levStor;
    Array1<GeomGroup*> tetra, measTetra;
    Array1<GeomMaterial*> matl, measMatl;
    Array1<MaterialHandle> Materials;
    GeomGroup* measAuxTetra;
    GeomGroup* measGroup;
    GeomGroup* regGroup;
    GeomMaterial* measAuxMatl;
    GeomSwitch* regSwitch;
    GeomSwitch* measSwitch;
    GeomSwitch* measAuxSwitch;
    Point bmin, bmax;

    Array1<double> volMeas, aspMeas, sizeMeas;
    CrowdMonitor geom_lock;

    Array1<MaterialHandle> genMat;
public:
    MeshView(const clString& id);
    MeshView(const MeshView&, int deep);
    void initGroups();
    virtual ~MeshView();
    virtual Module* clone(int deep);
    virtual void execute();
    void doChecks(const MeshHandle& Mesh,
			   const ColorMapHandle& genColors);
    void updateInfo(const MeshHandle& Mesh);
    void getElements(const MeshHandle& mesh, 
			   const ColorMapHandle& genColors);
    void initList();
    void addTet(int row, int ind);
    void makeLevels(const MeshHandle&);
    void getTetra(const MeshHandle&);
    int setToDraw(const MeshHandle&);
    int doClip(int n, const MeshHandle& mesh);

    void makeEdges(const MeshHandle& mesh);
    void calcMeasures(const MeshHandle& mesh, double *min, double *max);
    int getMeas(const MeshHandle& mesh, const ColorMapHandle& col);
    double volume(Point p1, Point p2, Point p3, Point p4);
    double aspect_ratio(Point p1, Point p2, Point p3, Point p4);
    double calcSize(const MeshHandle& mesh, int ind);
    void get_sphere(Point p1, Point p2, Point p3, Point p4, double& rad);
    double getDistance(Point p0, Point p1, Point p2, Point p3);
    int findElement(Point p, const MeshHandle& mesh, const ColorMapHandle&);
    void geom_release(GeomPick*, void*);
    void geom_moved(GeomPick*, int, double, const Vector&, void*);
    int doEdit(const MeshHandle& mesh);
    int newTesselation(const MeshHandle& mesh);
    int findNode(const MeshHandle& mesh);
};

extern "C" {
Module* make_MeshView(const clString& id)
{
    return new MeshView(id);
}
}

static clString mesh_name("Mesh");
static clString widget_name("Crosshair Widget");

/***************************************************************************
 * Constructor for the MeshView module.
 */
MeshView::MeshView(const clString& id)
: Module("MeshView", id, Filter), numLevels("numLevels", id, this),
  clipX("clipX", id, this), clipY("clipY", id, this), 
  clipZ("clipZ", id, this), clipNX("clipNX", id, this), 
  clipNY("clipNY", id, this), clipNZ("clipNZ", id, this), 
  allLevels("allLevels", id, this), editMode("editMode", id, this),
  elmMeas("elmMeas", id, this), elmSwitch("elmSwitch", id, this),
  tech("tech", id, this), inside("inside", id, this), 
  render("render", id, this), display("display", id, this),
  radius("radius", id, this), select("select", id, this)
{

    // Create an input port, of type Mesh
    inport=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);

    // Create a ColorMap input port
    colorPort = new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(colorPort);


    // Create the output port for geometry
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    // Create output port for writing the mesh
    oport = new MeshOPort(this, "Outmesh", MeshIPort::Atomic);
    add_oport(oport);

    // Initialize the 'old' values
    oldSeed = -1; toEdit = 0;
    oldClipY = 10000; oldClipX = 10000; oldClipZ = 10000;
    oldClipNY = 10000; oldClipNX = 10000; oldClipNZ = 10000;
    oldMeas = 10; oldEdit = 0;
    oldMin = Point(0.0, 0.0, 0.0);
    oldMax = Point(0.0, 0.0, 0.0);
    oldNumTet = 0; oldElm = -1;
    oldmMin = 10000; oldmMax = 10000;
    showing = 0; oldTech = 2; oldNL = -1; 


    Seed = 0;
    // Set up Material Properties
    genMat.grow(4);
    genMat[0]=new Material(Color(.5, .5, .5), Color(.5, .5, .5),
		      Color(.5, .5, .5), 20);
//    genMat[0] = new Material(Color(0,0,0), Color(0,.6,0),Color(.5,.5,.5),20);
    genMat[1]=new Material(Color(1, 0, 0), Color(1, 0, 0),
		      Color(.5, .5, .5), 20);
    genMat[2]=new Material(Color(0, 1, 1), Color(0, 1, 1),
		      Color(.5, .5, .5), 20);
    genMat[3] = new Material(Color(1, 1, 0), Color(1, 1, 0),
			     Color(.5, .5, .5), 20);
    haveVol = haveAsp = haveSize = 0;
    switchSet = 0; oldInside = 0;
    oldALL = ALL; oldRender = VOLUME;
    oldDisplay = 0; oldRadius = 0.025;
    lastTech = -1;

    startingTet = new CrosshairWidget(this, &widget_lock, 0.1);

    initGroups();
}


/***************************************************************************
 * Initializes the geometry groups.  There are three main groups, one for 
 * the regular tetrahedron, and two for the quantitative techniques with one
 * for the elements inside the range and one for the wireframe mesh of those
 * elements outside the range.  Each group has a switch that the program 
 * turns on and off, depending on which group should be displayed.
 */
void MeshView::initGroups()
{
    // measTetra is used for the elements inside the range, for the
    // quantitative measures
    measTetra.grow(1);
    measTetra[0] = new GeomGroup;

    GeomTetra *dumT = new GeomTetra(Point(0,0,0), Point(0,0,0.1), 
				    Point(0,0.01,0), Point(0.1,0,0));

    measTetra[0] -> add(dumT);
    measMatl.grow(1);
    measMatl[0] = new GeomMaterial(measTetra[0], genMat[0]);
    measGroup = new GeomGroup;
    measGroup -> add(measMatl[0]);
    measSwitch = new GeomSwitch(measGroup, 0);

    GeomTetra *dumX = new GeomTetra(Point(0,0,0), Point(0,0,0.1), 
				    Point(0,0.01,0), Point(0.1,0,0));

    // measAuxTetra is used for the wireframe mesh of elements outside
    // the range.
    measAuxTetra = new GeomGroup;
    measAuxTetra -> add(dumX);
    measAuxMatl = new GeomMaterial(measAuxTetra, genMat[2]);
    measAuxSwitch = new GeomSwitch(measAuxMatl, 0);

    // tetra is used for the regular tetrahedron in the manipulative mode
    tetra.grow(1);
    tetra[0] = new GeomGroup;
    GeomTetra *dumY = new GeomTetra(Point(0,0,0), Point(0,0,0.1), 
				    Point(0,0.01,0), Point(0.1,0,0));
    
    tetra[0] -> add(dumY);
    matl.grow(1);
    matl[0] = new GeomMaterial(tetra[0], genMat[1]);
    regGroup = new GeomGroup;
    regGroup -> add(matl[0]);
    regSwitch = new GeomSwitch(regGroup, 0);

    GeomGroup *group = new GeomGroup;
    group -> add(regSwitch);
    group -> add(measSwitch);
    group -> add(measAuxSwitch);
    
    ogeom -> delAll();
    ogeom -> addObj(group, mesh_name, &geom_lock);
    ogeom -> addObj(startingTet -> GetWidget(), widget_name, &widget_lock);
}	

/***************************************************************************
 */
MeshView::MeshView(const MeshView& copy, int deep)
: Module(copy, deep), numLevels("numLevels", id, this),
  clipX("clipX", id, this), clipY("clipY", id, this), 
  clipZ("clipZ", id, this), clipNX("clipNX", id, this), 
  clipNY("clipNY", id, this), clipNZ("clipNZ", id, this), 
  allLevels("allLevels", id, this), editMode("editMode", id, this),
  elmMeas("elmMeas", id, this), elmSwitch("elmSwitch", id, this),
  tech("tech", id, this), inside("inside", id, this),
  render("render", id, this), display("display", id, this),
  radius("radius", id, this), select("select", id, this)
{
    NOT_FINISHED("MeshView::MeshView");
}

/***************************************************************************
 */
MeshView::~MeshView()
{
}

/***************************************************************************
 */
Module* MeshView::clone(int deep)
{
    return new MeshView(*this, deep);
}

/***************************************************************************
 * Main function of the MeshView module
 */
void MeshView::execute()
{
    MeshHandle mesh;
    ColorMapHandle genColors;

    // Wait until we have a mesh
    if(!inport->get(mesh))
	return;

    // If we don't have a ColorMap, then create a generic one.
    if (!colorPort -> get(genColors)){
	genColors = new ColorMap;
	genColors -> build_default();
    }

    // Check to see if anything has changed
    doChecks(mesh, genColors);

    // Get the elements to display
    getElements(mesh, genColors);

    geom_lock.write_unlock();
//    startingTet -> execute();

    oport->send(mesh);
    ogeom->flushViews();

    lastTech = tech.get();
}

/***************************************************************************
 * Function to determine whether anything in the interface has been changed
 * and then act appropriately.
 */
void MeshView::doChecks(const MeshHandle& mesh, const ColorMapHandle& cmap)
{
    int numTetra=mesh->elems.size();
    char buf[1000];
    changed = 0; // This is used to indicate whether something has been
                 // changed.

    // Get the position of the crosshair widget
    Point newP = startingTet -> GetPosition();

    // if it is a new point, find which element it is in
    if (newP != oldPoint)
    {
	int findit=findElement(newP, mesh, cmap);
	if (findit == -1)
	    cerr << "Didn't find the tetrahedron\n";
	else
	{
	    // See if we were selecting the edit element or seed
	    if (select.get() == 1)
	    {
		toEdit = findit;
		if (toEdit != oldEdit)
		{
		    updateInfo(mesh);
		    oldEdit = toEdit;
		}
		editPoint = newP;
	    }
	    else
		Seed = findit;
	    cerr << "Element = " << findit << endl;
	}
	oldPoint = newP;
    }

    // Check if we're editing the mesh
    if (tech.get() == 2)
    {
	oldElm = -1;
	doEdit(mesh);
    }

    // if we've read in a new file or changed the number of
    // elements, reset the slider
    if (oldNumTet != numTetra)
    {
	toDrawTet.remove_all();
	toDrawTet.grow(numTetra);
	toDrawMeas.remove_all();
	toDrawMeas.grow(numTetra);
        oldNumTet = numTetra;
	changed = 1;
	haveVol = haveAsp = haveSize = 0;
	Element* elem = mesh->elems[Seed];

	// Put the crosshair widget in the center of the seed
	Point p1(mesh->nodes[elem->n[0]]->p);
	Point p2(mesh->nodes[elem->n[1]]->p);
	Point p3(mesh->nodes[elem->n[2]]->p);
	Point p4(mesh->nodes[elem->n[3]]->p);

	Point center((p1.x() + p2.x() + p3.x() + p4.x()) / 4.,
		     (p1.y() + p2.y() + p3.y() + p4.y()) / 4.,
		     (p1.z() + p2.z() + p3.z() + p4.z()) / 4.);
	
	startingTet -> SetPosition(center);
	oldPoint = center;
    }

    mesh->get_bounds(bmin, bmax);

    // If the new bounding box values aren't equal to the old ones, reset
    //  the values on the sliders

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
	changed = 1;
	Vector v = bmin - bmax;
	double widget_scale = sqrt(v.x() * v.x() + v.y() * v.y() + 
				   v.z() * v.z());
	startingTet -> SetScale(widget_scale * .001);
    }

    // Check to see if the clipping surfaces have changed, and if so,
    // mark 'changed'.
    if (oldClipX != clipX.get() || oldClipNX != clipNX.get() || oldClipY !=
	clipY.get() || oldClipNY != clipNY.get() || oldClipZ != clipZ.get() ||
	oldClipNZ != clipNZ.get())
    {
	oldClipX = clipX.get(); oldClipY = clipY.get(); oldClipZ = clipZ.get();
	oldClipNX=clipNX.get(); oldClipNY=clipNY.get(); oldClipNZ=clipNZ.get();
	changed = 1;
    }

    // If we have a new seed tetrahedron, need to remake the level sets and
    // update the slider.
    if (oldSeed != Seed)
    {
	makeLevels(mesh);
	// Reset the slider with the new number of levels
        ostrstream str3(buf, 1000);
        str3 << id << " set_minmax_nl " << " " << 0 << " " << deep-1 << '\0';
        TCL::execute(str3.str());

	updateInfo(mesh);
    }

    int NL = numLevels.get();
    int AL = allLevels.get();

    // Check various variables to see if their values have changed, and if
    // so, mrk 'changed'.
    if (oldNL != NL)
	oldNL = NL; changed = 1;
    if (oldALL != AL)
	oldALL = AL; changed = 1;
    if (oldRender != render.get())
	oldRender = render.get(); changed = 1;
    if (oldDisplay != display.get())
	oldDisplay = display.get(); changed = 1;
    if (oldRadius != radius.get())
	oldRadius = radius.get(); changed = 1;


    geom_lock.write_lock();

    double measMin, measMax;

    // if we're looking at quantitative measures...
    if (tech.get() == 1)
    {
	// Calculate the measure
	calcMeasures(mesh, &measMin, &measMax);
    
	// if we're looking at a new measure, reset the histogram values
	if (oldMeas != elmMeas.get())
	{
	    oldMeas = elmMeas.get();
	    histo.SetTitle(id+" Histogram");
	    if (oldMeas == 1)
	    {
		histo.SetData(volMeas);
		histo.SetValueTitle("Volume");
	    }
	    else if (oldMeas == 2)
	    {
		histo.SetData(aspMeas);
		histo.SetValueTitle("Aspect Ratio");
	    }
	    else if (oldMeas == 3)
	    {
		histo.SetData(sizeMeas);
		histo.SetValueTitle("Size v Neighbor");
	    }
	    histo.ShowGrid();
	    histo.ShowRange();
	    // pop up the histogram
	    histo.ui();
	}
    }

}

/***************************************************************************
 * Procedure to determine which elements should be displayed.  Handles both
 * quantitative measures and manipulative techniques.
 */
void MeshView::getElements(const MeshHandle& mesh, 
			   const ColorMapHandle&)
{
    //int numTetra = mesh -> elems.size();
    //int nL = numLevels.get();
    //int aL = allLevels.get();
    //HashTable<Edge, int> edge_table;

    if (tech.get() == 1 || (tech.get() == 2 && lastTech == 1))
    {
	// If we're looking at quantitative measures, get the apppropriate
	// elements
	//int numGroups = getMeas(mesh, genColors);

	// Turn the regular group off and the measures on
	regSwitch -> set_state(0);
	measSwitch -> set_state(1);
	if (elmSwitch.get() == 2)
	    measAuxSwitch -> set_state(1);
	else
	    measAuxSwitch -> set_state(0);
    }

    else
    {
	// Otherwise, we're looking at the manipulative techniques, so if
	// something has changed, then get the appropriate elements
	if (changed)
	    getTetra(mesh);

	// Turn the regular tetra group on and the other two off.
	regSwitch -> set_state(1);
	measSwitch -> set_state(0);
	measAuxSwitch -> set_state(0);

    }

}

/***************************************************************************
 *
 * Procedure for calculating the levels of the mesh.  A level is defined as
 * all elements that share a face with an element on the previous level, 
 * starting with a specified tetrehedron.
 */

void MeshView::makeLevels(const MeshHandle& mesh)
{
    int counter = 0;
    int numTetra = mesh -> elems.size();

    // the array 'levels' is used to keep track of which level each 
    // element is at.
    levels.remove_all();
    levels.grow(numTetra);

    for (int i = 0; i < numTetra; i++)
        levels[i] = -1;

    // We will use a queue to keep track of elements that need to be put
    // into a level
    Queue<int> q;
//    q.append(seedTet.get());
    q.append(Seed);

    // Use the value -2 to specify we're at a new level
    q.append(-2);
	
    levStor.remove_all();
    levStor.grow(1);
    levStor[0].remove_all();
    deep = 0;
    while(counter < numTetra)
    {
        int x = q.pop();

	// if the value is -2, we're at a new level, so increase the counter
	// and grow the array
        if (x == -2) 
        {
            deep++;
            q.append(-2);
	    levStor.grow(1);
	    levStor[deep].remove_all();
        } 
	// Otherwise, x will be an element number.  If x's level has not
	// yet been set, then we add that element to the list of elements
	// at level 'deep', and set x's level.
        else if (levels[x] == -1)
        {
            levStor[deep].add(x);
	    levels[x] = deep;
            counter++;
            Element* e=mesh->elems[x];

	    // Get all 4 of x's neighbors, and add them to the queue
	    for(int i = 0; i < 4; i++)
            {
        	int neighbor=e->face(i);
        	if(neighbor !=-1 && levels[neighbor] == -1)
        	    q.append(neighbor);
            }
        }
    }
    deep++;
}

/***************************************************************************
 * Procedure to decide which elements should be displayed when in the
 * manipulative mode.
 */
void MeshView::getTetra(const MeshHandle& mesh)
{
    int numTetra = mesh -> elems.size();
    //int numGroups, nL = numLevels.get(), aL = allLevels.get();
    int i, j;

    // First have to remove all elements that were previously shown.
    regGroup -> remove_all();
    tetra.remove_all();

    int numGroups = setToDraw(mesh);

    // Switch to see if we're displaying elements in production mode or
    // normal mode.
    int RENDER = display.get();;
    if (RENDER)
	numGroups = 1;
    tetra.grow(numGroups + 1);
    for (i = 0; i < numGroups + 1; i++)
	tetra[i] = new GeomGroup;
    
    int toDo = render.get(); // Either looking at entire volume or surface
                             // faces only.

    HashTable<Edge, int> edge_table;
    int dummy = 0;
    for (i = 0; i < numTetra; i++)
    {
	if (toDrawTet[i] != -1)
	{
	    Element *e = mesh -> elems[i];
	    // If we're in production mode
	    if (RENDER)
	    {
		// and looking at entire volume
		if (toDo == VOLUME)
		{
		    // add each edge of the tetrahedron to the hashtable, if
		    // it's not already there
		    Edge e1(e->n[0], e->n[1]);
		    Edge e2(e->n[0], e->n[2]);
		    Edge e3(e->n[0], e->n[3]);
		    Edge e4(e->n[1], e->n[2]);
		    Edge e5(e->n[1], e->n[3]);
		    Edge e6(e->n[2], e->n[3]);
		    if (!(edge_table.lookup(e1, dummy)))
			edge_table.insert(e1, 0);
		    if (!(edge_table.lookup(e2, dummy)))
			edge_table.insert(e2, 0);
		    if (!(edge_table.lookup(e3, dummy)))
			edge_table.insert(e3, 0);
		    if (!(edge_table.lookup(e4, dummy)))
			edge_table.insert(e4, 0);
		    if (!(edge_table.lookup(e5, dummy)))
			edge_table.insert(e5, 0);
		    if (!(edge_table.lookup(e6, dummy)))
			edge_table.insert(e6, 0);
		}
		else 
		{
		    // if we're only looking at the surface, for each of
		    // the four neighbors, see if it is to be displayed,
		    // and if it is not, then add the 3 edges of that face
		    // to the hashtable
		    for (int k = 0; k < 4; k++)
		    {
			int nbr = e->face(k);
			if (nbr == -1 || toDrawTet[nbr] == -1)
			{  
			    int p1 = (k+1)%4;
			    int p2 = (k+2)%4;
			    int p3 = (k+3)%4;
			    Edge e1(e->n[p1], e->n[p2]);
			    Edge e2(e->n[p2], e->n[p3]);
			    Edge e3(e->n[p3], e->n[p1]);
			    if (!(edge_table.lookup(e1, dummy)))
				edge_table.insert(e1, dummy);
			    if (!(edge_table.lookup(e2, dummy)))
				edge_table.insert(e2, dummy);
			    if (!(edge_table.lookup(e3, dummy)))
				edge_table.insert(e3, dummy);
			}
		    }
		}
	    }
	    else
	    {
		// If we're in normal display mode, then check the four 
		// faces, and if we're in volume mode, or the neighbor is
		// not to be displayed, add the face triangle to the group.
		for (j = 0; j < 4; j++)
		{
		    
		    int neighbor = e->face(j);
		    if (toDo == VOLUME || neighbor == -1 || 
			toDrawTet[neighbor] == -1)
		    {
			int i1=e->n[(j+1)%4];
			int i2=e->n[(j+2)%4];
			int i3=e->n[(j+3)%4];
			Point p1(mesh->nodes[i1]->p);
			Point p2(mesh->nodes[i2]->p);
			Point p3(mesh->nodes[i3]->p);
			GeomTri *tri = new GeomTri(p1, p2, p3);
			tetra[toDrawTet[i]] -> add(tri);
		    }
		}
	    }
    
	}
    }
    HashTableIter<Edge, int> eiter(&edge_table);
    double rad = radius.get();

    // For each edge in the hashtable, make it into a cylinder and add it to
    // the group.
    for(eiter.first(); eiter.ok(); ++eiter)
    {
	Edge e(eiter.get_key());
	Point p1(mesh->nodes[e.n[0]]->p);
	Point p2(mesh->nodes[e.n[1]]->p);
	GeomCylinder* cyl = new GeomCylinder(p1, p2, rad, 8, 1);
	tetra[0] -> add(cyl);
    }


    matl.remove_all();
    matl.grow(numGroups+1);
    
    for (i = 0; i < numGroups+1; i++)
    {
	matl[i] = new GeomMaterial(tetra[i], Materials[i]);
	regGroup -> add(matl[i]);
    }
}

/***************************************************************************
 * Procedure to determine which elements are eligible to be displayed.  It
 * First checks to see if each element is in a desired level, and if so, 
 * sends it to another function to test the element against the clipping
 * surfaces.
 */
int MeshView::setToDraw(const MeshHandle& mesh)
{
    int numTetra = mesh -> elems.size();
    int numGroups, nL = numLevels.get(), aL = allLevels.get();
    double cX, cY, cZ, cNX, cNY, cNZ;

    cX = clipX.get(); cNX = clipNX.get();
    cY = clipY.get(); cNY = clipNY.get();
    cZ = clipZ.get(); cNZ = clipNZ.get();
    
    int needC, i, j;
    Point min(cX, cY, cZ), max(cNX, cNY, cNZ);
    
    // If the clipping points are the same as the bounding box, then we 
    // don't need to check for clipping, otherwise we do
    if ((min != bmin) || (max != bmax))
	needC = 1;
    else needC = 0;

    toDrawTet.remove_all();
    toDrawMeas.remove_all();
    toDrawTet.grow(numTetra);
    toDrawMeas.grow(numTetra);
    Materials.remove_all();
    for (i=0; i < numTetra; i++)
    {
	toDrawTet[i] = -1;
	toDrawMeas[i] = -1;
    }

    // If we're looking at the outer level
    if (aL == OUTER)
    {
	// For all the elements at that level, if they're inside the clipping
	// box, set them as needing to be displayed.
	for (i = 0; i < levStor[nL].size(); i++)
	    if ((needC == 0) || ((needC == 1) && 
				 doClip(levStor[nL][i], mesh)))
		if (levStor[nL][i] == Seed)
		    toDrawTet[levStor[nL][i]] = 1;
		else	
		    toDrawTet[levStor[nL][i]] = 0;
	Materials.grow(2);
	Materials[0] = genMat[0];
	Materials[1] = genMat[3];
	numGroups = 1;
    }
    else
    {
	// Otherwise we're looking at all of the levels, so run through all
	// the elements at each level, determining whether they need to be
	// displayed or not
	for (i = 0; i < nL; i++)
	    for (j = 0; j < levStor[i].size(); j++)
		if ((needC == 0) || ((needC == 1) && 
				     (doClip(levStor[i][j], mesh))))
		{
		    if (levStor[i][j] == Seed)
			toDrawTet[levStor[i][j]] = 2;
		    else
			toDrawTet[levStor[i][j]] = 0;
		} 
	
	for (i = 0; i < levStor[nL].size(); i++)
	    if ((needC == 0) || ((needC == 1) && 
				 doClip(levStor[nL][i], mesh)))
		if (levStor[nL][i] == Seed)
		    toDrawTet[levStor[nL][i]] = 2;
		else	
		    toDrawTet[levStor[nL][i]] = 1;
	
	Materials.grow(3);
	Materials[0] = genMat[1];
	Materials[1] = genMat[0];
	Materials[2] = genMat[3];
	numGroups = 2;
    }
    return numGroups;
}

/***************************************************************************
 * Procedure for calculating which elements are discarded due to clipping.
 * We look at each level individually, comparing the bounding box of that 
 * level with the clipping cube.  Only those levels that around outside the
 * box are sent here for clipping.
 */

int MeshView::doClip(int n, const MeshHandle& mesh)
{

    Element* e=mesh->elems[n];
    Point p1(mesh->nodes[e->n[0]]->p);
    Point p2(mesh->nodes[e->n[1]]->p);
    Point p3(mesh->nodes[e->n[2]]->p);
    Point p4(mesh->nodes[e->n[3]]->p);


    double cX, cY, cZ, cNX, cNY, cNZ;
    cX = clipX.get(); cNX = clipNX.get();
    cY = clipY.get(); cNY = clipNY.get();
    cZ = clipZ.get(); cNZ = clipNZ.get();


    // If all nodes are inside the box, then we add the element to
    // the group of elements to be rendered.
    
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
	return 1;
    }
    return 0;
}

/***************************************************************************
 * Procedure for calculating the measure, either the volume, aspect ratio,
 * or size versus neighbor.
 */
void MeshView::calcMeasures(const MeshHandle& mesh, double *min, double *max)
{
    int i, e = elmMeas.get();
    int numTetra=mesh->elems.size();


    *min = DBL_MAX;
    *max = DBL_MIN;
    if ((e == 1) && !haveVol)
    {
	volMeas.grow(numTetra);
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
	for (i = 0; i < numTetra; i++)
	    volMeas[i] = (volMeas[i] - *min) / (*max - *min);
    }
    else if ((e == 2) && !haveAsp)
    {
	aspMeas.grow(numTetra);
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
	// We need to know the volumes before we can calculate the size
	// versus neighbor
	sizeMeas.grow(numTetra);
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
	    if (sizeMeas[i] != 0)
	    {
		if (*min > sizeMeas[i])
		    *min = sizeMeas[i];
		if (*max < sizeMeas[i])
		    *max = sizeMeas[i];
	    }
	}
	haveSize = 1;
    }

}

/**************************************************************************
 * Procedure for determining which elements fall inside (or out) of the given
 * range for the particular measure
 */
int MeshView::getMeas(const MeshHandle& mesh, const ColorMapHandle& genColors)
{
    //GeomGroup *gr = new GeomGroup;
    int e = elmMeas.get();
    HashTable<Edge, int> edge_table;
    double min, max;
    histo.GetRange(min, max);
    int cSize = genColors -> colors.size();

    double meas;
    int numTetra=mesh->elems.size();
    int i, in = inside.get();

    // We only need to do calculations if something has changed, ie, we are
    // looking at a different measure, the range has changed, or we have
    // changed whether we want to be looking at the elements inside or outside
    // of the range.

    double mm1, mm2;
    if (in == 0)
	histo.GetMaxMin(mm1, mm2);

    if ((e != oldElm) || ((oldmMin != min) || (oldmMax != max)) ||
	(oldInside != in))
    {
	Materials.remove_all();
	Materials.grow(cSize + 1);
        measTetra.remove_all();
        measTetra.grow(cSize);
	measAuxTetra -> remove_all();

	for (i = 0; i < numTetra; i++)
	{
	    if (e == 1)
		meas = volMeas[i];
	    else if (e == 2)
	    {
		meas = aspMeas[i];
	    }	
	    else if (e == 3)
		meas = sizeMeas[i];
	    
	    // If the element is within the specified range then simply
	    // add the tetrahedron to the group
	    if ((in && (meas >= min) && (meas <= max)) ||
		(!in && ((meas < min) || (meas > max))))
	    {
		double rat;
		if (in == 0)
		    rat = (fabs) (meas - mm1) / (mm2 - mm1);
		else
		    rat = (fabs) (meas - min) / (max - min);
		int bucket = rat * cSize;
		Materials[bucket] = genColors -> lookup(rat);
		toDrawMeas[i] = bucket;
	    }

	    // Otherwise add the edges of the element into an edge table
	    else
	    {
		toDrawMeas[i] = cSize;
		Materials[cSize] = genMat[0];
	    }   
	}
	for (i = 0; i < cSize; i++)
	    measTetra[i] = new GeomGroup;

	for (i = 0; i < numTetra; i++)
	{
	    // If the element is not in the range to be drawn, then add
	    // its edges to the hashtable to be rendered for the wireframe
	    // mesh
	    if (toDrawMeas[i] == cSize)
	    {
		Element* elm=mesh->elems[i];
		Edge e1(elm->n[0], elm->n[1]);
		Edge e2(elm->n[0], elm->n[2]);
		Edge e3(elm->n[0], elm->n[3]);
		Edge e4(elm->n[1], elm->n[2]);
		Edge e5(elm->n[1], elm->n[3]);
		Edge e6(elm->n[2], elm->n[3]);
		
		int dummy=0;
		if (!(edge_table.lookup(e1, dummy)))
		    edge_table.insert(e1, 0);
		int dummy2=0;
		if (!(edge_table.lookup(e2, dummy2)))
		    edge_table.insert(e2, dummy2);
		int dummy3=0;
		if (!(edge_table.lookup(e3, dummy3)))
		    edge_table.insert(e3, dummy3);
		int dummy4=0;
		if (!(edge_table.lookup(e4, dummy4)))
		    edge_table.insert(e4, dummy4);
		int dummy5=0;
		if (!(edge_table.lookup(e5, dummy5)))
		    edge_table.insert(e5, dummy5);
		int dummy6=0;
		if (!(edge_table.lookup(e6, dummy6)))
		    edge_table.insert(e6, dummy6);
	    }
	    // Otherwise, if it is in the range, check each face to see if
	    // that neighbor is to be drawn, and if it is not, then add that
	    // face triangle
	    else if (toDrawMeas[i] != -1)
	    {
		Element *e = mesh -> elems[i];
		for (int j = 0; j < 4; j++)
		{
		    int neighbor = e->face(j);
		    if (neighbor == -1 || (toDrawMeas[neighbor] == -1 || 
					   toDrawMeas[neighbor] == cSize))
		    {
			int i1=e->n[(j+1)%4];
			int i2=e->n[(j+2)%4];
			int i3=e->n[(j+3)%4];
			Point p1(mesh->nodes[i1]->p);
			Point p2(mesh->nodes[i2]->p);
			Point p3(mesh->nodes[i3]->p);
			GeomTri *tri = new GeomTri(p1, p2, p3);
			measTetra[toDrawMeas[i]] -> add(tri);
		    }
		}
	    }
	}
	measMatl.remove_all();
	measMatl.grow(cSize);
	measGroup -> remove_all();
	
	for (i = 0; i < cSize; i++)
	{
	    measMatl[i] = new GeomMaterial(measTetra[i], Materials[i]);
	    measGroup -> add(measMatl[i]);
	}

	// For each edge in the hashtable, add a line for the wireframe.
	HashTableIter<Edge, int> eiter(&edge_table);
	for(eiter.first(); eiter.ok(); ++eiter)
	{
	    Edge e(eiter.get_key());
	    Point p1(mesh->nodes[e.n[0]]->p);
	    Point p2(mesh->nodes[e.n[1]]->p);
	    GeomLine* gline = new GeomLine(p1, p2);
	    measAuxTetra -> add(gline);
	    measAuxMatl -> setMaterial(Materials[cSize]);
	}

	measAuxMatl = new GeomMaterial(measAuxTetra, Materials[cSize]);

	oldElm = e;
	oldmMin = min;
	oldmMax = max;
	oldInside = in;

    }	
    return cSize;
}

/*************************************************************************
 * function to calculate the volume of a tetrahedron, given the four
 * vertices
 *
 * Volume = | Ax  Ay  Az  1 |
 *          | Bx  By  Bz  1 |
 *          | Cx  Cy  Cz  1 |
 *          | Dx  Dy  Dz  1 |
 */
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

    return fabs((a1 + a2 + a3 + a4) / 6.);
}

/**************************************************************************
 * function to calculate the aspect ratio of a tetrahedron
 *
 * Aspect ratio = 4 * sqrt(3/2 * (rho_k / h_k))  where rho_k is the
 * diameter of the sphere circumscribed about the tetrahedron and h_k is
 * the maximum distance between two vertices
 */
double MeshView::aspect_ratio(Point p1, Point p2, Point p3, Point p4)
{
    double rad, len;

    // First calculate the sphere circumscribing the tetrahedron
    get_sphere(p1, p2, p3, p4, rad);
    rad = sqrt(rad);
    double dia = rad * 2;

    // Calculate the maximum distance
    len = getDistance(p1, p2, p3, p4);
    
    double ar = sqrt(1.5) * (len / dia);

    return ar;
}

/**************************************************************************
 * function to calculate the size of an element compared to its 4 
 * neighbors.  We compare the test element with its four neighbors and find
 * the maximum and minimum differences.  The ratio of the maximum and 
 * minimum to the test element are compared, with the largest being returned.
 */
double MeshView::calcSize(const MeshHandle& mesh, int ind)
{
    double m1 = 10000, m2 = -10000, a, b, c, d;

    Element* e=mesh->elems[ind];
    int q = e->face(0);
    if (q != -1)
    {
	a=volMeas[q];
	if (a != 0)
	    m1 = Min(m1, a); m2 = Max(m2, a);
    }
    q = e->face(1);
    if (q != -1)
    {
	b=volMeas[q];
	if (b != 0)
	    m1 = Min(m1, b); m2 = Max(m2, b);
    }
    q = e->face(2);
    if (q != -1)
    {
	c=volMeas[q];
	if (c != 0)
	    m1 = Min(m1, c); m2 = Max(m2, c);
    }
    q = e->face(3);
    if (q != -1)
    {
	d=volMeas[q];
	if (d != 0)
	    m1 = Min(m1, d); m2 = Max(m2, d);
    }

    if (volMeas[ind] != 0)
	return Max(volMeas[ind] / m1, m2 / volMeas[ind]);
    else
	return 0;
}
    
/***************************************************************************
 * function to compare the four edges of a tetrahedron and find the longest
 * one.
 */
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
    return (dis / 2.0);

}

/***************************************************************************
 * Procedure to calculate the sphere circumscribing an element.
 */
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

/***************************************************************************
 * function to determine which element the crosshair point is in.
 */
int MeshView::findElement(Point p, const MeshHandle& mesh,
			  const ColorMapHandle& genColors)
{
    int j, which = (tech.get() == 2) ? lastTech : tech.get();

    int num;
    if (which == 1)
	num = toDrawMeas.size();
    else
	num = toDrawTet.size();
    for (int i = 0; i < num; i++)
    {
	int num2;
	if (which == 0)
	    num2 = toDrawTet[i];
	else if (toDrawMeas[i] == genColors -> colors.size())
	    num2 = -1;
	else
	    num2 = toDrawMeas[i];

	if (num2 != -1)
	{
	    Element* elem=mesh->elems[i];
	    
	    Point p1(mesh->nodes[elem->n[0]]->p);
	    Point p2(mesh->nodes[elem->n[1]]->p);
	    Point p3(mesh->nodes[elem->n[2]]->p);
	    Point p4(mesh->nodes[elem->n[3]]->p);
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
	    double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
	    double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
	    double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
	    double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
	    double iV6=1./(a1+a2+a3+a4);
	    
	    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
	    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
	    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
	    double s1=iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());
	    if(s1<-1.e-6){
		j=elem->face(0);
		if(j==-1)
		    return -1;
		continue;
	    }
	    
	    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
	    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
	    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
	    double s2=iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
	    if(s2<-1.e-6){
		j=elem->face(1);
		if(j==-1)
		    return -1;
		continue;
	    }
	    
	    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
	    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
	    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
	    double s3=iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
	    if(s3<-1.e-6){
		j=elem->face(2);
		if(j==-1)
		    return -1;
		continue;
	    }
	    
	    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
	    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
	    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
	    double s4=iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
	    if(s4<-1.e-6){
		j=elem->face(3);
		if(j==-1)
		    return -1;
		continue;
		
	    }
	    return i;
	}
    }
    return -1;
}

/***************************************************************************
 * Procedure to handle the editing options.  Will add a node in the center
 * of the given tetrahedra, or delete either 1 or 4 nodes.
 */
int MeshView::doEdit(const MeshHandle& mesh)
{
    Element* elem = mesh->elems[toEdit];
cerr << "Deleting: " << toEdit << endl; 

    // Add a point
    if (editMode.get() == 1)
    {
	// Find the center of the tetrahedron and add a node there
	Point p1(mesh->nodes[elem->n[0]]->p);
	Point p2(mesh->nodes[elem->n[1]]->p);
	Point p3(mesh->nodes[elem->n[2]]->p);
	Point p4(mesh->nodes[elem->n[3]]->p);

	Point center((p1.x() + p2.x() + p3.x() + p4.x()) / 4.,
		     (p1.y() + p2.y() + p3.y() + p4.y()) / 4.,
		     (p1.z() + p2.z() + p3.z() + p4.z()) / 4.);
	
	// Tesselate the mesh with this new point.
	mesh -> insert_delaunay(center);
	mesh -> compute_neighbors();
	mesh -> pack_all();
    }
    // Delete the four nodes of the given tetrahedron
    else if (editMode.get() == 2)
    {
	mesh->nodes.remove(elem->n[0]);
	mesh->nodes.remove(elem->n[1]);
	mesh->nodes.remove(elem->n[2]);
	mesh->nodes.remove(elem->n[3]);

	// Retesselate entire mesh
	newTesselation(mesh);
    }
    // Delete the node nearest the crosshair point
    else if (editMode.get() == 3)
    {
	mesh->nodes.remove(findNode(mesh));
	// Retesselate the entire mesh
	newTesselation(mesh);
    }

    // Since the mesh has changed, we need to redetermine what the seed
    // element is.
    oldSeed = -1; oldEdit = -1;
//    toEdit = Seed = findElement(startingTet -> GetPosition(), mesh);
    Seed = -1;
    if (Seed == -1)
    {
	Seed = 0; toEdit = 0;
	Element* elem = mesh->elems[Seed];

	Point p1(mesh->nodes[elem->n[0]]->p);
	Point p2(mesh->nodes[elem->n[1]]->p);
	Point p3(mesh->nodes[elem->n[2]]->p);
	Point p4(mesh->nodes[elem->n[3]]->p);

	Point center((p1.x() + p2.x() + p3.x() + p4.x()) / 4.,
		     (p1.y() + p2.y() + p3.y() + p4.y()) / 4.,
		     (p1.z() + p2.z() + p3.z() + p4.z()) / 4.);
	
	startingTet -> SetPosition(center);
    }

    return 1;
}

/***************************************************************************
 * Procedure to retessellate the mesh.  This is only done when a node is
 * deleted, and is necessary as we can not determine how far-reaching the 
 * effect of removing a node will be.
 */
int MeshView::newTesselation(const MeshHandle& mesh)
{
    mesh -> elems.remove_all();
    mesh -> compute_neighbors();
    

    // We first need to determine the bounding box of the mesh.
    int nn = mesh -> nodes.size();
    BBox bbox;
    for (int i = 0; i < nn; i++)
	bbox.extend(mesh->nodes[i] -> p);
    
    double eps = 1.e-4;
    
    bbox.extend(bbox.max() - Vector(eps, eps, eps));
    bbox.extend(bbox.min() + Vector(eps, eps, eps));
    
    Point center(bbox.center());
    double le=1.0001*bbox.longest_edge();
    
    Vector diag(le, le, le);
    Point bmin(center - diag/2.0);
    
    // Next, we create the first tetrahedron of the mesh that bounds all
    // of the nodes
    mesh -> nodes.add(new Node(bmin-Vector(le, le, le)));
    mesh -> nodes.add(new Node(bmin+Vector(le*5, 0, 0)));
    mesh -> nodes.add(new Node(bmin+Vector(0, le*5, 0)));
    mesh -> nodes.add(new Node(bmin+Vector(0, 0, le*5)));
    
    Mesh* m2 = mesh.get_rep();
    Element* e = new Element(m2, nn+0, nn+1, nn+2, nn+3);
    int onn=nn;
    
    // Orient the element so that it has a positive volume
    e -> orient();

    // Initialize the element to indicate it has no neighbors
    e -> faces[0] = e -> faces[1] = e -> faces[2] = e -> faces[3] = -1;
    mesh -> elems.add(e);
    
    // One by one add in all of the nodes
    for (int node=0; node < nn; node++)
    {
	if (!mesh->insert_delaunay(node))
	{
	    error("Mesher upset");
	    return 0;
	}
	
	if (node%200 == 0)
	{
	    mesh -> pack_elems();
	}
    }
    
    // Compute the neighbors of each element
    mesh -> compute_neighbors();

    // And remove the four extra points added for the bounding tetrahedron
    mesh -> remove_delaunay(onn, 0);
    mesh -> remove_delaunay(onn + 1, 0);
    mesh -> remove_delaunay(onn + 2, 0);
    mesh -> remove_delaunay(onn + 3, 0);
    mesh -> pack_all();

    return 1;
}

/***************************************************************************
 * function to determine which of the four nodes of the edit element is 
 * nearest to the crosshair picker. 
 */
int MeshView::findNode(const MeshHandle& mesh)
{
    Element* e = mesh -> elems[toEdit];
    
    Point p1(mesh->nodes[e->n[0]]->p);
    Point p2(mesh->nodes[e->n[1]]->p);
    Point p3(mesh->nodes[e->n[2]]->p);
    Point p4(mesh->nodes[e->n[3]]->p);

    Point newp = editPoint;

    Vector v1(newp - p1);
    Vector v2(newp - p2);
    Vector v3(newp - p3);
    Vector v4(newp - p4);

    // Calculate the distance from each point to the editPoint.
    double d1 = sqrt(v1.x() * v1.x() + v1.y() * v1.y() + v1.z() * v1.z());
    double d2 = sqrt(v2.x() * v2.x() + v2.y() * v2.y() + v2.z() * v2.z());
    double d3 = sqrt(v3.x() * v3.x() + v3.y() * v3.y() + v3.z() * v3.z());
    double d4 = sqrt(v4.x() * v4.x() + v4.y() * v4.y() + v4.z() * v4.z());

    // Find the smallest distance
    double m1 = Min(Min(Min(d1, d2), d3), d4);
    
    int node;

    // Determine which node it was
    if (m1 == d1) node = 0;
    else if (m1 == d2) node = 1;
    else if (m1 == d3) node = 2;
    else node = 3;

    return e->n[node];
}

/***************************************************************************
 * Procedure to update the information in the interface
 */
void MeshView::updateInfo(const MeshHandle& mesh)
{	
    char buf[1000];
    double m1, m2;
    if (!haveVol)
	calcMeasures(mesh, &m1, &m2);
    double t1, t2, t3, t4;
    t1 = volMeas[Seed]; t3 = volMeas[toEdit];
    if (!haveAsp)
    {
	Element* e=mesh->elems[Seed];
	t2 = aspect_ratio(mesh->nodes[e->n[0]]->p,
			  mesh->nodes[e->n[1]]->p,
			  mesh->nodes[e->n[2]]->p,
			  mesh->nodes[e->n[3]]->p);
	e=mesh->elems[toEdit];
	t4 = aspect_ratio(mesh->nodes[e->n[0]]->p,
			  mesh->nodes[e->n[1]]->p,
			  mesh->nodes[e->n[2]]->p,
			  mesh->nodes[e->n[3]]->p);
	
    }
    else
	t2 = aspMeas[Seed];
    
    ostrstream str4(buf, 1000);
    str4 << id << " set_info " << " " << mesh->nodes.size() << " " <<
	mesh->elems.size() << " " << Seed << " " << t1 << " " << t2 << 
	    " " << toEdit << " " << t3 << " " << t4 << '\0';
    TCL::execute(str4.str());
    oldSeed = Seed;
    changed = 1;
}

/***************************************************************************
 * Procedure for when the mouse button is released when using a widget
 */
void MeshView::geom_release(GeomPick*, void*)
{
    if(!abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}

/***************************************************************************
 * Dummy procedure for when the mouse button is moved during a widget action
 */
void MeshView::geom_moved(GeomPick*, int, double, const Vector&, void*)
{
}

#ifdef __GNUG__

#include <Classlib/HashTable.cc>

template class HashTable<Edge, int>;
template class HashTableIter<Edge, int>;
template class HashKey<Edge, int>;
template int Hash(Edge&, int);

#include <Classlib/Array1.cc>
template class Array1<GeomGroup*>;
template class Array1<Array1<int> >;

#endif
