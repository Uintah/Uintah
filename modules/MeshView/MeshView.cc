
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

#include <stdio.h>
#include <malloc.h>
#include <Classlib/HashTable.h>
#include <MeshView/MeshView.h>
#include <Geometry/Point.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>
#include <Classlib/Queue.h>

static Module* make_MeshView()
{
    return new MeshView;
}

static RegisterModule db1("Fields", "MeshView", make_MeshView);
static RegisterModule db2("Visualization", "MeshView", make_MeshView);

static clString mesh_name("Mesh");

MeshView::MeshView()
: UserModule("MeshView", Source)
{

    readDat();
    levels = (int *) malloc (numTetra * sizeof(int));
    oldLev = -1;
    numLevels=3;
    levSlide = new MUI_slider_int("Number of Levels",
				  &numLevels,
				  MUI_widget::Immediate, 1);
    add_ui(levSlide);
    deep = 4;
    levSlide -> set_minmax(0, deep);

    oldSeed = -1;
    seedTet = 0;
    MUI_slider_int *seedSlide = new MUI_slider_int("Starting Tetra", &seedTet,
					MUI_widget::Immediate, 1);
    add_ui(seedSlide);	
    seedSlide -> set_minmax(0, numTetra - 1);

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

    sched_state=SchedNewData;

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

}

MeshView::MeshView(const MeshView& copy, int deep)
: UserModule(copy, deep)
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
    int i;
  
    ogeom->delAll();
    if (oldSeed != seedTet)
    {
	makeLevels();
	levSlide -> set_minmax(0, deep);
	oldSeed = seedTet;
    }

    oldLev = numLevels;
    oldClipX = clipX;
    oldClipY = clipY;
    oldClipZ = clipZ;
    ObjGroup *group = new ObjGroup;
    for (i = 0; i < numTetra; i++)
    {
	if (((allLevels == 0) && (levels[i] == numLevels)) ||
	    ((allLevels == 1) && (levels[i] <= numLevels)))
        {
	    Point p1(data[tetra[i * 4] * 3], 
		     data[tetra[i * 4] * 3 + 1],
		     data[tetra[i * 4] * 3 + 2]);
	    Point p2(data[tetra[i * 4 + 1] * 3], 
		     data[tetra[i * 4 + 1] * 3 + 1],
		     data[tetra[i * 4 + 1] * 3 + 2]);
	    Point p3(data[tetra[i * 4 + 2] * 3], 
		     data[tetra[i * 4 + 2] * 3 + 1],
		     data[tetra[i * 4 + 2] * 3 + 2]);
	    Point p4(data[tetra[i * 4 + 3] * 3], 
		     data[tetra[i * 4 + 3] * 3 + 1],
		     data[tetra[i * 4 + 3] * 3 + 2]);

	    if (((p1.x() > clipX) && (p2.x() > clipX) && (p3.x() > clipX) &&
		 (p4.x() > clipX)) &&
		((p1.y() > clipY) && (p2.y() > clipY) && (p3.y() > clipY) &&
		 (p4.y() > clipY)) &&
		((p1.z() > clipZ) && (p2.z() > clipZ) && (p3.z() > clipZ) && 
		(p4.z() > clipZ)))
		group -> add(new Tetra(p1, p2, p3, p4));	
	}
    }
  
    ogeom -> addObj(group, mesh_name);
}	

void MeshView::initList()
{
    int i;

    for (i = 0; i < numVerts; i++)
    {
	list[i] = newList();
	list[i] -> next = NULL;
    }

}

void MeshView::addTet(int row, int ind)
{
    int fin = 0;
    LPTR newL, curr;

    curr = list[row];
    while ((curr -> next != NULL) && (!fin))
    {
        if (curr -> next -> tetra == ind)
	     fin = 1;
	else
	     curr = curr -> next;
    }

    if (!fin)
    {
		newL = newList();
		newL -> tetra = ind;
		newL -> next = NULL;
		curr -> next = newL;
    }
}

LPTR MeshView::newList()
{
    return (LPTR) malloc (sizeof (LIST));
}

void MeshView::makeLevels()
{
    int i, j, x; 
    Queue<int> q;
    LPTR curr;
    int *work, wCount;

    work = (int *) malloc (numTetra * sizeof(int));
    
    int counter = 0;
    
    for (i = 0; i < numTetra; i++)
	levels[i] = -1;

    q.append(seedTet);
    q.append(-2);
	
    deep = 0;
    while(counter < numTetra)
    {
	x = q.pop();
	if (x == -2)
	{
	    deep++;
	    q.append(-2);
	}
	else if (levels[x] == -1)
	{
	    levels[x] = deep;
	    counter++;
	    wCount = 0;
//	    for (i = 0; i < 4; i++)
//	    {
//		for (curr = list[tetra[x * 4 + i]] -> next; curr != NULL;
//		     curr = curr -> next)
//	        {
//		    q.append(curr -> tetra);
//		 
//		}	
	    for (i = 0; i < 4; i++)
	    {
		for (curr = list[tetra[x * 4 + i]] -> next; curr != NULL;
		     curr = curr -> next)
		{
		    if (work[curr -> tetra] == 0)
			wCount++;
		    work[curr -> tetra]++;
		}
	    }
	    cerr << "Count " << counter << ": " <<  wCount << endl;
	    for (i = 0; i < numTetra; i++)
	    {
		if (work[i] == 3)
		    q.append(i);
		work[i] = 0;
	    }
	}   
    }     
}	

	
void MeshView::readDat()
{
    FILE *dat, *tet;
    double x, y, z;
    int a, b, c, d;

    dat = fopen("/home/grad/cgitlin/classes/cs523/project/cube.pts","r");
    numVerts = 0;

    data = (double *) malloc (3 * 10000 * sizeof(double));
    tetra = (int *) malloc (4 * 50000 * sizeof(int));

    Xmin = Ymin = Zmin = 10000; 
    Xmax = Ymax = Zmax = -10000;
    while (!feof(dat))
    {
	fscanf(dat,"%lf %lf %lf", &x, &y, &z);
	data[numVerts * 3] = x;
	data[numVerts * 3 + 1] = y;
	data[numVerts * 3 + 2] = z;
	numVerts++;
	if (Xmin > x) Xmin = x;
	if (Xmax < x) Xmax = x;
	if (Ymin > y) Ymin = y;
	if (Ymax < y) Ymax = y;
	if (Zmin > z) Zmin = z;
	if (Zmax < z) Zmax = z;
    }

    
    tet = fopen("/home/grad/cgitlin/classes/cs523/project/cube.tetra", "r");
    list = (LPTR *) malloc (numVerts * sizeof(LPTR));

    initList();
    numTetra = 0;
   
    while (!feof(tet))
    {
	fscanf(tet, "%d %d %d %d", &a, &b, &c, &d);
	tetra[numTetra * 4] = a-1;
	tetra[numTetra * 4 + 1] = b-1;
	tetra[numTetra * 4 + 2] = c-1;
	tetra[numTetra * 4 + 3] = d-1;
	addTet(a-1, numTetra); 
	addTet(b-1, numTetra); 
	addTet(c-1, numTetra);
	addTet(d-1, numTetra);
	numTetra++;
    }

}
    
void MeshView::mui_callback(void*, int which)
{
    if ((which == 0) && (oldLev != numLevels))
	want_to_execute();

    else if ((which == 1) && (oldSeed != seedTet))
	want_to_execute();

    else if (oldClipX != clipX)
	want_to_execute();

    else if (oldClipY != clipY)
	want_to_execute();

    else if (oldClipZ != clipZ)
	want_to_execute();

    else if (which == 3)
	want_to_execute();
}
