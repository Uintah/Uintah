
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
#include <Field3D.h>
#include <Field3DPort.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_MeshView()
{
    return new MeshView;
}

static RegisterModule db1("Fields", "MeshView", make_MeshView);
static RegisterModule db2("Visualization", "MeshView", make_MeshView);

MeshView::MeshView()
: UserModule("MeshView", Source)
{
    FILE *dat, *tet;
    double x, y, z;
    int a, b, c, d;

    numLevels=1;
    MUI_slider_int *slide = new MUI_slider_int("Number of Levels", &numLevels,
			       MUI_widget::Immediate, 1);

	add_ui(slide);
	slide -> set_minmax(0, 5);

    dat = fopen("/home/grad/cgitlin/classes/cs523/project/cube.pts","r");
    numVerts = 0;

    data = (double *) malloc (3 * 10000 * sizeof(double));
    tetra = (int *) malloc (4 * 50000 * sizeof(int));

    while (!feof(dat))
	{
		fscanf(dat,"%lf %lf %lf", &x, &y, &z);
		data[numVerts * 3] = x;
		data[numVerts * 3 + 1] = y;
		data[numVerts * 3 + 2] = z;
		numVerts++;
	}

    data = (double *) realloc(data, 3 * numVerts * sizeof(double));
    
    tet = fopen("/home/grad/cgitlin/classes/cs523/project/cube.tetra", "r");
    list = (LPTR *) malloc (numVerts * sizeof(LPTR));
    // levels = (LPTR *) malloc (numVerts * sizeof(LPTR));
    // cerr << "levels=" << levels << endl;
    numCnct = (int *) malloc (numVerts * sizeof(int));

    initList();
    numTetra = 0;
   
	while (!feof(tet))
	{
		fscanf(tet, "%d %d %d %d", &a, &b, &c, &d);
		tetra[numTetra * 4] = a;
		tetra[numTetra * 4 + 1] = b;
		tetra[numTetra * 4 + 2] = c;
		tetra[numTetra * 4 + 3] = d;
        addTet(a, numTetra); 
        addTet(b, numTetra); 
        addTet(c, numTetra);
        addTet(d, numTetra);
		numTetra++;
	}
    
    //tetra = (int *) realloc(tetra, 4 * numTetra * sizeof(int));
	// makeLevels();	
    sched_state=SchedNewData;

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

MeshView::MeshView(const MeshView& copy, int levDeep)
: UserModule(copy, levDeep)
{
    NOT_FINISHED("MeshView::MeshView");
}

MeshView::~MeshView()
{
}

Module* MeshView::clone(int levDeep)
{
    return new MeshView(*this, levDeep);
}

void MeshView::execute()
{
  cerr << "Executing MeshView...\n";
	int i, j;
	LPTR curr;

    ogeom->delAll();

	ObjGroup *group = new ObjGroup;
	for (i = 0; i < numVerts; i++)
	{
		for (curr = list[i] -> next; curr != NULL; curr = curr -> next)
		{
			j = curr -> tetra;
			Point p1(data[tetra[j * 4] * 3], 
						data[tetra[j * 4] * 3 + 1],
						data[tetra[j * 4] * 3 + 2]);
			Point p2(data[tetra[j * 4 + 1] * 3], 
						data[tetra[j * 4 + 1] * 3 + 1],
						data[tetra[j * 4 + 1] * 3 + 2]);
			Point p3(data[tetra[j * 4 + 2] * 3], 
						data[tetra[j * 4 + 2] * 3 + 1],
						data[tetra[j * 4 + 2] * 3 + 2]);
			Point p4(data[tetra[j * 4 + 3] * 3], 
						data[tetra[j * 4 + 3] * 3 + 1],
						data[tetra[j * 4 + 3] * 3 + 2]);
			group -> add(new Tetra(p1, p2, p3, p4));	
		}
	}

	ogeom -> addObj(group);
}

void MeshView::initList()
{
    int i;

    for (i = 0; i < numVerts; i++)
	{
		numCnct[i] = 0;
		list[i] = newList();
		list[i] -> next = NULL;
	}

}

void MeshView::addTet(int row, int ind)
{
    int fin = 0;
    LPTR newL, curr;

    curr = list[row] -> next;
    while ((curr != NULL) && (!fin))
    {
        if (curr -> tetra == ind)
	     fin = 1;
	else
	     curr = curr -> next;
    }

    if (!fin)
    {
		newL = newList();
		newL -> tetra = ind;
		newL -> next = NULL;
		curr = list[row];
		while (curr -> next != NULL)
		    curr = curr -> next;
		curr -> next = newL;
		numCnct[row]++;
    }
}

LPTR MeshView::newList()
{
    return (LPTR) malloc (sizeof (LIST));
}

void MeshView::makeLevels()
{
	int i, j, counter = numTetra; 
	LPTR newL = newList(), curr, trav;
	int *visitedV, *visitedT;
	int levDeep = 0;

	visitedV = (int *) malloc (numVerts * sizeof(int));
	visitedT = (int *) malloc (numTetra * sizeof(int));

	for (i = 0; i < numVerts; i++)
		visitedV[i] = 0;

	for (i = 0; i < numTetra; i++)
		visitedT[i] = 0;


	newL -> tetra = 0;
	newL -> next = NULL;
	levels[levDeep] -> next = newL;
	visitedT[0] = 1;

	while (counter > 1)
	{
		for (trav = levels[levDeep] -> next; trav != NULL; trav = trav -> next)
		{
			j = trav -> tetra;
			for (i = 0; i < 4; i++)
			{
				if (visitedV[tetra[j * 4 + i]] == 0)
				{
					visitedV[tetra[j * 4 + i]] = 1;
					for (curr = list[tetra[j * 4 + i]] -> next; curr != NULL; 
					        	curr = curr -> next)
					{
						if (visitedT[curr -> tetra] == 0)
						{
							newL = newList();
							newL -> tetra = curr -> tetra;
							newL -> next = levels[levDeep + 1] -> next;
							levels[levDeep + 1] -> next = newL;
							visitedT[curr -> tetra] = 1;
							counter = counter - 1;
						      }
					      }
				      }
			      }
		      }
		levDeep = levDeep + 1;
	}
}
	
void MeshView::mui_callback(void*, int which)
{
    want_to_execute();
    cerr << "MeshView::mui_callback" << endl;
}
