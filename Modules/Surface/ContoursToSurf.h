
/*
 *  ContoursToSurf.h: Merge the input contour sets into a Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_ContoursToSurf_h
#define SCI_project_module_ContoursToSurf_h

#include <Dataflow/Module.h>
#include <Classlib/Array1.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ContourSetPort.h>

class Grid;

class ContoursToSurf : public Module {
    Array1<ContourSetIPort*> incontours;
    SurfaceOPort* osurface;
    BBox bbox;
    Grid *grid;
    void break_triangle(int tri_id, int pt_id, const Point& p, TriSurface*);
    void break_edge(int tri1,int tri2,int e1,int e2,int pt_id,const Point &p,
		    TriSurface*);
    void break_edge2(int tri1, int e1, int pt_id,const Point &p, TriSurface*);
    void lace_contours(const ContourSetHandle& contour, TriSurface* surf);
    void add_point(const Point& p, TriSurface* surf);
    double distance(const Point &p, Array1<int> &res, TriSurface*);
    void contours_to_surf(const Array1<ContourSetHandle> &contours, TriSurface*);
    Array1<int>* get_cubes_at_distance(int dist, int i, int j, int k, int imax,
				       int jmax, int kmax);
public:
    ContoursToSurf(const clString&);
    ContoursToSurf(const ContoursToSurf&, int deep);
    virtual ~ContoursToSurf();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
//    virtual void mui_callback(void*, int);
};

#endif
