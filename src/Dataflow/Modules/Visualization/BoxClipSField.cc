
/*
 *  BoxClipSField.cc:  Clip a field using a box widget
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Tester/RigorousTest.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Plane.h>
#include <Core/TclInterface/TCLvar.h>
#include <stdio.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class BoxClipSField : public Module {
    ScalarFieldIPort* ifield;
    ScalarFieldOPort* ofield;
    GeometryOPort *ogeom;
    CrowdMonitor widget_lock;
    int widget_id;
    ScaledBoxWidget *widget;
    int init;
    int find_field;
    int axis_allign_flag;
    TCLint interp;
    TCLint axis;
public:
    BoxClipSField(const clString& id);
    virtual ~BoxClipSField();
    virtual void execute();
    ScalarField* UGtoUG(const Point &, const Point &, const Vector &, 
			const Vector&, const Vector &, ScalarFieldUG *, int);
    ScalarField* RGtoRG_Alligned(int, int, int, const Point &, const Point &,
				 ScalarFieldRG*);
    ScalarField* RGtoRG_Unalligned(int, int, int, const Point &, 
				   const Vector &, const Vector &,
				   const Vector &, ScalarFieldRG*);
    virtual void widget_moved(int last);    
    virtual void tcl_command(TCLArgs&, void*);
    
    ScalarFieldHandle fldHandle;
    ScalarFieldRG* osf;
};

extern "C" Module* make_BoxClipSField(const clString& id) {
  return new BoxClipSField(id);
}

static clString module_name("BoxClipSField");

BoxClipSField::BoxClipSField(const clString& id)
: Module("BoxClipSField", id, Filter), widget_lock("BoxClipSField lock"),
    axis("axis", id, this), interp("interp", id, this)
{
    ifield=scinew ScalarFieldIPort(this, "SField", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    // Create the output port
    ofield=scinew ScalarFieldOPort(this, "SField", ScalarFieldIPort::Atomic);
    add_oport(ofield);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    widget=scinew ScaledBoxWidget(this, &widget_lock, 0.1);
    init=1;
    find_field=1;
    axis_allign_flag=1;
}

BoxClipSField::~BoxClipSField()
{
}

void BoxClipSField::execute()
{
    ScalarFieldHandle ifh;
    if(!ifield->get(ifh))
	return;
    if (init) {
	init=0;
	GeomObj *w=widget->GetWidget();
	widget_id = ogeom->addObj(w, module_name, &widget_lock);
	widget->Connect(ogeom);
	widget->SetRatioR(0.2);
	widget->SetRatioD(0.2);
	widget->SetRatioI(0.2);
    }
    double ld=ifh->longest_dimension();
    if (find_field) {
	find_field=0;
	Point min, max;
	ifh->get_bounds(min,max);
	Point center = min + (max-min)/2.0;
	Point right( max.x(), center.y(), center.z());
	Point down( center.x(), max.y(), center.z());
	Point in( center.x(), center.y(), max.z());
	widget->SetPosition( center, right, down, in);
	widget->SetScale(ld/20.);
	init=1;
    }
    if (axis_allign_flag) {
	widget->AxisAligned(axis.get());
	axis_allign_flag=0;
    }
    Point center, R, D, I;
    widget->GetPosition( center, R, D, I);
    Vector v1 = R - center,
    v2 = D - center,
    v3 = I - center;
    
    // calculate the corner and the
    // u and v vectors of the cutting plane
    Point minPt = center - v1 - v2 - v3;
    Point maxPt = center + v1 + v2 + v3;
    Vector u = v1 * 2.0,
    v = v2 * 2.0,
    w = v3 * 2.0;

    double u_fac = widget->GetRatioR();
    double v_fac = widget->GetRatioD();
    double w_fac = widget->GetRatioI();
    int u_num = (int)(u_fac*100);
    int v_num = (int)(v_fac*100);
    int w_num = (int)(w_fac*100);

    // Now we need to figure out whether our input was structured or 
    // unstructured, and whether our output was structured or unstructured,
    // and then call the right method to generate our new field.

    ScalarField* osf;
    ScalarFieldRG *sfrg=ifh->getRG();
    ScalarFieldUG *sfug=ifh->getUG();
    if (sfug) {
	osf=UGtoUG(minPt, maxPt, u, v, w, sfug, widget->IsAxisAligned());
    } else if (widget->IsAxisAligned()) {
	osf=RGtoRG_Alligned(u_num, v_num, w_num, minPt, maxPt, sfrg);
    } else {
	osf=RGtoRG_Unalligned(u_num, v_num, w_num, minPt, u, v, w, sfrg);
    }
    ofield->send(osf);
    return;
}

ScalarField* BoxClipSField::UGtoUG(const Point &minPt, const Point &maxPt,
				   const Vector &/*u*/, const Vector &/*v*/, 
				   const Vector &/*w*/, ScalarFieldUG *sfug, 
				   int isAlligned) {
    ScalarFieldUG* sf = scinew ScalarFieldUG(ScalarFieldUG::NodalValues);
    MeshHandle m=sfug->mesh;
    Array1<int> in_nodes(m->nodesize());
    Array1<int> in_elements(m->elemsize());
    //Array1<int> new_nodes;
    if (isAlligned) {
	for (int i=0; i<m->nodesize(); i++) {
	    const Point &p = m->point(i);
	    if (p.x()>=minPt.x() && p.y()>=minPt.y() && p.z()>=minPt.z() &&
		p.x()<=maxPt.x() && p.y()<=maxPt.y() && p.z()<=maxPt.z())
		in_nodes[i]=-2;
	    else 
		in_nodes[i]=-3;
	}
    } else {
	Array1<Point> pnt;
	pnt.add(Point(minPt.x(), minPt.y(), minPt.z()));
	pnt.add(Point(maxPt.x(), minPt.y(), minPt.z()));
	pnt.add(Point(maxPt.x(), maxPt.y(), minPt.z()));
	pnt.add(Point(minPt.x(), maxPt.y(), minPt.z()));
	pnt.add(Point(minPt.x(), minPt.y(), maxPt.z()));
	pnt.add(Point(maxPt.x(), minPt.y(), maxPt.z()));
	pnt.add(Point(maxPt.x(), maxPt.y(), maxPt.z()));
	pnt.add(Point(minPt.x(), maxPt.y(), maxPt.z()));
	Array1<Plane> pln;
	pln.add(Plane(pnt[0], pnt[1], pnt[2]));
	pln.add(Plane(pnt[6], pnt[5], pnt[4]));
	pln.add(Plane(pnt[1], pnt[5], pnt[6]));
	pln.add(Plane(pnt[7], pnt[4], pnt[0]));
	pln.add(Plane(pnt[2], pnt[6], pnt[7]));
	pln.add(Plane(pnt[4], pnt[5], pnt[1]));
	for (int i=0; i<m->nodesize(); i++) {
 	    const Point &p = m->point(i);
	    if (pln[0].eval_point(p)<=0 && pln[1].eval_point(p)<=0 &&
		pln[2].eval_point(p)<=0 && pln[3].eval_point(p)<=0 &&
		pln[4].eval_point(p)<=0 && pln[5].eval_point(p)<=0)
		in_nodes[i]=-2;
	    else
		in_nodes[i]=-3;
	}
    }
    int nelems=0;
    int i;
    for (i=0; i<m->elemsize(); i++) {
	if (in_nodes[m->element(i)->n[0]]>-3 && 
	    in_nodes[m->element(i)->n[1]]>-3 &&
	    in_nodes[m->element(i)->n[2]]>-3 && 
	    in_nodes[m->element(i)->n[3]]>-3) {
	    in_elements[i]=1;
	    nelems++;
	    if (in_nodes[m->element(i)->n[0]]==-2) 
		in_nodes[m->element(i)->n[0]] = -1;
	    if (in_nodes[m->element(i)->n[1]]==-2) 
		in_nodes[m->element(i)->n[1]] = -1;
	    if (in_nodes[m->element(i)->n[2]]==-2) 
		in_nodes[m->element(i)->n[2]] = -1;
	    if (in_nodes[m->element(i)->n[3]]==-2) 
		in_nodes[m->element(i)->n[3]] = -1;
	} else
	    in_elements[i]=0;
    }
    int nnodes=0;
    sf->mesh = scinew Mesh(nnodes, nelems);
    for (i=0; i<in_nodes.size(); i++) {
	if (in_nodes[i] == -1) {
	    in_nodes[i] = nnodes;
	    sf->mesh->nodes.add(new Node(m->point(i)));
	    sf->data.add(sfug->data[i]);
	    nnodes++;
	}
    }
    int currEl=0;
    for (i=0; i<in_elements.size(); i++) {
	if (in_elements[i]) {
	    sf->mesh->elems[currEl] =
	      new Element(sf->mesh.get_rep(),
			  in_nodes[m->element(i)->n[0]],
			  in_nodes[m->element(i)->n[1]],
			  in_nodes[m->element(i)->n[2]],
			  in_nodes[m->element(i)->n[3]]);
	    currEl++;
	}
    }
    cerr << "currEl="<<currEl<<" and nelems="<<nelems<<"\n";
    cerr << "New mesh has " << nnodes << "/" << m->nodesize();
    cerr <<" nodes and " << nelems << "/" << m->elemsize() << " elements.\n";
    if (sf->mesh->elemsize()>0) {
	sf->mesh->compute_neighbors();
    } else {
	free(sf);
	sf=0;
    }
    return sf;
}

ScalarField* BoxClipSField::RGtoRG_Alligned(int /*u_num*/, int /*v_num*/,
					    int /*w_num*/,
					    const Point &minPt, 
					    const Point &maxPt,
					    ScalarFieldRG* sfrg) {
    ScalarFieldRG* sf = scinew ScalarFieldRG;
    if (!interp.get()) {
	int u_min, v_min, w_min, u_max, v_max, w_max;
	sfrg->locate(minPt, u_min, v_min, w_min);
	sfrg->locate(maxPt, u_max, v_max, w_max);
cerr << "MinPt:"<<minPt<<" is at ("<<u_min<<", "<<v_min<<", "<<w_min<<")\n";
cerr << "MaxPt:"<<minPt<<" is at ("<<u_max<<", "<<v_max<<", "<<w_max<<")\n";
cerr << "Field is: "<<sfrg->grid.dim1()<<" x "<<sfrg->grid.dim2()<<" x "<<sfrg->grid.dim3()<<"\n";
	u_max++; v_max++; w_max++;
	if (u_min < 0) u_min=0;
	if (v_min < 0) v_min=0;
	if (w_min < 0) w_min=0;
	if (u_min > sfrg->grid.dim1()) u_min=sfrg->grid.dim1();
	if (v_min > sfrg->grid.dim2()) v_min=sfrg->grid.dim2();
	if (w_min > sfrg->grid.dim3()) w_min=sfrg->grid.dim3();
	if (u_max < 0) u_max=0;
	if (v_max < 0) v_max=0;
	if (w_max < 0) w_max=0;
	if (u_max > sfrg->grid.dim1()) u_max=sfrg->grid.dim1();
	if (v_max > sfrg->grid.dim2()) v_max=sfrg->grid.dim2();
	if (w_max > sfrg->grid.dim3()) w_max=sfrg->grid.dim3();
	sf->nx=u_max-u_min;
	sf->ny=v_max-v_min;
	sf->nz=w_max-w_min;
	if (sf->nx==0 || sf->ny==0 || sf->nz==0) {
	    free(sf);
	    sf=0;
	    return sf;
	}
	sf->grid.newsize(sf->nx, sf->ny, sf->nz);
	for (int i=u_min; i<u_max; i++) {
	    for (int j=v_min; j<v_max; j++) {
		for (int k=w_min; k<w_max; k++) {
		    sf->grid(i-u_min, j-v_min, k-w_min)=sfrg->grid(i,j,k);
		}
	    }
	}
	sf->set_bounds(sfrg->get_point(u_min, v_min, w_min),
		       sfrg->get_point(u_max-1, v_max-1, w_max-1));
	sf->compute_minmax();
    } else {
	NOT_FINISHED("BoxClipSField::RGtoRG_alligned w/ interpolation");
	free(sf);
	sf=0;
    }
    return sf;
}

ScalarField* BoxClipSField::RGtoRG_Unalligned(int /*u_num*/, int /*v_num*/, 
					      int /*w_num*/,
					      const Point &/*minPt*/, 
					      const Vector &/*u*/, 
					      const Vector &/*v*/,
					      const Vector &/*w*/, 
					      ScalarFieldRG* /*sfrg*/) {
    ScalarFieldRG *sf = scinew ScalarFieldRG;
    NOT_FINISHED("BoxClipSField::RGtoRG_unalligned");
    free(sf);
    sf=0;
    return sf;

/*
    for (int i = 0; i < u_num; i++)
        for (int j = 0; j < v_num; j++)
            for(int k = 0; k < w_num; k++)
                {
                    Point p = corner + u * ((double) i/(u_num-1)) + 
                        v * ((double) j/(v_num-1)) +
                            w * ((double) k/(w_num-1));
                    
                    // Query the vector field...
                    double val;
                    if (vfield->interpolate( p, val)){
			
                        if(have_sfield){
                            // get the color from cmap for p        
                            MaterialHandle matl;
                            double sval;
                            if (ssfield->interpolate( p, sval))
                                matl = cmap->lookup( sval);
                            else
                                {
                                    matl = outcolor;
                                }
                            arrows->add(p, vv*lenscale, matl, matl, matl);
                        } else {
                            arrows->add(p, vv*lenscale);
                        }
                    }
                }
    grid_id = ogeom->addObj(arrows, module_name);
    
    // delete the old grid/cutting plane
    if (old_grid_id != 0)
        ogeom->delObj( old_grid_id );
*/
}


void BoxClipSField::widget_moved(int last)
{
    if(last && !abort_flag)
        {
            abort_flag=1;
            want_to_execute();
        }
}


void BoxClipSField::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2)
        {
            args.error("BoxClipSField needs a minor command");
            return;
        }
    if(args[1] == "find_field")
        {
	    find_field=1;
            want_to_execute();
        }
    else if(args[1] == "axis_allign")
        {
	    axis_allign_flag=1;
	    want_to_execute();
	}
    else
        {
            Module::tcl_command(args, userdata);
        }
}

} // End namespace SCIRun

