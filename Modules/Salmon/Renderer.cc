
/*
 *  Renderer.h: Abstract interface to a renderer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <iostream.h>
#include <values.h>

static AVLTree<clString, RegisterRenderer*>* known_renderers=0;
static int db_trimmed=0;

RegisterRenderer::RegisterRenderer(const clString& name,
				   query_Renderer query,
				   make_Renderer maker)
: name(name), query(query), maker(maker)
{
    RegisterRenderer* tmp;
    if(!known_renderers)
	known_renderers=new AVLTree<clString, RegisterRenderer*>;
    if(known_renderers->lookup(name, tmp)){
	cerr << "Error: Two renderers of the same name!" << endl;
    } else {
	RegisterRenderer* that=this;
	known_renderers->insert(name, that);
    }
}

RegisterRenderer::~RegisterRenderer()
{
}

Renderer* Renderer::create(const clString& type)
{
    RegisterRenderer* rr;
    if(known_renderers->lookup(type, rr)){
	make_Renderer maker=rr->maker;
	return (*maker)();
    } else {
	return 0;
    }
}

AVLTree<clString, RegisterRenderer*>* Renderer::get_db()
{
    if(!db_trimmed){
	AVLTreeIter<clString, RegisterRenderer*> iter(known_renderers);
	for(iter.first();iter.ok();++iter){
	    query_Renderer query=iter.get_data()->query;
	    if(! (*query)()){
		// This renderer is not supported, remove it...
		known_renderers->remove(iter);
	    }
	}
	db_trimmed=1;
    }
    return known_renderers;
}


int Renderer::compute_depth(Roe* roe, const View& view,
			    double& znear, double& zfar)
{
    znear=MAXDOUBLE;
    zfar=-MAXDOUBLE;
    BBox bb;
    roe->get_bounds(bb);
    if(bb.valid()) {
	// We have something to draw...
	Point min(bb.min());
	Point max(bb.max());
	Point eyep(view.eyep);
	Vector dir(view.lookat-eyep);
	dir.normalize();
	double d=-Dot(eyep, dir);
	for(int ix=0;ix<2;ix++){
	    for(int iy=0;iy<2;iy++){
		for(int iz=0;iz<2;iz++){
		    Point p(ix?max.x():min.x(),
			    iy?max.y():min.y(),
			    iz?max.z():min.z());
		    double dist=Dot(p, dir)+d;
		    znear=Min(znear, dist);
		    zfar=Max(zfar, dist);
		}
	    }
	}
	if(znear <= 0){
	    if(zfar <= 0){
		// Everything is behind us - it doesn't matter what we do
		znear=1.0;
		zfar=2.0;
	    } else {
		znear=zfar*.001;
	    }
	}
	return 1;
    } else {
	return 0;
    }
}

