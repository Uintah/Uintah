
/*
 *  Raytracer.cc: A raytracer for Salmon
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/Salmon/Raytracer.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Salmon.h>
#include <Classlib/NotFinished.h>
#include <Geom/Group.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/Lighting.h>
#include <Geom/Light.h>
#include <Geometry/Ray.h>
#include <Math/Trig.h>
#include <strstream.h>

const int STRINGSIZE=200;

static Renderer* make_Raytracer()
{
    return new Raytracer;
}

static int query_Raytracer()
{
    return 1;
}

RegisterRenderer Raytracer_renderer("Raytracer", &query_Raytracer,
				    &make_Raytracer);

Raytracer::Raytracer()
{
    max_level=3;
    min_weight=0.01;
    strbuf=new char[STRINGSIZE];
}

Raytracer::~Raytracer()
{
    delete[] strbuf;
}

clString Raytracer::create_window(Roe* roe,
				  const clString& name,
				  const clString& width,
				  const clString& height)
{
    rend=roe->get_renderer("OpenGL");
    if(!rend){
	rend=roe->get_renderer("X11");
	if(!rend)
	    return "";
    }
    xres=rend->xres;
    yres=rend->yres;
    return rend->create_window(roe, name, width, height);
}

void Raytracer::redraw(Salmon* _salmon, Roe* _roe)
{
    salmon=_salmon;
    roe=_roe;
    rend->redraw(salmon, roe);

    WallClockTimer timer;
    timer.start();
    // Find all toplevel objects
    GeomGroup group(0);
    HashTableIter<int, PortInfo*> iter(&salmon->portHash);
    for (iter.first(); iter.ok(); ++iter) {
	HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
	HashTableIter<int, SceneItem*> serIter(serHash);
	for (serIter.first(); serIter.ok(); ++serIter) {
	    SceneItem *si=serIter.get_data();
	    
	    // Look up this object by name and see if it is supposed to be
	    // displayed...
	    ObjTag* vis;
	    if(roe->visible.lookup(si->name, vis)){
		if(vis->visible->get())
		    group.add(si->obj);
	    } 
	}
    }
    bgcolor=roe->bgcolor.get();
    bg_firstonly=1;

    // Compute viewing parameters...
    View view(roe->view.get());
    current_view=&view;
    Vector direction=view.lookat-view.eyep;
    double dist=direction.length();
    Vector v(Cross(direction, view.up));
    if(v.length2() == 0.0){
	// Ambiguous up direction...
	cerr << "Error: ambiguous up direction\n";
	return;
    }
    v.normalize();
    Vector u(Cross(v, direction));
    u.normalize();
    double aspect=double(xres)/double(yres);
    double width=aspect*2.0*dist*Tan(DtoR(view.fov*0.5))/yres;
    u*=width;
    v*=width;

    topmatl=salmon->default_matl;
    Color* scanline=new Color[xres];

    // Preprocess the scene
    topobj=&group;
    topobj->preprocess();
    for(int yc=0;yc<yres;yc++){
	while(salmon->process_event(0)) { /* Nothing... */ }
	if(roe->need_redraw)break;
	   
	if(yc%10==0 && yc>0){
	    ostrstream str(strbuf, STRINGSIZE);
	    str << "updatePerf " << roe->id << " \""
		<< int(double(yc)/double(yres)*100) << "% in "
		<< timer.time() << " seconds\" \"\";update idletasks" << '\0';
	    TCL::execute(str.str());
	}
	int y, nscan;
	inside_out(yres, yc, y, nscan);
	for(int x=0;x<xres;x++){
	    double screenx=x-double(xres)/2.0+0.5;
	    double screeny=y-double(yres)/2.0+0.5;
	    Vector sv(v*screenx);
	    Vector su(u*screeny);
	    Vector raydir(su+sv+direction);
	    raydir.normalize();
	    Ray ray(view.eyep, raydir);
	    scanline[x]=trace_ray(ray, 0, 1.0, 1.0);
	}
	rend->put_scanline(y, xres, scanline);
    }
    timer.stop();
    ostrstream str(strbuf, STRINGSIZE);
    str << "updatePerf " << roe->id << " \"Done in "
	<< timer.time() << " seconds\" \"\";update idletasks" << '\0';
    TCL::execute(str.str());
}

void Raytracer::hide()
{
}

void Raytracer::get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&)
{
    NOT_FINISHED("Raytracer::get_pick");
}

Color Raytracer::trace_ray(const Ray& ray, int level, double weight,
			   double ior)
{
    Hit hits;
    topobj->intersect(ray, topmatl.get_rep(), hits);
    if(hits.hit() > 0){
	return shade(ray, hits, level, weight, ior);
    } else {
	// Hit the background
	if(!bg_firstonly || level==0)
	    return bgcolor;
	else
	    return Color(0,0,0);
    }
}

Color Raytracer::shade(const Ray& ray, const Hit& hit,
		       int level, double weight, double ior)
{
    double nearest=hit.t();
    Point hit_position=ray.origin()+ray.direction()*nearest;

    // Transform to object space
    Point obj_hit_position=hit_position;
    Vector obj_normal=hit.prim()->normal(obj_hit_position, hit);
    // Transform normal back to world space
    Vector normal=obj_normal;
    if(Dot(normal, ray.direction()) > 0){
	normal=-normal;
    }

    // Apply lighting...
    Lighting& l=salmon->lighting;
    int nlights=l.lights.size();
    OcclusionData od(hit.prim(), this, level, current_view);
    Color amblight(0,0,0);
    Color difflight(0,0,0);
    Color speclight(0,0,0);
    MaterialHandle matl(hit.matl());
    double specpow=matl->shininess;
    for(int i=0;i<nlights;i++){
	Color light;
	Vector light_dir;
	l.lights[i]->lintens(od, hit_position, light, light_dir);
	double cos_theta=Dot(light_dir, normal);
	difflight+=light*cos_theta;
	Vector H=(light_dir-ray.direction())/2.;
	H.normalize();
	double cos_alpha=Dot(H, normal);
	if(cos_alpha > 0.0 && specpow > 0.0)
	    speclight+=light*Pow(cos_alpha, specpow);
    }
    Color surfcolor=
	matl->diffuse*difflight
	    + matl->specular*speclight;
    int new_level=level+1;
    if(new_level < max_level){
	// Compute reflected ray
	double incident_angle=-Dot(normal, ray.direction());
	Vector refl_dir=ray.direction()+normal*(2.*incident_angle);
	double refl_weight=matl->reflectivity*weight;
	if(refl_weight > min_weight){
	    Ray refl(hit_position, refl_dir);
	    Color rcolor=trace_ray(refl, new_level, refl_weight, ior);
	    surfcolor+=rcolor*refl_weight;
	}
    }
    return surfcolor;
}

void Raytracer::put_scanline(int, int, Color*, int)
{
    NOT_FINISHED("RayTracer:put_scanline");
}

void Raytracer::inside_out(int n, int a, int& b, int& r)
{
    int m=n;
    r=n;
    b=0;
    int k=1;
    while(k<n){
	if(2*a>=m){
	    if(b==0)
		r=k;
	    b+=k;
	    a-=(m+1)/2;
	    m/=2;
	} else {
	    m=(m+1)/2;
	}
	k*=2;
    }
    if(r>n-b)r=n-b;
}
 
double Raytracer::light_ray(const Point& from, const Point&,
			    const Vector& direction, double dist)
{
    Point f(from+direction*1.e-6);
    Ray ray(f, direction);
    Hit hits;
    topobj->intersect(ray, topmatl.get_rep(), hits);
    if(hits.hit() && hits.t() < dist)
	return 0;
    else
	return 1;
}
