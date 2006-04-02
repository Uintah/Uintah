
#include <Packages/DaveW/Core/Datatypes/CS684/Scene.h>
#include <Core/Containers/String.h>

#include <fstream>
#include <iostream>
using std::cerr;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace DaveW {
using namespace SCIRun;

extern int global_numbounces;
extern Mutex global_bounces_mutex;

Scene::Scene() {
}

Scene::~Scene() {
}

#define Scene_VERSION 1
void Scene::io(Piostream& stream) {
  using DaveW::Pio;
  using SCIRun::Pio;

  int version=stream.begin_class("Scene", Scene_VERSION);
  Pio(stream, lights);
  Pio(stream, obj);
  Pio(stream, numBounces);
  Pio(stream, attenuateFlag);
  if (version >= 2) {
    Pio(stream, mesh);
  } else {
    mesh.resize(0);
  }
  stream.end_class();
}

//a:=(x,y)->2*abs(y-.5)*x+(1-2*abs(y-.5))*(1-E^(-10*x));
//b:=(x,R)->(sign(R-.5)+1)/2*(1-a(1-x,1-R))+(1-sign(R-.5))/2*a(x,R);
//d:=R->((R-.5)*2)^3*20+.5;
//e:=(x,R)->b(x,d(R));

double rescaleReflection(double x, double R) {
  //    double Rminus = R-.5;
  //    double r1 = Rminus*Rminus*Rminus*160+0.5;
  //    double x1, R1;
  //    if (Rminus<0) {x1=x; R1=r1;} else {x1=1-x; R1=1-r1;}
  //    double a = 2*abs(R1-.5)*x + (1-2*abs(R-.5))*(1-exp(-10*x));
  //    if (Rminus<0) return a; else return 1-a;
  return x*R;
}

//void Scene::RT_shade(RTRay&R, const RTHit &hit, int bounce) {
//    RTObject* obj=hit.obj;
//    Array1<double> reflect(R.spectrum.size());
//    double reflC=obj->matl->base->reflectivity;
//    if (bounce<numBounces && (reflC != 0)) {
//	global_bounces_mutex.lock();
//	global_numbounces++;
//	global_bounces_mutex.unlock();
//	RTRay refl(R);		// copy everything, then fix direction
//	refl.dir=obj->BRDF(hit.p, hit.side, R.dir, hit.face);
//	refl.origin=hit.p;
//	refl.energy*=reflC;
//	reflect+=RT_trace(refl, bounce+1, 0);
//    }
//    return reflect*reflC + emit;
//}


//void Scene::RT_trace(RTRay& R, int bounce, int shadowFlag) {
//    RTHit hit;
//    if (findIntersect(R, hit, shadowFlag)) {
//	if (shadowFlag) {
//	    if (hit.t > (R.origin-light[shadowFlag-1].pos).length()) 
//		return(Color(0,0,0));	// hit something after light
//	    else
//		return(Color(1,1,1));	// hit something before light
//	}
//	RT_shade(R, hit, bounce);
//    } else {
//	return(Color(0,0,0));
//    }
//}


// choose a random point on a random light source.
// send a shadow ray from there and see if it hits us.
// if so, add in its direct light to R (using cos(theta), and area
// scaling; as well as distance to the fourth downscaling
void Scene::directDiffuse(RTRay& R, const RTHit &hit, int bounce) {
  //    cerr << "dirDiff - bounce="<<bounce<<"\n";
  if (lights.size()==0) {
    cerr << "No lights in this scene!\n";
    return;
  }
  double didx=drand48();
  int idx=lights[(int)(didx*lights.size())];	// randomly choose a light
  if (obj[idx]->visible==0) return;

  double r1,r2;
  r1=drand48();
  r2=drand48();
  Point p=obj[idx]->getSurfacePoint(r1,r2);
  RTRay lightRay(R);
  RTHit lightHit(hit);
  Vector v(hit.p-p);
  lightHit.t=v.length();
  //    double xxx=lightHit.t;
  lightRay.origin=p;	// light Ray starts on light and aims for us!
  v.normalize();
  lightRay.dir=v;
  if (!findIntersect(lightRay, lightHit, 1)) {
    double surfNormal=Dot(-lightRay.dir, hit.obj->normal(hit.p, 0, 
							 lightRay.dir, 0));
    double lightNormal=Dot(lightRay.dir, obj[idx]->normal(p, 0,
							  -lightRay.dir,
							  0));
    double dist2=lightHit.t*lightHit.t;
    double area=obj[idx]->area();
    double scale=surfNormal*lightNormal*area/(dist2/M_PI);


    int lll, mmm;
    lll=obj[idx]->matl->lidx;
    mmm=hit.obj->matl->midx;
    if (R.pixel) (R.pixel->D[bounce])[lll][mmm]+=scale;
    //	cerr << "Direct - bounce "<<bounce<<" from lightMatl "<<lll<<" to surfMatl "<<mmm<<" += "<<scale<<"\n";


    //	double scale=surfNormal*lightNormal*area/dist2;
    //	cerr << "scale="<<scale<<" ";

    //	vectorAddScale(R.spectrum, obj[idx]->matl->temp_emission, scale);

    vectorAddScale(R.spectrum, obj[idx]->matl->temp_emission, scale*(1-obj[idx]->matl->base->reflectivity));
    //    } else {
    //	if (hit.p.y() != 548.8) {
    //	    cerr << "r1="<<r1<<" r2="<<r2<<"  Light pt="<<p<< "\n";
    //	    cerr << "lHit.p="<<lightHit.p<<"  hit.p="<<hit.p<<"  p="<<p<<"\n";
    //	    cerr << "Hit.t="<<lightHit.t<<"  v.length="<<xxx<<" diff="<<lightHit.t-xxx<<"\n";
    //	}
  }
}

void Scene::trace(RTRay& R, int bounce, RTObject *from_obj) {
  //    if (drand48()>=R.energy) return;
  //    R.energy*=(numBounces/(numBounces+1.));
  //    cerr << "trace - bounce="<<bounce<<"\n";
  RTHit hit;
  if (!findIntersect(R, hit, 0)) {
    //	for (int i=0; i<R.spectrum.size(); i++) R.spectrum[i]=0;
    return;
  }

  // PUT IN INDIRECT (SQUARE) MATRIX ENTRIES HERE!
  // gotta add into the indirect matrix
  if (from_obj && !hit.obj->matl->emitting && bounce>0 && bounce<=numBounces) {
    int mfrom, mto;
    mto=hit.obj->matl->midx;
    mfrom=from_obj->matl->midx;
    //	(R.pixel->S[bounce-1])[mto][mfrom]++;
    //	cerr << "setting S"<<bounce-1<<"["<<mto<<"]["<<mfrom<<"]\n";
  }


  // RM: correlate this ->reflectivity value with ks and kd !!!!

  if (hit.obj->matl->base->reflectivity > hit.obj->mr()) {
    //	cerr << "SPEC ";
    radianceSpecular(R, hit, bounce);
  } else {
    //	cerr << "DIFF ";
    radianceDiffuse(R, hit, bounce);
  }
}

void Scene::radianceDiffuse(RTRay& R, const RTHit& hit, int bounce) {
  // if this was a primary ray and it hit an emitting light source, 
  // add it in and return
  //    cerr << "radDiff - bounce="<<bounce<<"\n";
  if (hit.obj->matl->emitting && (bounce==0)) {


    // RAY MATRIX INFO!!!
    int lll=hit.obj->matl->lidx;
    if (R.pixel) R.pixel->E[lll]++;
    cerr << "Emit - from lightMatl ++\n";


    vectorAddScale(R.spectrum, hit.obj->matl->temp_emission, 1.);
    return;
  }
    
  // if this was a secondary ray and it hit an emitting light source,
  // just ignore it and return (b/c direct lighting will look for this)
  if (hit.obj->matl->emitting) return;
    
  if (bounce<numBounces) {
    // the object we hit was not an emitting light source --
    // add in the contributions from a BRDF-biased reflection ray
    //   and direct light
    RTRay refl(R);		// copy everything, then fix direction
    refl.dir=hit.obj->BRDF(hit.p, hit.side, R.dir, hit.face);
    refl.origin=hit.p;
	
    trace(refl, bounce+1, hit.obj);
  }

  // RayMatrix will be updated in directDiffuse()
  directDiffuse(R, hit, bounce);

  //    double scale=(numBounces+1.)/numBounces;	// inverse prob of this ray
  //                                                // haing been cast.
  //    vectorScaleBy(R.spectrum, scale);
  vectorScaleBy(R.spectrum, hit.obj->matl->temp_diffuse);
}

void bldFrame(Vector &n, Vector &u, Vector &v) {
  Vector Up(0,1,0);
  n.normalize();
  double dott=Dot(Up,n);
  if (dott>.9) {
    Up=Vector(1,0,0);
    dott=Dot(Up,n);
  }
  v=Up-n*dott;
  v.normalize();
  u=Cross(n,v);
}
   
void Scene::radianceSpecular(RTRay& R, const RTHit& hit, int bounce) {

  cerr << "radSpec - bounce="<<bounce<<"\n";
  // find a reflection ray -- see if it's gonna hit a light;
  // if so, add in direct light; otherwise, just pass it on to trace...

  if (bounce>=numBounces) return;

  // add in the contribution from a specular reflection ray
  RTRay refl(R);		// copy everything, then reflect direction
  Vector nor=hit.obj->normal(hit.p, 0, refl.dir, 0);
  refl.dir=nor*-2*Dot(nor,refl.dir)+refl.dir;
  Vector u,v;
  bldFrame(refl.dir, u, v);
  double x=hit.obj->mr()*2-1;
  double y=hit.obj->mr()*2-1;
  //    cerr << "x="<<x<<" y="<<y;
  int shine=hit.obj->matl->base->shininess;
  shine = (shine/2)*2+1;
  //    cerr << " shine="<<shine<<" refl="<<hit.obj->matl->base->reflectivity;
  x=pow(x, shine);
  y=pow(y, shine);
  //    if (x<0) x+=1; else x-=1;
  //    if (y<0) y+=1; else y-=1;
  Vector specDir(refl.dir+u*x+v*y);
  specDir.normalize();
  //    cerr << "refl="<<refl.dir<<"  specDir="<<specDir<<" nor="<<nor<<" x="<<x<<" y="<<y<<" u="<<u<<" v="<<v<<"\n";
  //    cerr << " newx="<<x<<" newy="<<y<<"\n";
  //    cerr << "refl="<<refl.dir<<"  specDir="<<specDir<<"\n  nor="<<nor<<"\n x="<<x<<" y="<<y<<" u="<<u<<" v="<<v<<"\n";
  if (Dot(specDir, nor) < 0) {
    //	cerr << "Going with spec... \n";
    specDir=refl.dir;
  }
  refl.dir=specDir;
  refl.origin=hit.p;
  trace(refl, bounce+1);

  // see if we hit a light source... if so, add it in!
  RTHit h;
  if (!findIntersect(refl, h, 0)) return;
  if (!h.obj->matl->emitting) return;
  if (h.obj->visible == 0) return;
  double surfNormal=Dot(-refl.dir, hit.obj->normal(hit.p, 0, 
						   refl.dir, 0));
  double lightNormal=Dot(refl.dir, h.obj->normal(h.p, 0,
						 -refl.dir,
						 0));
  double dist2=h.t*h.t;
  double area=h.obj->area();
  double scale=surfNormal*lightNormal*area/(dist2/M_PI);


  int lll, mmm;
  lll=h.obj->matl->lidx;
  mmm=hit.obj->matl->midx;
  if (R.pixel) (R.pixel->D[bounce])[lll][mmm]+=scale;
  //    cerr << "Direct (spec) - bounce "<<bounce<<" from lightMatl "<<lll<<" to surfMatl "<<mmm<<" += "<<scale<<"\n";



  //    vectorAddScale(R.spectrum, h.obj->matl->temp_emission, scale);
  vectorAddScale(R.spectrum, h.obj->matl->temp_emission, scale*(h.obj->matl->base->reflectivity));
}

void Scene::setupTempSpectra(double min, double max, int num) {
  // since multiple objects can be pointing to the same materials, and
  // therefore the same Spectra, we destroy them all, and then
  // rebuild them one object at a time -- each time making sure that
  // the material spectra hasn't already been built

  int i;
  for (i=0; i<obj.size(); i++) {
    obj[i]->destroyTempSpectra();
  }
  for (i=0; i<obj.size(); i++) {
    obj[i]->buildTempSpectra(min, max, num);
  }
}

int Scene::findIntersect(RTRay& R, RTHit &hit, int shadowFlag) {

  RTObject* rto=hit.obj;
  if (shadowFlag) {
    int i=0;
    while (i<obj.size()) {
      if (obj[i]->visible && obj[i]->intersect(R,hit) && (rto!=hit.obj))
	return 1;
      i++;
    }
    return 0;
  } else {
    for (int i=0; i<obj.size(); i++) {
      if (obj[i]->visible) obj[i]->intersect(R, hit);
    }		
    return (hit.valid);
  }
}

#if 0
#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Core/Containers/Array1.cc>

static void _dummy_(Piostream& p1, Array1<RTObject*>& p2)
{
  Pio(p1, p2);
}

#endif
#endif
#endif
} // End namespace DaveW

