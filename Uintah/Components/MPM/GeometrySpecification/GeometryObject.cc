
#include <math.h>
#include "GeometryObject.h"
#include "BoxGeometryPiece.h"
#include "SphereGeometryPiece.h"
#include "CylinderGeometryPiece.h"
#include "TriGeometryPiece.h"   
#include "Material.h"
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using Uintah::Interface::DataWarehouseP;
using Uintah::Grid::ParticleVariable;
#include <fstream>
#include <string>
#include <iostream>
using std::cerr;
using std::ifstream;
using std::string;
using std::vector;

using namespace Uintah::Components;

GeometryObject::GeometryObject()
{
}
GeometryObject::~GeometryObject()
{
}

void GeometryObject::setObjInfo(int n,Point lo,Point hi,
				Vector dx,IntVector nppc)
{

  d_object_number=n;
  
  d_xyz = dx;
  d_num_par_per_cell  = nppc;
  

  d_upper_prob_coord = hi;
  d_lower_prob_coord = lo;
}

Point GeometryObject::getObjInfoBoundsLower()
{
  return d_lower_prob_coord;
}

Point GeometryObject::getObjInfoBoundsUpper()
{
  return d_upper_prob_coord;
}

void GeometryObject::setObjInfoBounds(Point lo, Point hi)
{
  d_lower_prob_coord = lo;
  d_upper_prob_coord = hi;    
}

int GeometryObject::getObjInfoNumPieces()
{
  return d_num_pieces;
}

Vector GeometryObject::getObjInfoDx()
{
  return d_xyz;
}

IntVector GeometryObject::getObjInfoNumParticlesPerCell()
{
  return d_num_par_per_cell;
}
   

void GeometryObject::addPieces(ProblemSpecP prob_spec)
{
  int pt,pn,mn,i,m,vf_num;
  double gb[7];
  string stuff;
  Vector init_cond_vel;
  Point origin;
  double radius;
  double length;
  Point lo, up;
  CylinderGeometryPiece::AXIS axis;

  std::string type;
  prob_spec->require("type",type);

  GeometryPiece *geom_piece;

  // NOTE: In the original code, we set the initial conditions, but this
  // hasn't been implemented yet.  Probably shouldn't be here either.

  if (type == "box") {
    Point min,max;
    prob_spec->require("min",min);
    prob_spec->require("max",max);
  
    geom_piece = new BoxGeometryPiece(min,max);

  }

  else if (type == "cylinder") {
    std::string ax;
    prob_spec->require("axis",ax);
    prob_spec->require("origin",origin);
    prob_spec->require("length",length);
    prob_spec->require("radius",radius);
    
    if (ax == "X") axis = CylinderGeometryPiece::X;
    if (ax == "Y") axis = CylinderGeometryPiece::Y;
    if (ax == "Z") axis = CylinderGeometryPiece::Z;

    geom_piece = new CylinderGeometryPiece(axis,origin,length,radius);

  }

  else if (type == "sphere") {
    prob_spec->require("origin",origin);
    prob_spec->require("radius",radius);

    geom_piece = new SphereGeometryPiece(radius,origin);
    
  }

  else if (type == "tri") {

    geom_piece = new TriGeometryPiece;

  }
    
  d_geom_pieces.push_back(geom_piece);
  
}


int GeometryObject::checkShapes(Point part_pos, int &np)
{
  int pp=0,ppold;
  double dlpp;
  
  dlpp=d_particle_spacing.x();
  if(dlpp<d_particle_spacing.y()){
    dlpp=d_particle_spacing.y();
  }
  if(dlpp<d_particle_spacing.z()){
    dlpp=d_particle_spacing.z();
  }
  
  for(int i=1;i<=d_geom_pieces.size();i++){
   
    //  "Positive" space objects go here
  
    if((pp<=0)&&(d_geom_pieces[i]->getPosNeg()>=1.0)){
       pp = d_geom_pieces[i]->checkShapesPositive(part_pos,np,i,d_particle_spacing, ppold);
       d_in_piece = d_geom_pieces[i]->getInPiece();

    }      /* if (pp<=0)           */


/* Negative stuff       */
    if((pp>0)&&(d_geom_pieces[i]->getPosNeg()==0.0)){
      ppold=pp;
      pp = d_geom_pieces[i]->checkShapesNegative(part_pos,np,i,d_particle_spacing, ppold);
    }      /* if (pp>0)            */

  }         /* for */
  
  return pp;

}

void GeometryObject::surface(Point part_pos,int surf[7], int &np)
{
  Point check_point;
  int next=1,last=6,ss;
  /*  Check the candidate points which surround the point just passed
      in.  If any of those points are not also inside the body
      described in SHAPE, the current point is on the surface */

 // Check to the left
  check_point = Point(part_pos.x()-d_particle_spacing.x(),part_pos.y(),part_pos.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  // Check to the right
  check_point = Point(part_pos.x()+d_particle_spacing.x(),part_pos.y(),part_pos.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  // Check below
  check_point = Point(part_pos.x(),part_pos.y()-d_particle_spacing.y(),part_pos.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  // Check above
  check_point = Point(part_pos.x(),part_pos.y()+d_particle_spacing.y(),part_pos.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  // Check behind
  check_point = Point(part_pos.x(),part_pos.y(),part_pos.z()-d_particle_spacing.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }
  
  // Check in front
  check_point = Point(part_pos.x(),part_pos.y(),part_pos.z()+d_particle_spacing.z());
  ss=checkShapes(check_point,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }


 

  return;
}

void GeometryObject::norm(Vector &norm, Point part_pos, int sf[7], 
			  int inPiece, int &np)
{

  Vector dir(0.0,0.0,0.0);
  norm = Vector(0.0,0.0,0.0);
  int small = 1;


 for(int i=1;i<=6;i++) {
    if(sf[i] < small) { 
      small = sf[i]; 
    }
  }
  
  // Not a surface point
  if(small == 1){ 
    return; 
  }		

  d_geom_pieces[inPiece]->computeNorm(norm, part_pos,sf,inPiece,np);


}



void GeometryObject::fillWithParticles(vector<Material *> &materials,
				vector<BoundCond> &BC,
				const Region* region,
				DataWarehouseP& dw)
{

#ifdef WONT_COMPILE_YET

    cerr << "In FillWParticles\n";
    ParticleVariable<Vector> pposition;
    dw->get(pposition, "p.x", region, 0);
    ParticleVariable<double> pvolume;
    dw->get(pvolume, "p.volume", region, 0);
    ParticleVariable<double> pmass;
    dw->get(pmass, "p.mass", region, 0);
    ParticleVariable<Vector> pvel;
    dw->get(pvel, "p.velocity", region, 0);
    ParticleVariable<Vector> pexternalforce;
    dw->get(pexternalforce, "p.externalforce", region, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    dw->get(pconmod, "p.conmod", region, 0);

    int ocount=pposition.getParticleSubset()->numParticles();
    for(int pass=0;pass<2;pass++){
	int count=ocount;
  int i,j,k,ix,jy,kz,n=0,iva;
  double xp[3];           // particle position
  Vector norm;           // surface normal
  int inObj=0,surflag[7],np;
  int mat_type,temp,mat_num,vf_num;
  double density,mass,volume;
  double icv[4];
  double mp[10];
  
  dxpp=dXYZ[1]/((double) numParPCell[1]);      // determine particle spacing
  dypp=dXYZ[2]/((double) numParPCell[2]);      // in x, y, and z
  dzpp=dXYZ[3]/((double) numParPCell[3]);
  
  volume = dxpp*dypp*dzpp;			// Particle volume
  
  kz=0,k=1;                                       // index of potential particles in z
  xp[2] = dzpp/2.0 + probBounds[5];               // starting point in z direction
  while(xp[2]<probBounds[6]) {
    xp[1] = dypp/2.0 + probBounds[3];       // starting point in y direction
    jy=0,j=1;                               // index of potential particles in y
    while(xp[1]<probBounds[4]) {
      xp[0] = dxpp/2.0 + probBounds[1]; // starting point in x direction
      ix=0,i=1; // index of potential particles in x
      while(xp[0]<probBounds[2]) {
	//                       determine if particle is within the
	//                       object
	inObj=checkShapes(xp,np);
	if(inObj>=1){
	  temp = inPiece;
  //  check to see if current point is a
  //  surface point
	  surface(xp,surflag,np);
	  inPiece = temp;
  //  Norm will find the surface normal at the current particle position.
  //  If the particle is not on the surface, (0,0,0) is returned.
          norm(norm,xp,surflag,inPiece,np);
	  mat_num =d_geom_pieces[inPiece].getMaterialNum();
	  vf_num  =d_geom_pieces[inPiece].getVFNum(); 
	  density = materials[mat_num-1]->getDensity();
	  mat_type= materials[mat_num-1]->getMaterialType();
	  mass = density*volume;	// Particle mass

	  for(iva=1;iva<=3;iva++){
	    icv[iva]=d_geom_pieces[inPiece].getInitVel(iva);
	  }

	  // Determine if the particle has a boundary condition that needs 
	  // to be associated with it.

	  double bcf[3];
	  bcf[0] = bcf[1] = bcf[2] = 0;
	  
	  for (unsigned int i = 0; i<BC.size(); i++) {
	    if (BC[i].getBCObj() == objectNumber ) {
	      if (BC[i].getBCPiece() == inPiece ) {
		for (int j = 1; j<=6; j++) {
		  if (BC[i].getBCSurf() == surflag[j] ) {
		    bcf[0] += BC[i].getBCForce(1);
		    bcf[1] += BC[i].getBCForce(2);
		    bcf[2] += BC[i].getBCForce(3);
		    break;
		  }
		}
	      }
	    }
	  }

	  materials[mat_num-1]->getMatProps(mp);
	  double vel[3], normal[3];
	  double t=300.0,K=100.0,c_p=2.0;
	  vel[0] = icv[1];
	  vel[1] = icv[2];
	  vel[2] = icv[3];
	  normal[0] = norm.x();
	  normal[1] = norm.y();
	  normal[2] = norm.z();

	  n++;                          // increment particle number

          int srf=0;
          for (int j = 1; j<=6; j++) {
	   if((surflag[j] == -91) && (vf_num == 1) || (surflag[j] == -9) && (vf_num == 2)){
//	   if((surflag[j] == -1) && (vf_num == 1) || (surflag[j] == -2) && (vf_num == 2)){
		srf++;
	   }
          }

#if 0
	  Particle tmp = Particle(xp, volume, mass, vf_num,
				  mat_type, mp, vel, bcf, normal,
				  t,K,c_p,n,srf);

	  particles.appendItem(tmp);
#else
	  if(pass == 0){
	      count++;
	  } else {
	      Vector xpos(xp[0], xp[1], xp[2]);
	      pposition[count]=xpos;
	      pvolume[count]=volume;
	      pmass[count]=mass;
	      Vector velocity(vel[0], vel[1], vel[2]);
	      pvel[count]=velocity;
	      Vector externalforce(bcf[0], bcf[1], bcf[1]);
	      pexternalforce[count]=externalforce;
	      CompMooneyRivlin cm(100000.0, 20000.0, 70000.0, 1320000.0);
	      pconmod[count]=cm;
	      count++;
	  }
#endif
	  
	}
	xp[0]=xp[0] + dxpp;            // increment next potential
	ix++;                          // particle position and
	if( (ix % numParPCell[1]) ==0){        // grid index in x (y & z below)
	  i++;
	}
      }
      xp[1]=xp[1] + dypp;
      jy++;
      if( (jy % numParPCell[2])==0){
	j++;
      }
    }
    xp[2]=xp[2] + dzpp;
    kz++;
    if( (kz % numParPCell[3])==0){
      k++;
    }
  }
    if(pass == 0){
	cerr << "Resizing to: " << count << "\n";
	// Ugly, I know - steve
	pposition.getParticleSubset()->resize(count);
	for(int i=0;i<count;i++)
	    pposition.getParticleSubset()->set(i, i);
	pposition.resize(count);
	pvolume.resize(count);
	pmass.resize(count);
	pvel.resize(count);
	pexternalforce.resize(count);
	pconmod.resize(count);
    }
    }
    dw->put(pposition, "p.x", region, 0);
    dw->put(pvolume, "p.volume", region, 0);
    dw->put(pmass, "p.mass", region, 0);
    dw->put(pvel, "p.velocity", region, 0);
    dw->put(pexternalforce, "p.externalforce", region, 0);
    dw->put(pconmod, "p.conmod", region, 0);

#endif
}


  
// $Log$
// Revision 1.4  2000/04/19 05:26:07  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.2  2000/03/20 17:17:14  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:36:05  jas
// Readded geometry specification source files.
//
// Revision 1.2  2000/02/27 07:48:41  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
// Revision 1.1  2000/02/24 06:11:56  sparker
// Imported homebrew code
//
// Revision 1.4  2000/01/27 06:40:30  sparker
// Working, semi-optimized serial version
//
// Revision 1.3  2000/01/26 01:08:06  sparker
// Things work now
//
// Revision 1.2  2000/01/25 18:28:44  sparker
// Now links and kinda runs
//
// Revision 1.1  2000/01/24 22:48:51  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/09/04 23:00:31  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.7  1999/09/03 23:02:17  guilkey
// Fixed some old errors with GeometryObject.cc in CheckShapes, specifically
// with "negative" cylindrical pieces.
//
// Revision 1.5  1999/08/31 15:52:25  jas
// Added functions to set the geometry bound for a geometry object.  This will
// allow a processor to query how big a space it has and then have the
// geometry object fill itself with only the particles that actually fill
// the space controlled by the processor.
//
// Revision 1.4  1999/08/25 17:52:48  guilkey
// Added functionality to allow for an "bond strength" momentum exchange
// routine to implemented.  Most of these functions are currently commented
// out in the files that I'm checking in.  These include applyExternalForce,
// exchangeMomentumBond(2),  and computeTractions.  Also, the printParticle
// function has been updated so that the grid values are now also printed out,
// again in tecplot format.  These files should also work with the SCIRun
// viz tools, although this hasn't been tested.
//
// Revision 1.3  1999/08/18 19:23:09  jas
// Now can return information about the problem specification and object
// bounds.  This is necessary for the creation of particles on the processor
// that they reside.
//
// Revision 1.2  1999/06/18 05:44:52  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.10  1999/05/31 19:36:14  cgl
// Work in stand-alone version of MPM:
//
// - Added materials_dat.cc in src/constitutive_model to generate the
//   materials.dat file for preMPM.
// - Eliminated references to ConstitutiveModel in Grid.cc and GeometryObject.cc
//   Now only Particle and Material know about ConstitutiveModel.
// - Added reads/writes of Particle start and restart information as member
//   functions of Particle
// - "part.pos" now has identicle format to the restart files.
//   mpm.cc modified to take advantage of this.
//
// Revision 1.9  1999/05/30 02:10:49  cgl
// The stand-alone version of ConstitutiveModel and derived classes
// are now more isolated from the rest of the code.  A new class
// ConstitutiveModelFactory has been added to handle all of the
// switching on model type.  Between the ConstitutiveModelFactory
// class functions and a couple of new virtual functions in the
// ConstitutiveModel class, new models can be added without any
// source modifications to any classes outside of the constitutive_model
// directory.  See csafe/Uintah/src/CD/src/constitutive_model/HOWTOADDANEWMODEL
// for updated details on how to add a new model.
//
// --cgl
//
// Revision 1.8  1999/05/26 20:30:50  guilkey
// Added the capability for determining which particles are on the surface
// and what the surface normal of those particles is.  This information
// is output to part.pos.
//
// Revision 1.7  1999/05/24 21:06:00  guilkey
// Added a new constitutive model, and tried to make it easier for
// others to add new models in the future.
//
// Revision 1.6  1999/02/25 22:32:32  guilkey
// Fixed some functions associated with the HyperElastic constitutive model.
//
// Revision 1.5  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.4  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
