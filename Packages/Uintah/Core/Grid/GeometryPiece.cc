
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>

#include <math.h>
#include <fstream>
#include <string>
#include <iostream>

using std::cerr;
using std::ifstream;
using std::string;
using std::vector;

using namespace SCIRun;
using namespace Uintah;

GeometryPiece::GeometryPiece()
{
}
GeometryPiece::~GeometryPiece()
{
}

#if 0
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
  
    geom_piece = scinew BoxGeometryPiece(min,max);

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

    geom_piece = scinew CylinderGeometryPiece(axis,origin,length,radius);

  }

  else if (type == "sphere") {
    prob_spec->require("origin",origin);
    prob_spec->require("radius",radius);

    geom_piece = scinew SphereGeometryPiece(radius,origin);
    
  }

  else if (type == "tri") {

    geom_piece = scinew TriGeometryPiece;

  }
    
  d_geom_pieces.push_back(geom_piece);
  
}
#endif

#ifdef FUTURE
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
#endif


#ifdef WONT_COMPILE_YET

void GeometryObject::fillWithParticles(vector<Material *> &materials,
				vector<BoundCond> &BC,
				const Patch* patch,
				DataWarehouseP& dw)
{


    cerr << "In FillWParticles\n";
    ParticleVariable<Vector> pposition;
    dw->get(pposition, "p.x", patch, 0);
    ParticleVariable<double> pvolume;
    dw->get(pvolume, "p.volume", patch, 0);
    ParticleVariable<double> pmass;
    dw->get(pmass, "p.mass", patch, 0);
    ParticleVariable<Vector> pvel;
    dw->get(pvel, "p.velocity", patch, 0);
    ParticleVariable<Vector> pexternalforce;
    dw->get(pexternalforce, "p.externalforce", patch, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    dw->get(pconmod, "p.conmod", patch, 0);

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
    dw->put(pposition, "p.x", patch, 0);
    dw->put(pvolume, "p.volume", patch, 0);
    dw->put(pmass, "p.mass", patch, 0);
    dw->put(pvel, "p.velocity", patch, 0);
    dw->put(pexternalforce, "p.externalforce", patch, 0);
    dw->put(pconmod, "p.conmod", patch, 0);

}

#endif
