
#include <math.h>
#include "GeometryObject.h"
#include "Material.h"
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/DataWarehouse.h>
#include "CompMooneyRivlin.h"
#include <SCICore/Geometry/Vector.h>
using SCICore::Geometry::Vector;
#include <fstream>
#include <string>
#include <iostream>
using std::cerr;
using std::ifstream;
using std::string;
using std::vector;

GeomObject::GeomObject()
{
}
GeomObject::~GeomObject()
{
}

void GeomObject::setObjInfo(int n, double bds[7],int np,double dx[4],int nppc[4])
{

  objectNumber=n;
  numPieces = np;
  for(int i=1;i<=3;i++){
    dXYZ[i]         = dx[i];
    numParPCell[i]  = nppc[i];
    probBounds[i]   = bds[i];
    probBounds[i+3] = bds[i+3];
  }
}

double * GeomObject::getObjInfoBounds()
{
	return probBounds;
}

void GeomObject::setObjInfoBounds(double bds[7])
{
      for (int i = 1; i<=6; i++) {
	  probBounds[i] = bds[i];
      }	
}

int GeomObject::getObjInfoNumPieces()
{
 	return numPieces;
}

double * GeomObject::getObjInfoDx()
{
	return dXYZ;
}

int * GeomObject::getObjInfoNumParticlesPerCell()
{
	return numParPCell;
}
   

void GeomObject::addPieces(ifstream &filename)
{
  int pt,pn,mn,i,m,vf_num;
  double gb[7];
  string stuff;
  double icv[4];

  for(i=1;i<=numPieces;i++){
    filename >> stuff;
    //cout << stuff << endl;
    filename >> pt >> pn >> mn >> vf_num;
    gP[i].setPieceType(pt);
    gP[i].setPosNeg(pn);
    gP[i].setMaterialNum(mn);
    gP[i].setVelFieldNum(vf_num);
    
    if(pt==1){              // Box
      filename >> stuff;;
      //cout << stuff << endl;
      filename >> gb[1]>>gb[2]>>gb[3]>>gb[4]>>gb[5]>>gb[6];
      gP[i].setGeomBounds(gb);
      filename >> stuff;
      //cout << stuff << endl;
      filename >> icv[1]>>icv[2]>>icv[3];
      gP[i].setInitialConditions(icv);
    }
    else if(pt==2){         // Cylinder
      filename >> stuff;
      //cout << stuff << endl;
      filename >> gb[1]>>gb[2]>>gb[3]>>gb[4]>>gb[5]>>gb[6];
      gP[i].setGeomBounds(gb);
      filename >> stuff;
      //cout << stuff << endl;
      filename >> icv[1]>>icv[2]>>icv[3];
      gP[i].setInitialConditions(icv);
    }
    else if(pt==3){         // Sphere
      filename >> stuff;
      //cout << stuff << endl;
      filename >>gb[1]>>gb[2]>>gb[3]>>gb[4];
      gP[i].setGeomBounds(gb);
      filename >> stuff;
      //cout << stuff << endl;
      filename >>icv[1]>>icv[2]>> icv[3];
      gP[i].setInitialConditions(icv);
    }
    else if(pt==4){         // Planes
      for(m=i;m<i+pn;m++){
	filename >> stuff;
	//cout << stuff << endl;
	filename >>gb[1]>>gb[2]>>gb[3]>>gb[4];
	gP[m].setGeomBounds(gb);
      }
      filename >> stuff;
      //cout << stuff << endl;
      filename >> icv[1] >> icv[2] >> icv[3];
      gP[i].setInitialConditions(icv);
      i=i+pn;
    }
  }
  
}
int GeomObject::CheckShapes(double x[3], int &np)
{
  int i,j,pp=0,ppold,numplanes,m;
  double xlow,ylow,zlow;
  double xhigh,yhigh,zhigh;
  double rsqr,rdpsqr,dlpp;
  double a,b,c,d,pln;
  double gb[7];
  
  dlpp=dxpp;
  if(dlpp<dypp){
    dlpp=dypp;
  }
  if(dlpp<dzpp){
    dlpp=dzpp;
  }
  
  for(i=1;i<=numPieces;i++){
    for(j=1;j<=6;j++){ gb[j]=gP[i].getGeomBounds(j); }
    if((pp<=0)&&(gP[i].getPosNeg()>=1.0)){ //  "Positive" space objects go here
      if(gP[i].getPieceType()==3.0){                /*sphere*/
	if(Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) <= Sqr(gb[1]))
	  {
	    pp = 3;
	    inPiece=i;
	  }
      }
      else if(gP[i].getPieceType()==2.0){              /*cylinder*/
	rsqr=Sqr(gb[6]);
	rdpsqr=Sqr(gb[6]+dlpp);
	if(gb[1]==1.0){
	  if(Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) <= rsqr
	     && gb[2]-dxpp-x[0] <= 0.0
	     && gb[2]+gb[5]+dxpp-x[0] >= 0.0){
	    pp = 2;
	    inPiece=i;
	    if((gb[2]-x[0] <= dxpp)&&(gb[2]-x[0]>0.0)){
	      pp = -7;
	    }
	    else if((gb[2]+gb[5]-x[0]>=-dxpp)&&
		    (gb[2]+gb[5]-x[0] < 0.0) ){
	      pp = -8;
	    }
	  }
	  else if(Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) > rsqr
		  && Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) <= rdpsqr
		  && x[0] >= gb[2] 
		  && x[0] < gb[2]+gb[5])
	    {
	      pp = -9;
	    }
	}
	
	else if(gb[1]==2.0){
	  if(Sqr(x[0]-gb[2])+Sqr(x[2]-gb[4]) <= rsqr
	     && gb[3]-dypp-x[1] <= 0.0
	     && gb[3]+gb[5]+dypp-x[1] >= 0.0){
	    pp = 2;
	    inPiece=i;
	    if((gb[3]-x[1] <= dypp)&&(gb[3]-x[1]>0.0)){
	      pp = -7;
	    }
	    else if((gb[3]+gb[5]-x[1]>=-dypp)&&
		    (gb[3]+gb[5]-x[1] < 0.0) ){
	      pp = -8;
	    }
	  }
	  else if(Sqr(x[0]-gb[2])+Sqr(x[2]-gb[4]) > rsqr
		  && Sqr(x[0]-gb[2])+Sqr(x[2]-gb[4]) <= rdpsqr
		  && x[1] >= gb[3]
		  && x[1] < gb[3]+gb[5])
	    {
	      pp = -9;
	    }
	}
	
	else if(gb[1]==3.0){
	  if(Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3]) <= rsqr
	     && gb[4]-dzpp-x[2] <= 0.0
	     && gb[4]+gb[5]+dzpp-x[2] >= 0.0){
	      pp = 2;
	      inPiece=i;
	    if((gb[4]-x[2] <= dzpp)&&(gb[4]-x[2]>0.0)){
	      pp = -7;
	    }
	    else if((gb[4]+gb[5]-x[2]>=-dzpp)&&
		    (gb[4]+gb[5]-x[2] < 0.0) ){
	      pp = -8;
	    }
	  }
	  else if(Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3]) > rsqr
		  && Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3]) <= rdpsqr
		  && x[2] >= gb[4]
		  && x[2] < gb[4]+gb[5])
	    {
	      pp = -9;
	    }
	}
      }
      else if(gP[i].getPieceType()==1.0){                   /*hexahedron*/
	xlow =gb[1];
	xhigh=gb[2];
	ylow =gb[3];
	yhigh=gb[4];
	zlow =gb[5];
	zhigh=gb[6];
	if     ((x[0]-xlow>=0.0) && (xhigh-x[0]>=0.0) &&
		(x[1]-ylow>=0.0) && (yhigh-x[1]>=0.0) &&
		(x[2]-zlow>=0.0) && (zhigh-x[2]>=0.0) ){
	  pp = 1;
//	  pp = 3; 
	  inPiece=i;
	} 
	else if(xlow-x[0] <= dxpp && 
		xlow-x[0] > 0.0){
	  pp= -1;
	}
	else if(xhigh-x[0] >= -dxpp &&
		xhigh-x[0] < 0.0){
	  pp= -2;
	}
	else if(ylow-x[1] <= dypp &&
		ylow-x[1] > 0.0){
	  pp= -3;
	}
	else if(yhigh-x[1] >= -dypp &&
		yhigh-x[1] < 0.0){
	  pp= -4;
	}
	else if(zlow-x[2] <= dzpp &&
		zlow-x[2] > 0.0){
	  pp= -5;
	}
	else if(zhigh-x[2] >= -dzpp &&
		zhigh-x[2] < 0.0){
	  pp= -6;
	}
      }
      else if(gP[i].getPieceType()==4.0){
	numplanes=gP[i].getPosNeg();
	for(m=i;m<i+numplanes;m++){
	  a=gP[m].getGeomBounds(1);
	  b=gP[m].getGeomBounds(2);
	  c=gP[m].getGeomBounds(3);
	  d=gP[m].getGeomBounds(4);
	  pln=a*x[0]+b*x[1]+c*x[2]+d;
	  if(pln>=0.0){
	    pp=4;
	    inPiece=i;
	  }
	  else{
	    pp=-100*m;
	    m=i+numplanes;
	  }
	}
	i=i+numplanes;
      }
    }      /* if (pp<=0)           */
    
/* Negative stuff       */
    if((pp>0)&&(gP[i].getPosNeg()==0.0)){
      ppold=pp;
      if(gP[i].getPieceType()==3.0){                   /*sphere*/
	if(Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) < Sqr(gb[1])) {
	  pp = -30;
	  np = i;
	}
      }
      else if(gP[i].getPieceType()==2.0){              /*cylinder*/
	if(gb[1]==1.0){
	  if(Sqr(x[1]-gb[3])+Sqr(x[2]-gb[4]) < Sqr(gb[6])) {
	    pp = -91;
	    np = i;
	    if((x[0]<gb[2])||(x[0]>(gb[2]+gb[5]))){
	      pp = ppold;
	    }
	    else if((x[0]-gb[2] <= dxpp) &&
		    (x[0]-gb[2] > 0.0 )  &&
		    (pp != -91)){
	      pp = -71;
	    }
	    else if((x[0]-(gb[2]+gb[5]) >= -dxpp) &&
		    (x[0]-(gb[2]+gb[5]) < 0.0 )   &&
		    (pp != -91)){
	      pp = -81;
	    }
	  }
	}
	else if(gb[1]==2.0){
	  if(Sqr(x[0]-gb[2])+Sqr(x[2]-gb[4]) < Sqr(gb[6])) {
	    pp = -91;
	    np = i;
	    if((x[1]<gb[3])||(x[1]>(gb[3]+gb[5]))){
	      pp = ppold;
	    }
	    else if((x[1]-gb[3] <= dypp) &&
		    (x[1]-gb[3] > 0.0 )  &&
		    (pp != -91)){
	      pp = -71;
	    }
	    else if((x[1]-(gb[3]+gb[5]) >= -dypp) &&
		    (x[1]-(gb[3]+gb[5]) < 0.0 )  &&
		    (pp != -91)){
	      pp = -81;
	    }
	  }
	}
	else if(gb[1]==3.0){
	  if(Sqr(x[0]-gb[2])+Sqr(x[1]-gb[3]) < Sqr(gb[6])) {
	    pp = -91;
	    np = i;
	    if((x[2]<gb[4])||(x[2]>(gb[4]+gb[5]))){
	      pp = ppold;
	    }
	    else if((x[2]-gb[4] <= dzpp) &&
		    (x[2]-gb[4] > 0.0 )  &&
		    (pp != -91)){
	      pp = -71;
	    }
	    else if((x[2]-(gb[4]+gb[5]) >= -dzpp) &&
		    (x[2]-(gb[4]+gb[5]) < 0.0 )  &&
		    (pp != -91)){
	      pp = -81;
	    }
	  }
	}
      }
      else if(gP[i].getPieceType()==1.0){                   /*hexahedron*/
	xlow =gb[1];
	xhigh=gb[2];
	ylow =gb[3];
	yhigh=gb[4];
	zlow =gb[5];
	zhigh=gb[6];
	if((x[0]-xlow>0.0) && (xhigh-x[0]>0.0) &&
	   (x[1]-ylow>0.0) && (yhigh-x[1]>0.0) &&
	   (x[2]-zlow>0.0) && (zhigh-x[2]>0.0) ){
	  pp = -10;
	  np = i;
	}
	
	if(xlow-x[0] >= -dxpp &&
	   xlow-x[0] < 0.0   &&
	   x[1]-ylow>0.0 && yhigh-x[1]>0.0 &&
	   x[2]-zlow>0.0 && zhigh-x[2]>0.0 ){
	  pp= -11;
	  np = i;
	}
	if(xhigh-x[0] <= dxpp &&
	   xhigh-x[0] > 0.0   &&
	   x[1]-ylow>0.0 && yhigh-x[1]>0.0 &&
	   x[2]-zlow>0.0 && zhigh-x[2]>0.0 ){
	  pp= -21;
	  np = i;
	}
	if(ylow-x[1] >= -dypp &&
	   ylow-x[1] < 0.0    &&
	   x[0]-xlow>0.0 && xhigh-x[0]>0.0 &&
	   x[2]-zlow>0.0 && zhigh-x[2]>0.0 ){
	  pp= -31;
	  np = i;
	}
	if(yhigh-x[1] <= dypp &&
	   yhigh-x[1] > 0.0  &&
	   x[0]-xlow>0.0 && xhigh-x[0]>0.0 &&
	   x[2]-zlow>0.0 && zhigh-x[2]>0.0 ){
	  pp= -41;
	  np = i;
	}
	if(zlow-x[2] >= -dzpp &&
	   zlow-x[2] < 0.0    &&
	   x[0]-xlow>0.0 && xhigh-x[0]>0.0 &&
	   x[1]-ylow>0.0 && yhigh-x[1]>0.0){
	  pp= -51;
	  np = i;
	}
	if(zhigh-x[2] <= dzpp &&
	   zhigh-x[2] > 0.0   &&
	   x[0]-xlow>0.0 && xhigh-x[0]>0.0 &&
	   x[1]-ylow>0.0 && yhigh-x[1]>0.0){
	  pp= -61;
	  np = i;
	}
	
	
      }     /* if (which shape)     */
    }      /* if (pp>0)            */
  }         /* for */
  
  return pp;
}

void GeomObject::Surface(double x[3],int surf[7], int &np)
{
  double cp[3];
  int next=1,last=6,ss;
  /*  Check the candidate points which surround the point just passed
      in.  If any of those points are not also inside the body
      described in SHAPE, the current point is on the surface */

  cp[0]=x[0]-dxpp;        /* Check to the left */
  cp[1]=x[1];
  cp[2]=x[2];
  ss=CheckShapes(cp,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  cp[0]=x[0]+dxpp;        /* Check to the right */
  cp[1]=x[1];
  cp[2]=x[2];
  ss=CheckShapes(cp,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  cp[0]=x[0];
  cp[1]=x[1]-dypp;        /* Check below */
  cp[2]=x[2];
  ss=CheckShapes(cp,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  cp[0]=x[0];
  cp[1]=x[1]+dypp;        /* Check above  */
  cp[2]=x[2];
  ss=CheckShapes(cp,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }

  cp[0]=x[0];
  cp[1]=x[1];
  cp[2]=x[2]-dzpp;        /* Check behind */
  ss=CheckShapes(cp,np);
  if(ss<1){
    surf[next]=ss;
    next=next+1;
  }
  else{
    surf[last]=ss;
    last=last-1;
  }
  
  cp[0]=x[0];
  cp[1]=x[1];
  cp[2]=x[2]+dzpp;        /* Check in front */
  ss=CheckShapes(cp,np);
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

void GeomObject::Norm(Vector &norm, double x[3], int sf[7], int inPiece, int &np)
{

  Vector dir(0.0,0.0,0.0);
  double gb[7],len;
  norm = Vector(0.0,0.0,0.0);
  int small = 1;

  for(int i=1;i<=6;i++){
	if(sf[i] < small){ small = sf[i]; }
  }
  if(small == 1){ return; }		// Not a surface point


  for(int j=1;j<=6;j++){ gb[j]=gP[inPiece].getGeomBounds(j); }
  int ptype = gP[inPiece].getPieceType();

  if(ptype == 1){			// hexahedron
	for(int i=1;i<=6;i++){
	  if(sf[i]==(-1)){		// low x
		dir = Vector(-1.0,0.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-2)){		// high x
		dir = Vector(1.0,0.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-3)){		// low y
		dir = Vector(0.0,-1.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-4)){		// high y
		dir = Vector(0.0,1.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-5)){		// low z
		dir = Vector(0.0,0.0,-1.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-6)){		// high z
		dir = Vector(0.0,0.0,1.0);
		norm+=dir;
	  }
	}
  }
  else if(ptype == 2){			// cylinder
     if(gb[1]==1.0){			// x-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-7)){		// low x
		dir = Vector(-1.0,0.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-8)){		// high x
		dir = Vector(1.0,0.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-9)){		// curved surface
		dir = Vector(0.0,2.0*(x[1]-gb[3]),2.0*(x[2]-gb[4]));
		dir.normalize();
		norm+=dir;
          }
	}
     }
     else if(gb[1]==2.0){		// y-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-7)){		// low y
		dir = Vector(0.0,-1.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-8)){		// high y
		dir = Vector(0.0,1.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-9)){		// curved surface
		dir = Vector(2.0*(x[0]-gb[2]),0.0,2.0*(x[2]-gb[4]));
		dir.normalize();
		norm+=dir;
          }
	}
     }
     else if(gb[1]==3.0){		// z-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-7)){		// low z
		dir = Vector(0.0,0.0,-1.0);
		norm+=dir;
          }
          if(sf[i]==(-8)){		// high z
		dir = Vector(0.0,0.0,1.0);
		norm+=dir;
          }
          if(sf[i]==(-9)){		// curved surface
		dir = Vector(2.0*(x[0]-gb[2]),2.0*(x[1]-gb[3]),0.0);
		dir.normalize();
		norm+=dir;
          }
	}
     }
  }
  else if(ptype == 3){
	for(int i=1;i<=6;i++){
	  if(sf[i]==(0)){		// sphere's surface
		dir = Vector(2.0*(x[0]-gb[1]),2.0*(x[1]-gb[2]),2.0*(x[2]-gb[3]));
		dir.normalize();
		norm+=dir;
	  }
	}
  }

  if(small < -10){	// The point is on the surface of a "negative" object.
    // Get the geometry information for the negative object so we can determine
    // a surface normal.
    for(int j=1;j<=6;j++){ gb[j]=gP[np].getGeomBounds(j); }
     int ptype = gP[np].getPieceType();

     if(ptype == 1){			// hexahedron
	for(int i=1;i<=6;i++){
	  if(sf[i]==(-11)){			// low x
		dir = Vector(1.0,0.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-21)){		// high x
		dir = Vector(-1.0,0.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-31)){		// low y
		dir = Vector(0.0,1.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-41)){		// high y
		dir = Vector(0.0,-1.0,0.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-51)){		// low z
		dir = Vector(0.0,0.0,1.0);
		norm+=dir;
	  }
	  else if(sf[i]==(-61)){		// high z
		dir = Vector (0.0,0.0,-1.0);
		norm+=dir;
	  }
	}
     }
     else if(ptype == 2){	// cylinder
      if(gb[1]==1.0){			// x-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-71)){		// low x
		dir = Vector(1.0,0.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-81)){		// high x
		dir = Vector(-1.0,0.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-91)){		// curved surface
		dir = Vector(0.0,-2.0*(x[1]-gb[3]),-2.0*(x[2]-gb[4]));
		dir.normalize();
		norm+=dir;
          }
	}
      }
      else if(gb[1]==2.0){		// y-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-71)){		// low y
		dir = Vector(0.0,1.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-81)){		// high y
		dir = Vector(0.0,-1.0,0.0);
		norm+=dir;
          }
          if(sf[i]==(-91)){		// curved surface
		dir = Vector(-2.0*(x[0]-gb[2]),0.0,-2.0*(x[2]-gb[4]));
		dir.normalize();
		norm+=dir;
          }
	}
      }
      else if(gb[1]==3.0){		// z-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-71)){		// low z
		dir = Vector(0.0,0.0,1.0);
		norm+=dir;
          }
          if(sf[i]==(-81)){		// high z
		dir = Vector(0.0,0.0,-1.0);
		norm+=dir;
          }
          if(sf[i]==(-91)){		// curved surface
		dir = Vector(-2.0*(x[0]-gb[2]),-2.0*(x[1]-gb[3]),0.0);
		dir.normalize();
		norm+=dir;
          }
	}
     }
    }
    else if(ptype == 3){
	for(int i=1;i<=6;i++){
	  if(sf[i]==(-30)){		// sphere's surface
		dir = Vector(-2.0*(x[0]-gb[1]),-2.0*(x[1]-gb[2]),-2.0*(x[2]-gb[3]));
		dir.normalize();
		norm+=dir;
	  }
	}
    }

  }

  // Normalize the surface normal vector
  len = norm.length();
  if(len > 0.0){ norm*=1./len; }

  return;
}

void GeomObject::FillWParticles(vector<Material *> materials,
				vector<BoundCond> BC,
				const Region* region,
				DataWarehouseP& dw)
{
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
	inObj=CheckShapes(xp,np);
	if(inObj>=1){
	  temp = inPiece;
  //  check to see if current point is a
  //  surface point
	  Surface(xp,surflag,np);
	  inPiece = temp;
  //  Norm will find the surface normal at the current particle position.
  //  If the particle is not on the surface, (0,0,0) is returned.
          Norm(norm,xp,surflag,inPiece,np);
	  mat_num =gP[inPiece].getMaterialNum();
	  vf_num  =gP[inPiece].getVFNum(); 
	  density = materials[mat_num-1]->getDensity();
	  mat_type= materials[mat_num-1]->getMaterialType();
	  mass = density*volume;	// Particle mass

	  for(iva=1;iva<=3;iva++){
	    icv[iva]=gP[inPiece].getInitVel(iva);
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
}
  
// $Log$
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
