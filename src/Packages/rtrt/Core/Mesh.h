#ifndef MESH_H
#define MESH_H

#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Util.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Util.h>

#define MAXITER 7
#define MESH_ROOT_TOL 1E-3

namespace rtrt {
class Mesh;
}
namespace SCIRun {
void Pio(Piostream&, rtrt::Mesh*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Dot;
using SCIRun::Cross;

class Mesh : public SCIRun::Persistent {
public:
  Mesh (int, int);
  ~Mesh();
  Mesh *Copy();
  Mesh() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Mesh*&);

  void Out(int, int);
  void Print();
  Point **getPts(Vector &, Vector &, Vector &);

  inline int get_scratchsize() {
    return (int)(msize*nsize*sizeof(Point));
  }

  inline void get_scratch(double* &thetau, double* &thetav,
			  double* &dthetau, double* &dthetav,
			  PerProcessorContext *ppc)
  {
    double *scratch = (double *)ppc->getscratch(2*msize+2*nsize);
    
    thetau = scratch;
    thetav = &scratch[nsize];
    dthetau = &scratch[nsize+msize];
    dthetav = &scratch[nsize+msize+nsize];
  }

  inline void SetPoint (Point p, int i, int j)
  {
    mesh[i][j] = p;
  }

  inline void RowSplit(int row, Mesh *newm, double w)
  {
    newm->mesh[row][nsize-1] = mesh[row][nsize-1];

    for (int j=1; j<nsize; j++) {
      for (int k=nsize-1; k>=j; k--)
	//mesh[row][k].blend(mesh[row][k-1],mesh[row][k],w);
	mesh[row][k] = SCIRun::Interpolate(mesh[row][k-1], mesh[row][k], w);
      newm->mesh[row][nsize-1-j] = mesh[row][nsize-1];
    }
  }

  inline void ColSplit(int col, Mesh *newm, double w)
  {
    newm->mesh[msize-1][col] = mesh[msize-1][col];

    for (int j=1; j<msize; j++) {
      for (int k=msize-1; k>=j; k--)
	//mesh[k][col].blend(mesh[k-1][col],mesh[k][col],w);
	mesh[k][col] = SCIRun::Interpolate(mesh[k-1][col], mesh[k][col], w);
      newm->mesh[msize-1-j][col] = mesh[msize-1][col];
    }
  }

  inline void SubDivide(Mesh **nw, Mesh **ne, Mesh **se, Mesh **sw,
			double w)
  {
    int i,j;
    
    *nw = this;
    *ne = new Mesh(msize,nsize);
    for (i=0; i<msize; i++) 
      (*nw)->RowSplit(i,*ne,w);
    
    *sw = new Mesh(msize,nsize);
    *se = new Mesh(msize,nsize); 
    for (j=0; j<nsize; j++) {
      (*nw)->ColSplit(j,*sw,w);
      (*ne)->ColSplit(j,*se,w);
    }
  }

  inline void RowSplit(int row, double w, Point *sub)
  {
    Point *m=mesh[row];
      
    sub = &(sub[row*nsize]);
      
    sub[0] = m[0];
    for (int j=1; j<nsize; j++, m=sub)
      for (int k=nsize-1; k>=j; k--)
	sub[k] = SCIRun::Interpolate(m[k-1], m[k], w);
  }
  
  inline void ColSplit(int col, double w, Point *sub)
  {
    
    int ki, ki_1, k;
    sub = &(sub[col]);
    
    for (int j=1; j<msize; j++)
      for (k=msize-1, ki=k*nsize, ki_1=ki-nsize; 
	   k>=j; 
	   k--, ki=ki_1, ki_1-=nsize) {
	// This function had to be changed in the merger.  The points
	// had to reversed to comply with the new function
	sub[ki] = SCIRun::Interpolate(sub[ki_1], sub[ki], w);
      }
  }
    
  inline void Eval(double u, double v, Point &S,
		   Vector &Su, Vector &Sv, PerProcessorContext *ppc)
  {
    int i;
    //double wu,wv;
    Point *sub = (Point *) ppc->getscratch(get_scratchsize());
    Point *Pi_1j_1, *Pi_1j_2, *Pi_2j_1;
    
    for (i=0; i<msize; i++)
      RowSplit(i,u,sub);
    ColSplit(nsize-1,v,sub);
    ColSplit(nsize-2,v,sub);
    
    Pi_1j_1 = &sub[msize*nsize-1];
    Pi_1j_2 = &sub[msize*nsize-2];
    Pi_2j_1 = &sub[(msize-1)*nsize-1];

    S = *Pi_1j_1;
    
    Su = (*Pi_1j_1 - *Pi_1j_2)*((nsize-1)/(u));
    Sv = (*Pi_1j_1 - *Pi_2j_1)*((msize-1)/(v));
  }
    
  inline void compute_bounds(BBox *b) {
    Point P;
    
    for (int i=0; i<msize; i++) {
      for (int j=0; j<nsize; j++) {
	P = 
	  Point(mesh[i][j].x(),mesh[i][j].y(),mesh[i][j].z());
	b->extend(P);
      }
    }
  }
  
    
  inline Vector getNormal(double u, double v, PerProcessorContext *ppc)
  {
    Point S;
    Vector Su,Sv;

    Eval(u,v,S,Su,Sv,ppc);

    Vector N(Cross(Su, Sv));
    
    N.normalize();
    
    return N;
  }
    
  inline void F(const Point &S, const Vector &p1, const double p1d,
		const Vector &p2, const double p2d,
		double &f1, double &f2)
  {
    f1 = Dot(S, p1)+p1d;
    f2 = Dot(S, p2)+p2d;
  }
    
  inline void Fu(const Vector &Su, const Vector &p1, const Vector &p2,
		 double &d0, double &d1)
  {
    d0 = Dot(Su, p1);
    d1 = Dot(Su, p2);
  }

  inline void Fv(const Vector &Sv, const Vector &p1, const Vector &p2,
		 double &d0, double &d1)
  {
    d0 = Dot(Sv, p1);
    d1 = Dot(Sv, p2);
  }

  inline void ray_planes(const Ray &r, Vector &p1, double &p1d,
			 Vector &p2, double &p2d)
  {
    Point ro(r.origin());
    Vector rdir(r.direction());
    double rdx,rdy,rdz;
    double rdxmag, rdymag, rdzmag;

    rdx = rdir.x();
    rdy = rdir.y();
    rdz = rdir.z();

    rdxmag = fabs(rdx);
    rdymag = fabs(rdy);
    rdzmag = fabs(rdz);

    if (rdxmag > rdymag && rdxmag > rdzmag) 
      p1 = Vector(rdy,-rdx,0);
    else
      p1 = Vector(0,rdz,-rdy);

    p2 = Cross(p1, rdir);

    // Each plane contains the ray origin
    p1d = -Dot(p1, ro);
    p2d = -Dot(p2, ro);
  }

  inline void gentheta(double *theta, double t, int n) 
  {
    int i;
    double pow_w1=1., pow_w2=1.;
    double w1=t, w2=(1.-t);

    for (i=0; i<n; i++)	
      theta[i] = comb_table[n-1][i];
	
    for (i=0; i<n; i++) {

      theta[i] *= pow_w1;
      theta[n-i-1] *= pow_w2;

      pow_w1 *= w1;
      pow_w2 *= w2;
    }
  }

  inline void gendtheta(double *dtheta, double t, int n) 
  {
    ASSERT(n >= 2);

    gentheta(dtheta,t,n-1);
    
    dtheta[n-1] = n * dtheta[n-2];
    for (int i=nsize-2; i>0; i--)
      dtheta[i] = n*(dtheta[i-1]-dtheta[i]);
    dtheta[0] = n * (- dtheta[0]);
  }

  inline void EvalS(double u, double v, Point &S, PerProcessorContext *ppc)
  {
    
    Point P(0,0,0);
    double *thetau, *thetav, *dthetau, *dthetav;

    get_scratch(thetau,thetav,dthetau,dthetav,ppc);
    
    gentheta(thetau,u,nsize);
    gentheta(thetav,v,msize);

    for (int i=0; i<msize; i++) {
      for (int j=0; j<nsize; j++) {
	// P += mesh[i][j]*thetau[j]*thetav[i]
	P.addscaled(mesh[i][j],thetau[j]*thetav[i]);
      }
    }
    S = P;
  }

  inline void EvalAll(double u, double v, Point &S, Vector &Su, Vector &Sv,
		      PerProcessorContext *ppc) {
    
    double tu, dtu;
    double tv, dtv;
    double blend, blendu, blendv;
    Point Pu(0,0,0), Pv(0,0,0);
    Point P(0,0,0);
    double *thetau, *thetav, *dthetau, *dthetav;

    get_scratch(thetau,thetav,dthetau,dthetav,ppc);

    gentheta(thetau,u,nsize);
    gentheta(thetav,v,msize);
    gendtheta(dthetau,u,nsize);
    gendtheta(dthetav,v,msize);

    for (int i=0; i<msize; i++) {
      tv = thetav[i];
      dtv = dthetav[i];
      for (int j=0; j<nsize; j++) {
	dtu = dthetau[j];
	tu = thetau[j];
	blend = tu*tv;
	blendu = dtu*tv;
	blendv = tu*dtv;
	P.addscaled(mesh[i][j],blend);
	Pu.addscaled(mesh[i][j],blendu);
	Pv.addscaled(mesh[i][j],blendv);
      }
    }

    S = P;
    Su = Pu.vector();
    Sv = Pv.vector();
  }
  inline double calc_t(const Ray &r, Point &S)
  {
    Vector d(r.direction());
    Vector oS(S-r.origin());

    return Dot(d, oS)/Dot(d, d);
  }
    
  inline int Hit(const Ray &r, double &u, double &v, double &t,
		 double ulow, double uhigh, double vlow, double vhigh,
		 PerProcessorContext *ppc)
  {
    int i;
    double j11, j12, j21, j22;
    double f,g;
    //double fold, gold;
    double rootdist;
    //double oldrootdist;
    double detJ, invdetJ;
    //int tdiv=0;
        
    // Planes containing ray
    Vector p1,p2;
    double p1d=0,p2d=0;
        
    ray_planes(r,p1,p1d,p2,p2d);
        
    Point S;
    Vector Su, Sv;

    Eval(u,v,S,Su,Sv,ppc);
    F(S,p1,p1d,p2,p2d,f,g);
    rootdist = fabs(f) + fabs(g);
    for (i=0; i<MAXITER; i++)
    {
      Fu(Su,p1,p2,j11,j21);
      Fv(Sv,p1,p2,j12,j22);

      detJ = (j11*j22-j12*j21);

      if (detJ*detJ < 1E-9)
	return 0;

      invdetJ = 1./detJ;
	      
      u -= invdetJ*(j22*f-j12*g);
      v -= invdetJ*(j11*g-j21*f);
	      
      /*if ((u<=ulow) || (u>=uhigh) ||
	(v<=vlow) || (v>=vhigh))
	return 0;*/
	      
      Eval(u,v,S,Su,Sv,ppc);
      //fold = f;
      //gold = g;
      //oldrootdist = rootdist;
	      
      F(S,p1,p1d,p2,p2d,f,g);
	      
      rootdist = fabs(f)+fabs(g);
	      
      /*if (rootdist > oldrootdist)
	return 0;*/
      if (rootdist < MESH_ROOT_TOL) {
	if ((u<=ulow) || (u>=uhigh) ||
	    (v<=vlow) || (v>=vhigh))
	  return 0;
	t = calc_t(r,S);
                
	return 1;
      }
    }
    return 0;
  }

  //Uses Broyden's Method....
  inline int Hit_Broyden(const Ray &r, double &u, double &v, double &t,
			 double ulow, double uhigh, double vlow, double vhigh,
			 PerProcessorContext *ppc)
  {
    int i=0;
    double Bi_1[2][2], Bi[2][2];
    double f[2],f_1[2],df[2];
    double rootdist, rootdist_1;
    double detJ, invdetJ;
    //int tdiv=0;
    double u_1, v_1;
    double dx[2];
    double invdx2;

    // Planes containing ray
    Vector p1,p2;
    double p1d=0,p2d=0;
    
    ray_planes(r,p1,p1d,p2,p2d);
    
    Point S;
    Vector Su, Sv;
    
    // We start out with a single Newton iteration.
    u_1 = u;
    v_1 = v;
    
    //Eval(u_1,v_1,S,Su,Sv,ppc);
    EvalAll(u_1,v_1,S,Su,Sv,ppc);
    
    F(S,p1,p1d,p2,p2d,f_1[0],f_1[1]);
    
    rootdist_1 = fabs(f_1[0]) + fabs(f_1[1]);
    
    if (rootdist_1 > MESH_ROOT_TOL) {
      
      // Calculate the Jacobian for the -1th iteration
      Fu(Su,p1,p2,Bi_1[0][0],Bi_1[1][0]);
      Fv(Sv,p1,p2,Bi_1[0][1],Bi_1[1][1]);

      detJ = (Bi_1[0][0]*Bi_1[1][1]-Bi_1[0][1]*Bi_1[1][0]);
      if (detJ*detJ < 1E-16)
	return 0;

      invdetJ = 1./detJ;
      
      u -= invdetJ*(Bi_1[1][1]*f_1[0]-Bi_1[0][1]*f_1[1]);
      v -= invdetJ*(Bi_1[0][0]*f_1[1]-Bi_1[1][0]*f_1[0]);
      
      EvalS(u,v,S,ppc);
      
      F(S,p1,p1d,p2,p2d,f[0],f[1]);
      
      rootdist = fabs(f[0]) + fabs(f[1]);
      
      // Failure conditions
      /*if ((rootdist > rootdist_1) ||
	(u<ulow) || (u>uhigh) ||
	(v<vlow) || (v>vhigh))
	return 0;*/
      
      // Continue condition
      if (rootdist > MESH_ROOT_TOL) {
	
	// We move on to Broyden's method.
	for (; i<MAXITER; i++) {
	  double denom=((u-u_1)*(u-u_1)+(v-v_1)*(v-v_1));
	  if (denom == 0.) return 0;	  
	  invdx2 = 1./denom;
	  
	  df[0] = f[0]-f_1[0];
	  df[1] = f[1]-f_1[1];
	  
	  dx[0] = u - u_1;
	  dx[1] = v - v_1;
	  
	  // Broyden's approximate Jacobian, Bi
	  Bi[0][1] = Bi[0][0] =
	    (df[0] - (Bi_1[0][0]*dx[0]+Bi_1[0][1]*dx[1]))*invdx2;
	  Bi[1][1] = Bi[1][0] =
	    (df[1] - (Bi_1[1][0]*dx[0]+Bi_1[1][1]*dx[1]))*invdx2;
	  
	  Bi[0][0] = Bi[0][0]*dx[0] + Bi_1[0][0];
	  Bi[0][1] = Bi[0][1]*dx[1] + Bi_1[0][1];
	  Bi[1][0] = Bi[1][0]*dx[0] + Bi_1[1][0];
	  Bi[1][1] = Bi[1][1]*dx[1] + Bi_1[1][1];

	  detJ = (Bi[0][0]*Bi[1][1]-Bi[0][1]*Bi[1][0]);
	  if (detJ * detJ < 1E-16)
	    return 0;

	  invdetJ = 1./detJ;
	  
	  u_1 = u;
	  v_1 = v;
	  
	  u -= invdetJ*(Bi[1][1]*f[0]-Bi[0][1]*f[1]);
	  v -= invdetJ*(Bi[0][0]*f[1]-Bi[1][0]*f[0]);
	  
	  EvalS(u,v,S,ppc);

	  f_1[0] = f[0];
	  f_1[1] = f[1];
	  rootdist_1 = rootdist;
	  
	  F(S,p1,p1d,p2,p2d,f[0],f[1]);
	  rootdist = fabs(f[0])+fabs(f[1]);
	  
	  // Failure conditions
	  /*if ((rootdist > rootdist_1) ||
	    (u<ulow) || (u>uhigh) ||
	    (v<vlow) || (v>vhigh))
	    return 0;*/
	  
	  // Success condition
	  if (rootdist < MESH_ROOT_TOL)
	    break;
	  
	  Bi_1[0][0] = Bi[0][0];
	  Bi_1[1][0] = Bi[1][0];
	  Bi_1[0][1] = Bi[0][1];
	  Bi_1[1][1] = Bi[1][1];
	}
      }
    }
    
    // Took too long!
    if ((i >= MAXITER) || 
	(u<ulow) || (u>uhigh) ||
	(v<vlow) || (v>vhigh))
      return 0;
    
    t = calc_t(r,S);

    return 1;
  }
  
  Point **mesh;

private:

  int msize, nsize;
    
};

	
} // end namespace rtrt

#endif
