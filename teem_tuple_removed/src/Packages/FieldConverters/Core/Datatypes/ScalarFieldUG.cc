/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ScalarFieldUG.cc: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <FieldConverters/Core/Datatypes/ScalarFieldUG.h>
#include <Core/Util/NotFinished.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

#ifdef _WIN32
#include <stdlib.h>
#define drand48() rand()
#endif

namespace FieldConverters {

using namespace SCIRun;

static Persistent* maker()
{
    return scinew ScalarFieldUG(ScalarFieldUG::NodalValues);
}

PersistentTypeID ScalarFieldUG::type_id("ScalarFieldUG", "ScalarField", maker);

ScalarFieldUG::ScalarFieldUG(Type typ)
: ScalarField(UnstructuredGrid), typ(typ)
{
}

ScalarFieldUG::ScalarFieldUG(const MeshHandle& mesh, Type typ)
  : ScalarField(UnstructuredGrid), mesh(mesh),
    typ(typ)
{
  switch(typ){
  case NodalValues:
    data.resize(mesh->nodes.size());
    break;
  case ElementValues:
    data.resize(mesh->elems.size());
    break;
  }
}

ScalarFieldUG::~ScalarFieldUG()
{
}

ScalarField* ScalarFieldUG::clone()
{
    NOT_FINISHED("ScalarFieldUG::clone()");
    return 0;
}

void ScalarFieldUG::compute_bounds()
{
    if(have_bounds || mesh->nodes.size() == 0)
	return;
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}

#define SCALARFIELDUG_VERSION 2

void ScalarFieldUG::io(Piostream& stream)
{

    int version=stream.begin_class("ScalarFieldUG", SCALARFIELDUG_VERSION);
    // Do the base class....
    ScalarField::io(stream);

    if(version < 2){
        typ=NodalValues;
    } else {
        int* typp=(int*)&typ;
	stream.io(*typp);
    }

    SCIRun::Pio(stream, mesh);
    SCIRun::Pio(stream, data);
}

void ScalarFieldUG::compute_minmax()
{

    if(have_minmax || data.size()==0)
	return;
    double min=data[0];
    double max=data[1];
    for(int i=0;i<data.size();i++){
	min=Min(min, data[i]);
	max=Max(max, data[i]);
    }
    data_min=min;
    data_max=max;
    have_minmax=1;
}

int ScalarFieldUG::interpolate(const Point& p, double& value, double epsilon1, double epsilon2)
{
    int ix=0;
    if(!mesh->locate(p, ix, epsilon1, epsilon2))
	return 0;
    if(typ == NodalValues){
      double s1,s2,s3,s4;
      Element *e = mesh->elems[ix];
      mesh->get_interp(e, p, s1, s2, s3, s4);
      value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    } else {
	value=data[ix];
    }
    return 1;
}

// exhaustive is used to find points in "broken" meshes.
// if exhaustive == 1, we will try locate with 5 random seeds.
// if that fails and exhaustive == 2, we will do a full brute-force search.

int ScalarFieldUG::interpolate(const Point& p, double& value, int& ix, double epsilon1, double epsilon2, int exhaustive)
{
    if (!mesh->locate(p, ix, epsilon1, epsilon2)) {
	if (exhaustive > 0) {
	    MusilRNG mr;
	    ix=(int)(mr()*mesh->nodes.size());
	    int cntr=0;
	    while(!mesh->locate(p, ix, epsilon1, epsilon2) && cntr<5) {
		ix=(int)(mr()*mesh->nodes.size());
		cntr++;
	    }
	    if (cntr==5) {
		if (exhaustive == 2) {
		    if(!mesh->locate2(p, ix, 0))
			return 0;
		} else {
		    return 0;
		}
	    }
	} else {
	    return 0;
	}
    }
    if(typ == NodalValues){
        double s1,s2,s3,s4;
	Element* e=mesh->elems[ix];
	mesh->get_interp(e, p, s1, s2, s3, s4);
	value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    } else {
	value=data[ix];
    }   
    return 1;
}

Vector ScalarFieldUG::gradient(const Point& p)
{
    int ix;
    if(!mesh->locate(p, ix))
	return Vector(0,0,0);
    Vector g1, g2, g3, g4;
    Element* e=mesh->elems[ix];
    mesh->get_grad(e, p, g1, g2, g3, g4);
    return g1*data[e->n[0]]+g2*data[e->n[1]]+g3*data[e->n[2]]+g4*data[e->n[3]];
}

void ScalarFieldUG::get_boundary_lines(Array1<Point>& lines)
{
    mesh->get_boundary_lines(lines);
}

/*
 * Random point in a tetrahedra
 *  
 * S(P0,P1,P2,P3) is a tetrahedra, find a random point
 * internaly
 * 
 * A = P0 + Alpha(P2-P0)
 * B = P0 + Alpha(P1-P0)
 * C = P0 + Alpha(P3-P0)
 *
 * S(A,B,C) is a triangle "pushed" from point P0 by Alpha
 * now find a random point on this triangle (cube root of random var)
 *
 * D = A + Beta(B-A)
 * E = A + Beta(C-A)
 *
 * S(D,D) is a line segment pushed by Beta from A (square root)
 *
 * F = D + Gamma(E-D)
 *
 * F is a random point on the interval [D,E], and iside the tet
 *
 * Combining the above you get the following: (weights for nodes)
 *
 * W0 = 1-Alpha
 * W1 = BetaAlpha(1-Gamma)
 * W2 = Alpha(1-Beta)
 * W3 = BetaAlphaGamma
 *
 * (you just need 3, they sum to one...)
 */ 


inline Point RandomPoint(Mesh *mesh, Element *e)
{
  
  const Point &p0 = mesh->nodes[e->n[0]]->p;
  const Point &p1 = mesh->nodes[e->n[1]]->p;
  const Point &p2 = mesh->nodes[e->n[2]]->p;
  const Point &p3 = mesh->nodes[e->n[3]]->p;
  double alpha,gamma,beta; // 3 random variables...

  alpha = pow(drand48(),1.0/3.0);
  beta = sqrt(drand48());
  gamma = drand48();

  // let the combiler do the sub-expression stuff...

  return AffineCombination(p0,1-alpha,
			   p1,beta*alpha*(1-gamma),
			   p2,alpha*(1-beta),
			   p3,beta*alpha*gamma);
}

// tries to compute nsamp random points...
// this is just assigns the weights (ie: volumes)
// and then it calls the function that does the optimization thing...

void ScalarFieldUG::compute_samples(int nsamp)
{
  cerr << "Computing Samples UG\n";

  aug_elems.remove_all();
  samples.remove_all();
  
  Mesh* mymesh = mesh.get_rep(); // handles can be bad...

  // starting from scratch...
  
  aug_elems.resize(mymesh->elems.size());
  double total_volume=0.0;

  for(int i=0;i<mymesh->elems.size();i++) {
    if (mymesh->elems[i]) {
      aug_elems[i].importance = mymesh->elems[i]->volume();
      total_volume += aug_elems[i].importance;
    }
  }

  samples.resize(nsamp);
}

void ScalarFieldUG::distribute_samples()
{
  cerr << "Distributing Samples UG\n";
  // number is already assigned, but weights have changed

  double total_importance =0.0;
  Mesh* mymesh = mesh.get_rep(); // handles can be bad...
  Array1<double> psum(mymesh->elems.size());
  
  int i;
  for(i=0;i<mymesh->elems.size();i++) {
    if (mymesh->elems[i]) {
      total_importance += aug_elems[i].importance;
      psum[i] = total_importance;
      aug_elems[i].pt_samples.remove_all();
    } else {
      psum[i] = -1;  // bad, so it will just skip over this...
    }
  }

  // now just jump into the prefix sum table...
  // this is a bit faster, especialy initialy...

  int pi=0;
  int nsamp = samples.size();
  double factor = 1.0/(nsamp-1)*total_importance;

  for(i=0;i<nsamp;i++) {
    double val = (i*factor);
    while ( (pi < aug_elems.size()) && 
	   (psum[pi] < val))
      pi++;
    if (pi == aug_elems.size()) {
      cerr << "Over flow!\n";
    } else {
      aug_elems[pi].pt_samples.add(i);
      samples[i].loc = RandomPoint(mymesh, mymesh->elems[pi]);
    }
  }
}

// this stuff is for augmenting the random distributions...

void ScalarFieldUG::fill_gradmags() // these guys ignor the vf
{

  cerr << "Filling the gradient image UG\n";

  total_gradmag = 0.0;
  
  grad_mags.resize(mesh->elems.size());

  // MAKE PARALLEL

  for(int i=0;i<mesh->elems.size();i++) {
    Element *e = mesh->elems[i];
    if (e) {
      Vector g0,g1,g2,g3,grad;
      Point p;
      mesh->get_grad(e,p,g0,g1,g2,g3);
      grad = g0*data[e->n[0]] +g1*data[e->n[1]] +g2*data[e->n[2]] +
	g3*data[e->n[3]];
      
      grad_mags[i] = grad.length();
      
      total_gradmag += grad_mags[i];
    } else { // make it 0
      grad_mags[i] = 0;
    }
  }
}

void ScalarFieldUG::over_grad_augment(double vol_wt, double grad_wt, 
				      double /*crit_scale*/)
{
  // this is done in 2 passes, first just get gradients and compute totals
  double vol_total=0.0;
  double grad_total=0.0;
  Array1<double> grads(grad_mags.size());
  double max=-1;
  Array1<int> zeros;
  
  int i;
  for(i=0;i<grad_mags.size();i++) {
    Element *e = mesh->elems[i];  // have to grab these...
    if (e) {
      grads[i] = grad_mags[i]; // already compute this...

      if (grads[i] == 0.0) {
	cerr << "Have a 0...\n";
	zeros.add(i);
      } else {
	grads[i] = 1.0/grads[i];
	if (grads[i] > max)
	  max = grads[i];
      }
      vol_total += aug_elems[i].importance;
      grad_total += grad_mags[i];
    }
  }

  // ok...  now handle zeros...

  for(i=0;i<zeros.size();i++) {
    grad_total += max;
    grads[zeros[i]] = max;
  }

  // now do a second pass, normalizing the above...

  // normalization is done through recomputing the weights...

  vol_wt /= vol_total;
  grad_wt /= grad_total;

  for(i=0;i<mesh->elems.size();i++) {
    Element *e = mesh->elems[i];
    if (e) {
      aug_elems[i].importance =
	vol_wt*aug_elems[i].importance + grads[i]*grad_wt;
    }
  }

  cerr << vol_total << " Volume " << grad_total << " Gradient\n";
}

} // end namespace FieldConverters
