/*
 *  SourceLocalize.cc:  Builds the global finite element matrix
 *
 *  Written by:
 *   Ruth Nicholson Klepfer
 *   Department of Bioengineering
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/BitArray1.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/Matrix.h>

#include <Datatypes/ScalarFieldUG.h>

#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>
#include <Datatypes/GeometryPort.h>

#include <Geom/Line.h>
#include <Geom/Sphere.h>
#include <Geom/Group.h>
#include <Geom/Pt.h>
#include <Geom/Material.h>

#include <Widgets/PointWidget.h>

// this is per-proccesor information
// used for solving 

using sci::Mesh;
using sci::MeshHandle;
using sci::Element;
using sci::Node;
using sci::NodeHandle;
using sci::DirichletBC;

const int NUM_DOF=5;
const int NUM_SAMPLES=1000; // should there be more???

#define SPECIAL_CG

/*
 * Dipole Problem:
 *
 * We would like to model a dipole source wihtout having
 * to reconstruct the finite element matrix.  Instead we
 * will keep track of all of the nodes that would be affected 
 * by inserting the dipole nodes into the system.  The FakeRow
 * structure effectively does this (one per affected node).
 * the row for the given node is broken down into 3 parts:
 *
 * cols/cvals  -  these are nodes that are not fakeRows or sources
 * ncols/nvals -  indeces into fakeRow arrays - other fakeRows
 * src/svals   -  indeces into the inserted dipole sources
 *
 * src/svals can be compressed into a single value - contribution 
 * from the source.
 *
 * Matrix vector multiplies are performed by zeroing out all of
 * the fake nodes in the lhs vector (neccesitating the need for 
 * the fakeRow to keep track of its value).  Then the contribution
 * from each node is added to its neighboring nodes (cols), and
 * the row (node) is computed (cols/ncols/src) and updated.
 */

struct FakeRow {

  int nodeid;             // which node this is

  double cval;            // value - lhs is zeroed out...
  double diag;            // diagonal term - preconditioning

  Array1<int>     cols;   // columns that are connected to other fakeRows
  Array1<double>  cvals;  // data value at given columns

  Array1<int>     ncols;  // non-node cols - for fix up
  Array1<double>  nvals;  // data values for above

  double       scontrib;  // sum of all source contributions...
};

// you might want to just do 1 random point, and then move out
// from there instead...

struct SourceCoefs {
  // this structure represents the coeficients
  // for a given source...

  Array1<double> coefs; // coeficients x,y,z,orientation - etc.

  void CreateDipole(Point &p0, Vector &v);

  double err; // result of evaluating function with these coefs


  // below is the result of descritizing this source(s)
  
  int isValid; // if the point is a valid potential
               // solution - if a point fails, use the old coefs...

  Array1<double> oldCoefs; // old coeficients - copy back if invalid...

  Array1<FakeRow> fake;  // bogus rows
  Array1<double>  fakeS; // sources - really not needed...
};

void SourceCoefs::CreateDipole(Point &p0, Vector &v)
{
  double theta,phi,mag;

  p0.x(coefs[0]);
  p0.y(coefs[1]);
  p0.z(coefs[2]);

  theta = coefs[3];// - ((int)coefs[3]);
  phi = coefs[4];// - ((int)coefs[4]);

  double sinphi = sin(phi*2*M_PI);

  v.x(cos(theta*2*M_PI)*sinphi);
  v.y(sin(theta*2*M_PI)*sinphi);
  v.z(cos(phi*2*M_PI));
}

inline double EvaluateDipole(Point &p, Point &d, Vector &v)
{
  Point p0 = d + v; // assume that "d" is 2
  Point p1 = d - v;

  double d0 = (p-p0).length();
  double d1 = (p-p1).length();

  return 10000*(1/d0 - 1/d1);
}

struct AmoebaData {

  AmoebaData() { rhs = diag = R = Z = P = 0; }

  Array1< SourceCoefs > samples; // N+1 points
  Array1< double > psum;         // prefix sum table for coefs size N
  
  inline void ComputePSUM(); // computes sum for every coef...

  inline void amotry(double fac); // extrapolates from highest point

  void Contract(); // contracts around lowest point...

  int ihi;   // worst sample pt
  int inhi;  // second worst sample pt
  int lowi;  // "best" value

  // these are per-proc data - 

  BitArray1 *nodesUsed; // bitmasks for nodes and elements
  BitArray1 *elemsUsed;

  Mesh      *local_mesh; // local mesh for this thing
  
  Array1< int > nodeRemap; // node remaping array out of local mesh
  Array1< int > elem_check;// temp variables for local mesh construction 
  Array1< int > actual_remap; // temp for fake row construction

  SymSparseRowMatrix* local_matrix; // just reflect local mesh patch
  ColumnMatrix *local_rhs; // for local matrix

  ColumnMatrix *rhs; // rhs for this matrix...
  ColumnMatrix *lhs; // lhs for this matrix - can be previous or zero...

  ColumnMatrixHandle Hlhs; // handle for above...

  ColumnMatrix *diag;

  ColumnMatrix *R;   // variables used in CG
  ColumnMatrix *Z;
  ColumnMatrix *P;

  ScalarFieldUG *sug; // just used for random point stuff...

  Array1<int> colidx;
  Array1<int> mycols;
};

inline void AmoebaData::ComputePSUM()
{
  psum.resize(NUM_DOF);
  for(int i=0;i<NUM_DOF;i++) {
    psum[i] = 0.0;
    for(int j=0;j<samples.size();j++)
      psum[i] += samples[j].coefs[i];
  }
}



void AmoebaData::Contract()
{
  for(int i=0;i<samples.size();i++) {
    if (i != lowi) {
      for(int j=0;j<samples[i].coefs.size();j++) {
	samples[i].coefs[j] = 0.5*(samples[lowi].coefs[j] + 
				   samples[i].coefs[j]);
      }
    }
  }
}

class SourceLocalize : public Module {
  MeshIPort* inmesh;
  MeshOPort* outmesh;
  MatrixIPort * inmatrix;
  // MatrixOPort * outmatrix;
  ColumnMatrixOPort* rhsoport;
  ColumnMatrixOPort* solport;
  GeometryOPort* ogeom;

  int init;
  CrowdMonitor widget_lock;
  PointWidget *widget;
  int widgetMoved;

  // build a local mesh from the big mesh - you
  // have to correctly remap the elements, nodes
  // and face indices of the mesh.
  // this mesh will contain all of the elements whose
  // circumspheres intersect either point

  int build_local_mesh(Point &p1, Point &p2, // dipole locations
		       Mesh *&tmesh, // local region mesh
		       Array1< int > &nodeRemap, // back to original mesh
		       BitArray1 &nodesUsed, // nodes used
		       BitArray1 &elemsUsed, // elems used
		       Array1< int > &elem_check);

  // this assumes local mesh has been created and
  // fakeS field (voltage value for fake sources) has
  // has been filled in.  Fills in FakeRow's for all 
  // the required nodes...

  void build_fake_rows(int proc, int which);

  // this function constructs a matrix from a mesh

  void build_matrix(Array1<int> &colidx,
		    Array1<int> &mycols,
		    SymSparseRowMatrix *&cmat,
		    ColumnMatrix *&rhs,
		    Mesh *msh);

  void build_local_matrix(Element*, double lcl[4][4],
			  Mesh*);
  void add_lcl_gbl(Matrix&, double lcl[4][4],
		   ColumnMatrix&, int, const Mesh*);
  void add_lcl_gbl(Matrix&, double lcl[4][4],
		   ColumnMatrix&, int, const Mesh*,
		   int s, int e);

  void InitAmoeba(int proc, int init_dipole=1);
  void DoAmoeba(int proc); 

  void ComputeSamples(int proc); // could use multiple threads here...
  void ComputeSample(int proc,int which); // could use multiple threads here...

  void ConfigureSample(int proc, int which);

  double SolveStep(int proc, double fac);

  Array1<int> sources;  // array of the sources that are being used...
  Array1<double> svals; // values for those sources...

  Array1< AmoebaData > ambs; // per-proc stuff....

  Array1< GeomSphere* > err_spheres;
  Array1< GeomSphere* > amb_spheres;
  Array1< GeomLine* > amb_orient;

  GeomLines *lines;

  int* rows;
  Array1<int> colidx;
  Array1<int> mycols;
  int* allcols;
  Mesh* mesh;
  SymSparseRowMatrix* gbl_matrix;
  ColumnMatrix* rhs;
  ColumnMatrix* Lhs;

  ColumnMatrixHandle Hrhs; // ditto...

  double OptimizeInterp(Array1<double> &trial,double &mag); // returns error...

  Array1<Point>  interp_pts;   // interpolation points
  Array1<int>    interp_elems; // elements for interpolation...
  Array1<double> interp_vals;  // values at interpolated points...

  void SolveMatrix(); // this does the matrix solve 
                      // sources and svals must be set up.

  void SolveMatrix(AmoebaData& amb, int idx,ColumnMatrix &lhs); 
  // you have to compute all of the discrete stuff every time the above
  // function is called...

  inline void CleanupMMult(ColumnMatrix &x,
			   ColumnMatrix &b,
			   SourceCoefs &me);

  inline void PreMultFix(ColumnMatrix &x,
			 SourceCoefs &me);

public:
  SourceLocalize(const clString& id);
  SourceLocalize(const SourceLocalize&, int deep);
  virtual void widget_moved(int last);    
  virtual ~SourceLocalize();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
  Module* make_SourceLocalize(const clString& id)
    {
      return scinew SourceLocalize(id);
    }
};


SourceLocalize::SourceLocalize(const clString& id)
: Module("SourceLocalize", id, Filter),lines(0)
{
  // Create the input ports
  inmesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
  add_iport(inmesh);

  inmatrix = scinew MatrixIPort(this,"Matrix",MatrixIPort::Atomic);
  add_iport(inmatrix);

#if 0
  // Create the output ports
  outmatrix=scinew MatrixOPort(this, "FEM Matrix", MatrixIPort::Atomic);
  add_oport(outmatrix);
#endif
  solport=scinew ColumnMatrixOPort(this, "LHS", ColumnMatrixIPort::Atomic);
  add_oport(solport);

  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);

  rhsoport=scinew ColumnMatrixOPort(this, "RHS", ColumnMatrixIPort::Atomic);
  add_oport(rhsoport);

  outmesh = scinew MeshOPort(this, "Mesh", MeshIPort::Atomic);
  add_oport(outmesh);

  init = 0;
  widgetMoved=1;
}

SourceLocalize::SourceLocalize(const SourceLocalize& copy, int deep)
: Module(copy, deep)
{
  NOT_FINISHED("SourceLocalize::SourceLocalize");
}

SourceLocalize::~SourceLocalize()
{
}

Module* SourceLocalize::clone(int deep)
{
  return scinew SourceLocalize(*this, deep);
}

double SourceLocalize::SolveStep(int proc, double fac)
{
  AmoebaData &amb = ambs[proc];

  double fac1 = (1 - fac)/NUM_DOF;
  double fac2 = fac1 - fac;
  
  amb.ComputePSUM(); // just do it every time...

  amb.samples[amb.ihi].oldCoefs = amb.samples[amb.ihi].coefs;

  // save off the coefs...

  for(int i=0;i<NUM_DOF;i++) {
    amb.samples[amb.ihi].coefs[i] = amb.psum[i]*fac1 - 
      amb.samples[amb.ihi].coefs[i]*fac2;
  }
  int ovalid = amb.samples[amb.ihi].isValid;
  double yold =  amb.samples[amb.ihi].err;
  ConfigureSample(proc,amb.ihi);
  if (amb.samples[amb.ihi].isValid)
    ComputeSample(proc,amb.ihi);
  else 
    {
      cerr << "Out of range!\n";
      const double LARGE_ERROR=10000000000;
      amb.samples[amb.ihi].err += LARGE_ERROR;  
    }
  double ytry =  amb.samples[amb.ihi].err;
  
  if (ytry < yold) { // it is better...
    for(i=0;i<NUM_DOF;i++) {
      amb.psum[i] += amb.samples[amb.ihi].coefs[i] - 
	amb.samples[amb.ihi].oldCoefs[i];
    }
  } else { // copy back...
    amb.samples[amb.ihi].coefs = amb.samples[amb.ihi].oldCoefs;
    amb.samples[amb.ihi].err = yold;
  }
  return ytry;
}

void SourceLocalize::ConfigureSample(int proc, int which)
{
  AmoebaData &amb = ambs[proc];
  
  // for now we are just doing point sources...
  
  // wait - this has to be a dipole...
  
  Point p0;
  Vector v0;
  
  amb.samples[which].CreateDipole(p0,v0); // generate the dipole...

  Point p1 = p0 + v0*2;
  Point p2 = p0 - v0*2; // larger baseline???
  
  amb.samples[which].isValid = 1;

  //cerr << p1 << " " << p2 << endl;

  if (build_local_mesh(p1,p2,
		       amb.local_mesh,
		       amb.nodeRemap,
		       (*amb.nodesUsed),
		       (*amb.elemsUsed),
		       amb.elem_check)) { // valid dipole
#ifdef SPECIAL_CG
    build_fake_rows(proc,which);
#else
    build_matrix(amb.colidx,amb.mycols,gbl_matrix,amb.rhs,mesh);
#endif
  } else { // point is out of the mesh!
    amb.samples[which].isValid = 0;
  }
}

void SourceLocalize::ComputeSamples(int proc)
{
  AmoebaData &amb = ambs[proc];
  
  for(int i=0;i<amb.samples.size();i++) {
    if (!amb.samples[i].isValid) {
      cerr << "Woah - bad initial sample!\n";
    } else { // compute this puppy!
      ComputeSample(proc,i);
      //cerr << i << " " << amb.samples[i].err << endl;
    }
  }
}

void SourceLocalize::DoAmoeba(int proc)
{
  AmoebaData &amb = ambs[proc];
  cerr << "Doing Amoeba!\n";

  const int NMAX=150; // 100 function evaluations max...

  amb.ComputePSUM();

  int nfunk=0; // number of function evaluations...

  while(1) { // loop forever...
    amb.lowi = 0;
    if (amb.samples[0].err > amb.samples[1].err) {
      amb.ihi = 0;
      amb.inhi = 1;
    } else {
      amb.ihi = 1;
      amb.inhi = 0;
    }

    for(int i=0;i<amb.samples.size();i++) {
      if (amb.samples[i].err <= amb.samples[amb.lowi].err) amb.lowi = i;
      if (amb.samples[i].err > amb.samples[amb.ihi].err) {
	amb.inhi = amb.ihi;
	amb.ihi = i;
      } else if (amb.samples[i].err > amb.samples[amb.inhi].err &&
		 (i != amb.ihi)) amb.inhi = i;
    }

    double rtol=2*fabs(amb.samples[amb.ihi].err-amb.samples[amb.lowi].err)/
      (fabs(amb.samples[amb.ihi].err) + fabs(amb.samples[amb.lowi].err));

    const double ftol=0.000005;
    
    if (rtol < ftol) {
      cerr << rtol << "\n\ntolerance did it!\n";
      SourceCoefs tmpC = amb.samples[amb.lowi];
      amb.samples[amb.lowi] = amb.samples[0];
      amb.samples[0] = tmpC; // save the best one...
      return;
    }

    if (nfunk >= NMAX) {
      cerr << rtol << " Done max number of function calls!\n";
      SourceCoefs tmpC = amb.samples[amb.lowi];
      amb.samples[amb.lowi] = amb.samples[0];
      amb.samples[0] = tmpC; // save the best one...
      cerr << amb.samples[0].err << endl;
      
      return;
    }

    nfunk += 2;

    double ytry = SolveStep(proc,-1.0); // try and extrapolate through high pt

    if (ytry <= amb.samples[amb.lowi].err) { // try another contraction
      //cerr << "\n\nContracting again!\n\n\n";
      ytry = SolveStep(proc,2.0);
    } else if (ytry >= amb.samples[amb.inhi].err) { // do a 1d contraction...
      //cerr << "1D Contract!\n";
      double ysave = amb.samples[amb.ihi].err;
      ytry = SolveStep(proc,0.5);
      
      if (ytry >= ysave) { // do contraction around lowest point...
	amb.Contract(); // contract around the low point...
	//cerr << "Contract around low!\n";
	for(i=0;i<amb.samples.size();i++) {
	  if (i != amb.lowi) {
	    ComputeSample(proc,i);
	  }
	}
	nfunk += NUM_DOF;
	amb.ComputePSUM();
      }
    } else {
      nfunk--;
    }
  }
}

// this guy assumes that the sample is actualy valid!

void SourceLocalize::ComputeSample(int proc, int which)
{
  AmoebaData &amb = ambs[proc];

  SolveMatrix(amb,which,(*(amb.lhs)));

  // now that the matrix has been solved, compute the error...

  Array1<double> trial(interp_vals.size());

  Point p0;
  Vector v0;

  amb.samples[which].CreateDipole(p0,v0);

  // 1st compute the "offset"

  double offset=0;

  for(int i=0;i<trial.size();i++) {
#if 0
    Element *e = mesh->elems[interp_elems[i]];
    double a[4];
    mesh->get_interp(e,interp_pts[i],a[0],a[1],a[2],a[3]);

    trial[i] = 0.0;
    for(int j=0;j<4;j++) {
      trial[i] += (*amb.lhs)[e->n[j]]*a[j];
    }
#else
    offset += (*amb.lhs)[interp_elems[i]]; // just boundary points
#endif
  }

  offset /= trial.size(); // make sure this is correct...

  for(i=0;i<trial.size();i++) {
    trial[i] = (*amb.lhs)[interp_elems[i]] - offset;
  }

  // fix up the entire lhs!

  if (trial.size())
    for(i=0;i<(*amb.lhs).nrows();i++) {
      (*amb.lhs)[i] -= offset;
    }

  double mag;
  amb.samples[which].err = OptimizeInterp(trial,mag);

  //return;

  // now update the "lines"

  static int sphere_id=0;

  if (sphere_id &&
      amb_spheres.size() != amb.samples.size()) { // just remove this....
    ogeom->delObj(sphere_id);
  }

  if (amb_spheres.size() <= 1) {
    amb_spheres.resize(amb.samples.size());
    amb_orient.resize(amb.samples.size());
    GeomGroup *grp = scinew GeomGroup();

    for(int i=0;i<amb_spheres.size();i++) {
      amb_spheres[i] = scinew GeomSphere(15,7);

      Color curC(0.2,i/(amb_spheres.size()-1.),
		 0.1 + 
		 0.3*(amb_spheres.size()-1. - i)/(amb_spheres.size()-1.));

      if (amb_spheres.size() == 1) curC = Color(0,1,0);

      GeomGroup *sgrp = scinew GeomGroup();

      sgrp->add(amb_spheres[i]);

      // do dipole directions...

      Point p2 = p0+Vector(0,0,1);

      amb_orient[i] = new GeomLine(p0,p2);

      sgrp->add(amb_orient[i]);

      GeomMaterial *mtl = scinew GeomMaterial(sgrp,curC);
      grp->add( mtl );
    }
    sphere_id = ogeom->addObj(grp,"Amoeba Spheres");
  }

  int li=0;
  double minE=0,maxE=1000;
  for(i=0;i<amb.samples.size();i++) {

    amb.samples[i].CreateDipole(p0,v0);

    amb_spheres[i]->cen = p0;
    amb_orient[i]->p1 = p0;
    amb_orient[i]->p2 = p0 + v0*35;

    if (amb.samples[i].err < minE) minE = amb.samples[i].err;
    if (amb.samples[i].err > maxE) maxE = amb.samples[i].err;

    for(int j=i+1;j<amb.samples.size();j++) {
      Point p1;

      p1.x(amb.samples[j].coefs[0]);
      p1.y(amb.samples[j].coefs[1]);
      p1.z(amb.samples[j].coefs[2]);

      lines->pts[2*li] = p0;
      lines->pts[2*li + 1] = p1;
      li++;
    }
  }

  // also update the "sample" spheres...
  if (amb.samples.size() == 1)
    amb_spheres[0]->rad = 10.0;
  else
    for(i=0;i<amb.samples.size();i++) {
      amb_spheres[i]->rad = 2.0 + 8.0*((amb.samples[i].err-minE)/(maxE-minE));
    }

  ogeom->flushViews(); // flush this puppy...

}

void SourceLocalize::InitAmoeba(int proc, int init_dipole)
{
  AmoebaData &amb = ambs[proc];

#ifndef SPECIAL_CG
  int csize = gbl_matrix->nrows()+2;
#else
  int csize = gbl_matrix->nrows();
#endif
 
  if (amb.rhs && amb.rhs->nrows() != csize) {
    amb.rhs->resize(csize);
    amb.lhs->resize(csize);
    amb.diag->resize(csize);
    amb.R->resize(csize);
    amb.Z->resize(csize);
    amb.P->resize(csize);

    cerr << "\n\nWoah - mesh changed!\n\n\n";
  }

  if (!amb.rhs) {
    amb.rhs = scinew ColumnMatrix(csize);
    amb.lhs = scinew ColumnMatrix(csize);

    amb.lhs->zero(); // start with no guess - assign it later if you want to...

    amb.Hlhs = amb.lhs; // set the handle...

    amb.diag = scinew ColumnMatrix(csize);
    amb.R = scinew ColumnMatrix(csize);
    amb.Z = scinew ColumnMatrix(csize);
    amb.P = scinew ColumnMatrix(csize);

    amb.sug = scinew ScalarFieldUG(ScalarFieldUG::NodalValues);

    amb.sug->mesh = mesh;  // only once...

    // lets just choose some number of "starting" points
    // the initial solution will always be some subset of
    // these - spatialy - orientation is something else...

    amb.sug->compute_samples(NUM_SAMPLES);
    amb.sug->distribute_samples(); // could bias this???

    amb.nodesUsed = new BitArray1(mesh->nodes.size(),0);
    amb.elemsUsed = new BitArray1(mesh->elems.size(),0);
    
    amb.local_mesh = new Mesh(0,0); // copy conductivity tensors!

    amb.local_mesh->cond_tensors = mesh->cond_tensors;
    amb.local_matrix = 0;
    amb.local_rhs = scinew ColumnMatrix(500); // how many nodes?

    amb.nodeRemap.resize(500);
    amb.elem_check.resize(500);
    amb.actual_remap.resize(500);
  }
  int alines=0;
  if (!lines) {
    lines = scinew GeomLines();
    alines=1;
  }
  lines->pts.resize(2*((NUM_DOF+1)*NUM_DOF)/2); // just connect the points...

  // now you just have to assign starting points
  //  - orientations later...  for the "source"
  
  if (init_dipole)
    amb.samples.resize(NUM_DOF+1); //X,Y,Z,???

  if (init_dipole) {
    for(int i=0;i<amb.samples.size();i++) {
      // alternatively just do a offset from this array...
      Point p = amb.sug->samples[drand48()*(amb.sug->samples.size()-0.5)].loc;
      
      amb.samples[i].coefs.resize(NUM_DOF);
      amb.samples[i].coefs[0] = p.x();
      amb.samples[i].coefs[1] = p.y();
      amb.samples[i].coefs[2] = p.z();
      
      for(int j=3;j<NUM_DOF;j++) {
	//amb.samples[i].coefs[j] = 1.0; // [0,1) random value...
	amb.samples[i].coefs[j] = drand48(); // [0,1) random value...
      }
      amb.samples[i].oldCoefs = amb.samples[i].coefs; // just copy it...
      
    }
  }

  for(int i=0;i<amb.samples.size();i++) {
    ConfigureSample(proc,i); // set up the neccesary stuff...
    if (!amb.samples[i].isValid) {
      cerr << "Initial value is bogus???\n";
    } else {
      ComputeSample(proc,i);
    }
  }
#if 0
  if (alines)
    ogeom->addObj(lines,"Amoeba");
#endif
}

double SourceLocalize::OptimizeInterp(Array1<double> &trial,double &mag)
{
  // trial has been computed based on interp_pts/elems...

  double pisi=0.0;
  double si2=0.0;

  double minE=10000000.0;
  double maxE=-1.0;

  for(int i=0;i<interp_vals.size();i++) {
    pisi += interp_vals[i]*trial[i];
    si2 += trial[i]*trial[i];
  }

  mag = pisi/si2; // don't allow the dipole to flip...

  double rmag = mag;

  double err=0;

  for(i=0;i<interp_vals.size();i++) {
    double crr = (interp_vals[i] - trial[i]*mag)*
      (interp_vals[i] - trial[i]*mag);
    err += crr;
    if (crr < minE) minE = crr;
    if (crr > maxE) maxE = crr;
  }

  for(i=0;i<interp_vals.size();i++) {
    double crr = (interp_vals[i] - trial[i]*mag)*
      (interp_vals[i] - trial[i]*mag);
    err_spheres[i]->rad = 1.0 + 20.0*((crr-minE)/(maxE-minE));
  }

  ogeom->flushViews();

  //cerr << minE << " " << maxE << " -> " << err << endl;

  return err;
}

inline void SourceLocalize::CleanupMMult(ColumnMatrix &x,
					 ColumnMatrix &b,
					 SourceCoefs &me)
{
#ifndef SPECIAL_CG
  return;
#endif
  // assume that the values in x for these nodes
  // have been zeroed before the multiply - make
  // sure b is correct...

  for(int i=0;i<me.fake.size();i++) {
    if (x[me.fake[i].nodeid] != 0.0) {
      cerr << "Bad Iter: " << x[me.fake[i].nodeid] << endl;
    }

    x[me.fake[i].nodeid] = me.fake[i].cval; // was zero'd before...

    b[me.fake[i].nodeid] = 0.0; // compute this

    for(int j=0;j<me.fake[i].cols.size();j++)
      b[me.fake[i].nodeid] += me.fake[i].cvals[j]*
	me.fake[me.fake[i].cols[j]].cval;

    // now add in the non-fake nodes, and update them as well...
    for(j=0;j<me.fake[i].ncols.size();j++) {
      b[me.fake[i].nodeid] += me.fake[i].nvals[j]*
	x[me.fake[i].ncols[j]];
      
      // your contribution to other rows in the matrix...
      b[me.fake[i].ncols[j]] += me.fake[i].nvals[j]*me.fake[i].cval;
    }
  }

}

inline void SourceLocalize::PreMultFix(ColumnMatrix &x,
				       SourceCoefs &me)
{
#ifndef SPECIAL_CG
  return;
#endif
  // zero out appropriate parts of x,
  // copy x's values into "cval"

  for(int i=0;i<me.fake.size();i++) {
    me.fake[i].cval = x[me.fake[i].nodeid];
    x[me.fake[i].nodeid] = 0;
  }
}

void SourceLocalize::SolveMatrix(AmoebaData& amb, int idx, ColumnMatrix &lhs)
{
  Matrix *matrix = (Matrix*)gbl_matrix;
  ColumnMatrix &Rhs = (*(amb.rhs));
  ColumnMatrix &diag = (*(amb.diag));
  ColumnMatrix &R = (*(amb.R));
  ColumnMatrix &Z = (*(amb.Z));
  ColumnMatrix &P = (*(amb.P));

  SourceCoefs &me = amb.samples[idx];

  int i;

  // assume that lhs is properly initialized...
#ifdef SPECIAL_CG
  Rhs.zero(); 

  // zero out portion of lhs...

  for(i=0;i<me.fake.size();i++) {
    Rhs[me.fake[i].nodeid] = me.fake[i].scontrib;
    //cerr <<  me.fake[i].nodeid <<  " " << me.fake[i].scontrib << " RHS\n";
  }
  lhs.zero(); // will this help any?

#else

  Array1<int> rhsnz;

  for(i=0;i<Rhs.nrows();i++) {
    if (Rhs[i] != 0.0) {
      //cerr << i << " " << Rhs[i] << " RHS\n";
      rhsnz.add(i); // watch these guys every time...
    }
  }

  // also throw in the correct values for the sources...

  lhs[lhs.nrows()-2] = Rhs[lhs.nrows()-2];
  lhs[lhs.nrows()-1] = Rhs[lhs.nrows()-1];

#endif
  int *mrows = matrix->get_row();
  int *mcols = matrix->get_col();

  double *mvals = matrix->get_val();

  int size=matrix->nrows();

#ifdef SPECIAL_CG
  for(i=0;i<me.fake.size();i++) {
    // print out this row...
    int cnt=0;
    //cerr << me.fake[i].nodeid << endl;
    for(int j=0;j<me.fake[i].cols.size();j++) {
      //cerr << me.fake[me.fake[i].cols[j]].nodeid << " " << me.fake[i].cvals[j] << endl;
      cnt++;
    }
    for(j=0;j<me.fake[i].ncols.size();j++) {
      //cerr << me.fake[i].ncols[j] << " " << me.fake[i].nvals[j] << endl;
      cnt++;
    }
    //cerr << me.fake[i].scontrib << endl;
    //cerr << cnt << endl;
    //cerr << endl;
  }
#else
  for(i=0;i<Rhs.nrows()-2;i++) {
    if (Rhs[i] != 0.0) {
      int cnt=0;
      int starti = mrows[i];
      int endi = mrows[i+1];
      cerr << i << endl;
      for(int j=starti;j<endi;j++) {
	cerr << mcols[j] << " " << mvals[j] << endl;
	cnt++;
      }
      cerr << cnt << endl;
      cerr << endl;
    }
  }
#endif

  int target = 354;

  // We should try to do a better job at preconditioning...
  
  for(i=0;i<size;i++){
    diag[i]=1./matrix->get(i,i);
    if (i == target) {
      //cerr << "Diagnonal: " << diag[i] << endl;
    }
  }

  for(i=0;i<me.fake.size();i++){
    diag[me.fake[i].nodeid] = me.fake[i].diag;
    if (me.fake[i].nodeid == target) {
      //cerr << "Diagnonal: " << diag[target] << endl;
    }
  }

  int flop=0;
  int memref=0;

  double dnrm = diag.vector_norm(flop,memref);

  //cerr << dnrm << " : " << sqrt(dnrm*dnrm + 2) << endl;
  

  PreMultFix(lhs,me); // zeros out fake nodes
  matrix->mult(lhs, R, flop, memref);
  CleanupMMult(lhs,R,me); // fixes R and lhs

  //cerr << "\n\n\n";

#ifdef SPECIAL_CG
  for(i=0;i<me.fake.size();i++)
    if (me.fake[i].nodeid == target) {
      //cerr << me.fake[i].nodeid << " " << R[me.fake[i].nodeid] << endl;
    }
#else
  for(i=0;i<rhsnz.size();i++) 
    if (rhsnz[i] == target)
      cerr << rhsnz[i] << " " << R[rhsnz[i]] << endl;
#endif

  Sub(R, Rhs, R, flop, memref);

  double bnorm=Rhs.vector_norm(flop, memref);

#ifdef SPECIAL_CG
  bnorm = sqrt(bnorm*bnorm + 1000*1000*2); // the sources...
#endif

  //cerr << "Bnorm: " << bnorm << endl;

  PreMultFix(R,me);
  matrix->mult(R, Z, flop, memref);
  CleanupMMult(R,Z,me);
  
  //cerr << "1st Mult: " << Z.vector_norm(flop,memref) << endl;

  //cerr << "\n\n\n";

#ifdef SPECIAL_CG
  for(i=0;i<me.fake.size();i++)
    if (me.fake[i].nodeid == target) {
      //cerr << me.fake[i].nodeid << " " << Z[me.fake[i].nodeid] << endl;
    }
#else
  for(i=0;i<rhsnz.size();i++) 
    if (rhsnz[i] == target)
      cerr << rhsnz[i] << " " << Z[rhsnz[i]] << endl;
#endif

  double bkden=0;
  double err=R.vector_norm(flop, memref)/bnorm; 

  if(err == 0){
    lhs=Rhs;
    cerr << "Zero error?\n";
    return;
  }

  int niter=0;
  int toomany=500; // something else???
  if(toomany == 0)
    toomany=2*size;
  double max_error=0.0000001; // something else???
  
  while(niter < toomany){
    niter++;

    double new_error=0.000001;

    if(err < max_error)
      break;
    
    // Simple Preconditioning...
    Mult(Z, R, diag, flop, memref);	

    // Calculate coefficient bk and direction vectors p and pp
    double bknum=Dot(Z, R, flop, memref);
    
    //cerr << "bknum: " << bknum << endl;

    if(niter==1){
      P=Z;
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, flop, memref);
    }
    bkden=bknum;

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    PreMultFix(P,me);
    matrix->mult(P, Z, flop, memref);
    CleanupMMult(P,Z,me);
    
#ifdef SPECIAL_CG
  for(i=0;i<me.fake.size();i++)
    if (me.fake[i].nodeid == target) {
      //cerr << niter << " " << Z[me.fake[i].nodeid] << endl;
    }
#else
  for(i=0;i<rhsnz.size();i++) 
    if (rhsnz[i] == target)
      cerr << niter << " " << Z[rhsnz[i]] << endl;
#endif

    double akden=Dot(Z, P, flop, memref);
    double ak=bknum/akden;

    //cerr << "Ak stuff: " << akden << " " << ak << endl;

    ScMult_Add(lhs, ak, P, lhs, flop, memref);
    ScMult_Add(R, -ak, Z, R, flop, memref);
    
    err=R.vector_norm(flop, memref)/bnorm;

    //cerr << lhs[target] << " " << err << "\n";
  }
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


inline Point RandomPoint(Element* e)
{
  Point p0(e->mesh->nodes[e->n[0]]->p);
  Point p1(e->mesh->nodes[e->n[1]]->p);
  Point p2(e->mesh->nodes[e->n[2]]->p);
  Point p3(e->mesh->nodes[e->n[3]]->p);
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

void SourceLocalize::build_matrix(Array1<int> &colidx,
				  Array1<int> &mycols,
				  SymSparseRowMatrix *&cmat,
				  ColumnMatrix *&rhs,
				  Mesh *msh)
{
  if (cmat) {
    delete cmat; // you have to free it up...
  }

  int nnodes=msh->nodes.size();
  int *rows=scinew int[nnodes+1];

  int np=1;//Task::nprocessors();
  colidx.resize(np+1);

  int start_node=0; // nnodes*proc/np;
  int end_node=nnodes; // *(proc+1)/np;
  int ndof=end_node-start_node;
  
  int r=start_node;
  int i;
  mycols.resize(0);
  for(i=start_node;i<end_node;i++){
    rows[r++]=mycols.size();
    if(msh->nodes[i]->bc){
      mycols.add(i); // Just a diagonal term
    } else {
      msh->add_node_neighbors(i, mycols);
    }
  }
  colidx[0]=mycols.size();
  int st=0;
  for(i=0;i<np;i++){
    int ns=st+colidx[i];
    colidx[i]=st;
    st=ns;
  }
  colidx[np]=st;
  allcols=scinew int[st];
  
  int s=colidx[0];
  
  int n=mycols.size();
  for(i=0;i<n;i++){
    allcols[i+s]=mycols[i];
  }
  for(i=start_node;i<end_node;i++){
    rows[i]+=s;
  }
  
  rows[nnodes]=st;
  cmat=scinew SymSparseRowMatrix(nnodes, nnodes, rows, allcols, st);
  rhs->resize(nnodes);
  
  double* a=cmat->a;
  for(i=start_node;i<end_node;i++){
    (*rhs)[i]=0;
  }
  int ns=colidx[0];
  int ne=colidx[1];
  for(i=ns;i<ne;i++){
    a[i]=0;
  }
  double lcl_matrix[4][4];
  
  int nelems=msh->elems.size();
  for (i=0; i<nelems; i++){
    Element* e=msh->elems[i];
    if (e) {
      build_local_matrix(msh->elems[i],lcl_matrix,msh);
      add_lcl_gbl(*cmat,lcl_matrix,*rhs,i,msh, start_node, end_node);
    }
  }
  for(i=start_node;i<end_node;i++){
    if(msh->nodes[i]->bc){
      int id=rows[i];
      a[id]=1;
      (*rhs)[i]=msh->nodes[i]->bc->value;
    }
  }
}

void SourceLocalize::execute()
{
  MeshHandle mesh;
  static Mesh* tmpM=0;

  if(!inmesh->get(mesh))
    return;

  if (tmpM ==mesh.get_rep()) {
    cerr << "Same Mesh!\n";
    //return;
  } 
  tmpM = mesh.get_rep();

  MatrixHandle mHandle;

  if (!init) {
    init=1;
    widget=scinew PointWidget(this, &widget_lock, 0.2);
    GeomObj *w=widget->GetWidget();
    ogeom->addObj(w, "Source Location", &widget_lock);
    widget->Connect(ogeom);

    Point p0,p1;

    mesh->get_bounds(p0,p1);

    widget->SetPosition(Interpolate(p0,p1,0.5));
        
    widget->SetScale(20.0);
  }

  if (widgetMoved) {
    widgetMoved=0;
  }

  if (!inmatrix->get(mHandle)) { // build it from the mesh

    this->mesh=mesh.get_rep();

    build_matrix(colidx,mycols,gbl_matrix,rhs,this->mesh);

    Lhs = scinew ColumnMatrix(rhs->nrows());
    Hrhs = Lhs; // assign it...
  } else {
    this->mesh=mesh.get_rep();
    gbl_matrix = mHandle->getSymSparseRow();
    if (init == 1) { // first time through...
      init = 1;
      rhs=scinew ColumnMatrix(mesh->nodes.size());
      Lhs = scinew ColumnMatrix(rhs->nrows());
      Lhs->zero();

      Hrhs = Lhs; // assign it...
    }
  }

  // 1st create a single dipole source

  int elem = drand48()*(mesh->elems.size()-1);
  Point pt = RandomPoint(mesh->elems[elem]);

  ambs.resize(1);
  ambs[0].samples.resize(1); // just do 1 dipole source for now...

  SourceCoefs &testCF = ambs[0].samples[0];

  pt = widget->ReferencePoint();

  testCF.coefs.resize(NUM_DOF);
  testCF.coefs[0] = pt.x();
  testCF.coefs[1] = pt.y();
  testCF.coefs[2] = pt.z();

#if 0
  testCF.coefs[3] = drand48();
  testCF.coefs[4] = drand48(); // random orientation!
#else
  testCF.coefs[3] = 0.5;
  testCF.coefs[4] = 0.5; // random orientation!
#endif
  Point p0;
  Vector v0;

  testCF.CreateDipole(p0,v0);
  
  cerr << "start: " << p0 << " " << v0 << endl;

  int ix=0;
  testCF.isValid = 1;
  if (mesh->locate(p0,ix)) {
    Element *e = mesh->elems[ix]; // this is the element
    
    Point pa,pb;

    pa = p0 + v0*2;
    pb = p0 - v0*2;

    GeomGroup *grp = scinew GeomGroup();
    grp->add(scinew GeomSphere(p0,10.0));

    // now throw in the gradient and the dipole vector...

    GeomLines *gl = scinew GeomLines();

    gl->add(p0,p0 + v0*60);
    grp->add(gl);

    ogeom->addObj(grp,"Source");
    
    // this means you can compute stuff now...
    
  } else { // point is out of the mesh!
    testCF.isValid = 0;
    cerr << "Woah - initial point is bogus????\n";
  }

  InitAmoeba(0,0);

  // now get the boundary points...

  static Array1<int> bpts;

  static int bdry_id=0;

  if (!bdry_id)
    mesh->get_boundary_nodes(bpts);

  double aval=0.0;

#if 1

  GeomPts *Gpts;

  if (!bdry_id)
    Gpts = scinew GeomPts(bpts.size());

  for(int i=0;i<bpts.size();i++) {
    aval += (*ambs[0].lhs)[bpts[i]];
    if (!bdry_id)
      Gpts->add(mesh->nodes[bpts[i]]->p);
  }
  
  cerr << " Average is: " << aval/bpts.size() << endl;

  if (!bdry_id)
    bdry_id  = ogeom->addObj(Gpts,"Boundary Points");
#endif

  // send the "correct" solution
  //rhsoport->send(ColumnMatrixHandle(Lhs));

#if 1
  // now pick some random points
  // and evaluate them in this field...

  double offset = aval/bpts.size();

  // just send this field down the pipe...

  (*Lhs) = (*ambs[0].lhs);

  for(i=0;i<Lhs->nrows();i++) {
    (*Lhs)[i] -= offset;
  }

  rhsoport->send(Hrhs); 
#ifdef SPECIAL_CG
  static MeshHandle rmesh;

  rmesh = ambs[0].local_mesh;

  outmesh->send(rmesh);
#endif
  //return;

  interp_pts.resize(bpts.size()); // just do the "boundaries" of the cube
  interp_elems.resize(interp_pts.size());
  interp_vals.resize(interp_pts.size());
  
  // make them just be nodes for now...

  err_spheres.resize(interp_pts.size());

  GeomGroup *grp = scinew GeomGroup();

  for(i=0;i<interp_pts.size();i++) {
    interp_pts[i] = mesh->nodes[bpts[i]]->p;
    interp_elems[i] = bpts[i];
    
    interp_vals[i] = (*ambs[0].lhs)[bpts[i]] - offset;

    aval += (interp_vals[i]*interp_vals[i]); 

    err_spheres[i] = scinew GeomSphere(interp_pts[i],10.0,6,4);
    grp->add(err_spheres[i]);
  }

  cerr << "Bdry Err: " << aval << endl;

  ogeom->addObj(grp,"Error Spheres");

  ambs.resize(3); // try this for now...

  double minErr=10000;
  
  //lines = 0;
  
  for(i=0;i<ambs.size();i++) {
    InitAmoeba(i); // give each amoeba a random starting point...
    
    DoAmoeba(i); // get it started...
    // ok - now throw the real point in and see what happens
    
    if (ambs[i].samples[0].err < minErr)
      minErr = ambs[i].samples[0].err;
    
    Point tp;
    Vector tv;
    
    ambs[i].samples[0].CreateDipole(tp,tv);
    
    cerr << "Spatial: " << (tp-p0) << endl;
    
    cerr << "Original Orientation: " << v0 << endl;
    cerr << tv << " Orientation: " << 360*acos(Dot(tv,v0))/(2*M_PI) << endl;
    
    cerr << ambs[i].samples[0].coefs[3] << " ";
    cerr << ambs[i].samples[0].coefs[4] << endl;
    
    cerr << ambs[i].samples[0].err << endl;
    
    solport->send(ambs[i].Hlhs);
    
    cerr << minErr << " Finished - don't close me!\n";
  }
  

#endif

#if 1
  // outmatrix->send(MatrixHandle(gbl_matrix));
  solport->send(Hrhs); 
#endif

  // this->mesh=0;
}

// this builds the mesh which is the union of the elements around
// each vertex for every vertex that is part of any element whose
// circumsphere contains either dipole point.  This effectively
// pads the mesh so that any node that would be changed - ie: is
// connected to either dipole node, can be fully computed based on
// this local mesh (the "boundary" elements don't change)

int SourceLocalize::build_local_mesh(Point &p1, Point &p2, // dipole
				     Mesh *&tmesh, 
				     // these remap nodes/elems
				     // into the global space
				     Array1< int > &nodeRemap,
				     BitArray1 &nodesUsed, // nodes used
				     BitArray1 &elemsUsed, // elems used
				     Array1< int > &elem_check)
{
  int start_e1,start_e2;

  if (!mesh->locate(p1,start_e1))
    return 0; // nothing to do...
  start_e2 = start_e1; // pts are close - might even be equal...
  if (!mesh->locate(p2,start_e2))
    return 0;

  // both points are in the mesh...

  nodesUsed.clear_all();
  elemsUsed.clear_all();
  nodeRemap.resize(0); // nothing yet...

  Array1<int> inter_elems; // elems that intersect...

  //tmesh->clear_nodes_elems(); // clear out nodes and elems...

  for(int ii=0;ii<tmesh->nodes.size();ii++) tmesh->nodes[ii] = 0;
  for(ii=0;ii<tmesh->elems.size();ii++)
    if (tmesh->elems[ii]) delete tmesh->elems[ii];
  tmesh->nodes.resize(0);
  tmesh->elems.resize(0);

  elem_check.resize(0);

  elem_check.add(start_e1);

  // see if you need to do
  if (start_e1 != start_e2) {
    elem_check.add(start_e2);
    elemsUsed.set(start_e2); // this guy is used...
    //cerr << "Inside 2 seperate elements!\n";
  }

  int i=0;

  Point cen;
  double rad2;
  double err;

  while(i<elem_check.size()) {
    int cur_ei = elem_check[i]; // element you are working on...

    Element *cur_e = mesh->elems[cur_ei]; // look at this guy
    cur_e->get_sphere2(cen,rad2,err);
    
    int good=0;

    if (((p1-cen).length2() < rad2+err) ||
	((p2-cen).length2() < rad2+err)) {
      good = 1; // do this one...
      //cerr << cur_ei << " Circumsphere intersected!\n";
      inter_elems.add(cur_ei);
    } 
    
    // really means elem has been checked...
    elemsUsed.set(cur_ei); // this guy is used...

    int add_i = tmesh->elems.size();
    
    // ok - add the nodes if neccesary....
    
    int nremap[4];
    
    for(int j=0;j<4;j++) {
      if (nodesUsed.is_set(cur_e->n[j])) { // already there...
	
	nremap[j] = -1;
	for(int match=0;match <  nodeRemap.size();match++) {
	  if (nodeRemap[match] == cur_e->n[j]) {
	    nremap[j] = match;
	    break;
	  }
	}

	if (nremap[j] == -1) {
	  cerr << cur_e->n[j] << " Error in nodesUsed bitmask!\n";
	  nremap[j] = nodeRemap.size();
	  nodeRemap.add(cur_e->n[j]);
	  tmesh->nodes.add(new Node(mesh->nodes[cur_e->n[j]]->p));
	  if (!good) nodesUsed.set(cur_e->n[j]); // won't recurse...
	}
      } else { // set flag in next pass...
	nremap[j] = nodeRemap.size();
	nodeRemap.add(cur_e->n[j]);
	tmesh->nodes.add(new Node(mesh->nodes[cur_e->n[j]]->p));
	if (!good) nodesUsed.set(cur_e->n[j]); // won't recurse...
      }
    }    

    tmesh->elems.add(new Element(tmesh,nremap[0],nremap[1],
				 nremap[2],nremap[3]));
    
    // now loop through these nodes adding there ring of elements...
    // only if this element happend to be "good"
    
    if (good) {
      for(j=0;j<4;j++) {
	//	if (!nodesUsed.is_set(cur_e->n[j])) {
	  nodesUsed.set(cur_e->n[j]);

	  int passed=0;
	  for(int ii=0;ii<nodeRemap.size();ii++) {
	    if (nodeRemap[ii] == cur_e->n[j])
	      passed = 1;
	  }
	  if (!passed) {
	    cerr <<  cur_e->n[j] << " Woah - something wacked!\n";
	  }

	  // now loop through elements
	  Array1<int> &Nelems = mesh->nodes[cur_e->n[j]]->elems;
	  for(int k=0;k<Nelems.size();k++) {
	    if (!elemsUsed.is_set(Nelems[k])) { // stick it on!
	      elemsUsed.set(Nelems[k]);
	      elem_check.add(Nelems[k]);
	    }
	    //  }
	  }
      }
      for(j=0;j<4;j++) {
	if (cur_e->face(j) != -1) {
	  if (!elemsUsed.is_set(cur_e->face(j))) {
	    cerr << cur_e->face(j) << " What happened here?\n";
	    elemsUsed.set(cur_e->face(j));
	    elem_check.add(cur_e->face(j));
	  }
	}
      }
     
    
    } // element intersect a point
    i++; // add the next element to the mesh...
  }

  for(int kk=0;kk<inter_elems.size();kk++) {
    Element *ce = mesh->elems[inter_elems[kk]];
    if (!elemsUsed.is_set(inter_elems[kk])) {
      cerr << inter_elems[kk] << " Is not set!\n";
    }
    for(int j=0;j<4;j++) {
      if (ce->face(j) != -1 &&
	  !elemsUsed.is_set(ce->face(j))) {
	cerr << ce->face(j) << " Is not in mesh!\n";
      }
      if (!nodesUsed.is_set(ce->n[j])) {
	cerr << ce->n[j] << " Node is not there???\n";
      }
    }
  }

  // now the local mesh has been created - however the
  // connectivity is bogus...

#ifdef SPECIAL_CG

  tmesh->compute_neighbors(); // compute on the new mesh...
  tmesh->compute_face_neighbors();

  int p1i = tmesh->nodes.size();

  if (!tmesh->insert_delaunay(p1,0)) {
    cerr << "Woah!  couldn't insert!\n";
    return 0;
  }

  // now that you have inserted that node, make sure everything 
  // is cosher...

  tmesh->compute_neighbors(); // compute on the new mesh...
  tmesh->compute_face_neighbors();

  // now run through all of the elements - if any of them
  // that are within p2 have any boundary faces, we have
  // a problem!

  for(i=0;i<tmesh->elems.size();i++) {
    Element *test_e = tmesh->elems[i];

    if (test_e) {
      test_e->get_sphere2(cen,rad2,err);

      if (((p2-cen).length2() < rad2-err)) {
	// circumsphere intersects!
	int ok=1;
	for(int j=0;j<4;j++) {
	  if (test_e->face(j) < 0) {
	    ok=0;
	    break;
	  }
	}

	if (!ok)
	  cerr << "Woah - bad point after first insertion!\n";

      }
    }
  }
#if 1
  if (!tmesh->insert_delaunay(p2,0)) {
    cerr << "Woah!  couldn't insert! 2\n";
    return 0;
  }
#endif
  // what the heck, make sure everything is build that
  // needs to be - this is cheap compared to CG anyways...
  
  tmesh->compute_neighbors(); // compute on the new mesh...
  tmesh->compute_face_neighbors();

  // set boundary conditions

  p1i = tmesh->nodes.size()-2;
  int p2i = tmesh->nodes.size()-1;

  const double DIPOLE_POTENTIAL = 1000;

  tmesh->nodes[p1i]->bc = new DirichletBC(0,DIPOLE_POTENTIAL);
  tmesh->nodes[p2i]->bc = new DirichletBC(0,-DIPOLE_POTENTIAL);

  //cerr << "LM:" << tmesh->nodes[p1i]->p << " " <<  tmesh->nodes[p2i]->p << endl;

#else

  if (!mesh->insert_delaunay(p1,0)) {
    cerr << "Woah!  couldn't insert!\n";
    return 0;
  }
#if 1
  if (!mesh->insert_delaunay(p2,0)) {
    cerr << "Woah!  couldn't insert! 2\n";
    return 0;
  }
#endif
  // set boundary conditions

  int p1i = mesh->nodes.size()-2;
  int p2i = mesh->nodes.size()-1;

  const double DIPOLE_POTENTIAL = 1000;

  mesh->nodes[p1i]->bc = new DirichletBC(0,DIPOLE_POTENTIAL);
  mesh->nodes[p2i]->bc = new DirichletBC(0,-DIPOLE_POTENTIAL);
#endif

  return 1;
}

// this function fills in the fake row structure for
// the specified "node" of the amoeba.
// is assumes that this procs local data is created
// with respect to which

// it first has to figure out all of the 

void SourceLocalize::build_fake_rows(int proc, int which)
{
  AmoebaData &amb = ambs[proc];
  SourceCoefs &me = amb.samples[which];

  Mesh *msh = ambs[proc].local_mesh;

  BitArray1 &nodesUsed = (*amb.nodesUsed);
  
  Array1< int > &actual_nodes = (amb.elem_check); // double duty...
  Array1< int > &actual_remap = amb.actual_remap;

  build_matrix(amb.colidx,amb.mycols,
	       amb.local_matrix,amb.local_rhs,amb.local_mesh);
  
  nodesUsed.clear_all();

  actual_nodes.resize(0);
  actual_remap.resize(amb.local_mesh->nodes.size());

  // these nodes are broken down into 3 categories...

  int ngood=0;
  int src_idx_start = amb.local_mesh->nodes.size()-2; // CHANGE
  for(int i=0;i<amb.local_mesh->elems.size();i++) {
    int has_src=0;
    Element *test_e = amb.local_mesh->elems[i];
    if (test_e) {
      for(int j=0;j<4;j++) {
	if (test_e->n[j] >= src_idx_start) {
	  has_src++;
	  break; // out of for loop
	}
      }

      if (has_src) { // build array of potential nodes...
	for(int j=0;j<4;j++) {
	  if ((test_e->n[j] < src_idx_start) && // don't add sources...
	      !nodesUsed.is_set(test_e->n[j])) {
	    nodesUsed.set(test_e->n[j]);
	    ngood++;
	    actual_remap[test_e->n[j]] = actual_nodes.size();
	    actual_nodes.add(test_e->n[j]);
	  }
	}
      }
    }
  }

  //cerr << actual_nodes.size() << " Fictious nodes!\n";

  int *mrows = amb.local_matrix->get_row();
  int *mcols = amb.local_matrix->get_col();
  double *mvals = amb.local_matrix->get_val();

  me.fake.resize(actual_nodes.size());

  for(i=0;i<actual_nodes.size();i++) {
    int row_idx = mrows[actual_nodes[i]];
    int next_idx = mrows[actual_nodes[i]+1];

    me.fake[i].cols.resize(0);
    me.fake[i].cvals.resize(0);

    me.fake[i].ncols.resize(0);
    me.fake[i].nvals.resize(0);

    me.fake[i].scontrib = (*amb.local_rhs)[actual_nodes[i]];
    me.fake[i].nodeid = amb.nodeRemap[actual_nodes[i]]; // in global space
    me.fake[i].cval = 0; // could assign to scontrib?  fraction of it?
    me.fake[i].diag = 1.0;
    for(int j=row_idx;j<next_idx;j++) {
      if (mcols[j] == actual_nodes[i]) { // for pre-conditioning
	me.fake[i].diag = 1.0/mvals[j];
      }
      if (mcols[j] >= src_idx_start) { // shouldn't be here!
	cerr << "Coef wi/respect src!\n";
      }
      if (nodesUsed.is_set(mcols[j])) { // reference another fake node
	me.fake[i].cols.add(actual_remap[mcols[j]]); // index into fakeRow's
	me.fake[i].cvals.add(mvals[j]);
      } else { // references a "normal" node
	me.fake[i].ncols.add(amb.nodeRemap[mcols[j]]); // global index
	me.fake[i].nvals.add(mvals[j]);
      }
    }
    //cerr << me.fake[i].cols.size() << " Node Refs\n";
    //cerr << me.fake[i].ncols.size() << " Non-Node Refs\n";

    // check to see if any of these nodes are on the boundary....

    int nbad=0;
    for(int ii=0;ii<msh->nodes[actual_nodes[i]]->elems.size();ii++) {
      Element *check_e = msh->elems[msh->nodes[actual_nodes[i]]->elems[ii]];
      if (check_e) {
	for(j=0;j<4;j++) {
	  if (check_e->n[j] != actual_nodes[i]) {
	    if (check_e->face(j) == -1) {
	      //cerr << amb.nodeRemap[check_e->n[j]] << " Woah - fucked up face!\n";
	      nbad++;
	      break;
	    }
	  }
	}
      } else {
	cerr << amb.nodeRemap[actual_nodes[i]] << "What - bad element!\n";
      }
    }
    //cerr << nbad << " : " <<  amb.nodeRemap[actual_nodes[i]] << endl;
  }


}

void SourceLocalize::build_local_matrix(Element *elem, 
					double lcl_a[4][4],
					Mesh* mesh)
{
  Point pt;
  Vector grad1,grad2,grad3,grad4;
  double vol = mesh->get_grad(elem,pt,grad1,grad2,grad3,grad4);
  if(vol < 1.e-10){
    cerr << "Skipping element..., volume=" << vol << endl;
    for(int i=0;i<4;i++)
      for(int j=0;j<4;j++)
	lcl_a[i][j]=0;
    return;
  }
  

  double el_coefs[4][3];
  // this 4x3 array holds the 3 gradients to be used 
  // as coefficients for each of the four nodes of the 
  // element
  
  el_coefs[0][0]=grad1.x();
  el_coefs[0][1]=grad1.y();
  el_coefs[0][2]=grad1.z();
  
  el_coefs[1][0]=grad2.x();
  el_coefs[1][1]=grad2.y();
  el_coefs[1][2]=grad2.z();

  el_coefs[2][0]=grad3.x();
  el_coefs[2][1]=grad3.y();
  el_coefs[2][2]=grad3.z();

  el_coefs[3][0]=grad4.x();
  el_coefs[3][1]=grad4.y();
  el_coefs[3][2]=grad4.z();

  // cond_tensors are the sigma values for this element.
  // where:
  //  [0] => sigma xx
  //  [1] => sigma xy and sigma yx
  //  [2] => sigma xz and sigma zx
  //  [3] => sigma yy
  //  [4] => sigma yz and sigma zy
  //  [5] => sigma zz

  double el_cond[3][3];
  // in el_cond, the indices tell you the directions
  // the value is refering to. i.e. 0=x, 1=y, and 2=z
  // so el_cond[1][2] is the same as sigma yz
  el_cond[0][0] = mesh->cond_tensors[elem->cond][0];
  el_cond[0][1] = mesh->cond_tensors[elem->cond][1];
  el_cond[1][0] = mesh->cond_tensors[elem->cond][1];
  el_cond[0][2] = mesh->cond_tensors[elem->cond][2];
  el_cond[2][0] = mesh->cond_tensors[elem->cond][2];
  el_cond[1][1] = mesh->cond_tensors[elem->cond][3];
  el_cond[1][2] = mesh->cond_tensors[elem->cond][4];
  el_cond[2][1] = mesh->cond_tensors[elem->cond][4];
  el_cond[2][2] = mesh->cond_tensors[elem->cond][5];

  // build the local matrix
  for(int i=0; i< 4; i++) {
    for(int j=0; j< 4; j++) {
      lcl_a[i][j] = 0.0;
      for (int k=0; k< 3; k++){
	for (int l=0; l<3; l++){
	  lcl_a[i][j] += 
	    el_cond[k][l]*el_coefs[i][k]*el_coefs[j][l];
	}
      }
      lcl_a[i][j] *= vol;
    }
  }
}


void SourceLocalize::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
				 ColumnMatrix& rhs,
				 int el, const Mesh* mesh)
{

  for (int i=0; i<4; i++) // this four should eventually be a
    // variable ascociated with each element that indicates 
    // how many nodes are on that element. it will change with 
    // higher order elements
    {	  
      int ii = mesh->elems[el]->n[i];
      NodeHandle& n1=mesh->nodes[ii];
      if(!n1->bc){
	for (int j=0; j<4; j++) {
	  int jj = mesh->elems[el]->n[j];
	  NodeHandle& n2=mesh->nodes[jj];
	  if(!n2->bc){
	    gbl_a[ii][jj] += lcl_a[i][j];
	  } else {
	    // Eventually look at nodetype...
	    rhs[ii]-=n2->bc->value*lcl_a[i][j];
	  }
	}
      }
    }
}

void SourceLocalize::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
				 ColumnMatrix& rhs,
				 int el, const Mesh* mesh,
				 int s, int e)
{

  for (int i=0; i<4; i++) // this four should eventually be a
    // variable ascociated with each element that indicates 
    // how many nodes are on that element. it will change with 
    // higher order elements
    {	  
      int ii = mesh->elems[el]->n[i];
      if(ii >= s && ii < e){
	NodeHandle& n1=mesh->nodes[ii];
	if(!n1->bc){
	  for (int j=0; j<4; j++) {
	    int jj = mesh->elems[el]->n[j];
	    NodeHandle& n2=mesh->nodes[jj];
	    if(!n2->bc){
	      gbl_a[ii][jj] += lcl_a[i][j];
	    } else {
	      // Eventually look at nodetype...
	      rhs[ii]-=n2->bc->value*lcl_a[i][j];
	    }
	  }
	}
      }
    }
}

void SourceLocalize::widget_moved(int last)
{
  if (last)
    want_to_execute();
}

