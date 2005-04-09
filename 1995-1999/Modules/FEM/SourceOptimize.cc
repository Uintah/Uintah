/*
 *  SourceLocalize.cc:  Builds the global finite element matrix
 *
 *  Written by:
 *   Peter-Pike Sloan
 *   Department of Computer Science
 *   University of Utah
 *   April 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/BitArray1.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/Matrix.h>

#include <Datatypes/KludgeMessage.h>
#include <Datatypes/KludgeMessagePort.h>

#include <Datatypes/ScalarFieldUG.h>

#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Multitask/ITC.h>
#include <Multitask/Task.h>

// this is per-proccesor information
// used for solving 

using sci::Mesh;
using sci::MeshHandle;
using sci::Element;
using sci::Node;
using sci::NodeHandle;
using sci::DirichletBC;

const int NUM_DOF=5;
const int NUM_SAMPLES=100000; // should there be more???
const int NUM_PROC=6; // for now...

const double DIPOLE_POTENTIAL = 1000;

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

struct SourceCoefs2 {
  // this structure represents the coeficients
  // for a given source...

  SourceCoefs2():generation(0) {};

  Array1<double> coefs; // coeficients x,y,z,orientation - etc.

  void CreateDipole(Point &p0, Vector &v);

  double err; // result of evaluating function with these coefs

  double mag; // charge for specified dipole...

  // below is the result of descritizing this source(s)
  
  int isValid; // if the point is a valid potential
               // solution - if a point fails, use the old coefs...

  Array1<double> oldCoefs; // old coeficients - copy back if invalid...

  Array1<FakeRow> fake;  // bogus rows
  Array1<double>  fakeS; // sources - really not needed...

  int generation; // for the given source...

};

void SourceCoefs2::CreateDipole(Point &p0, Vector &v)
{
  double theta,phi;

  p0.x(coefs[0]);
  p0.y(coefs[1]);
  p0.z(coefs[2]);

  theta = coefs[3];// - ((int)coefs[3]);
  phi = coefs[4];// - ((int)coefs[4]);

  double sinphi = sin(phi);

  v.x(cos(theta)*sinphi);
  v.y(sin(theta)*sinphi);
  v.z(cos(phi));

  //cerr << "Dipole: "<< p0 << " " << v << endl;

}

inline double EvaluateDipole(Point &p, Point &d, Vector &v)
{
  Point p0 = d + v; // assume that "d" is 2
  Point p1 = d - v;

  double d0 = (p-p0).length();
  double d1 = (p-p1).length();

  return DIPOLE_POTENTIAL*(1/d0 - 1/d1);
}

struct AmoebaData2 {

  AmoebaData2():generation(0),Cgen(0) { rhs = diag = R = Z = P = 0; }

  Array1< SourceCoefs2 > samples; // N+1 points
  Array1< double > psum;         // prefix sum table for coefs size N
  
  inline void ComputePSUM(); // computes sum for every coef...

  inline void amotry(double fac); // extrapolates from highest point

  void Contract(); // contracts around lowest point...

  int ihi;   // worst sample pt
  int inhi;  // second worst sample pt
  int lowi;  // "best" value

  // these are per-proc data - 

  // error functional has to move to per-proc as well...

  int generation; // for updating stuff...

  int Cgen;

  Array1<int>    interp_pts;
  Array1<double> interp_vals;

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

  Array1<int> colidx;
  Array1<int> mycols;
};

inline void AmoebaData2::ComputePSUM()
{
  psum.resize(NUM_DOF);
  for(int i=0;i<NUM_DOF;i++) {
    psum[i] = 0.0;
    for(int j=0;j<samples.size();j++)
      psum[i] += samples[j].coefs[i];
  }
}



void AmoebaData2::Contract()
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

class SimplexManager;
class AmoebaWorker;

class SourceOptimize : public Module {

  friend class SimplexManager;
  friend class AmoebaWorker;

  MeshIPort            *inmesh;
  MatrixIPort          *inmatrix;

  KludgeMessageIPort   *inkludge; // messages - about source and junk like that

  AmoebaMessageOPort   *outamoeba; // incremental updates
  AmoebaMessageOPort   *outbest;   // the best solution to date

  int init;

  KludgeMessage *lastKludge;

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

  void InitAmoeba(int proc);
  void DoAmoeba(int proc); 
  void AmoebaFinished(int proc); // flushes stuff...

  void AmoebaUpdate(int proc,int which=-1);   // tries to update stuff...
  
  void ComputeSamples(int proc); // could use multiple threads here...
  void ComputeSample(int proc,int which); // could use multiple threads here...

  void ConfigureSample(int proc, int which);

  double SolveStep(int proc, double fac);

  Array1< AmoebaData2 > ambs; // per-proc stuff....

  Mutex AmoebaInfo;          // lock for data that is sent
  int   do_execute;
  Array1< AmoebaRecord > amoebers; // data to be sent..

  AmoebaRecord           best_source; // the best guy to date...

  int ship_it;                     // set if you want data shipped
  int new_best;

  Semaphore NewData;         // manager thread chomps on this

  Semaphore SendSomething;   // infinite loop for main thread...

  CrowdMonitor dataLock;     // lock for new data...

  int curGen;                // for updating

  Mutex workQLock;           // lock for workQ
  
  Semaphore workS;           // semaphore for work stuff...

  int numTrials;             // total number of starts...

  void GetWork(int proc);    // fills in info for this proc

  Mutex updateLock;          // for updating amoeba records
  

  ScalarFieldUG *sug; // just used for random point stuff...

  Mesh* mesh;
  SymSparseRowMatrix* gbl_matrix;

  double OptimizeInterp(int proc,
			Array1<double> &trial,double &mag); // returns error...

  Array1<Point>  interp_pts;   // interpolation points
  Array1<int>    interp_elems; // elements for interpolation...
  Array1<double> interp_vals;  // values at interpolated points...

  Array1<double> next_interp_vals; // force a transistion
  Array1<int>    next_interp_pts;  // point indeces - if size == above, single nodes
  Array1<double> next_interp_wts;  // weights for above

  Array1<int>    bdry_nodes;   // boundary nodes for given mesh...

  void SolveMatrix(); // this does the matrix solve 
                      // sources and svals must be set up.

  void SolveMatrix(AmoebaData2& amb, int idx,ColumnMatrix &lhs); 
  // you have to compute all of the discrete stuff every time the above
  // function is called...

  inline void CleanupMMult(ColumnMatrix &x,
			   ColumnMatrix &b,
			   SourceCoefs2 &me);

  inline void PreMultFix(ColumnMatrix &x,
			 SourceCoefs2 &me);

public:
  SourceOptimize(const clString& id);
  SourceOptimize(const SourceOptimize&, int deep);
  virtual void widget_moved(int last);    
  virtual ~SourceOptimize();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
  Module* make_SourceOptimize(const clString& id)
    {
      return scinew SourceOptimize(id);
    }
};


SourceOptimize::SourceOptimize(const clString& id)
: Module("SourceOptimize", id, Filter),NewData(0),workS(0),
  lastKludge(0),curGen(0),numTrials(0),SendSomething(0)
{
  // Create the input ports
  inmesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
  add_iport(inmesh);

  inmatrix = scinew MatrixIPort(this,"Matrix",MatrixIPort::Atomic);
  add_iport(inmatrix);

  inkludge = scinew KludgeMessageIPort(this, "Kludge Message", KludgeMessageIPort::Atomic);
  add_iport(inkludge);

  // create output ports

  outamoeba = scinew AmoebaMessageOPort(this,"Amoeba Output", AmoebaMessageIPort::Atomic);
  add_oport(outamoeba);

  outbest = scinew  AmoebaMessageOPort(this,"Best Amoeba Output", AmoebaMessageIPort::Atomic);
  add_oport(outbest);
  init = 0;
  
}

SourceOptimize::SourceOptimize(const SourceOptimize& copy, int deep)
: Module(copy, deep),NewData(0),workS(0),SendSomething(0)
{
  NOT_FINISHED("SourceOptimize::SourceOptimize");
}

SourceOptimize::~SourceOptimize()
{
}

Module* SourceOptimize::clone(int deep)
{
  return scinew SourceOptimize(*this, deep);
}

double SourceOptimize::SolveStep(int proc, double fac)
{
  AmoebaData2 &amb = ambs[proc];

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
      //cerr << "Out of range!\n";
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

void SourceOptimize::ConfigureSample(int proc, int which)
{
  AmoebaData2 &amb = ambs[proc];
  
  // for now we are just doing point sources...
  
  // wait - this has to be a dipole...
  
  Point p0;
  Vector v0;
  
  amb.samples[which].CreateDipole(p0,v0); // generate the dipole...

  Point p1 = p0 + v0*4;
  Point p2 = p0 - v0*4; // larger baseline???
  
  //p2 = p1; // CHANGE

  amb.samples[which].isValid = 1;

  if (build_local_mesh(p1,p2,
		       amb.local_mesh,
		       amb.nodeRemap,
		       (*amb.nodesUsed),
		       (*amb.elemsUsed),
		       amb.elem_check)) { // valid dipole
    build_fake_rows(proc,which);
  } else { // point is out of the mesh!
    amb.samples[which].isValid = 0;
    //cerr << p1 << " " << p2 << endl;
  }
}

void SourceOptimize::ComputeSamples(int proc)
{
  AmoebaData2 &amb = ambs[proc];
  
  for(int i=0;i<amb.samples.size();i++) {
    if (!amb.samples[i].isValid) {
      cerr << "Woah - bad initial sample!\n";
    } else { // compute this puppy!
      ComputeSample(proc,i);
      //cerr << i << " " << amb.samples[i].err << endl;
    }
  }
}

void SourceOptimize::DoAmoeba(int proc)
{
  AmoebaData2 &amb = ambs[proc];
  //cerr << proc << " Doing Amoeba!\n";

  const int NMAX=110 + drand48()*50; // 100 function evaluations max...

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
      SourceCoefs2 tmpC = amb.samples[amb.lowi];
      amb.samples[amb.lowi] = amb.samples[0];
      amb.samples[0] = tmpC; // save the best one...
      return;
    }

    if (nfunk >= NMAX) {
      //cerr << rtol << " Done max number of function calls!\n";
      SourceCoefs2 tmpC = amb.samples[amb.lowi];
      amb.samples[amb.lowi] = amb.samples[0];
      amb.samples[0] = tmpC; // save the best one...
      //cerr << amb.samples[0].err << endl;
      
      return;
    }

    nfunk += 2;

    double ytry = SolveStep(proc,-1.0); // try and extrapolate through high pt

    int didupdate=0;

    if (ytry <= amb.samples[amb.lowi].err) { // try another contraction
      //cerr << amb.samples[amb.lowi].err << " Contracting again!\n";
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
	//cerr << ytry << " : " << ysave << endl;
	AmoebaUpdate(proc);
	didupdate=1;
      }
    } else {
      nfunk--;
    }
    if (!didupdate) {
      AmoebaUpdate(proc,amb.inhi);
    }

  }
}

// this guy assumes that the sample is actualy valid!

void SourceOptimize::ComputeSample(int proc, int which)
{
  AmoebaData2 &amb = ambs[proc];

  SolveMatrix(amb,which,(*(amb.lhs)));

  // now that the matrix has been solved, compute the error...

  Array1<double> trial(amb.interp_vals.size());

  Point p0;
  Vector v0;

  amb.samples[which].CreateDipole(p0,v0);

  // 1st compute the "offset"

  double offset=0;

  for(int i=0;i<bdry_nodes.size();i++) {
#if 0
    Element *e = mesh->elems[interp_elems[i]];
    double a[4];
    mesh->get_interp(e,interp_pts[i],a[0],a[1],a[2],a[3]);

    trial[i] = 0.0;
    for(int j=0;j<4;j++) {
      trial[i] += (*amb.lhs)[e->n[j]]*a[j];
    }
#else
    offset += (*amb.lhs)[bdry_nodes[i]]; // just boundary points
#endif
  }

  offset /= bdry_nodes.size(); // make sure this is correct...

  //cerr << offset << endl;

  for(i=0;i<trial.size();i++) {
    trial[i] = (*amb.lhs)[amb.interp_pts[i]] - offset;
    //cerr << trial[i] << endl;
  }

  // fix up the entire lhs!

  if (trial.size())
    for(i=0;i<(*amb.lhs).nrows();i++) {
      (*amb.lhs)[i] -= offset;
    }

  amb.samples[which].err = OptimizeInterp(proc,trial,amb.samples[which].mag);
}

void SourceOptimize::InitAmoeba(int proc)
{
  AmoebaData2 &amb = ambs[proc];

  int csize = gbl_matrix->nrows();
 
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

    amb.nodesUsed = new BitArray1(mesh->nodes.size(),0);
    amb.elemsUsed = new BitArray1(mesh->elems.size(),0);
    
    amb.local_mesh = new Mesh(0,0); // copy conductivity tensors!

    amb.local_mesh->cond_tensors = mesh->cond_tensors;

    cerr << amb.local_mesh->cond_tensors.size() << endl;

    for(int i=0;i<mesh->cond_tensors.size();i++) {
      for(int j=0;j<mesh->cond_tensors[i].size();j++) {
	cerr << mesh->cond_tensors[i][j] << endl;
      }
    }
    
    cerr << "Now for mine!\n";

    for(i=0;i<amb.local_mesh->cond_tensors.size();i++) {
      for(int j=0;j<amb.local_mesh->cond_tensors[i].size();j++) {
	cerr << amb.local_mesh->cond_tensors[i][j] << endl;
      }
    }

    amb.local_matrix = 0;
    amb.local_rhs = scinew ColumnMatrix(500); // how many nodes?

    amb.nodeRemap.resize(500);
    amb.elem_check.resize(500);
    amb.actual_remap.resize(500);
  }

  for(int i=0;i<amb.samples.size();i++) {

    //cerr << "Configuring a sample!\n";

    ConfigureSample(proc,i); // set up the neccesary stuff...
    while (!amb.samples[i].isValid) { // re randomize this...
      Point trypt = sug->samples[drand48()*(sug->samples.size()-0.5)].loc;
      amb.samples[i].coefs[0] = trypt.x();
      amb.samples[i].coefs[1] = trypt.y();
      amb.samples[i].coefs[2] = trypt.z();
      ConfigureSample(proc,i); // set up the neccesary stuff...
    } 
    ComputeSample(proc,i);
  }
}

double SourceOptimize::OptimizeInterp(int proc,
				      Array1<double> &trial,double &mag)
{
  
  AmoebaData2 &amb = ambs[proc];
  
  // trial has been computed based on interp_pts/elems...

  double pisi=0.0;
  double si2=0.0;

  for(int i=0;i<amb.interp_vals.size();i++) {
    pisi += amb.interp_vals[i]*trial[i];
    si2 += trial[i]*trial[i];

    //cerr << i << ":" << amb.interp_vals[i] << " - " << trial[i] << endl;

  }

  mag = pisi/si2; // don't allow the dipole to flip...
#if 0
  if (mag < 0) 
    mag = -1.0;
  else
    mag = 1.0;
#endif

  double err=0;

  for(i=0;i<amb.interp_vals.size();i++) {
    double crr = (amb.interp_vals[i] - trial[i]*mag)*
      (amb.interp_vals[i] - trial[i]*mag);
    err += crr;
  }
  
  //cerr << mag << " " << err << endl;

  return err;
}

//#define PETE_DO_DEBUG 0

inline void SourceOptimize::CleanupMMult(ColumnMatrix &x,
					 ColumnMatrix &b,
					 SourceCoefs2 &me)
{
  // assume that the values in x for these nodes
  // have been zeroed before the multiply - make
  // sure b is correct...

  for(int i=0;i<me.fake.size();i++) {
#ifdef PETE_DO_DEBUG
    if (x[me.fake[i].nodeid] != 0.0) {
      cerr << "Bad Iter: " << x[me.fake[i].nodeid] << endl;
    }
#endif

    x[me.fake[i].nodeid] = me.fake[i].cval; // was zero'd before...

    b[me.fake[i].nodeid] = 0.0; // compute this
  }
  for(i=0;i<me.fake.size();i++) {
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

inline void SourceOptimize::PreMultFix(ColumnMatrix &x,
				       SourceCoefs2 &me)
{
  // zero out appropriate parts of x,
  // copy x's values into "cval"

  for(int i=0;i<me.fake.size();i++) {
    me.fake[i].cval = x[me.fake[i].nodeid];
    x[me.fake[i].nodeid] = 0;
  }
}

void SourceOptimize::SolveMatrix(AmoebaData2& amb, int idx, ColumnMatrix &lhs)
{
  Matrix *matrix = (Matrix*)gbl_matrix;
  ColumnMatrix &Rhs = (*(amb.rhs));
  ColumnMatrix &diag = (*(amb.diag));
  ColumnMatrix &R = (*(amb.R));
  ColumnMatrix &Z = (*(amb.Z));
  ColumnMatrix &P = (*(amb.P));

  SourceCoefs2 &me = amb.samples[idx];

  int i;

  // assume that lhs is properly initialized...

  Rhs.zero(); 

  // zero out portion of lhs...

  for(i=0;i<me.fake.size();i++) {
    Rhs[me.fake[i].nodeid] = me.fake[i].scontrib;
    //cerr << me.fake[i].nodeid << " " << me.fake[i].scontrib << endl;
  }
  lhs.zero(); // will this help any?

  int *mrows = matrix->get_row();
  int *mcols = matrix->get_col();

  double *mvals = matrix->get_val();

  int size=matrix->nrows();

  // We should try to do a better job at preconditioning...
  
  for(i=0;i<size;i++){
    diag[i]=1./matrix->get(i,i);
  }

  for(i=0;i<me.fake.size();i++){
    diag[me.fake[i].nodeid] = me.fake[i].diag;
    //cerr << "Fake Diag: " << me.fake[i].nodeid << " " << me.fake[i].diag;
    //cerr << " " << me.fake[i].scontrib << endl;
  }

  int flop=0;
  int memref=0;

  //cerr << "Diag: " << diag.vector_norm(flop,memref) <<endl;

  PreMultFix(lhs,me); // zeros out fake nodes
  matrix->mult(lhs, R, flop, memref);
  CleanupMMult(lhs,R,me); // fixes R and lhs

  Sub(R, Rhs, R, flop, memref);

  double bnorm=Rhs.vector_norm(flop, memref);

  // you have to add in the influence of the sources...

  bnorm = sqrt(bnorm*bnorm + DIPOLE_POTENTIAL*DIPOLE_POTENTIAL*2);

  PreMultFix(R,me);
  matrix->mult(R, Z, flop, memref);
  CleanupMMult(R,Z,me);
  
  double bkden=0;
  double err=R.vector_norm(flop, memref)/bnorm; 

  if(err == 0){
    lhs=Rhs;
    cerr << "Zero error?\n";
    return;
  }

  //cerr << err << " bnorm: " << bnorm <<endl;

  int niter=0;
  int toomany=500; // something else???
  if(toomany == 0)
    toomany=2*size;
  double max_error=0.0000001; // something else???
  
  while(niter < toomany){
    niter++;

    if(err < max_error)
      break;
    
    // Simple Preconditioning...
    Mult(Z, R, diag, flop, memref);	

    // Calculate coefficient bk and direction vectors p and pp
    double bknum=Dot(Z, R, flop, memref);
    
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
    
    double akden=Dot(Z, P, flop, memref);
    double ak=bknum/akden;

    ScMult_Add(lhs, ak, P, lhs, flop, memref);

    if ((niter%50) == 0) { // fix it back again...
      
      PreMultFix(lhs,me); // zeros out fake nodes
      matrix->mult(lhs, R, flop, memref);
      CleanupMMult(lhs,R,me); // fixes R and lhs
      
      Sub(R, Rhs, R, flop, memref);

    } else { // just incriment it...
      ScMult_Add(R, -ak, Z, R, flop, memref);
    }
    
    err=R.vector_norm(flop, memref)/bnorm;

#ifdef PETE_DO_DEBUG
    if ((niter %10) == 0) {
      //cerr << lhs.vector_norm(flop,memref) << " ";
      cerr << niter << " " << err << endl;
    }
#endif
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

void SourceOptimize::build_matrix(Array1<int> &colidx,
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
  int *allcols=scinew int[st];
  
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

class SimplexManager : public Task {
public:
  SourceOptimize *owner;
  virtual int body(int);
  SimplexManager(SourceOptimize *, int);
};

SimplexManager::SimplexManager(SourceOptimize *o, int)
  :Task("Simplex Manager"),owner(o)
{

}

int SimplexManager::body(int)
{
  while(1) {
    owner->NewData.down(); // block on getting data...

    // this means stuff will change...

    owner->dataLock.write_lock();

#if 0
    if (!owner->curGen) { //1st time - build source!
      owner->ambs[0].samples[0].coefs[0] = 250;
      owner->ambs[0].samples[0].coefs[1] = 250;
      owner->ambs[0].samples[0].coefs[2] = 250;

      owner->ambs[0].samples[0].coefs[3] = drand48()*2*M_PI;
      owner->ambs[0].samples[0].coefs[4] = drand48()*M_PI;
      
      owner->InitAmoeba(0);
      //owner->ComputeSample(0,0);

      // this gives you the source to work with!

      owner->next_interp_pts.resize(owner->bdry_nodes.size());
      owner->next_interp_vals.resize(owner->bdry_nodes.size());

      double berr=0;
      double offset=0;
      for(int i=0;i<owner->bdry_nodes.size();i++) {
	owner->next_interp_pts[i] = owner->bdry_nodes[i];
	offset += (*(owner->ambs[0].lhs))[owner->bdry_nodes[i]];
      }
      offset /= owner->bdry_nodes.size();
      //cerr << offset<< " Current Offset!\n";
      for(i=0;i<owner->bdry_nodes.size();i++) {
	(*(owner->ambs[0].lhs))[owner->bdry_nodes[i]] -= offset;
	owner->next_interp_vals[i] = (*(owner->ambs[0].lhs))[owner->bdry_nodes[i]];
	berr += owner->next_interp_vals[i]*owner->next_interp_vals[i];
      }
      //cerr << "New BOundary: " << berr << endl;

    }
#endif
    owner->curGen++;

    if (owner->next_interp_pts.size()) {
      
      owner->interp_elems = owner->next_interp_pts;
      owner->interp_vals = owner->next_interp_vals;
      //owner->interp_pts = owner->next_interp_pts;
      owner->next_interp_pts.resize(0);

      // let them fire away on this...

      cerr << "We have data!\n";

      owner->workQLock.lock(); // set this stuff up...
      owner->numTrials = 500; // something small for now
      owner->workQLock.unlock();

      for(int i=0;i<owner->ambs.size();i++)
	owner->workS.up(); // wake everybody up...
    }

    owner->dataLock.write_unlock();
  }
}

class AmoebaWorker : public Task {
public:
  SourceOptimize *owner;
  virtual int body(int);
  AmoebaWorker(SourceOptimize *, int);
};

AmoebaWorker::AmoebaWorker(SourceOptimize *o, int proc)
  :Task("Amobea Worker"),owner(o)
{

}

int AmoebaWorker::body(int proc)
{
  while(1) { // always try and grab stuff...
    owner->GetWork(proc);
    
    owner->InitAmoeba(proc);
    //cerr << proc << " Finished Init!\n";

    owner->AmoebaUpdate(proc); // do everybody...
    owner->DoAmoeba(proc);

    owner->AmoebaFinished(proc); // do its thing...
  }
}

void SourceOptimize::GetWork(int proc)
{
  workQLock.lock();

  while (numTrials <= 0) {
    workQLock.unlock();
    workS.down(); // sleep until you have something...
    workQLock.lock(); // grab lock and see if there is something...
  }
  
  numTrials--; // grab one...
  
  workQLock.unlock();

  dataLock.read_lock(); // make sure you are synchronized...

  if (curGen != ambs[proc].Cgen) {

    //cerr << "Got work!\n";

    ambs[proc].interp_pts = interp_elems;
    ambs[proc].interp_vals = interp_vals;
    ambs[proc].Cgen = curGen;
  }

  // 1st check 

  // ok - set up random values for this proc...
  
  for(int i=0;i<ambs[proc].samples.size();i++) {
    Point pt = sug->samples[drand48()*(sug->samples.size()-0.5)].loc;

    ambs[proc].samples[i].coefs[0] = pt.x();
    ambs[proc].samples[i].coefs[1] = pt.y();
    ambs[proc].samples[i].coefs[2] = pt.z();
    

    for(int j=3;j<NUM_DOF;j++) {
      ambs[proc].samples[i].coefs[j] =drand48()*2*M_PI;
    }
    ambs[proc].samples[i].oldCoefs = ambs[proc].samples[i].coefs;
  }

  dataLock.read_unlock();
  
}

// this allows you to cause the module
// to be fired off again to look at intermediate
// steps.  It also allows you to

void SourceOptimize::AmoebaUpdate(int proc, int which)
{
  AmoebaInfo.lock();
  ship_it++;

  if (which == -1) { // just do them all...
    
    for(int i=1;i<ambs[proc].samples.size();i++) {
      Point pt(ambs[proc].samples[i].coefs[0],
	       ambs[proc].samples[i].coefs[1],
	       ambs[proc].samples[i].coefs[2]);
      amoebers[proc].sources[i].loc = pt;
      amoebers[proc].sources[i].theta = ambs[proc].samples[i].coefs[3];
      amoebers[proc].sources[i].phi = ambs[proc].samples[i].coefs[4];
      
      amoebers[proc].sources[i].v = ambs[proc].samples[i].mag;
      amoebers[proc].sources[i].err = ambs[proc].samples[i].err;
    }
  } else if (proc != 0) {
    Point pt(ambs[proc].samples[which].coefs[0],
	     ambs[proc].samples[which].coefs[1],
	     ambs[proc].samples[which].coefs[2]);
    amoebers[proc].sources[which].loc = pt;
    amoebers[proc].sources[which].theta = ambs[proc].samples[which].coefs[3];
    amoebers[proc].sources[which].phi = ambs[proc].samples[which].coefs[4];
    
    amoebers[proc].sources[which].v = ambs[proc].samples[which].mag*DIPOLE_POTENTIAL;
    amoebers[proc].sources[which].err = ambs[proc].samples[which].err;

    // check for the "best"

    
    if (amoebers[proc].sources[which].err < best_source.sources[0].err) {
      best_source.sources[0] = amoebers[proc].sources[which];
      new_best++;
      cerr <<  ambs[proc].samples[which].mag << " " << amoebers[proc].sources[which].err << " Best!\n";
    }
  }
  
  //cerr << ship_it << " " << do_execute << " Did update!\n";

  amoebers[proc].generation++; // bump this up...

  if (!do_execute && proc) {
    do_execute = 1;
    //cerr << "Want to execute!\n";
    //want_to_execute(); // compressed stuff would make a lot of sense...
    SendSomething.up();
  }
  
  AmoebaInfo.unlock();
}

void SourceOptimize::AmoebaFinished(int proc)
{
  //ok - bump the generation!
  ambs[proc].generation++;

  AmoebaInfo.lock();

  if (amoebers[proc].sources[0].err < best_source.sources[0].err) {
    best_source.sources[0] = amoebers[proc].sources[0];
    new_best++;
  }
  if (!do_execute) {
    do_execute = 1;
    //want_to_execute(); // compressed stuff would make a lot of sense...
    SendSomething.up();
  }
  
  AmoebaInfo.unlock();
}

void SourceOptimize::execute()
{
  MeshHandle mesh;

  if(!inmesh->get(mesh)) {
    return;
  }

  if (!bdry_nodes.size()) { // see if these need to be built
    this->mesh = mesh.get_rep();
    if (this->mesh) {
      this->mesh->get_boundary_nodes(bdry_nodes);
    }
  }

  MatrixHandle mHandle;

  if (!inmatrix->get(mHandle)) { // build it from the mesh
    cerr << "Bailing - matrix\n";
    return; // you need a matrix to do anything...
  } else {
    if (init == 0) { // first time through...
      init = 2;
      this->mesh=mesh.get_rep();
      gbl_matrix = mHandle->getSymSparseRow();
    }
  }

  if (init == 2) { // start up stuff...
    init=1;

    ship_it=0;
    new_best=0;
    do_execute=0;

    curGen=0;

    ambs.resize(NUM_PROC+1); // ready for multiple threads...
    amoebers.resize(NUM_PROC+1);

    best_source.sources.resize(1);
    best_source.sources[0].err = 1000000000000; 

    // create the ScalarField...

    sug = scinew ScalarFieldUG(mesh,ScalarFieldUG::NodalValues);

    sug->compute_samples(NUM_SAMPLES);
    sug->distribute_samples();

    // also start the "manager" thread
    
    SimplexManager *manager = scinew SimplexManager(this,0);
    manager->activate(1);

    // and the amb threads...

    for(int i=0;i<ambs.size();i++) {
      if (i)
	ambs[i].samples.resize(NUM_DOF+1);
      else 
	ambs[i].samples.resize(1); // 1st guy!
      amoebers[i].sources.resize(NUM_DOF+1);
      amoebers[i].generation = 0;
      for(int j=0;j<ambs[i].samples.size();j++) {
	ambs[i].samples[j].coefs.resize(NUM_DOF);
      }
    }
    //NewData.up();
    for(i=1;i<ambs.size();i++) {
      AmoebaWorker *awork = scinew AmoebaWorker(this,i);
      awork->activate(i);
    }

  }

  KludgeMessageHandle khandle;

  int send_it=0;

  if (inkludge->get(khandle)) { // some work to do???
    // just look for source information for now...


    if (lastKludge != khandle.get_rep() &&
	khandle->surf_pots.size()) {
      dataLock.write_lock();
      next_interp_vals = khandle->surf_pots;
      next_interp_pts = khandle->surf_pti;
      next_interp_wts = khandle->surf_wts;

      cerr << khandle->surf_pti.size() << endl;

      dataLock.write_unlock();

      NewData.up(); // let the threads get to work on this...
      cerr << "NewData?\n";
      lastKludge = khandle.get_rep();
      //return; // any data we would have is probably bogus..
    }

    if (lastKludge != khandle.get_rep() &&
	(khandle->surf_pots.size() == 0) &&
	(khandle->src_mag.size() == 0) &&
	(khandle->src_recs.size() == 0)) {
      //send_it = 1; 
    }

    if (khandle->src_mag.size()) { // re-seed the random starting points...

    }
    lastKludge = khandle.get_rep();
  }

  int bestInc=100;

  double lastbest=8000;

  while(1) {
    SendSomething.down();

    // if we get here, see if we need to send
    // stuff...

    AmoebaInfo.lock();

    if (new_best >= 25) { // see
      new_best=0;
      AmoebaMessage *camb = scinew  AmoebaMessage();
      camb->amoebas.resize(1); // this is just a single record
      camb->amoebas[0].sources.resize(1);
      camb->amoebas[0].sources[0] = best_source.sources[0];
    
      camb->generation=-1;

      cerr << "Shipping Best!\n";

      outbest->send_intermediate(AmoebaMessageHandle(camb));
    }
    if (ship_it || send_it) { 
      AmoebaMessage *camb = new  AmoebaMessage();
      //camb->amoebas = amoebers;
      camb->amoebas.resize(amoebers.size()-1); // don't send 0!
#if 1
      for(int ii=1;ii<amoebers.size();ii++) {
	camb->amoebas[ii-1].sources.resize(amoebers[ii].sources.size());
	for(int jj=0;jj<amoebers[ii].sources.size();jj++)
	  camb->amoebas[ii-1].sources[jj] = amoebers[ii].sources[jj];
	camb->amoebas[ii-1].generation = amoebers[ii].generation;
      }
#endif    

      AmoebaMessageHandle amh = camb;

      outamoeba->send_intermediate(amh); // ship this stuff off...
      ship_it=0;
      if (new_best > 0) { // something is waiting...
	if (bestInc) {
	  --bestInc;
	} else {
	  bestInc = 60;
	  ++new_best;
	}

	if (best_source.sources[0].err < lastbest) {
	  lastbest = best_source.sources[0].err*0.7; // shrink it some
	  new_best += 40;
	}

      }

    }
    do_execute = 0; // clear it...
    AmoebaInfo.unlock();
  }
}

// this builds the mesh which is the union of the elements around
// each vertex for every vertex that is part of any element whose
// circumsphere contains either dipole point.  This effectively
// pads the mesh so that any node that would be changed - ie: is
// connected to either dipole node, can be fully computed based on
// this local mesh (the "boundary" elements don't change)

int SourceOptimize::build_local_mesh(Point &p1, Point &p2, // dipole
				     Mesh *&tmesh, 
				     // these remap nodes/elems
				     // into the global space
				     Array1< int > &nodeRemap,
				     BitArray1 &nodesUsed, // nodes used
				     BitArray1 &elemsUsed, // elems used
				     Array1< int > &elem_check)
{
  int start_e1,start_e2;

  if (!mesh->locate(p1,start_e1)) {
    if (!mesh->locate2(p1,start_e1)) {
      //cerr << p1 << " Outside!\n";
      return 0; // nothing to do...
    }
  }
  start_e2 = start_e1; // pts are close - might even be equal...
  if (!mesh->locate(p2,start_e2)) {
    if (!mesh->locate2(p2,start_e2)) {
      //cerr << p2 << " Outside!\n";
      return 0; // nothing to do...
    }
  }

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
	  cerr << cur_e->n[j] << " " << mesh->nodes[ cur_e->n[j]]->p << " Error in nodesUsed bitmask!\n";
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

    Element *ne = new Element(tmesh,nremap[0],nremap[1],
			      nremap[2],nremap[3]);

    ne->cond = cur_e->cond;

    tmesh->elems.add(ne);
    
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
#if 0
      for(j=0;j<4;j++) {
	if (cur_e->face(j) != -1) {
	  if (!elemsUsed.is_set(cur_e->face(j))) {
	    cerr << cur_e->face(j) << " What happened here?\n";
	    elemsUsed.set(cur_e->face(j));
	    elem_check.add(cur_e->face(j));
	  }
	}
      }
#endif     
    
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
#if 0
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
#endif
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

  tmesh->nodes[p1i]->bc = new DirichletBC(0,DIPOLE_POTENTIAL);
  tmesh->nodes[p2i]->bc = new DirichletBC(0,-DIPOLE_POTENTIAL);

  //cerr << "LM:" << tmesh->nodes[p1i]->p << " " <<  tmesh->nodes[p2i]->p << endl;

  return 1;
}

// this function fills in the fake row structure for
// the specified "node" of the amoeba.
// is assumes that this procs local data is created
// with respect to which

// it first has to figure out all of the 

void SourceOptimize::build_fake_rows(int proc, int which)
{
  AmoebaData2 &amb = ambs[proc];
  SourceCoefs2 &me = amb.samples[which];

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
#if 0
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
#endif
  }


}

void SourceOptimize::build_local_matrix(Element *elem, 
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

  int nzero=0;

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
      if (lcl_a[i][j] == 0.0)
	nzero++;
    }
  }
#if 0
  if (nzero == 16)
    cerr << "Error in local matrix!\n";
#endif
}


void SourceOptimize::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
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

void SourceOptimize::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
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
	      //cerr << n2->bc->value << " - " << lcl_a[i][j] << endl;
	    }
	  }
	}
      }
    }
}

void SourceOptimize::widget_moved(int last)
{
  if (last)
    want_to_execute();
}

