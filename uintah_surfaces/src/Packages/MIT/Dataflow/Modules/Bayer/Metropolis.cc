/*
 *  Metropolis.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Jun-27-2001
 *
 */

#include <stdlib.h>
#include <math.h>

extern "C" {
#include <unuran.h>

#include <cvode/llnltyps.h>
#include <cvode/cvode.h>
#include <cvode/cvdense.h>
#include <cvode/nvector.h>
#include <cvode/cvdense.h>
}


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Thread/Thread.h>
//#include <Core/Algorithms/DataIO/GuiFile.h>
#include <Core/2d/LockedPolyline.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Graph.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

#include <Packages/MIT/Dataflow/Modules/Bayer/Bayer.h>

#include <Packages/MIT/share/share.h>

extern "C" {
  void dpotrf_( char &, int &, double *, int &, int &  ); 
  void bayer_( double &, double *, double *);
  void zufalli_( int &);
  void zufall_( int& , double *);
  void normalen_(int&, double *);
}
  
namespace MIT {

using namespace SCIRun;

FILE *rlog;
double min( double a, double b ) { return a<b ? a : b; }
double sqr( double x ) { return x*x; }
void swap( double * &a, double * &b) { double *t = a; a = b; b = t; }

class MITSHARE Metropolis : public Module {
private:
  // input
  MeasurementsIPort *m_port;
  DistributionIPort *d_port;
  ResultsOPort *r_port;

  MeasurementsHandle measurements;
  DistributionHandle pd;

  GuiInt gui_burning, gui_monitor, gui_thin;
  GuiDouble gui_kappa;
  GuiInt gui_use_cvode;

  Graph *graph;
  CrowdMonitor *monitor_;

  bool user_ready;

  // theta_init;
  int seed;
  double kappa;
  int burning;
  int monitor;
  int thin;
  int iterations;

  double mu0, eta0;
  double sigma_mu, sigma_eta;

  int nparms;
  int n_observations;

  double *theta;
  double *star;

  Array2<double> lkappa;

  double vp1, vp2, vp3, sp1, sp2, sp3;

  //  Bayer bayer;
  Diagram *diagram;
  ResultsHandle results;
  Array1<LockedPolyline *> poly;

  UNUR_GEN *gen;

  Bayer bayer;

public:
  Metropolis(const string& id);

  virtual ~Metropolis();

  virtual void execute();

  void init();
  void reset();
  void metropolis();
  void pdsim( double [], double [] );
  double logpost( double [] );
  double logprior( double p[] );
  double loglike( double [] );
    
    
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MITSHARE Module* make_Metropolis(const string& id) {
  return scinew Metropolis(id);
}

Metropolis::Metropolis(const string& id)
  : Module("Metropolis", id, Source, "Bayer", "MIT"),
    gui_burning("burning", id, this),
    gui_monitor("monitor", id, this),
    gui_thin("thin", id, this),
    gui_kappa("kappa", id, this),
    gui_use_cvode("use-cvode", id, this)
{
  init();
}

Metropolis::~Metropolis() 
{
}

void Metropolis::execute() 
{
  update_state(NeedData);
  
  d_port = (DistributionIPort *) get_iport("Distribution");
  if ( !d_port) {
    cerr << "Metropolis: can not get input port \"Distribution\".";
    return;
  }
  d_port->get(pd);

  m_port = (MeasurementsIPort *) get_iport("Measurements");
  if ( !m_port) {
    cerr << "Metropolis: can not get input port \"Measurements\".";
    return;
  }
  m_port->get(measurements);

  if ( !measurements.get_rep() || ! pd.get_rep() ) {
    cerr << "missing input\n";
    return;
  }

  r_port = (ResultsOPort *) get_oport("Posterior");
  
  update_state(JustStarted);

  reset();

  if ( user_ready ) {
    metropolis();
    user_ready = false;
  }
}


void
Metropolis::init()
{
  mu0 = -5;
  sigma_mu = 2;

  eta0 = -5;
  sigma_eta = 2;

  vp1 = -20;
  vp2 = 10;
  vp3 = 1.5;

  sp1 = -5;
  sp2 = 2;
  sp3 = 2;

  seed = 6;

  nparms = 0;
  theta = 0;
  star = 0;

  monitor_ = scinew CrowdMonitor( id.c_str() );
  graph = scinew Graph( id+"-Graph" );
  diagram = scinew Diagram("Metropolis");
  graph->add("Theta", diagram);

  user_ready = false;
}


void
Metropolis::reset()
{
  int m = pd->sigma.dim1();
  if ( m != nparms ) {
    delete [] theta;
    delete [] star;
    
    theta = scinew double[m];
    star = scinew double[m];
    
    lkappa.newsize( m, m );

    results = scinew Results;

    srand48(0);
    for (int i=nparms; i<m; i++) {
      LockedPolyline *p = scinew LockedPolyline( i );
      p->set_lock( monitor_ );
      p->set_color( Color( drand48(), drand48(), drand48() ) );
      poly.add( p );
      diagram->add( p );
    }

    nparms = m;
  }

  n_observations = measurements->t.size();


  // init random number generator
  srand48(seed);

  // init unuran
  UNUR_DISTR *distr = unur_distr_normal(0,0);
  UNUR_PAR *par = unur_arou_new(distr);
  gen = unur_init(par);
  if ( ! gen ) 
    cerr << "Error, cannot create generator object\n";
  unur_distr_free(distr);
  gui_kappa.set( pd->kappa );
  reset_vars();
}

void Metropolis::metropolis()
{
  // init Cholesky of proposal distribution covariance matrix

  double *A = scinew double[nparms*nparms];
  for (int i=0; i<nparms; i++) 
    for (int j=0; j<nparms; j++) 
      A[i*nparms+j] = pd->sigma(i,j);
  
  int info;

  char cmd = 'L';  // 'L' if fortran means 'U' in C
  dpotrf_( cmd, nparms, A, nparms, info ); 
  if ( info != 0 ) {
    cerr << "Cholesky factorization error = " << info << endl;
    return;
  }

  kappa = pd->kappa;

  double k2 = sqrt(kappa);
  for (int i=0; i<nparms; i++) {
    lkappa(i,i) = A[i*nparms+i]*k2;
    for (int j=i+1; j<nparms; j++) {
      lkappa(j,i) = A[i*nparms+j]*k2;
      lkappa(i,j) = 0;
    }
  }
    
  // input probable distribution
  for (int i=0; i<nparms; i++) 
    theta[i] = pd->theta[i]; // we should let the user change it too.

  //zufalli_(seed);
  
  int len = (gui_monitor.get() - gui_burning.get()) / gui_thin.get();

  results->k_.reserve( len );
  results->data_.resize( nparms );

  for (int i=0; i<nparms; i++) {
    results->data_[i].reserve( len );
    results->data_[i].remove_all();
    poly[i]->clear();
  }


  //   FILE *ulog = fopen( "u.log", "r");
  //   rlog = fopen( "r.log", "r");

  double post = logpost( theta );
  for (int k=1; k<gui_monitor.get(); k++) {

    pdsim( theta, star);

    double u;

    //fscanf( ulog, " %lf", &u);
    //int i = 1;
    //zufall_(i, &u);
    u = drand48();

    double new_post = logpost( star );
    
    if ( new_post - post  <= 500 && exp(new_post-post) >= u ) {
      swap( theta, star );   // yes
      post = new_post;
    }
    
    // 
    if ( k > gui_burning.get()  
	 && fmod((double) k-gui_burning.get(), gui_thin.get() ) == 0 ) 
    {
      results->k_.add(k) ;

      for (int i=0; i<nparms; i++) {
	results->data_[i].add(theta[i]);
	poly[i]->add(theta[i]);
      }

      graph->need_redraw();
    }
    
    //    r_port->send_intermediate( results );
  }
  
  if ( !r_port ) 
    cerr << " no output port\n";
  else {
    for (int i=0; i<poly.size(); i++)
      results->color_.add(poly[i]->get_color());
    r_port->send( results );
  }
}


void Metropolis::pdsim( double theta[], double star[] )
{
  //double r[nparms];
  Array1 <double> r(nparms);
  //   fscanf( rlog, "%lf %lf %lf %lf %lf", 
  // 	  &r[0], &r[1], &r[2], &r[3], &r[4] );

  for (int i=0; i<nparms; i++)
    r[i] = unur_sample_cont(gen);

  //   int tmp=5;
  //   normalen_(tmp, r);

  for (int i=0; i<nparms; i++) {
    star[i] = theta[i];
    for (int j=0; j<nparms; j++ )
      star[i] += lkappa(i,j) * r[j];
  }
}


double 
Metropolis::logpost( double theta[] )
{
  double prior = logprior( theta );
  double like = loglike( theta );
  
  return prior+like;
}

double 
Metropolis::logprior( double theta[] )
{
  double mu = theta[0];
  double eta = theta[1];
  double l1 = theta[2];
  double l2 = theta[3];
  double l3 = theta[4];

  double prior = -0.5*( (sqr((mu-mu0)/sigma_mu)) +
			(sqr((eta-eta0)/sigma_eta)) +
			(sqr((vp1-l1)/sp1)) +
			(sqr((vp2-l2)/sp2)) +
			(sqr((vp3-l3)/sp3)));

  return prior;
}
		

double 
Metropolis::loglike( double theta[] )
{
  int neq = n_observations;
  //double ym1[neq];
  //double ym2[neq];

  Array1 <double> ym1(neq);
  Array1 <double> ym2(neq);
  double ymi[100];
  double dt = exp(theta[4]);
  
  double reltol = 1e-6;   // should be input from the UI
  double abstol = 1e-6;
  double f = 0.1335;

  double p[6];

  p[0] = theta[0];
  p[1] = theta[1];
  p[2] = theta[2];
  p[3] = log(0.0249);
  p[4] = log(0.0842);
  p[5] = theta[3];


  if ( gui_use_cvode.get() == 1 ) {
    bayer.set_p( p );
    
    // get y0 
    N_Vector y = N_VNew( neq, 0 );
    bayer.get_y0( y );
    
    long int iopt[OPT_SIZE];
    
    for (int i=0; i<OPT_SIZE; i++) iopt[i] = 0;
    iopt[MXSTEP] = 2000;
    
    // init integrator
    void *cvode_mem = CVodeMalloc( neq, bayer.f(),        // size and function
				   0, y,                  // t0 and y
				   ADAMS, NEWTON,         // methods
				   SS, &reltol, &abstol,  // tolerances
				   &bayer,                // f_data
				   0,                     // err file
				   TRUE, iopt, 0, 0       // optIn, iopt, ropt 
				   ); 
    
    if ( !cvode_mem ) {
      cerr << "CVodeMalloc failed\n";
      return -1; // should exit!!
    }
    
    // select linear solver
    CVDense( cvode_mem, 0, 0 );  // use internal jacobian
    
    
    for (int i=0; i<neq; i++) { 
      double tf = measurements->t[i]+dt;
      double t;
      
      int status = CVode( cvode_mem, tf, y, &t, NORMAL );
      
      if ( status != SUCCESS ) {
	cerr << "CVode fail. status = " << status << endl;
	Thread::niceAbort();
	return -1; // should exit or report error!!
      }
      
      ym1[i] = N_VIth(y,0);
      ym2[i] = N_VIth(y,neq-2);
    }
    
    N_VFree( y );
  }
  else {
    for (int i=0; i<neq; i++) { 
      double tf = measurements->t[i]+dt;
      bayer_( tf, p, ymi );
      ym1[i] = ymi[0];
      ym2[i] = ymi[5];
    }

  }

  double l1 = 0.05*0.05;
  double l2 = 0.015*0.015;

  double logl1 = log(l1);
  double logl2 = log(l2);

  double sum = 0;
  for (int i=0; i< neq; i++) 
    sum += sqr(measurements->concentration(0,i) - ym1[i]/f);


  double like = -0.5*( 
		      7*logl1  +
		      sum/l1  +
		      3*logl2 +
		      (sqr( measurements->concentration(1,0) - ym2[1])+
		       sqr( measurements->concentration(1,1) - ym2[3])+
		       sqr( measurements->concentration(1,2) - ym2[6])
		       ) / l2
		      );
  //  cerr << "like = " << like << endl;
  

  return like;
}
    
  



void Metropolis::tcl_command(TCLArgs& args, void* userdata)
{
  if ( args[1] == "graph-window" ) {
    cerr << "Metropolis:: graph set window " << args[2] << endl;
    graph->set_window( args[2] ); 
  } 
  else if ( args[1] == "exec" ) {
    user_ready = true;
    want_to_execute();
  }
  else
    Module::tcl_command(args, userdata);
}


} // End namespace MIT


