/*
 *  Sampler.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <stdlib.h>
#include <math.h>

extern "C" {
  void dpotrf_( char &, int &, double *, int &, int &  ); 
}

#include <Core/Parts/Part.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geom/Color.h>
//#include <Core/Thread/Thread.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>
#include <Packages/MIT/share/share.h>

#include <Packages/MIT/Dataflow/Modules/Metropolis/SamplerInterface.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/PriorPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/LikelihoodPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>

#include <Core/Util/Signals.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/SamplerGui.h>

namespace MIT {

using namespace SCIRun;


extern "C" MITSHARE Module* make_Sampler(const string& id) {
  return scinew Sampler(id);
}

Sampler::Sampler(const string& id)
  : Module("Sampler", id, Source, "Metropolis", "MIT"),
    Part( 0, "Sampler")
{
  seed = 6;

  nparms = 0;
  theta = 0;
  star = 0;

  // init random number generator
  srand48(seed);

  // init unuran
  UNUR_DISTR *distr = unur_distr_normal(0,0);
  UNUR_PAR *par = unur_arou_new(distr);
  gen = unur_init(par);
  if ( ! gen ) 
    cerr << "Error, cannot create generator object\n";
  unur_distr_free(distr);

  user_ready = false;
  
  // install Interface
  interface_ = new SamplerInterface( this, 0 );
  Part::interface_ = interface_;

  // install Likelihood and Prior parts
  prior_ = new PriorPart( interface_, "Prior");
  likelihood_ = new LikelihoodPart( interface_, "Likelihood" );
}

Sampler::~Sampler() 
{
}

void
Sampler::go()
{
  cerr << "Sampler exec\n";
  Module *module = dynamic_cast<Module*>(this);
  module->want_to_execute();
}

void Sampler::execute() 
{
  update_state(NeedData);

  // Distribution
  d_port_ = (DistributionIPort *) get_iport("Distribution");
  if ( !d_port_) {
    cerr << "Sampler: can not get input port \"Distribution\".";
    return;
  }
  d_port_->get(distribution_);

  // Measurements
  m_port_ = (MeasurementsIPort *) get_iport("Measurements");
  if ( !m_port_) {
    cerr << "Metropolis: can not get input port \"Measurements\".";
    return;
  }
  m_port_->get(measurements_);

  if ( !measurements_.get_rep() || ! distribution_.get_rep() ) {
    cerr << "missing input\n";
    return;
  }

  r_port_ = (ResultsOPort *) get_oport("Posterior");
  
  update_state(JustStarted);

  reset();

//   if ( user_ready ) {
    metropolis();
//     user_ready = false;
//   }
}


void
Sampler::reset()
{
  int m = distribution_->sigma.dim1();
  if ( m != nparms ) {
    delete [] theta;
    delete [] star;
    
    theta = scinew double[m];
    star = scinew double[m];
    
    lkappa.newsize( m, m );

    results = scinew Results;

    srand48(0);
    nparms = m;
  }

  likelihood_->measurements( measurements_ );
}

void Sampler::metropolis()
{
  // init Cholesky of proposal distribution covariance matrix

  double A[nparms][nparms];

  for (int i=0; i<nparms; i++) 
    for (int j=0; j<nparms; j++) 
      A[i][j] = distribution_->sigma(i,j);
  
  int info;

  char cmd = 'L';  // 'L' if fortran means 'U' in C
  dpotrf_( cmd, nparms, (double *)A, nparms, info ); 
  if ( info != 0 ) {
    cerr << "Cholesky factorization error = " << info << endl;
    return;
  }

  kappa = distribution_->kappa;

  double k2 = sqrt(kappa);
  for (int i=0; i<nparms; i++) {
    lkappa(i,i) = A[i][i]*k2;
    for (int j=i+1; j<nparms; j++) {
      lkappa(j,i) = A[i][j]*k2;
      lkappa(i,j) = 0;
    }
  }
    
  // input probable distribution
  for (int i=0; i<nparms; i++) 
    theta[i] = distribution_->theta[i]; // we should let the user change it too.

  int len = (interface_->monitor() - interface_->burning()) 
    / interface_->thin();

  results->k_.reserve( len );
  results->data_.resize( nparms );

  for (int i=0; i<nparms; i++) {
    results->data_[i].reserve( len );
    results->data_[i].remove_all();
  }

  double post = logpost( theta );
  for (int k=1; k<interface_->monitor(); k++) {

    pdsim( theta, star);

    double u;

    u = drand48();

    double new_post = logpost( star );
    
    if ( new_post - post  <= 500 && exp(new_post-post) >= u ) {
      swap( theta, star );   // yes
      post = new_post;
    }
    
    if ( k > interface_->burning()  
	 && fmod(k-interface_->burning(), interface_->thin() ) == 0 ) 
    {
      results->k_.add(k) ;

      for (int i=0; i<nparms; i++) {
	results->data_[i].add(theta[i]);
      }
    }
  }
  
  if ( !r_port_ ) 
    cerr << " no output port\n";
  else {
    for (int i=0; i<nparms; i++)
      results->color_.add(Color( drand48(), drand48(), drand48() ));
    r_port_->send( results );
  }
}


void Sampler::pdsim( double theta[], double star[] )
{
  double r[nparms];

  for (int i=0; i<nparms; i++)
    r[i] = unur_sample_cont(gen);

  for (int i=0; i<nparms; i++) {
    star[i] = theta[i];
    for (int j=0; j<nparms; j++ )
      star[i] += lkappa(i,j) * r[j];
  }
}


double 
Sampler::logpost( double theta[] )
{
  double prior = prior_->compute( theta );
  double like = likelihood_->compute( theta );
  
  return prior+like;
}
  

void 
Sampler::tcl_command( TCLArgs &args, void *data)
{
  if ( args[1] == "set-window" ) {
    SamplerGui *gui = new SamplerGui( id+"-gui" );
    gui->set_window( args[2] );
    connect( gui->burning, interface_, &SamplerInterface::burning );
    connect( gui->monitor, interface_, &SamplerInterface::monitor );
    connect( gui->thin, interface_, &SamplerInterface::thin );
    connect( gui->go, interface_, &SamplerInterface::go);
  }
  else Module::tcl_command( args, data );
}

} // End namespace MIT


