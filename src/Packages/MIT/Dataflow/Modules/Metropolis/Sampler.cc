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
#include <vector>

extern "C" {
  void dpotrf_( char &, int &, double *, int &, int &  ); 
}

#include <Core/Parts/Part.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geom/Color.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>
#include <Packages/MIT/share/share.h>

#include <Packages/MIT/Dataflow/Modules/Metropolis/SamplerInterface.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/PriorPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/LikelihoodPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IGaussianPDSimPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IUniformPDSimPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/ItPDSimPart.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/MultivariateNormalDSimPart.h>
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
    Part( 0, "Sampler"),
    pdsim_( 0, "PDSim" )
{
  seed = 6;

  nparms = 0;
  theta.resize(2);
  old = 0;
  star = 1;

  // init random number generator
  srand48(seed);

  user_ready = false;
  
  // install Interface
  interface_ = new SamplerInterface( this, 0 );
  Part::interface_ = interface_;

  // install Likelihood and Prior parts
  prior_ = new PriorPart( interface_, "Prior");
  likelihood_ = new LikelihoodPart( interface_, "Likelihood" );

  pdsim_.set_parent( interface_ );
  pdsim_.add_part( new MultivariateNormalDSimPart( this, 0, "MVNormal") );
  pdsim_.add_part( new IUniformPDSimPart( this, 0, "IU" ) );
  pdsim_.add_part( new IGaussianPDSimPart( this, 0, "IG" ) );
  pdsim_.add_part( new ItPDSimPart( this, 0, "It" ) );

  graph_ = 0;

}

Sampler::~Sampler() 
{
}

void
Sampler::go()
{
  Module *module = dynamic_cast<Module*>(this);
  if ( ! graph_ ) {
    graph_ = new GraphPart( interface_, "Progress");
    graph_->set_num_lines(nparms);
  }
  module->want_to_execute();
}

void 
Sampler::execute() 
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

    theta[old].resize(m);
    theta[star].resize(m);
    
    has_lkappa_ = false;

    results = scinew Results;

    srand48(0);
    nparms = m;

  }

  if ( graph_ )
    graph_->set_num_lines(nparms);
  likelihood_->measurements( measurements_ );
}

Array2<double> &
Sampler::get_lkappa()
{
  if ( !has_lkappa_ ) {
    // init Cholesky of proposal distribution covariance matrix
  
    kappa = distribution_->kappa;
    
    lkappa.newsize( nparms, nparms );

    Array2<double> A(nparms, nparms);
  
    for (int i=0; i<nparms; i++) 
      for (int j=0; j<nparms; j++) 
	A(i,j) = distribution_->sigma(i,j)*kappa; 
    
    int info;
    
    char cmd = 'U';  // 'U' in C means 'L' in Fortran
    dpotrf_( cmd, nparms, *(A.get_dataptr()), nparms, info ); 
    if ( info != 0 ) {
      cerr << "Cholesky factorization error = " << info << endl;
      return lkappa; // trow exeption ?
    }
    
    for (int i=0; i<nparms; i++) {
      lkappa(i,i) = A(i,i);
      for (int j=i+1; j<nparms; j++) {
	lkappa(i,j) = A(i,j);
	lkappa(j,i) = 0;
      }
    }
    
    has_lkappa_ = true;
  }

  return lkappa;
}

void Sampler::metropolis()
{
  // input probable distribution
  for (int i=0; i<nparms; i++) 
    theta[old][i] = distribution_->theta[i]; // we should let the user change 
                                             // it too
  
  int len = (interface_->monitor() - interface_->burning()) 
            / interface_->thin();

  results->k_.reserve( len );
  results->data_.resize( nparms );

  for (int i=0; i<nparms; i++) {
    results->data_[i].reserve( len );
    results->data_[i].remove_all();
  }

  double post = logpost( theta[old] );
  double lpr = pdsim_->lpr(theta[old]);

  for (int k=1; k<interface_->monitor(); k++) {

    pdsim_->compute( theta[old], theta[star]);
    double new_lpr = pdsim_->lpr( theta[star] );
    double new_post = logpost( theta[star] );
    double sum = new_post - post + lpr - new_lpr;
    double u = drand48();

    if ( sum <= 500 && exp(sum) >= u ) {
      swap( old, star );   // yes
      post = new_post;
      lpr = new_lpr;
    }
    
    if ( k > interface_->burning()  
	 && fmod((double) k-interface_->burning(), interface_->thin() ) == 0 ) 
    {
      results->k_.add(k) ;

      vector<double> v(nparms);
      //v.resize(1);
      for (int i=0; i<nparms; i++) {
	results->data_[i].add(theta[old][i]);
	v[i]=theta[old][i];
      }
      if ( graph_ ) 
	graph_->add_values(nparms,v);      
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


double 
Sampler::logpost( vector<double> &theta )
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
    connect( interface_->has_child, (PartGui* )gui, &PartGui::add_child );

    interface_->report_children( (PartGui* )gui, &PartGui::add_child );
  }
  else Module::tcl_command( args, data );
}


} // End namespace MIT


