#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;

#include <tabprops/prepro/mixmdl/MixMdlHelper.h>
#include <tabprops/StateTable.h>

#include <tabprops/prepro/mixmdl/PresumedPDFMixMdl.h>
#include <tabprops/prepro/mixmdl/BetaMixMdl.h>
#include <tabprops/prepro/mixmdl/ClippedGaussMixMdl.h>
#include <tabprops/prepro/mixmdl/Integrator.h>
#include <tabprops/prepro/mixmdl/GaussKronrod.h>
#include <tabprops/prepro/mixmdl/MixMdlFunctor.h>

bool test_pdf( PresumedPDFMixMdl* mm, Integrator* integrator )
{
  cout << "Testing integrators with presumed pdf." << endl;
  
  //
  // Test integrator with the provided PDF model.  Three functions are used as the
  // convolution function, testing the requirements that
  //
  //   1. the convoluation of the PDF with the unit function must integrate to unity
  //
  //   2. the convoluation of the PDF with a linear function must integrate to the mean
  //
  //   3. the convoluation of the PDF with the appropriate quadratic function must
  //   integrate to the variance.
  //
  // Results are written to three files: integral.dat, mean.dat, and variance.dat.
  //
  
  static const double SMALL = 1.0e-8;

  //
  // Set the functor that we want to convolute with the PDF.
  // Here we will use three seperate functions which test the integration.
  // Hook each functor up to the mixing model, then perform the integration.
  //
  FunctorDoubleBase * testPdfFunc  =
    new FunctorDouble<PresumedPDFMixMdl>( mm, &PresumedPDFMixMdl::test_pdf  );
  
  FunctorDoubleBase * testMeanFunc =
    new FunctorDouble<PresumedPDFMixMdl>( mm, &PresumedPDFMixMdl::test_mean );
  
  FunctorDoubleBase * testVarFunc  =
    new FunctorDouble<PresumedPDFMixMdl>( mm, &PresumedPDFMixMdl::test_var  );

  ofstream iout("integral.dat");
  ofstream mout("mean.dat");
  ofstream vout("variance.dat");

  iout << "#         mean      scalVar      integral      rel err   ncalls  nIntervals" << endl;
  mout << "#         mean      scalVar      integral      rel err   ncalls  nIntervals" << endl;
  vout << "#         mean      scalVar      integral      rel err     abs err   ncalls  nIntervals" << endl;

  //
  //  Perform some integrals with various test functions.
  //
  const int nmean = 21;
  const int nvar  = 21;
  double mean, varscal, ig, err;
  for( int imn=0; imn<nmean; imn++ ){

    // set the mean 
    mean = double(imn)/double(nmean-1);
    mm->set_mean( mean );

    // set singularities at mean and at extrema
    integrator->add_singul( mean );
    integrator->add_singul( 0.0 );
    integrator->add_singul( 1.0 );

    for( int ivr=0; ivr<nvar; ivr++ ){
      varscal = double(ivr)/double(nvar-1);
      mm->set_scaled_variance( varscal );

      if( varscal > 0.0 && varscal < 1.0 && mean > 0.0 && mean < 1.0 ){

	testPdfFunc->reset_n_calls();
	mm->set_convolution_func( testPdfFunc );
	ig = mm->integrate();
	err = fabs(1.0-ig);
	iout << setw(13) << mean
	     << setw(13) << varscal
	     << setw(13) << ig
	     << setw(13) << err
	     << setw(8) << testPdfFunc->get_n_calls()
	     << setw(8) << integrator->get_n_intervals()
	     << endl;

	testMeanFunc->reset_n_calls();
	mm->set_convolution_func( testMeanFunc );
	ig = mm->integrate();
	err = fabs(mean-ig)/(mean+SMALL);
	mout << setw(13) << mean
	     << setw(13) << varscal
	     << setw(13) << ig
	     << setw(13) << err
	     << setw(8)  << testMeanFunc->get_n_calls()
	     << setw(8) << integrator->get_n_intervals()
	     << endl;

	testVarFunc->reset_n_calls();
	mm->set_convolution_func( testVarFunc );
	ig = mm->integrate();
	err = fabs(mm->get_variance()-ig)/(mm->get_variance()+SMALL);
	vout << setw(13) << mean
	     << setw(13) << varscal
	     << setw(13) << ig
	     << setw(13) << err
	     << setw(13) << fabs(mm->get_variance()-ig)
	     << setw(8) << testVarFunc->get_n_calls()
	     << setw(8) << integrator->get_n_intervals()
	     << endl;
      }
    }
    cout << imn << " " << flush;
    iout << endl;
    mout << endl;
    vout << endl;
  }
  cout << "  DONE" << endl;

  return true;
}
//--------------------------------------------------------------------
bool test_mix_rxn_mdl_1( StateTable & rxnMdl )
{
  // hard-coded for one-variable reaction model based on mixture fraction

  const int order = 1;
  const int nfpts = 101;
  const int ngpts = 51;

  // create the mixing model.  Here we use the clipped-Gaussian PDF.  This is preferred
  // over the Beta PDF because of its better numerical properties.  It can be integrated
  // much more accurately.  At high variance, you may not get reasonable results from the
  // Beta PDF.
  PresumedPDFMixMdl * mixMdl = new ClipGauss();
  
  // create a mixing model helper.  This will build the mixing model for us.  We must
  // provide the mixing model, a state table (reaction model), and the name of the
  // variable that we are convoluting.  Later, we will impose the mesh that we want the
  // model on.
  MixMdlHelper mm( *mixMdl, rxnMdl, "MixtureFraction", order );

  // define the mesh in the mixture fraction dimension.  No need to have it uniformly
  // spaced, but it must be constant w.r.t. the variance dimension.
  vector<double> vec(nfpts);
  for( int i=0; i<nfpts; ++i )
    vec[i] = double(i)/double(nfpts-1);

  // advertise the mesh to the mixing model helper
  mm.set_mesh( "MixtureFraction", vec );

  // define the mesh in the variance dimension.
  vec.resize(ngpts);
  for( int i=0; i<ngpts; ++i )
    vec[i] = double(i)/double(ngpts-1);
  
  // advertise the mesh to the mixing model helper
  mm.set_mesh( "MixtureFractionVariance", vec );

  // implement the model.  This convolutes the tabular function defined by rxnMdl over the
  // PDF defined by mixMdl for each variable in rxnMdl.  The result is written to disk as
  // a StateTable database in HDF5 format.
  mm.implement();
  
  return true;
}
//--------------------------------------------------------------------
int main()
{
  Integrator * const integrator = new GaussKronrod( 0, 1 );
  PresumedPDFMixMdl * mixMdl = new ClipGauss();
  //  mixMdl = new BetaMixMdl();
  //  integrator->interrogate();
  
  mixMdl->set_integrator( integrator );
  
  test_pdf( mixMdl, integrator );

  delete mixMdl;
  delete integrator;

  return 0;
}
