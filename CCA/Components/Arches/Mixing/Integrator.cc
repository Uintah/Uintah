//----- Integrator.cc --------------------------------------------------


#include <Packages/Uintah/CCA/Components/Arches/Mixing/Integrator.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFShape.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/BetaPDFShape.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>
using namespace std;
using namespace Uintah;
PDFMixingModel* pdfMixModel;

//****************************************************************************
// Default constructor for Integrator
//****************************************************************************
Integrator::Integrator(int tableDimension, PDFMixingModel* mixmodel,
		       ReactionModel* rxnmodel, MixRxnTableInfo* tableInfo):
  d_tableDimension(tableDimension), 
  d_rxnModel(rxnmodel), d_tableInfo(tableInfo)
{
  pdfMixModel = mixmodel; // global variable required for fortran fun call
  //d_meanValues = vector<double>(tableDimension);
  d_keyValues = vector<double>(d_tableDimension, 0.0);
  //d_keyValues = d_meanValues;
  d_varsHFPi = vector<double>(pdfMixModel->getNumMixVars()+pdfMixModel->getNumRxnVars()+
			      !(pdfMixModel->isAdiabatic()), 0.0);
}

//****************************************************************************
// Destructor
//****************************************************************************
Integrator::~Integrator()
{
}

//****************************************************************************
// Problem Setup for Integrator
//****************************************************************************
void 
Integrator::problemSetup(const ProblemSpecP& params)
{
  //ProblemSpecP db = params->findBlock("Integrator");
  // dimPDF need to define based on mix fraction and variance or
  // BetaPDFShape can be assumed to be a univariate PDF, higher
  // dim beta pdf can be called multivariateBetaPDF
  int dimPDF = pdfMixModel->getNumMixStatVars();
  if (pdfMixModel->getPDFShape() == "Beta") 
    d_mixingPDF = new BetaPDFShape(dimPDF);
  else
    throw InvalidValue("PDF shape not implemented "
		       + pdfMixModel->getPDFShape());
}


//****************************************************************************
// Integrate
//****************************************************************************
Stream
Integrator::integrate(int* tableKeyIndex)
{
  int mixIndex = 0;
  if (!(pdfMixModel->isAdiabatic()))
    ++mixIndex;
  // checking for mixing variance
  int mixVarIndex = mixIndex + pdfMixModel->getNumMixVars();
  int rxnIndex = mixVarIndex + pdfMixModel->getNumMixStatVars();
  // compute meanValues from tableKeyIndex
  vector<double> meanValues(d_tableDimension);
  convertKeytoFloatValues(tableKeyIndex, meanValues);
  // d_keyValues is used for Integral function evaluation
  d_keyValues = meanValues;
  // Write independent variables, excluding variance, to separate vector
  // for use by reaction model
  int inc = 0;
  if (!(pdfMixModel->isAdiabatic()))
    {
      d_varsHFPi[inc] = meanValues[0];
      ++inc;
    }
  for (int ii = mixIndex; ii < mixIndex+pdfMixModel->getNumMixVars(); ii++)
    {
      d_varsHFPi[inc++] = meanValues[ii];
    }
  for (int ii = rxnIndex; ii < rxnIndex+pdfMixModel->getNumRxnVars(); ii++)
    {
     d_varsHFPi[inc++] = meanValues[ii];  
    }
  // Compute mean values, store in vector
  vector<double> mixVars(pdfMixModel->getNumMixVars());
  for (int jj = 0; jj < pdfMixModel->getNumMixVars(); jj++)
    mixVars[jj] = d_keyValues[mixIndex+jj];
  d_unreactedStream = pdfMixModel->speciesStateSpace(mixVars);
  d_rxnModel->getRxnStateSpace(d_unreactedStream,d_varsHFPi,
				   d_meanSpaceVars);
  vector<double> meanStateSpaceVars = d_meanSpaceVars.convertStreamToVec();
  //cout << "meanStateSpaceVars = " << meanStateSpaceVars[0] << endl;  
// Compute pdf function for given values of mean and variance
  d_mixingPDF->computePDFFunction(&meanValues[mixIndex], meanValues[mixVarIndex]);
   // store integral values in vector
  vector<double> resultStateSpaceVars(pdfMixModel->getTotalVars());
  if (!(d_mixingPDF->validIntegral())) { 
    // If gammafnc is not valid, return mean values
    cout << "Invalid Gamma Function" << endl;
    resultStateSpaceVars = meanStateSpaceVars;
  }
  else {
    // Gammafnc valid, so perform integration
    // Do initialization of parameters for guassian quadrature
    // These parameters are described in qagpe.f
    //cout << "Defining params for dqagpe" << endl;
    double lowLimit, upLimit, epsrel, epsabs, result, abserr;
    double points[NPTS];
    //double pts[NPTS],alist[LIMITS],blist[LIMITS],rlist[LIMITS],elist[LIMITS];
    int leniw = LIMITS;
    int lenw = LIMITS*2; 
    double work[LIMITS*2];
    int iwork[LIMITS];
    int npts2, limit, neval, ier, last;
    //int level[LIMITS], ndin[NPTS], iord[LIMITS];
    lowLimit = 0.0; // Limits for Integration
    upLimit = 1.0;
    epsabs = 1e-05; 
    epsrel = 1e-04; // relative error
    npts2 = 6;
    //double fstoic = d_tableInfo->getStoicValue(mixIndex);
    double fstoic = pdfMixModel->getStoicPt();
    //double fstoic = 0.05516706 ; // Kluge to run methane
    //fstoic = 6.218489e-02; // Kluge to run heptane
    limit = LIMITS;
    bool integralSuccess = false;
   points[0] = 0;
   points[1] = meanValues[mixIndex];
   points[2] = fstoic;
   points[3] = 1.0;
    // Assumption:
    // Only integrating over mixing variable (mix_dim == 1); rxn_dim = 0
    // For details see gammafn.C
    int densityIndex = d_tableInfo->getDensityIndex();
    int temperatureIndex = d_tableInfo->getTemperatureIndex();
    for (d_count = 0; d_count < resultStateSpaceVars.size(); d_count++)
      {
	// fnc is a member function
	//dqagpe_(fnc, &lowLimit, &upLimit, &npts2, points, &epsabs, &epsrel,
	//&limit, &result, &abserr, &neval, &ier, alist, blist, rlist,
	//	elist, pts, level, ndin, iord, &last); 
	dqagp(fnc, &lowLimit, &upLimit, &npts2, points, &epsabs, &epsrel, &result,
	       &abserr, &neval, &ier, &leniw, &lenw, &last, iwork, work);
	if (ier == 0) {
	  integralSuccess = true;
	  resultStateSpaceVars[d_count] = result;
	  if (d_count == densityIndex) {
	    resultStateSpaceVars[d_count] = 1.0/resultStateSpaceVars[d_count];
	    if (resultStateSpaceVars[densityIndex] > 1.5)
	      cout << "Density too high: " << resultStateSpaceVars[densityIndex] << endl;
	  }			
	  if (d_count == temperatureIndex) {
	    resultStateSpaceVars[d_count] *= resultStateSpaceVars[densityIndex];
	    if (resultStateSpaceVars[temperatureIndex] > 2400)
	      cout << "Temperature too high: " << resultStateSpaceVars[temperatureIndex] << endl;
	  }
          //cout << "after integration: " << resultStateSpaceVars[d_count] << " "<< result << endl;
	}
	else {
	  switch (ier) {
	  case 2:   //  roundoff errors - continue anyway
	    cout << "NOTE: Roundoff errors encountered in integration" << endl;
	    cout << "Requested erorr tol: " << epsabs << " Actual error estimate: " << 
	      abserr << endl;
	  case 4:  //  integral didn't converge - continue anyway
	    cout << "NOTE: Integration did not sucessfully converge. " <<
	      "Best estimate returned." << endl;
	  case 5:
	    cout << "NOTE: Integration did not sucessfully converge. " <<
	      "Divergence possible." << endl;
	  default:
	    cout << "ERROR number " << ier << " in dqagp" << endl;
	  }
	  integralSuccess = false;
	}
	if (!(integralSuccess)) 
	  {
	    cout << "convergence problems for the variable " << d_count << endl;
	    resultStateSpaceVars[d_count] = meanStateSpaceVars[d_count];
	  }
      } // for(d_count=0)
  } //Gammafnc valid, so perform integration 
  // Copy integral values from vector to stream
  d_resultStateVars.convertVecToStream(resultStateSpaceVars,
               pdfMixModel->getNumMixVars(), pdfMixModel->getNumRxnVars(),
	       d_meanSpaceVars.d_lsoot);
  cout << "Integrator::h = "<<meanValues[0]<<endl;
  cout << "Integrator::f = "<<meanValues[1]<<endl;
  cout << "Integrator::gf = "<<meanValues[2]<<endl;
  d_resultStateVars.print(cout);
  //resultStateVars.d_drhodf = 0.0;
  return d_resultStateVars;

}

void
Integrator::convertKeytoFloatValues(int tableKeyIndex[], vector<double>& indepVars) {
  for (int i = 0; i < d_tableDimension; i++)
    if (tableKeyIndex[i] <= d_tableInfo->getNumDivsBelow(i))
      indepVars[i] = tableKeyIndex[i]*d_tableInfo->getIncrValueBelow(i) + 
	d_tableInfo->getMinValue(i);
    else
      indepVars[i] = (tableKeyIndex[i]-d_tableInfo->getNumDivsBelow(i))*	
	d_tableInfo->getIncrValueAbove(i) + d_tableInfo->getStoicValue(i);
  return;
}

double
fnc(double *x) {
  return pdfMixModel->getIntegrator()->fun(x);
}

double
Integrator::fun(double *x) {
  int mixIndex = !(pdfMixModel->isAdiabatic());
  d_keyValues[mixIndex] = *x;
    d_varsHFPi[mixIndex] = *x;
    vector<double> mixVars(pdfMixModel->getNumMixVars());
    for (int jj = 0; jj < pdfMixModel->getNumMixVars(); jj++) 
      mixVars[jj] = d_keyValues[mixIndex+jj]; 
    // Kluge for stanjan as it blows up if mixture fraction too close to 1
    //if (mixVars[jj] > 0.99) 
    //mixVars[jj] = 1.0;
    //}
    Stream unreactedStream = pdfMixModel->speciesStateSpace(mixVars);
    Stream stateVars;
    d_rxnModel->getRxnStateSpace(unreactedStream, d_varsHFPi, stateVars);
    //cout << "stateVars = " << stateVars.d_density << " " << stateVars.d_temperature << endl;
    // from the stream class get relevant variable corresponding to count
    double integrand = stateVars.getValue(d_count);
    //cout << "mixVar = "<< d_varsHFPi[mixIndex] << " count = " << d_count << " integrand = " << integrand << endl;
    // pass only mixing variables
    integrand *= d_mixingPDF->computeShapeFunction(&d_keyValues[mixIndex]); 
    //integrand = d_mixingPDF->computeShapeFunction(&d_keyValues[mixIndex]); 
    //cout << "Integrand is " << integrand << endl;
    return integrand;

}
  





