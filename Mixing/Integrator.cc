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
  d_meanValues = vector<double>(tableDimension);
  d_keyValues = d_meanValues;
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
  ProblemSpecP db = params->findBlock("Integrator");
  db->require("favre",d_lfavre);
  string pdfType;
  db->require("PDFSHAPE",pdfType);
  // dimPDF need to define based on mix fraction and variance or
  // BetaPDFShape can be assumed to be a univariate PDF, higher
  // dim beta pdf can be called multivariateBetaPDF
  int dimPDF = pdfMixModel->getNumMixStatVars();
  if (pdfType == "BetaPDFShape") 
    d_mixingPDF = new BetaPDFShape(dimPDF);
  else
    throw InvalidValue("PDF shape not implemented "
		       + pdfType);
}


//****************************************************************************
// Integrate
//****************************************************************************
Stream
Integrator::integrate(int* tableKeyIndex)
{
  // Do initialization of parameters for guassian quadrature
  // These parameters are described in qagpe.f

  // Should I check for validIntegral first before doing all this stuff???
  double lowLimit, upLimit, epsrel, epsabs, result, abserr;
  double points[NPTS], pts[NPTS], alist[LIMITS], blist[LIMITS], 
    rlist[LIMITS],elist[LIMITS];
  int npts2, limit, neval, ier, last;
  int level[LIMITS], ndin[NPTS], iord[LIMITS];
  lowLimit = 0.0; // Limits for Integration
  upLimit = 1.0;
  // this can be moved to problem setup
  epsabs = 0.0;
  epsrel = 0.005; // relative error
  //epsrel = 0.0005; // relative error
  npts2 = 3;
  int mixIndex = 0;
  if (!(pdfMixModel->isAdiabatic()))
    ++mixIndex;
  int rxnIndex = mixIndex + pdfMixModel->getNumMixStatVars()+1;
  double fstoic = d_tableInfo->getStoicValue(mixIndex);
  points[0] = fstoic;
  limit = LIMITS;
  bool integralSuccess = false;
  // ***Modify for higher order integration***
  // checking for mixing variance
  int mixVarIndex = mixIndex + pdfMixModel->getNumMixVars();
  // compute d_meanValues from tableKeyIndex
  convertKeytoMeanValues(tableKeyIndex);
  // d_keyValues is used for Integral function evaluation
  d_keyValues = d_meanValues;
  //NOTE-Can change line below to speed up integration!!!
  //if (d_meanValues[mixVarIndex] > 0.1) {
  if (d_meanValues[mixVarIndex] > 0.005) {
    npts2 = 4;
    // points calculated for Gaussian Quadrature
    double nearEdge = (fstoic - lowLimit) / 20.;
    double midEdge = (upLimit - fstoic) / 10.;
    double farEdge = (upLimit - fstoic) / 30.;
    if (d_meanValues[mixIndex] < 0.5) {
      npts2 = 4;
      points[0] = lowLimit + nearEdge;
      points[1] = fstoic;
    }
    else {
      npts2 = 6;
      points[0] = lowLimit + nearEdge;
      points[1] = fstoic;
      points[2] = upLimit - midEdge;
      points[3] = upLimit - farEdge;
    }
  }
  // Write independent variables, excluding variance, to separate vector
  // for use by reaction model
  int inc = 0;
  if (!(pdfMixModel->isAdiabatic()))
    {
      d_varsHFPi[inc] = d_meanValues[0];
      ++inc;
    }
  for (int ii = mixIndex; ii < mixIndex+pdfMixModel->getNumMixVars(); ii++)
    {
      d_varsHFPi[inc++] = d_meanValues[ii];
    }
  for (int ii = rxnIndex; ii < rxnIndex+pdfMixModel->getNumRxnVars(); ii++)
    {
     d_varsHFPi[inc++] = d_meanValues[ii];  
    }
  // store mean values in vector
  vector<double> mixVars(pdfMixModel->getNumMixVars());
  for (int jj = 0; jj < pdfMixModel->getNumMixVars(); jj++)
    mixVars[jj] = d_keyValues[mixIndex+jj];
  Stream unreactedStream = pdfMixModel->speciesStateSpace(mixVars);
  Stream meanSpaceVars;
  d_rxnModel->getRxnStateSpace(unreactedStream,d_varsHFPi,
				   meanSpaceVars);
  // compute pdf function for given values of mean and variance
  d_mixingPDF->computePDFFunction(&d_meanValues[mixIndex], d_meanValues[mixVarIndex]);
  cout<<"Int::after call to Beta statVar = "<<d_meanValues[mixVarIndex]<<endl;
  // if lfavre temp = temp/density...replaced logical with lsoot
  //bool lsoot = d_rxnModel->getSootBool();
  //vector<double> meanStateSpaceVars = meanSpaceVars.convertStreamToVec(lsoot);
  vector<double> meanStateSpaceVars = meanSpaceVars.convertStreamToVec();
  // store integral values in vector and then copy to stream
  vector<double> resultStateSpaceVars(pdfMixModel->getTotalVars());
  // Assumption:
  // Only integrated over mixing variable; rxn_dim = 0
  // check if gammafnc is valid
  if (d_mixingPDF->validIntegral()) { 
    for (d_count = 0; d_count < resultStateSpaceVars.size(); d_count++)
    //for (d_count = 0; d_count < 2; d_count++)
      {
	// assuming mix_dim == 1
	// for details see gammafn.C
	// fnc is a member function
	//cout<<"Int::begin integration keyValues are = "<<d_keyValues[0]<<" "<<d_keyValues[1]<<endl;
	dqagpe_(fnc, &lowLimit, &upLimit, &npts2, points, &epsabs, &epsrel,
		&limit, &result, &abserr, &neval, &ier, alist, blist, rlist,
		elist, pts, level, ndin, iord, &last);
	
	if (ier == 0) {
	  integralSuccess = true;
	  resultStateSpaceVars[d_count] = result;
	  //cout << "Int::result = " << result << endl;
	}
	else
	  integralSuccess = false;
	if (!(integralSuccess)) 
	  {
	    cout << "convergence problems for the variable " << d_count << endl;
	    resultStateSpaceVars[d_count] = meanStateSpaceVars[d_count];
	  }
      }
    // favre averaged
  }
  else {
    cout << "Invalid Gamma Function" << endl;
    resultStateSpaceVars = meanStateSpaceVars;
  }
  
  //fix it! not right for cases with invalid gammafn      
#if 0
  if ((d_lfavre)&&(integralSuccess))
    resultStateSpaceVars[d_temperatureIndex] *= resultStateSpaceVars[d_densityIndex];
#endif
  Stream resultStateVars;
  resultStateVars.convertVecToStream(resultStateSpaceVars, d_lfavre, 
               pdfMixModel->getNumMixVars(), pdfMixModel->getNumRxnVars(),
	       meanSpaceVars.d_lsoot);
  cout << "Integrator::h = "<<d_meanValues[0]<<endl;
  cout << "Integrator::f = "<<d_meanValues[1]<<endl;
  cout << "Integrator::gf = "<<d_meanValues[2]<<endl;
  resultStateVars.print(cout);
  return resultStateVars;

}

Stream
Integrator::computeMeanValues(int* tableKeyIndex)
{
  // compute d_meanValues from tableKeyIndex; store mean values in 
  // vector d_meanValues
  convertKeytoMeanValues(tableKeyIndex);
  int mixIndex = 0;
  if (!(pdfMixModel->isAdiabatic()))
    ++mixIndex;
  int rxnIndex = mixIndex + pdfMixModel->getNumMixStatVars() + 1;
  vector<double> mixVars(pdfMixModel->getNumMixVars());
  for (int jj = 0; jj < pdfMixModel->getNumMixVars(); jj++)
    mixVars[jj] = d_meanValues[mixIndex+jj];
  Stream unreactedStream = pdfMixModel->speciesStateSpace(mixVars);
  //cerr << "unreacted Stream: " << d_meanValues[0] << std::endl;
  ChemkinInterface* rxnData = d_rxnModel->getChemkinInterface();
  //unreactedStream.print(cerr, rxnData); 
  // Write independent variables, excluding variance, to separate vector
  // for use by reaction model
  //*****FIX THIS****I think it is now
  int inc = 0;
  if (!(pdfMixModel->isAdiabatic()))
    {
      d_varsHFPi[inc] = d_meanValues[0];
      ++inc;
      //cout << "Int::varsHFPi = " << d_varsHFPi[0] << endl;
    }
  for (int ii = mixIndex; ii < mixIndex+pdfMixModel->getNumMixVars(); ii++)
    {
      d_varsHFPi[inc++] = d_meanValues[ii];
      //cout << "Int::varsHFPi = " << d_varsHFPi[ii] << endl;
    }
  for (int ii = rxnIndex; ii < rxnIndex+pdfMixModel->getNumRxnVars(); ii++)
    {
     d_varsHFPi[inc++] = d_meanValues[ii];
     //cout << "Int::varsHFPi = " << d_varsHFPi[ii] << endl;
    }
 
  Stream outStream;
  d_rxnModel->getRxnStateSpace(unreactedStream, d_varsHFPi, outStream);
  return outStream;
  }


void
Integrator::convertKeytoMeanValues(int tableKeyIndex[]) {
  for (int i = 0; i < d_tableDimension; i++)
    if (tableKeyIndex[i] <= d_tableInfo->getNumDivsBelow(i))
      d_meanValues[i] = tableKeyIndex[i]*d_tableInfo->getIncrValueBelow(i) + 
	d_tableInfo->getMinValue(i);
    else
      d_meanValues[i] = (tableKeyIndex[i]-d_tableInfo->getNumDivsBelow(i))*	
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
    for (int jj = 0; jj < pdfMixModel->getNumMixVars(); jj++) {
      mixVars[jj] = d_keyValues[mixIndex+jj]; 
      // Kluge for stanjan as it blows up if mixture fraction too close to 1
      if (mixVars[jj] > 0.99) 
	mixVars[jj] = 1.0;
    }
    Stream unreactedStream = pdfMixModel->speciesStateSpace(mixVars);
    Stream stateVars;
    d_rxnModel->getRxnStateSpace(unreactedStream, d_varsHFPi, stateVars);
    // from the stream class get relevant variable corresponding to count
    // if lfavre then get temp = temp/density
    double integrand = stateVars.getValue(d_count, d_lfavre);
    // pass only mixing variables
    integrand *= d_mixingPDF->computeShapeFunction(&d_keyValues[mixIndex]);
    return integrand;
}
  





