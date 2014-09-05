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
  npts2 = 3;
  int mixIndex = 0;
  if (!(pdfMixModel->isAdiabatic()))
    ++mixIndex;
  double fstoic = d_tableInfo->getStoicValue(mixIndex);
  points[0] = fstoic;
  limit = LIMITS;
  bool integralSuccess = false;
  // modify for higher order integration
  // checking for mixing variance
  int mixVarIndex = mixIndex + pdfMixModel->getNumMixVars();
  // compute d_meanValues from tableKeyIndex
  computeKeyValues(tableKeyIndex);
 // d_keyValues is used for Integral function evaluation
  d_keyValues = d_meanValues;
  if (d_meanValues[mixVarIndex] > 0.1) {
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
  // store mean values in vector
  Stream unreactedStream = pdfMixModel->speciesStateSpace(d_meanValues);
  Stream meanSpaceVars = d_rxnModel->computeRxnStateSpace(unreactedStream);
  // compute pdf function for given values of mean and variance
  d_mixingPDF->computePDFFunction(&d_meanValues[mixIndex], d_meanValues[mixVarIndex]);
  // if lfavre temp = temp/density
  vector<double> meanStateSpaceVars = meanSpaceVars.convertStreamToVec(d_lfavre);
  // store integral values in vector and then copy to stream
  vector<double> resultStateSpaceVars(pdfMixModel->getTotalVars());
  // Assumption:
  // Only integrated over mixing variable; rxn_dim = 0
  // check if gammafnc is valid
  if (d_mixingPDF->validIntegral()) { 
    for (d_count = 0; d_count < resultStateSpaceVars.size(); d_count++) 
      {
	// assuming mix_dim == 1
	// for details see gammafn.C
	// fnc is a member function
	dqagpe_(fnc, &lowLimit, &upLimit, &npts2, points, &epsabs, &epsrel,
		&limit, &result, &abserr, &neval, &ier, alist, blist, rlist,
		elist, pts, level, ndin, iord, &last);
	
	if (ier == 0) {
	  integralSuccess = true;
	  resultStateSpaceVars[d_count] = result;
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
  resultStateVars.convertVecToStream(resultStateSpaceVars, d_lfavre);
  return resultStateVars;

}
Stream
Integrator::computeMeanValues(int* tableKeyIndex)
{
  // compute d_meanValues from tableKeyIndex
  computeKeyValues(tableKeyIndex);
 // d_keyValues is used for Integral function evaluation
  d_keyValues = d_meanValues;
  // store mean values in vector
  Stream unreactedStream = pdfMixModel->speciesStateSpace(d_meanValues);
  cerr << "unreacted Stream: " << d_keyValues[0] << std::endl;
  unreactedStream.print(cerr);
  return d_rxnModel->computeRxnStateSpace(unreactedStream);
}


void
Integrator::computeKeyValues(int tableKeyIndex[]) {
  for (int i = 0; i < d_tableDimension; i++) 
    d_meanValues[i] = tableKeyIndex[i]*d_tableInfo->getIncrValue(i) + 
      d_tableInfo->getMinValue(i);
  return;
}

double
fnc(double *x) {
  return pdfMixModel->getIntegrator()->fun(x);
}

double
Integrator::fun(double *x) {
    double y;
    //    d_keyValues[0] = *x; // stores key values to look-up reaction table
    int mixIndex = pdfMixModel->isAdiabatic();
    d_keyValues[mixIndex] = *x;
    // change it to account for reaction progress variable
    //    for (int i = d_numMixVar; i < d_numTableDim; i++)
    //      d_keyValues[i] = d_stateSpaceMeanVars[i];
    //    vector<REAL> stateSpaceVars = d_rxnModel->tableLookUp(d_keyValues);
    //    y = stateSpaceVars[count];
    //    for (int i = d_numMixVar; i < d_numTableDim; i++)
    //      d_keyValues[i] = d_stateSpaceMeanVars[i];
    Stream unreactedStream = pdfMixModel->speciesStateSpace(d_keyValues);
    Stream stateVars = d_rxnModel->computeRxnStateSpace(unreactedStream);
    // from the stream class get relevant variable corresponding to count
    // if lfavre then get temp = temp/density
    y = stateVars.getValue(d_count, d_lfavre);
    // pass only mixing variables
    y *= d_mixingPDF->computeShapeFunction(&d_keyValues[mixIndex]);
    return y;
}
  





