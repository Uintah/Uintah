//----- BetaPDFShape.cc --------------------------------------------------


#include <Packages/Uintah/CCA/Components/Arches/Mixing/BetaPDFShape.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <vector>
#include <iostream>
#include <math.h>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for BetaPDFShape
//****************************************************************************
BetaPDFShape::BetaPDFShape(int dimPDF):PDFShape(),d_dimPDF(dimPDF)
{
  d_coef = vector<double>(d_dimPDF+1);
}

//****************************************************************************
// Destructor
//****************************************************************************
BetaPDFShape::~BetaPDFShape()
{
}

//****************************************************************************
// Problem Setup for BetaPDFShape
//****************************************************************************
void 
BetaPDFShape::problemSetup(const ProblemSpecP& /* params*/)
{
}


//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
BetaPDFShape::computePDFFunction(double* meanMixVar,
				 double statVar)
{
  d_validGammaValue = true;
  //cout<<"Beta::meanMixVar ="<<meanMixVar[0]<<" "<<statVar<<endl;
  if ((meanMixVar[0]<0.0000001)||(meanMixVar[0]>0.999999)) {
     d_validGammaValue = false;
     return;
  }
  if (statVar <= 1e-05*meanMixVar[0]) {
     d_validGammaValue = false;
     return;
  }
  //  cout << " gamma value " << d_validGammaValue << endl;
  d_gammafnValue = computeBetaPDFShape(meanMixVar, statVar);
  //  cout<<"Beta::PDFFunction statVar = "<<statVar<<endl;
  //  cout << "LN(gammafnValue) = " << d_gammafnValue << endl; 
  //if (d_gammafnValue < SMALL)  //Old check before Wing put in
  // robust gammafn
  //  d_validGammaValue = false;
  // ***Jennifer's note- This kluge will only work for integration over
  // ONE variable***
  //What if mixVar=0 or 1??; right now, if var too high, it is reduced
  // If var too low, mean values will be returned (see next line)
  return;
}

double
BetaPDFShape::computeBetaPDFShape(double* meanMixVar,
				  double statVar) {
  //--------------------------------------------------------
  //This function calculates the log of the gamma function
  //part of the the BetaPDF.  It is commonly used in 
  //commercial software like Matlab and MathCad.  In order 
  //to handle low variance (large ai values), log of gamma is 
  //used instead.
  //--------------------------------------------------------
  double factor;
  double new_factor = 0.0, new_d_statVar = 0.0;
  double sumMixVars = 0.0;
  double sumSqrMixVars = 0.0;
  //Because multi-dim PDF formulation is used here, need to
  //correct statVar for 1-D PDF
  if (d_dimPDF == 1)
    statVar = 2.0*statVar;
  for (int i = 0; i < d_dimPDF; i++) {
    sumMixVars += meanMixVar[i];
    sumSqrMixVars += meanMixVar[i]*meanMixVar[i];
  }
  double lastMixVars = 1.0 - sumMixVars; // 1.0 - fi's
  sumSqrMixVars += lastMixVars*lastMixVars; //SUM(sqr(fi's))
  //cout<<"statVar = "<<statVar<<endl;
  if (statVar < SMALL) 
    factor = 0.0;
  else
      // (1-S)/Q - 1
    factor = (1.0 - sumSqrMixVars)/statVar - 1.0;
  double sumCoefs = 0.0;
  double multGamma = 0.0;
  vector<double> gammafn(d_dimPDF+1);
  for (int i = 0; i < d_dimPDF; i++) {
    d_coef[i] = meanMixVar[i]*factor; // ai = fi*(1-S)/Q - 1
    //cout << meanMixVar[i] << " Beta(" << i << ") = " << d_coef[i] << endl;

    if (d_coef[i] <= 0.0)
      //upper limit, if Q is greater than (1-S), ai becomes negative.  So we reset the new Q = 0.9*fi*(1-fi)/2
      {
	//	cout << "meanmixvar " << meanMixVar[i] << endl;
	new_d_statVar = 0.9*meanMixVar[i]*(1-meanMixVar[i])/2.0;
	cout << "newstatvar = " << new_d_statVar << endl;
	new_factor = (1.0 - sumSqrMixVars)/new_d_statVar - 1.0;
	d_coef[i] = meanMixVar[i]*new_factor;
	cout << "beta is negative, new beta is calculated" << endl;
      }
    gammafn[i] = dgammaln_(&d_coef[i]); // gammaln(ai)
    //cout << "gammaln(" << i << ") = " << gammafn[i] << endl;
    multGamma += gammafn[i]; // LN(gamma(a1))+LN(gamma(a2))...LN(gamma(a(N-1)))
    sumCoefs += d_coef[i];  // a1+a2+...+a(N-1)
  }
  d_coef[d_dimPDF] = lastMixVars*factor; // computing aN
  if (d_coef[d_dimPDF] <= 0.0)
    {
      new_d_statVar = 0.9*lastMixVars*(1-lastMixVars)/2.0;
      cout << "lastMixVar = "<<lastMixVars<<" mean[d_dimPDF] = "<<meanMixVar[d_dimPDF]<<endl;
      new_factor = (1.0 - sumSqrMixVars)/new_d_statVar - 1.0;
      d_coef[d_dimPDF] = lastMixVars*new_factor;
      cout << "Beta is negative, new beta is calculated" << endl;
    }    

  cout << lastMixVars << " Beta(" << d_dimPDF << ") = " << d_coef[d_dimPDF] << endl;
  gammafn[d_dimPDF] = dgammaln_(&d_coef[d_dimPDF]); // computing gamma(aN)
  //  cout << "gammaln(" << i << ") = " << d_gammafn[i] << endl;
  //  cout << "*******************" << endl;
  multGamma += gammafn[d_dimPDF];    // LN(gamma(a1))+...+LN(gamma(aN))
  //cout << "LN(mult) = " << multGamma << endl;
  sumCoefs += d_coef[d_dimPDF];
  //  cout << "sumCoefs = " << sumCoefs << endl;
  double gammafnNum = dgammaln_(&sumCoefs);   // LN(gamma(a1+a2+...+aN))
  //cout << "LN(sum) = " << gammafnNum << endl;
  return (gammafnNum-multGamma);   
} // gamma(a1+a2+...+aN)/(gamma(a1)*...*gamma(aN))

double
BetaPDFShape::computeShapeFunction(double *var) {
 //---------------------------------------------------
  //This function combines the log gamma function with 
  //the rest of the PDF.  It returns the exponential of
  //of the result (which is the correct value).
  //---------------------------------------------------
  double sumMixVars = 0.0;
  double pdffn = d_gammafnValue;
  for (int i = 0; i < d_dimPDF; i++) {
    sumMixVars += var[i];
    pdffn += (d_coef[i] - 1.0)*log(var[i]);
  }
  double lastMixVar = 1.0 - sumMixVars;
  pdffn += (d_coef[d_dimPDF] - 1.0)*log(lastMixVar);
  return exp(pdffn);
}
    



//
// $Log$
// Revision 1.4  2001/08/25 07:32:45  skumar
// Incorporated Jennifer's beta-PDF mixing model code with some
// corrections to the equilibrium code.
// Added computation of scalar variance for use in PDF model.
// Properties::computeInletProperties now uses speciesStateSpace
// instead of computeProps from d_mixingModel.
//
// Revision 1.3  2001/07/16 21:15:37  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.2  2001/02/02 01:54:33  rawat
// cnges made for checkpointing to work
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//

