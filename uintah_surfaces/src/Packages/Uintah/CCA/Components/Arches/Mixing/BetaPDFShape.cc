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
  d_vars = vector<double>(2);
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
				 double normStatVar)
{
  d_validGammaValue = true;
  // IMPORTANT: statVar being passed in has been normalized. It needs to be
  // unnormalized for use in this class  
  double maxVar = meanMixVar[0]*(1.0 - meanMixVar[0]);
  double statVar = normStatVar*maxVar; 
  // If var > 0.9maxVar, this function is not called. Table entries
  // are computed from the values at 0.9maxVar and the unreacted
  // values at 1.0maxVar
  // If var is too high, it is reduced in computeBetaPDFShape
  // If var too low, mean values will be returned  
  //if (statVar < SMALL_VARIANCE*maxVar) {
  if (statVar < SMALL_VARIANCE) {
    d_validGammaValue = false;
    //cout << "WARNING: Variance too small" << endl;
    return;
  }
#if 0
  if (statVar > 0.995*maxVar) {
     d_validGammaValue = false;
    return;
  } 
#endif
  if ((meanMixVar[0]<CLOSETOZERO)||(meanMixVar[0]>CLOSETOONE)) {
     d_validGammaValue = false;
     return;
  }
  d_gammafnValue = computeBetaPDFShape(meanMixVar, statVar);
  //  cout<<"Beta::PDFFunction statVar = "<<statVar<<endl;
  //  cout << "LN(gammafnValue) = " << d_gammafnValue << endl; 
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
  // Data needed by computeShapeFunction
  d_vars[0] = meanMixVar[0];
  d_vars[1] = statVar;
  //Because multi-dim PDF formulation is used here, need to
  //correct statVar for 1-D PDF
  if (d_dimPDF == 1)
    statVar = 2.0*statVar;
  //cout << "in computeBetaPDFShape, d_vars = " << d_vars[0] << " " << d_vars[1] << endl;
  for (int i = 0; i < d_dimPDF; i++) {
    sumMixVars += meanMixVar[i];
    sumSqrMixVars += meanMixVar[i]*meanMixVar[i];
  }
  double lastMixVars = 1.0 - sumMixVars; // 1.0 - fi's
  sumSqrMixVars += lastMixVars*lastMixVars; //SUM(sqr(fi's))
  if (statVar < SMALL_VARIANCE) 
    factor = 0.0;
  else 
    factor = (1.0 - sumSqrMixVars)/statVar - 1.0;  // (1-S)/Q - 1
  double sumCoefs = 0.0;
  double multGamma = 0.0;
  vector<double> gammafn(d_dimPDF+1);
  for (int i = 0; i < d_dimPDF; i++) {
    d_coef[i] = meanMixVar[i]*factor; // ai = fi*(1-S)/Q - 1
    //cout << meanMixVar[i] << " Beta(" << i << ") = " << d_coef[i] << endl;
    if (d_coef[i] <= 0.0)
      //Upper limit, if Q is greater than (1-S), ai becomes negative.  
      //So we reset the new Q = 0.9*fi*(1-fi)*2
      {
	new_d_statVar = 0.9*meanMixVar[i]*(1-meanMixVar[i])*2.0;
	if (new_d_statVar < SMALL_VARIANCE)
	  new_factor = 0.0;
	else
	  new_factor = (1.0 - sumSqrMixVars)/new_d_statVar - 1.0;
	d_coef[i] = meanMixVar[i]*new_factor;
	//cout << "new_factor = " << new_factor << endl;
	cout << "beta is negative, new beta is calculated" << endl;
        cout << "f, gf =" << meanMixVar[0] << " " << statVar<< endl;   
    }
    gammafn[i] = dgammaln(&d_coef[i]); // gammaln(ai)
    //cout << "gammaln(" << i << ") = " << gammafn[i] << endl;
    multGamma += gammafn[i]; // LN(gamma(a1))+LN(gamma(a2))...LN(gamma(a(N-1)))
    sumCoefs += d_coef[i];  // a1+a2+...+a(N-1)
  }
  d_coef[d_dimPDF] = lastMixVars*factor; // computing aN
  if (d_coef[d_dimPDF] <= 0.0)
    {
      new_d_statVar = 0.9*lastMixVars*(1-lastMixVars)*2.0;
      if (new_d_statVar < SMALL_VARIANCE)
         new_factor = 0.0;
      else
	 new_factor = (1.0 - sumSqrMixVars)/new_d_statVar - 1.0;
      cout << "lastMixVar = "<<lastMixVars<<" mean[d_dimPDF] = "<<meanMixVar[d_dimPDF]<<endl;
      d_coef[d_dimPDF] = lastMixVars*new_factor;
      //cout << "new_factor = " << new_factor << endl;
      cout << "Beta is negative, new beta is calculated" << endl;
      cout << "f, gf =" << meanMixVar[0] << " " << statVar<< endl;
    }    

  //cout << lastMixVars << " Beta(" << d_dimPDF << ") = " << d_coef[d_dimPDF] << endl;
  gammafn[d_dimPDF] = dgammaln(&d_coef[d_dimPDF]); // computing gamma(aN)
  //  cout << "gammaln(" << i << ") = " << d_gammafn[i] << endl;
  //  cout << "*******************" << endl;
  multGamma += gammafn[d_dimPDF];    // LN(gamma(a1))+...+LN(gamma(aN))
  //cout << "LN(mult) = " << multGamma << endl;
  sumCoefs += d_coef[d_dimPDF];
  //  cout << "sumCoefs = " << sumCoefs << endl;
  double gammafnNum = dgammaln(&sumCoefs);   // LN(gamma(a1+a2+...+aN))
  //cout << "LN(sum) = " << gammafnNum << endl;
  //cout << "gammafn = " << gammafnNum-multGamma << endl;
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
  // Implement James Sutherlands' checks; only works for 
  // one mixture fraction
  double gf = d_vars[1];
  if ( gf < SMALL_VARIANCE) {
    cout << "GF is SMALL" << d_vars[0] << " " << d_vars[1] << endl;
    if ((var[0] <= (d_vars[0] + 100.0*SMALL_VARIANCE))&&
	(var[0] >= (d_vars[0] - 100.0*SMALL_VARIANCE))) 
      return 1.0e9;
    else {
      cout << "returning 0" << endl;
      return 0.0;
    }
  }
  else {
    if (var[0] < SMALL_VARIANCE)
      var[0] = SMALL_VARIANCE;
    else if (var[0] > (1.0-SMALL_VARIANCE))
      var[0] = 1.0 - SMALL_VARIANCE;
  }
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
// Revision 1.10  2003/01/22 00:43:04  spinti
// Added improved BetaPDF mixing model and capability to create a betaPDF table a priori. Cleaned up favre averaging and streamlined sections of code.
//
// Revision 1.9  2002/06/21 16:29:16  dav
// xlC AIX fix: removed _ from fortran names.  This probably will break sgi/linux.  However, need to be consistent and do something link #define to add _ as is done in some files.
//
// Revision 1.8  2001/11/08 19:13:43  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.5  2001/08/26 06:31:48  spinti
// 1. Changed Petsc's convergence criterion to not depend on absolute norm
// 2. Removed zeroing of pressref in PressureSolver
// 3. Added a requires in Properties.cc to make it consistent with taskgraph
// 4. Changed criteria for computing PDF
// 5. Initialized reqwrk array to zero in Stanjan
// 6. Moved location of common block in stanlib.F
//
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

