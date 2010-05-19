#include "RadCoeff.h"
#include "Consts.h"
#include <cmath>
#include <iostream>
#include <cstdlib>

using namespace std;

RadCoeff::RadCoeff(const double &OPL_) {

  OPL = OPL_;
  pi1 = 1 / pi;
  PATM = PRES / PRES;
  OPF = 3.281 * OPL;
  
}


RadCoeff::~RadCoeff(){
}



// as the properties only change with x, so calculate x's first
// then simply assign these values to ys and zs.

void RadCoeff::PrepCoeff(const double *CO2,
			 const double *H2O,
			 const double *SFV,
			 const double *T_Vol,
			 double *kl_Vol,
			 const int &VolElementNo, const int &TopBottomNo,
			 const int &xno, const int &yno, const int &zno) {
  
  
  // C-----------------------------------------------------------------------
  // C	This section calculates CO2-H2O mixture gas emissivities based
  // C	on a combination of a Hottel chart curve fit (originally done
// C	by Phil Smith and students) and a curve fit of high temperature
// C	exponential wide band spectral predictions by Coppalle and 
// C	Vervisch (Combustion and Flame, Vol 49, pp. 101-108, 1983).
// C	If the gas temperature is less than 1200 K, use PJS model;
// C	if the temperature is greater than 2100 K use C & V model;  
// C	if the temperature is 1200-2100 K use linear interpolation
// C	between the Hottel chart point at 1200 K and the C & V point
// C	at 2100 K. This makes the routine run a little slower, but it
// C	is still much faster than Modak's model and just as accurate 
// C	(over a wider temperature range, i.e., 300-3000 K).
// C
// C	Note: Since most temperatures will be 1200-2100 K, do both
// C	      models for all temperatures and determine appropriate
// C	      emissivity to use at end. This improves run time by
// C	      eliminating a nested IF statement. Also, run times can be
// C	      cut almost in half if only one IF option is used with the
// C	      C & V model, e.g., only Pw/Pc = 1 or TG < 2500 K is used.
// C----------------------------------------------------------------------

  int i_index;
  
  for ( int i = 0; i < xno; i ++ ) {

    EMG = 0;
    PCO2 = CO2[i] * PATM;
    PH2O = H2O[i] * PATM;

    PCO2F = PCO2 * OPF;
    PH2OF = PH2O * OPF;
    PSUMF = max((PCO2F + PH2OF),0.0001);
    PAVE = (PATM + PH2O) * 0.5;
    PML = PSUMF / 3.281;
    PWPC = PH2O / (PCO2+small);
    
    if (PWPC <= 1.5) {
      AEK[0] = 0.464* PML;
      AEK[1] = 3.47 * PML;
      AEK[2] = 121.6 * PML;
    }
    else {
      AEK[0] = 0.527 * PML;
      AEK[1] = 3.78 * PML;
      AEK[2] = 99.54 * PML;
    }
    
// C
// C	EICO2 and EIH2O use Hottel charts to calculate (curve-fit)
// C	emissivities for carbon dioxide and water vapor, respectively;
// C	PLA units are atm-ft, Pc and Pw are in atm, TGI is in K
// C
    
    TGI = min(T_Vol[i],1200.0);
    ECO2 = EICO2(TGI, PCO2F, ECO2);
    EH2O = EIH2O(TGI, PH2OF, EH2O);

// C	Calculate total gas emissivity accounting for spectral overlap
// C	and pressure correction
// C
    
    CCO2 = 1.0;
    CH2O = 1.0;
    
    if (PCO2F >= 0.5) 
      CCO2 = 1.0 + 0.14427 * log(PATM);
    else if (PCO2F > 0.0)
      CCO2 = 1.0 + 0.24663 * log(PATM);
	          
    if (PH2OF > 1.0) 
      CH2O = 1.29703 + 0.43168 * log(PAVE);
    else if (PH2OF > 0.0) 
      CH2O = 1.46701 + 0.6663 * log(PAVE);
	         
    if (TGI >= 1005.0) {
      DELE = 0.03047 + 0.01717 * log(PSUMF);
      DELE = min(DELE,0.06);
      DELE = max(DELE,0.0);
    }
    else{	         
      DELE = 0.01544 + 0.01045 * log(PSUMF);
      DELE = min(DELE,0.035);
      DELE = max(DELE,0.0);
    }
    
    EMGI = CCO2 * ECO2 + CH2O * EH2O - DELE;

// C	Use Coppalle and Vervisch model to calculate CO2-H2O mixture 
// C	total emissivity by summing 3 gray gases (plus one clear).
// C	Compute emissivity weighting factors for CO2-H2O mixture
// C	based on linear fit of temperature (for Ptot = 1 atm, TGF in K).
// C	First coefficient set is for Pw/Pc = 1, second is for Pw/Pc = 2.
// C
    
    TGF = max(T_Vol[i],2100.0);
    
    if (TGF < 2500.0) {
      AEM[0] = 0.136 + 0.0000726 * TGF;
      AEM[1] = 0.516 - 0.000163 * TGF;
      AEM[2] = 0.0517 - 0.0000176 * TGF;
    }
    else{
      AEM[0] = 0.464 - 0.0000596 * TGF;
      AEM[1] = 0.336 - 0.0000909 * TGF;
      AEM[2] = 0.0245 - 0.00000654 * TGF;
    }
    

    if (TGF < 2500.0 && PWPC >= 1.5) {
      AEM[0] = 0.132 + 0.0000725 * TGF;
      AEM[1] = 0.547 - 0.000171 * TGF;
      AEM[2] = 0.0489 - 0.0000176 * TGF;
    }
    else if (PWPC >=  1.5) {
      AEM[0] = 0.430 - 0.0000472 * TGF;
      AEM[1] = 0.370 - 0.000101 * TGF;
      AEM[2] = 0.0184 - 0.00000511 * TGF;
    }
    
    EMGF = AEM[0] * (1.0 - exp(-AEK[0]))
      + AEM[1] * (1.0 - exp(-AEK[1]))
      + AEM[2] * (1.0 - exp(-AEK[2]));

// C	Determine which model is appropriate for given gas temperature
// C
    if ( T_Vol[i] <= 1200.0)
      EMG = EMGI;
    else if (T_Vol[i] >= 2100.0)
      EMG = EMGF;
    else
      EMG = EMGI + (EMGF-EMGI) 
	* ((T_Vol[i]-1200.0) / 900.0);



// c WARNING! The second of the following limits is needed to make sure
// c negative argument will not go in log function. Gautham needs to investigate
// c why emg becomes greater then one.
    
    EMG = max (EMG, 0.0);
    EMG = min (EMG, 0.9);
    
// C
// C	Calculate absorption coefficient from Bouguer's Law
// C
    
// ABSKG = kl_Vol;
    
    kl_Vol[i] = -log (1.0 - EMG) / OPL;

    
// C-----------------------------------------------------------------------
// C	Soot model is based on a correlation given by Sarofim and Hottel
// C	in "Radiative Transfer in Combustion Chambers: Influence of 
// C	Alternative Fuels," Sixth Intl. Heat Transfer Conf., Toronto,
// C	Vol 6, pp 199-217, August, 1978.  Soot formation is based on
// C	mixture fraction, density and moles of carbon at each local 
// C	cell. Soot destruction is based on local equivalence ratio.
// C	Scattering effects due to soot agglomerates are not considered
// C	(only soot absorption/emission). 
// C	THIS MODEL IS NOT YET DESIGNED TO BE USED WITH PARTICLE
// C	(COAL) CALCULATIONS.
// C-----------------------------------------------------------------------

    FV = SFV[i];
    kl_Vol[i] = kl_Vol[i] +
      min( 25.0,( (4.0/OPL) * log(1.0 + 350.0 * FV * T_Vol[i] * OPL)) );

    

    for ( int j = 0; j < zno; j ++ ) {

      for ( int k = 0; k < yno; k ++ ) {

	i_index = i + xno * k + TopBottomNo * j;
	kl_Vol[i_index] = kl_Vol[i];
      } // end of for k

    } // end of for j
    

  } // end of for i


}
  
  



void RadCoeff::FIND(double ARG, double ANS,
		    const double x[], const double y[],
		    int NPTS, int I){

  bool flag;
  flag = 0;
  
  do{
    if (ARG <= x[0]){
      ANS = y[0];
      flag = 1;
    }
    else if (ARG >= x[NPTS -1]){
      ANS = y[NPTS-1];
      flag = 1;
    }
    else if (ARG <= x[I]){
      ANS = y[I] - (y[I] - y[I-1]) * (x[I] - ARG) / (x[I] - x[I-1]);
      flag = 1;
    }
    else
      I = I + 1;
  }while ( !flag );
  


  
}







  
double RadCoeff::EIH2O(const double &TK, const double &PLA, double &EH2O){

  double TR,PL,ALOGPL,TMP,A,B,C;
  int ITEMP;
  
// C	TR = DEG R, TK = DEG K; PLA = FT-ATM

  // ITEMP = 1; // in fortran
  ITEMP = 0; // in c++
  TR = TK * 1.8;

// C	Set bounds, interpolate Hottel chart to find H2O emissivity

  if (PLA >= 0.005) {
    PL = min(PLA, 20.0);
    TMP = max(TR, 500.0);
    TMP = min(TR, 5000.0);
    FIND(TMP, A, EIH2O_TEMP, EIH2O_AA, 5, ITEMP);
    FIND(TMP, B, EIH2O_TEMP, EIH2O_BB, 5, ITEMP);
    FIND(TMP, C, EIH2O_TEMP, EIH2O_CC, 5, ITEMP);
    ALOGPL = log(PL);
    EH2O = exp(A + ALOGPL * (B + ALOGPL * C));
  }
  else
    EH2O = 0;

  return EH2O;
  
}





double RadCoeff::EICO2(const double &TK, const double &PLA, double &ECO2){

  double TR,PL,ALOGPL,TMP,A,B,C;
  int ITEMP;

// C	TR = DEG R, TK = DEG K; PLA = FT-ATM

  ITEMP = 1;
  TR = TK * 1.8;

// C	Set up bounds, interpolate Hottel chart to find CO2 emissivity

  if (PLA >= 0.001){
    PL = min(PLA, 5.0);
    TMP = max(TR, 500.0);
    TMP = min(TR, 5000.0);
    FIND( TMP, A, EICO2_TEMP, EICO2_AA, 7, ITEMP);
    FIND( TMP, B, EICO2_TEMP, EICO2_BB, 7, ITEMP);
    FIND( TMP, C, EICO2_TEMP, EICO2_CC, 7, ITEMP);
    ALOGPL = log(PL);
    ECO2 = exp(A + ALOGPL * (B + ALOGPL * C));
  }
  else
    ECO2 = 0;

  return ECO2;
 
}
  
  
