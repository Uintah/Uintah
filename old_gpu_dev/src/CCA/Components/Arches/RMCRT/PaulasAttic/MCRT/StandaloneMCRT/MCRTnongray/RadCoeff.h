#ifndef RadCoeff_H
#define RadCoeff_H

#include <cmath>

const double SIG = 5.669e-8;
//const double pi = acos(-1);
const double PRES = 1.01325e5;
const double small = 1e-4;
const double EIH2O_TEMP[5] =
  {500.0, 1000.0, 1500.0, 2000.0, 5000.0};
const double EIH2O_AA[5] =
  {-1.03668,-1.26274,-1.3823,-1.5577,-2.7500};
const double EIH2O_BB[5] =
  {0.33619, 0.37853,0.4274,0.4708,0.6896};
const double EIH2O_CC[5] =
  {-0.04208,-0.04386,-0.04796,-0.053,-0.09528};
const double EICO2_TEMP[7] =
  { 500, 1000, 1500, 2000, 2500, 3000, 5000};
const double EICO2_AA[7] =
  { -1.86189, -2.04189, -1.89565, -1.83814, -1.94996, -2.14107, -3.01083};
const double EICO2_BB[7] =
  { 0.18990,  0.22385,  0.18790,  0.21910, 0.25793,  0.30845,  0.38896};
const double EICO2_CC[7] =
  { -0.02613, -0.02004, -0.03814, -0.04054, -0.03876, -0.02882, -0.02949};


class RadCoeff {

public:
  
  RadCoeff(const double &OPL_);
  ~RadCoeff();
  
  void PrepCoeff(const double *CO2,
		 const double *H2O,
		 const double *SFV,
		 const double *T_Vol,
		 double *kl_Vol,
		 const int &VolElementNo, const int &TopBottomNo,
		 const int &xno, const int &yno, const int &zno);


private:
  double OPL;
  double OPF; // conversion of OPL in Meters to feet
  double pi1; // 1/pi
  double PATM;
  double EMG;
  double PCO2, PH2O;
  double CCO2, CH2O, ECO2, EH2O,DELE;
  double PCO2F, PH2OF;
  double PSUMF;
  double PAVE;
  double PML, PWPC;
  double TGI, TGF, EMGF, EMGI;
  double AEK[3], AEM[3];
  double FV;
 
  double EIH2O(const double &TK, const double &PLA, double &EH2O);
  
  double EICO2(const double &TK, const double &PLA, double &ECO2);
  
  void FIND(double ARG, double ANS,
	    const double x[], const double y[],
	    int NPTS, int I);  
  
};

#endif
