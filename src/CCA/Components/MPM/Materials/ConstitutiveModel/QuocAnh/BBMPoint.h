#pragma once
#include "BBMMatrix.h"

class BBMPoint
{
public:
	// these variables are made public because they will be used quite a lot and
	// using them by functions is slower and possibly not that clear.
	double stress[6]; //there is the stress state at the point
	double strain[7]; // there is the strain state at the point + suction
	double plastic_strain[6]; //values of plastic strain, no suction included
	double state[3]; //there are the state variables:
		/*
		state [0] - p0* - preconsolidation stress at suction==0
		state [1] - s0 - max suction experienced
		state [2] - specific volume  //yes, it is possible to calculate specific volume but it takes time and is not obvious
		*/
	double MicroStress[6];
	double MicroStrain[7];
	double MicroState[3];
	double MicroPlasticStrain[6];

	BBMPoint(void);
	~BBMPoint(void);
	void read ();
	void write ();
	void Copy (BBMPoint* Result);
	void Update (double* PlasticStrainInc, double* Strain_Incr, double *Stress_Incr, double dPZeroStar);
	double GetMeanStress ();
	double GetShearStress ();
	double GetPStar ();
	double GetSpecVol ();
	double GetSuction ();
	double GetYieldSuction ();
	double GetTheta ();
	double GetThetaDeg ();
	double GetThetaDeg_0 ();
	void UpdatePStar (double value); //sets P0Star parameter
	void SetSuction (double value); //sets suction
	void SetSpecificVolume (double value); //sets specific volume
	void SetPStar (double value);
	void SetYieldSuction (double value);


	//Invariants functions
	double GetFirstInvariant ();
	double GetSecondInvariant ();
	double GetThirdInvariant ();
	double GetFirstDevInvariant();
	double GetSecondDevInvariant();
	double GetThirdDevInvariant();

	//Eigenvalues
	void GetEigen(double Eigen[3]);
	void GetEigen(double Eigen[3],BBMMatrix* EigenVectors);
	void GetEigen(BBMMatrix* EigenValues, BBMMatrix* EigenVectors);
	void SetStressEigen (double Eigen[3]);

    bool CheckIfFinite ();
};
