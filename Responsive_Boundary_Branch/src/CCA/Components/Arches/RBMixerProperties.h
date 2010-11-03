// This header file defines the RBMixerProperties class:


#ifndef Uintah_Components_Arches_RBMixerProperties_h
#define Uintah_Components_Arches_RBMixerProperties_h

#include<iostream>
#include<cmath>
#include<vector>
#include<string>
#include<CCA/Components/Arches/RBComponentProperties.h>
//#include"RBComponentProperties.h"

using namespace std;


namespace Uintah{





//-------------------------------------------------------

/** 
* @class RBMixerProperties
* @author Weston Eldredge (weseldredge@yahoo.com)
* @date February 2009
*
* @brief This class supports Responsive Boundary class, it calculates liquid
* fuel mixture properties (bubble point, Vapor/Liquid equilibrium, burn rates,
* activity coefficients etc.) 
*
*/

//-------------------------------------------------------






class RBMixerProperties {

	public:

	//Default Constructor:

	RBMixerProperties();

	//Standard Constructor:

	RBMixerProperties(vector<bool>);

	//Standard Destructor:

	~RBMixerProperties();

	//Getter Methods:

	int getNOC(){ return M_NOC; }
	double getGamma(int i) { return M_Gamma.at(i); }
	double getY(int i) { return M_filmComp.at(i); }
	double getSumY() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum +  getY(i); } return sum;}
	double getM(int i) { return M_MassFilmComp.at(i); }
	double getSumM() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum + getM(i);} return sum;}
	double getLMC(int i) { return M_LiquidMassComp.at(i); }
	double getDD(int i, int j) { return M_DD[i][j]; }
	double getDMS(int i, int j) {return M_DMS[i][j];}
	double getBB(int i, int j) { return M_BB[i][j]; }
	double getBI(int i, int j) { return M_BI[i][j]; }
	double getGG(int i, int j) { return M_GG[i][j]; }
	double getFD(int i, int j) { return M_FD[i][j]; } 
	double getDab(int i, int j) { return M_Dab[i][j]; }
	double getKC(int i, int j) { return M_KC[i][j]; }
	double getRR(int i, int j) { return M_RR[i][j]; }
	double getK(int i, int j) { return M_K[i][j]; }
	double getPHI(int i, int j) {return M_phi[i][j]; }
	double getTHETA(int i, int j) { return M_theta[i][j]; }
	double getBETA(int i, int j) { return M_BETA[i][j]; }
	double getNflux(int i) { return M_NFLUX.at(i); }
	double getMflux(int i) { return M_MFLUX.at(i); }
	double getEflux(int i) { return M_EFLUX.at(i); }
        double getHevap(int i) { return M_HEVAP.at(i); }
	double getSumNflux() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum + getNflux(i);} return sum;}
	double getSumMflux() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum + getMflux(i);} return sum;}
	double getSumEflux() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum + getEflux(i);} return sum;}
	double getRes(int i) { return M_Res.at(i); }
	double getResNorm() { double sum = 0; for (int i = 0; i < M_NOC; i++){sum = sum + abs(getRes(i));} return sum;}
	double getFConc() { return M_filmConc; }
	string getName(int i) { return M_Component[i]->getName(); }
	double getLDEN() { return M_LDEN; }
	double getLMCon() { return M_LMC; }
	double getEnthalpy(double T, double Tref, int i) { return M_Component[i]->liquidSensibleHeat(T,Tref); }
        double getHeatTransferCoeff() { return M_HeatTransferCoeff; }

	//Print Methods: (used for diagnostic purposes in developing this code, not actually needed for the operation of the code)
	void printLTHK(double T){
	 for (int i = 0; i < M_NOC; i++){
	  cout << M_Component[i]->getName() << " thermal conductivity at " << T << " kelvins: ";
	  cout << M_Component[i]->liquidThermalConductivity(T) << "\n";}}
	 
	void printLCP(double T){
	 for (int i = 0; i < M_NOC; i++){
	  cout << M_Component[i]->getName() << " heat capacity at " << T << " kelvins: ";
	  cout << M_Component[i]->liquidHeatCapacity(T) << "\n";}}

	void printLD(double T, double P){
	  for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " liquid density at " << T << " kelvins: ";
	   cout << M_Component[i]->liquidDensity(T,P) << "\n";}}

	void printFilmCP(double T){
	  for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " gas heat capacity at " << T << " kelvins: ";
	   cout << M_Component[i]->gasHeatCapacity(T) << "\n";}}
	
	void printFilmTHK(double T){
	  for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " gas thermal conductivity at " << T << " kelvins: ";
	   cout << M_Component[i]->gasThermalConductivity(T) << "\n";}}

	void printFilmVis(double T){
	  for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " gas viscosity at " << T << " kelvins: ";
	   cout << M_Component[i]->gasViscosity(T) << "\n";}}

	void printVP(double T){
	 for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " vapor pressure at " << T << " kelvins: ";
	   cout << M_Component[i]->vaporPressure(T) << "\n";}}
	
	void printLvis(double T){
	 for (int i = 0; i < M_NOC; i++){
	   cout << M_Component[i]->getName() << " liquid viscosity at " << T << " kelvins: ";
	   cout << M_Component[i]->liquidViscosity(T) << "\n";}}


	//Mixture Property Methods:
	double mix_liquidThermalConductivity(double, vector<double>);
	double mix_liquidHeatCapacity(double, vector<double>);
	double mix_liquidDensity(double, double, vector<double>);
	double film_heatCapacity(double);
	double film_thermalConductivity(double);
	double film_Viscosity(double);
	double film_Density(double, double);
	void Wilson(double, double, vector<double>);
	void Wilson2(double, double, vector<double>);
	void film_VLE(double, double, vector<double>);
	double BubblePointT(double, vector<double>);
	void diluteDiffusivity(double, double);
	void Fickian(double, double, vector<double>);
	void gasDiffusivity(double, double);
	void MTCoefficient(double, double, double, double, double);
	void MassFraction();
	void MassFraction(vector<double>);
        vector<double> MoleFraction(vector<double>);
        void film_Fluxes(double, double, vector<double>, double, double, double, double, double,double); //boiling
	void film_Fluxes(double,double,double,vector<double>,vector<double>,double,double,double); //non-boiling
	void film_Fluxes(vector<double>);
        void film_Flux_Main(double,double,double,vector<double>,vector<double>,double,double,double,double);
	void Residual(vector<double>,vector<double>);
	void computeTheta(int);

	//Math Methods: (The matrix methods here may not be the most efficient available)
	double KD(int i, int j){if (i == j){return 1.0;} else{return 0.0;}};
	void BBinvert();	
	void RRinvert();
	double Fact(double N) { if (N == 1) {return N;}
	   if ( N > 1 ) { return (N * Fact(N-1)); }};




	private:

	//members:
	vector<bool> M_fuelvec;
	int M_NOC; //number of components
	vector<RBComponentProperties*> M_Component;
	vector<double> M_Gamma; //vector for activity coefficients
	vector<double> M_filmComp; // vapor film composition (mole fraction) not including air
	vector<double> M_MassFilmComp; // vapor film compsoition (mass fraction) not including air
	vector<double> M_LiquidMassComp; // liquid mass fractions
	double M_DD[20][20]; //dilute diffusivity matrix
	double M_DMS[20][20]; //non-dilute diffusivity matrix
	double M_GG[20][20]; //Thermodynamic Gamma Matrix
	double M_BB[20][20]; // M_S diffusion matrix
	double M_BI[20][20]; // Inverse of BB.
	double M_FD[20][20]; // Fickian Diffusivity Matrix.

	//members associated with film interpahse mass transfer:
	double M_Dab[20][20]; 
	double M_KC[20][20];
	double M_RR[20][20];
	double M_K[20][20];
	double M_phi[20][20];
	double M_theta[20][20];
	double M_BETA[20][20];
	vector<double> M_NFLUX;
	vector<double> M_MFLUX;
	vector<double> M_EFLUX;
        vector<double> M_HEVAP;
	vector<double> M_Res;
	double M_filmConc;
	double M_HeatTransferCoeff; 
        bool M_convergeFluxProblem;       


	double M_LDEN;
	double M_LMC;
        double M_Kfactor;  //Adjustment parameter for mass transfer coefficients (for validation purposes)		

};

}  //namespace Uintah

#endif


