/** @brief This file (RBMixerProperties.cc)  defines the various methods for the RBMixerProperties class. */


#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include<CCA/Components/Arches/RBMixerProperties.h>
//#include"RBMixerProperties.h"


using namespace std;
using namespace Uintah;

//Standard Constructor:

RBMixerProperties::RBMixerProperties(vector<bool> fv):M_fuelvec(fv)
{
	
	M_NOC = 0;
	for (unsigned int i = 0; i < M_fuelvec.size(); i++)
	{

	 fuel f1 = n_Octane;
	 f1 = static_cast<fuel>(f1 + i);

	 if (M_fuelvec.at(i) == 1){
	   M_Component.push_back(new RBComponentProperties(f1));
	   M_NOC++;}
	}

//	cout << "The system is a " << getNOC() << "-component system." << "\n";
//	cout << "The Components are: " << "\n";
//	for (int i = 0; i < M_NOC; i++){
//	 cout << "(" << (i + 1) << "):  " <<  M_Component[i]->getName() << "\n";}

        M_convergeFluxProblem = 0;
}


//Standard Destructor:

RBMixerProperties::~RBMixerProperties(){}



//Methods:

//*********************************************************************************************************************
//*********************************************************************************************************************
/* @brief The method RBMixerProperties::mix_liquidThermalConductivity estimates the thermal conductivity of a liquid 
 mixture using the power law of Vredeveld (1973). The method has been tested well for non-aqueous systems of two and 
 three components in which the ratio of pure conductivities is no larger than 2.  For details see pg. 10.60 of the 
 5th edition of Poling et al.
 Inputs for the method are system Temperature (Kelvins) and sytem mass fractions. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::mix_liquidThermalConductivity(double T, vector<double> m)
{

	int n = M_NOC;
	vector<double> plthk(n);

// first obtain pure liquid component Thermal Conductivities:
	for (int i = 0; i < n; i++)
	{
	 plthk[i] = M_Component[i]->liquidThermalConductivity(T);
	}

	double mthk;
	double sum = 0;

	for (int i = 0; i < n; i++)
	{
	 sum = sum + m[i]/pow(plthk[i],2);
	}

	mthk = pow(sum,-0.5);

	return mthk;
}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBMixerProperties::mix_liquidHeatCapacity calculates the heat capacity of a liquid mixture.  
 It assumes negligible mixing effects. The input to the method is the system temperature (Kelvins) and the mass 
 fractions of the components. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::mix_liquidHeatCapacity(double T, vector<double> m)
{

	int n = M_NOC;
	vector<double> plcp(n);

//obtain pure liquid heat capacities:
	for (int i = 0; i < n; i++) 
	{
	 plcp[i] = M_Component[i]->liquidHeatCapacity(T);
	}

	double mlcp;
	double sum = 0;

	for (int i = 0; i < n; i++)
	{
	 sum = sum + m[i]*plcp[i];
	}

	mlcp = sum;

	return mlcp; // (J/kg-K)

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/* @brief The method RMBixerProperties::mix_liquidDensity calculates the density of a liquid mixture using 
 the method described by Thomson, Brobst, and Hankinson (1982) (AICHE journal v.28 (671)).  The inputs to 
 the method are system temperature (K), the system pressure (Bar) and the component mole fractions. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::mix_liquidDensity(double T, double P, vector<double> x)
{

	int n = M_NOC;

//obtain parameters from compoent classes:


        double Vstar[n], Wsrk[n], Tc[n], MW[n];
	double dummy;
	double R = 83.14; //gas constant (Bar-cm^3/(K-mol))
	

	for (int i = 0; i < n; i++) {
	 dummy = M_Component[i]->liquidDensity(T,P);
	 Vstar[i] = M_Component[i]->getVstar();
	 Wsrk[i] = M_Component[i]->getWSRK();
	 Tc[i] = M_Component[i]->getTC();
	 MW[i] = M_Component[i]->getMW();}
 	
	
	double VT[n][n];

	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
		 VT[i][j] = pow((Vstar[i]*Tc[i]*Vstar[j]*Tc[j]),0.5);}}

//calculate mixture Vstar, Tc, and Wsrk:

	double Vm, Tcm, Wsrkm, MWM;
	double sum1 = 0;
	double sum2 = 0;
	double sum3 = 0;
	double Wsum = 0;
	double MWsum = 0;

	for (int i = 0; i < n; i++){
	  sum1 = sum1 + x[i]*Vstar[i];
	  sum2 = sum2 + x[i]*pow(Vstar[i],2./3.);
	  sum3 = sum3 + x[i]*pow(Vstar[i],1./3.);
	  Wsum = Wsum + x[i]*Wsrk[i];
	  MWsum = MWsum + x[i]*MW[i];}

	MWM = MWsum;
	Wsrkm = Wsum;
	Vm = 0.25 * (sum1 + 3*sum2*sum3);

	double sum4 = 0;

	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
		  sum4 = sum4 + x[i]*x[j]*VT[i][j];}}

	Tcm = sum4/Vm;

	double Zcm, Pcm;

	Zcm = 0.291 - 0.080 * Wsrkm;

	Pcm = Zcm * R * Tcm / Vm;


//mixture reduced variables:
	double Trm, Prm, beta, alpha, Prm0, Prm1, Psm;

	Trm = T/Tcm;

	alpha = 35.0  - 36.0 / Trm - 96.736*log10(Trm) + pow(Trm,6);

	beta = log10(Trm) + 0.03721754*alpha;

	Prm0 = 5.8031817 * log10(Trm) + 0.07608141 * alpha;
	
	Prm1 = 4.86601 * beta;

	double lPrm = Prm0 + Wsrkm * Prm1;

	Prm = pow(10,lPrm);

//saturation pressure of the mixture:
	Psm = Prm * Pcm;

//Saturated Molar Volume of Liquid Mixture

	double Vs, Vr0, Vr1;

	double a = -1.52816;
        double b = 1.43907;
        double c = -0.81446;
        double d = 0.190454;
        double e = -0.296123;
        double f = 0.386914;
        double g = -0.0427258;
        double h = -0.0480645;
	
	double tau = (1 - Trm);

	Vr0 = 1 + a*pow(tau,1./3.) + b*pow(tau,2./3.) + c*tau + d*pow(tau,4./3.);

	Vr1 = (e + f*Trm + g*pow(Trm,2) + h*pow(Trm,3))/(Trm - 1.00001);

	Vs = Vm*Vr0*(1 - Wsrkm*Vr1);


//Now use the Tait equation to obtain the compressed molar volume:


	double ca = -9.070217;
        double cb = 62.45326;
        double cd = -135.1102;
        double cf = 4.79594;
        double cg = 0.250047;
        double ch = 1.14188;
        double cj = 0.0861488;
        double ck = 0.0344483;

	
	double C = cj +ck*Wsrkm;
        double eee, B, v, p;

        eee = exp(cf +cg*Wsrkm + ch*pow(Wsrkm,2.));

        B = (Pcm)*(-1. + ca*pow(tau,1./3.) + cb*pow(tau,2./3.) + cd*tau + eee*pow(tau,4./3.));
        v = Vs*(1. - C*log((B + P)/(B + Psm))); // The compressed molar volume (cm^3/mol)

	//Convert to density

	double density, molarvolume;

	M_LMC = 1/v*1000000; //convert to concentration (mol/m^3)
	density = MWM/v*1000; // Converted to kg/m^3;
	M_LDEN = density;
	

	return density;


}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBMixerProperties::film_heatCapacity calcuates film heat capacity for the liquid pool model, 
 it uses a mass fraction averaged approach.  The inputs are the system temperature (Kelvins), and fuel mass fractions 
 (calculated from film_VLE). */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::film_heatCapacity(double T)
{

	int n = M_NOC;
	double Cp[n];

	double sum = 0;

//calculate mass fraction of air in the film:

	for (int i = 0; i < n; i++){
	   Cp[i] = M_Component[i]->gasHeatCapacity(T);}

	 
	double ma = 1 - getSumM();



//calculate film Cp:

	double FHC = 1000*ma;

	for (int i = 0; i < n; i++){
	   FHC = FHC + Cp[i]*getM(i);}

	return FHC;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************









//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBMixerProperties::film_thermalConductivity estimates the thermal conductivity of 
 gas mixtures using the Wassilijewa Equation and Mason and Saxena Modification.  Details are found in 
 Poling et al. 5th edition ppg. 10.30-31.  Inputs are film Temperature (kelvins), and mole fractions of 
 fuel species (calculated from film_VLE). */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::film_thermalConductivity(double T)
{

	int n = M_NOC;
	double thk[n+1], Tc[n+1], Pc[n+1], MW[n+1], X[n+1];


	double sum = 0;
	
	for (int i = 0; i < n; i++){
	 X[i] = getY(i);
	 thk[i] = M_Component[i]->gasThermalConductivity(T);
	 Tc[i] = M_Component[i]->getTC();
	 Pc[i] = M_Component[i]->getPC();
	 MW[i] = M_Component[i]->getMW();}

	X[n] = 1 - getSumY();

//air properties:
	Tc[n] = 132.2; //Kelvins
	Pc[n] = 37.45; //Bar
	MW[n] = 29.1;  //g/mol


	//Air Thermal Conductivity

        thk[n] = 1.5524e-3 + 8.5032e-5*T - 2.3810e-9*T*T - 2.2222e-11*T*T*T; //(W/m-K)



	//Film Thermal Conductivity Calculation:
        double G[n+1], L[n+1], A[n+1][n+1], sum1, thcond;

        for (int i = 0; i < (n+1); i++){
        G[i] = 210*pow(Tc[i]*pow(MW[i],3)/pow(Pc[i],4),(1./6.));
        L[i] = (exp(0.0464*Tc[i]/T) - exp(-0.2412*Tc[i]/T))/G[i];}

        for (int i = 0; i < (n+1); i++){ for (int j = 0; j < (n+1); j++){
                A[i][j] = pow((1 + sqrt(L[i]/L[j])*pow(MW[i]/MW[j],0.25)),2)/sqrt(8*(1 + MW[i]/MW[j]));}}

        sum = 0.0;

        for (int i = 0; i < (n+1); i++){ sum1 = 0.0;
           for (int j = 0; j < (n+1); j++){ sum1 = sum1 + X[j]*A[i][j]; }
                sum = sum + X[i]*thk[i]/sum1; }

        thcond = sum; // (W/m-K)

        return thcond;


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBMixerProperties::film_Viscosity calculates the viscosity of the gas mixture 
  in the film above the pool surface (in kg/m/s.)  This mixture calculation uses the method of Reichenberg 
  as described in Poling et al. 5th edition, pg. 9.15.  The inputs are the film temperature (Kelvins) and 
  the film composition (mole fractions).  The film composition is now computed from film_VLE, and is no 
  longer a direct input to this method. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::film_Viscosity(double T)
{

	int n = M_NOC;

	double Tc[n+1], Pc[n+1], MW[n+1], dp[n+1], vis[n+1], X[n+1];
	double sum;

	sum = 0;

	for (int i = 0; i < n; i++){
	 Tc[i] = M_Component[i]->getTC();
	 Pc[i] = M_Component[i]->getPC();
 	 MW[i] = M_Component[i]->getMW();
	 dp[i] = M_Component[i]->getDP();
	 vis[i] = M_Component[i]->gasViscosity(T);
	 X[i] = getY(i);}


	// Air properties:

	Tc[n] = 132.2; //Kelvins
	Pc[n] = 37.45; //Bar
	MW[n] = 29.1; //g/mol
	dp[n] = 0.0; //debye
        vis[n] = 5.0477e-7 + 7.2825e-8*T - 4.8524e-11*T*T + 1.7778e-14*T*T*T; // kg/m-s
	X[n] = 1.0 - getSumY();


	// Calculate Mixture Viscosity
	
        double mur[n+1], Tr[n+1][n+1], murr[n+1][n+1], Fr[n+1][n+1], U[n+1], C[n+1], H[n+1][n+1], K[n+1];
        double sum1, sum2, sum3, num, den, den1, den2, den3, viscosity;


        //Main Body:
        for (int i = 0; i < (n+1); i++){ mur[i] = 52.46*pow(dp[i],2)*Pc[i]/pow(Tc[i],2); }

        for (int i = 0; i < (n+1); i++){ for(int j = 0; j < (n+1); j++) {
                Tr[i][j] = T/sqrt(Tc[i]*Tc[j]);
                murr[i][j] = sqrt(mur[i]*mur[j]); }}

        for (int i = 0; i < (n+1); i++) { for (int j = 0; j < (n+1); j++) {
                num = (pow(Tr[i][j],3.5) + (10*pow(murr[i][j],7)));
                den = pow(Tr[i][j], 3.5)*(1. + (10*pow(murr[i][j],7)));
                Fr[i][j] = num/den; }}

        for (int i = 0; i < (n+1); i++){
                num = pow(1. + 0.36*Tr[i][i]*(Tr[i][i] - 1.),(1./6.))*Fr[i][i];
                U[i] = num/sqrt(Tr[i][i]);
                C[i] = pow(MW[i], 0.25)/sqrt(vis[i]*U[i]);}

        for (int i = 0; i < (n+1); i++) { for (int j = 0; j < (n+1); j++) {
                den = pow((MW[i] + MW[j]), 3);
                den1 = pow((C[i] + C[j]),2);
                den2 = pow((1.0 + 0.36*Tr[i][j]*(Tr[i][j] - 1.0)),(1./6.));
                den3 = Fr[i][j]/sqrt(Tr[i][j]);
                H[i][j] = sqrt((MW[i]*MW[j])/32/den)*den1*den2*den3;}}

        for (int i = 0; i < (n+1); i++) {
                sum = 0.0; for (int k = 0; k < (n+1); k++) { if ( k != i ) {
                sum = sum + X[k]*H[i][k]*(3 + (2*MW[k]/MW[i]));}}
                K[i] = X[i]*vis[i]/(X[i] + vis[i]*sum);}

        sum1 = 0.0;
        for (int i = 0; i < (n+1); i++) { sum2 =0.0;
                for (int j = 0; j < i; j++) { sum2 = sum2 + H[i][j]*K[j]; }
                sum3 = 0.0;
                for (int j = 0; j < (n+1); j++) { if (j != i) {
                        for (int k = 0; k < (n+1); k++) { if (k != i) {
                                sum3 = sum3 + H[i][j]*H[i][k]*K[j]*K[k]; }}}}
                sum1 = sum1 + K[i]*(1 + 2*sum2 + sum3);}

                viscosity = sum1;

                return viscosity;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The simple method RBMixerProperties::film_Density calculates gas mixture density for the film above 
 the pool surface using a simple mole fraction weighted approach.  Assumes ideal gas conditions.  
 Inputs are film temperature (Kelvins), system pressure (bar),
 and the film composition (mole fractions).  Film composition is now computed in film_VLE and will no longer be
 used as a direct input to this method. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double 
RBMixerProperties::film_Density(double T, double P)
{

	int n = M_NOC;
	double MW[n];

	double sum;

	sum = 0;

	for (int i = 0; i < n; i++){
	 MW[i] = M_Component[i]->getMW();}
	 

	//Air properties:
	double xa = 1.000 - getSumY();
	double MWa = 29.1;

	double MWmix = xa*MWa;

	for (int i = 0; i < n; i++){
	 MWmix = MWmix + getY(i)*MW[i];}

	

	// Mixture density calculation:

	double Film_Density;
	double R = 8.314e-5; // gas constant (Bar - m^3/k-mol)

	Film_Density = P*MWmix/R/T/1000; // density in kg/m^3

	return Film_Density;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBMixerProperties::Wilson calculates the liquid phase activity coefficients for a liquid mixture using the 
 Wilson equation (Wilson, 1964).  The Wilson coefficients are estimated using the method of 
 Gothard et al. (Ind. Eng. Chem. V(15) N.2, pg. 333, 1976).  The inputs are system Temperature (Kelvins),
 system Pressure (Bar), and liquid composition (mole fraction). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::Wilson(double T, double P, vector<double> x)
{ 

	int n = M_NOC;

	double MW[n], Hv[n], Vm[n], SP[n];
	double R = 8.314; // gas constant (J/k-mol)
	double dummy;

	M_Gamma.clear();

	for (int i = 0; i < n; i++){
	 MW[i] = M_Component[i]->getMW();
//	 Hv[i] = M_Component[i]->latentHeat(T)*MW[i]/1000; //access latent heat of vaporization and convert to (J/mol)
	 Hv[i] = M_Component[i]->getHV();
	 dummy = M_Component[i]->liquidDensity(T,P);
	 Vm[i] = (M_Component[i]->getLMV())*1e-6; //access molar volume and convert to m^3/mol
	 SP[i] = sqrt((Hv[i] - R*T)/Vm[i]);} // compute solubility parameters (J/m^3)^1/2


	//Estimate Wilson Parameters with correlation from Gothard et al.
	
	double L[n][n], g[n][n];

	for (int i = 0; i < n; i++) { for (int j = i; j < n; j++){
	 double delta = abs(SP[i] - SP[j]);
 	 g[i][j] = -400.95 + 0.28127*delta - 1.41756e-5*pow(delta,2.0); // J/mol
	 g[j][i] = 186.17 + 0.14849*delta + 8.32908e-6*pow(delta,2.0); // J/mol
	
	 if (i != j){
	 L[i][j] = Vm[j]/Vm[i]*exp(-g[i][j]/R/T);
	 L[j][i] = Vm[i]/Vm[j]*exp(-g[j][i]/R/T);}
	
	 else { L[i][j] = 1.0; L[j][i] + 1.0;}}}

	
	//Now to calcuate the activity coefficients using the multi-component form of the Wilson equation:

	double sum1, sum2, sum3, gamma;

	for (int i = 0; i < n; i++){

	 sum1 = 0;

	 for (int j = 0; j < n; j++){
	  sum1 = sum1 + x[j]*L[i][j];}

	 sum1 = -log(sum1);

	 sum2 = 0;

	 for (int k = 0; k < n; k++){

	  sum3 = 0; 
	  for (int l = 0; l < n; l++){
	   sum3 = sum3 + x[l]*L[k][l];}

	  sum2 = sum2 + (x[k]*L[k][i])/sum3;}

	 gamma = exp(sum1 + 1 - sum2);
	
	 M_Gamma.push_back(gamma);}
	

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief  The method RBMixerProperties:: Wilson2  also makes use of the Wilson equation in order to calculate the 
 coefficients of the Gamma Matrix for use in Multi-Component mass diffusion in the liquid phase for details of the computation see Taylor
 and Krishna (1993) pg. 544.  Inputs are system temperature (Kelvins), system pressure (Bar), and liquid composition
 (mole fraction). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::Wilson2(double T, double P, vector<double> x)
{

	int n = M_NOC;

        double MW[n], Hv[n], Vm[n], SP[n];
        double R = 8.314; // gas constant (J/k-mol)
        double dummy;


        for (int i = 0; i < n; i++){
         MW[i] = M_Component[i]->getMW();
//         Hv[i] = M_Component[i]->latentHeat(T)*MW[i]/1000; //access latent heat of vaporization and convert to (J/mol)
	 Hv[i] = M_Component[i]->getHV();
         dummy = M_Component[i]->liquidDensity(T,P);
         Vm[i] = (M_Component[i]->getLMV())*1e-6; //access molar volume and convert to m^3/mol
         SP[i] = sqrt((Hv[i] - R*T)/Vm[i]);} // compute solubility parameters (J/m^3)^1/2

        //Estimate Wilson Parameters with correlation from Gothard et al.

        double L[n][n], g[n][n];

        for (int i = 0; i < n; i++) { for (int j = i; j < n; j++){
         double delta = abs(SP[i] - SP[j]);
         g[i][j] = -400.95 + 0.28127*delta - 1.41756e-5*pow(delta,2.0); // J/mol
         g[j][i] = 186.17 + 0.14849*delta + 8.32908e-6*pow(delta,2.0); // J/mol

         if (i != j){
         L[i][j] = Vm[j]/Vm[i]*exp(-g[i][j]/R/T);
         L[j][i] = Vm[i]/Vm[j]*exp(-g[j][i]/R/T);}

         else { L[i][j] = 1.0; L[j][i] + 1.0;}}}


	//Parameters for the Gamma Matrix:

	double S[n], Q[n][n];

	for (int i = 0; i < n; i++){
	  double sum = 0;
	  for (int j = 0; j < n; j++){
	   sum = sum + x[j]*L[i][j];}
	  S[i] = sum;}

	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	 double sum = 0;
	 for (int k = 0; k < n; k++){ sum = sum + x[k]*L[k][i]*L[k][j]/pow(S[k],2); }
	 Q[i][j] = -L[i][j]/S[i] - L[j][i]/S[j] + sum;}}


	//Now compute the Gamma Matrix (GG): 

	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	 M_GG[i][j] = KD(i,j) + x[i]*(Q[i][j] - Q[i][(n-1)]);}}


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @ brief Description of RBMixerProperties::film_VLE:
 The following method computes the composition of the gas film over the liquid pool using the equillibrium condition
 (i.e. fugacity of species in gas and liquid phases are equal).  Since this model is designed to work at or near
 atmospheric pressure, the Poynting factor and the ratio of liquid and gas phase fugacities are assumed to be
 unity.  Ideal liquid solution is not assumed.  Inputs are system Temperature (Kelvins), system Pressure (Bar),
 and liquid composition (mole fraction). Note that the film composition computed here includes only fuel species
 and not air.  To get the air mole fraction you must sum the fuel molefractions and subtract from unity. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::film_VLE(double T, double P, vector<double> x)
{

	M_filmComp.clear();

	int n = M_NOC;
	double Ps[n], y[n];

//compute activity coefficients:

	Wilson(T, P, x);

//compute vapor pressures and calculate gas phase composition:
	double sum = 0;

	for (int i = 0; i < n; i++){
	 Ps[i] = M_Component[i]->vaporPressure(T);
	 y[i] = getGamma(i)*Ps[i]*x.at(i)/P;
	 sum = sum + y[i];}
 
//	if (sum > 1.0) {for (int i = 0; i < n; i++){ y[i] = y[i]/sum; }}
	
	for (int i = 0; i < n; i++) { M_filmComp.push_back(y[i]); }
	MassFraction();  //also compute the mass fractions in the film

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::BubblePointT:
 The following method calculates the Bubble Point Temperature of a liquid mixture as a function of liquid 
 composition (mole fraction) and system Pressure (bar).  The Temperature (kelvins) is located using a 
 bisection root finding method. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBMixerProperties::BubblePointT(double P, vector<double> x)
{

	int n = M_NOC;
	double BP[n];
	double MaxBP = 0;
	double MinBP = 100000;

	//gather the component boiling points:
	for (int i = 0; i < n; i++) {
	 BP[i] = M_Component[i]->getNBP();
	 if (BP[i] > MaxBP) {MaxBP = BP[i];}
	 if (BP[i] < MinBP) {MinBP = BP[i];}}

	double Tn,Told,Tnew;
	double fn,fold,fnew;	

	Tn = (MaxBP/2 + MinBP/2);
        Told = Tn + 1.0;

	film_VLE(Tn, P, x);
	fn = getSumY() - 1.0;

	film_VLE(Told, P, x);
	fold = getSumY() - 1.0;


	bool sentinel = 0;
	int count = 0;
	int cmax = 21;

	while( sentinel == 0 ){
		count++;
	 	Tnew = Tn - fn*(Tn - Told)/(fn - fold);


		film_VLE(Tnew, P, x);
	        fnew = getSumY() - 1.0;

		Told = Tn;
		fold = fn;

		Tn = Tnew;
		fn = fnew;
		if (abs(fn) < 1e-9) { sentinel = 1;}
		if (count >= cmax) {sentinel = 1;}}
	return Tn;



}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/* @brief Description of RBMixerProperties::diluteDiffusivity: 
 The following method estimates the Diffusion coefficients for dilutes liquid solutions using the method of 
 Wilke and Chang.  Inputs are system temperature (kelvins), and system pressure (bar). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::diluteDiffusivity(double T, double P)
{

	int n = M_NOC;
	double MW[n], V1[n], Vis[n], phi[n];


	for (int i = 0; i < n; i++){
	 if (M_Component[i]->getType() != "Alcohol") { phi[i] = 1.00; }
	 else {
	  if (M_Component[i]->getName() == "Methanol") { phi[i] = 1.9; }
	  if (M_Component[i]->getName() == "Ethanol") { phi [i] = 1.5; }
	  if (M_Component[i]->getName() == "Isopropanol") { phi[i] = 1.5;}} // this quantity is not known here for sure, it is assumed the same as ethanol
	
	MW[i] = M_Component[i]->getMW();// molar mass (g/mol)
	V1[i] =1000*MW[i]/ (M_Component[i]->liquidDensity(M_Component[i]->getNBP(),P)); //molar volume (cm^3/mol)
	Vis[i] = M_Component[i]->liquidViscosity(T);} // mPa-s (cp)


	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	 if (i != j){
          M_DD[i][j] = 7.4e-12*pow(phi[j]*MW[j],0.5)*T/Vis[j]/pow(V1[i],0.6);}}} //units of m^2/s



}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::Fickian: 
 This method calculates the Multi-Component Fickian Diffustion Coefficients for use in species balance equations.
 The method for estimating liquid multi-component Maxwell-Stefan coefficients is described on pg. 91 (eq. 4.2.18)
 of Taylor and Krishna (1993). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::Fickian(double T, double P, vector<double> x)
{

	int n = M_NOC;
	int m = n - 1;

	//Obtain Dilute Diffusivities:

	diluteDiffusivity(T,P);

	// Calculate non-Dilute MS diffusivities:
	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	   M_DMS[i][j] = pow((M_DD[i][j]),(1 + x[j] - x[i])/2)*pow((M_DD[j][i]),(1 + x[i] - x[j])/2);}}


	//Form the B-Matrix
	for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
	   if (i == j) { 
	      double sum = 0;
	      for (int k = 0; k < n; k++){ 
	         if (i != k) {sum = sum + x[k]/M_DMS[i][k];}}
	       M_BB[i][j] = x[i]/M_DMS[i][m] + sum;}
	   else { M_BB[i][j] = -x[i]*(1/M_DMS[i][j] - 1/M_DMS[i][m]);}}}


	//Invert the B-Matrix
	BBinvert();


	//Obtain Gamma matrix and multiply to get Fick Matrix:
	Wilson2(T,P,x);


	for (int i = 0; i < m; i++){
	   for (int j = 0; j < m; j++){
	      double sum = 0;
	      for (int k = 0; k < m; k++){
	         sum = sum + M_BI[i][k]*M_GG[k][j];}
	      M_FD[i][j] = sum; }}
}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::BBinvert: 
 This method will invert the B matrix for the fickian matrix calculation using Gaussian Elimination. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::BBinvert()
{

	int m = M_NOC - 1;
	double I[m][m], l;
	double S[m][m+m];
	
	for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
	 S[i][j] = M_BB[i][j];
	 if (i == j) {I[i][j] = 1.0; }
	 else { I[i][j] = 0.0; }}
	 for (int j = m; j < (m+m); j++) { S[i][j] = I[i][(j - m)]; }}


	for (int i = 0; i < (m - 1); i++){ 
	   double div = S[i][i];
	   for (int j = 0; j < (m + m); j++){
	     S[i][j] = S[i][j]/div;}
	   for (int j = (i + 1); j < (m); j++){
	      l = S[j][i];
	      for (int k = 0; k < (m + m); k++){
	         S[j][k] = S[j][k] - l*S[i][k];}}}
	

	double div1 = S[m-1][m-1];
	for (int i = 0; i < (m + m); i++){
	 S[m - 1][i] = S[m - 1][i]/div1;}


	for (int i = (m-1); i > 0; i--){
	   for (int j = (i - 1); j >= 0; j--){
	      l = S[j][i];
	      for (int k = 0; k < (m+m); k++){
	         S[j][k] = S[j][k] - l*S[i][k];}}}


	for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
	   M_BI[i][j] = S[i][(j + m)];}}

	

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::gasDiffusivity: 
 The following method computes the diffusivity of each component with air and each other in  the gas phase using 
 the method of Wilke and Lee (1955).  See pg 11.10 of the 5th edition of Poling et al. for the details.  Inputs 
 include film Temperature (kelvins) and pressure (Bar).  Diffusivities are given in m^2/s. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::gasDiffusivity(double T, double P)
{

	int n = M_NOC;
	int m = n + 1;
	double MW[m];
	double Vc[m], sig[m], ek[m], Tb[m];
	double Vb[m], DB[m], delta[m];

	//Air properties:
	MW[n] = 29.1;
	Vc[n] = 84.8;
	ek[n] = 1.15*85;
	DB[n] = 0;
	delta[n] = 0;
	Vb[n] = 0.285*pow(Vc[n],1.048);
	sig[n] = 1.18*pow(Vb[n],1./3.);

	//gather pure properties:
	for (int i = 0; i < n; i++){
	   DB[i] = M_Component[i]->getDP();
	   MW[i] = M_Component[i]->getMW();
	   Vc[i] = M_Component[i]->getVC();
	   Tb[i] = M_Component[i]->getNBP();
	   Vb[i] = 0.285*pow(Vc[i],1.048);
	   delta[i] = 1.94e3*pow(DB[i],2)/Vb[i]/Tb[i];
	   if (abs(delta[i]) < 1e-12){ sig[i] = 1.18*pow(Vb[i],(1./3.));} 
	   else { sig[i] = pow(1.585*Vb[i]/(1 + 1.3*pow(delta[i],2)),(1./3.)); }
	   if (abs(delta[i]) < 1e-12) { ek[i] = 1.15*Tb[i]; }
	   else { ek[i] = 1.18*(1 + 1.3*pow(delta[i],2))*Tb[i]; }
	   }


	double dA = 1.06036;
        double dB = 0.15610;
        double dC = 0.19300;
        double dD = 0.47635;
        double dE = 1.03587;
        double dF = 1.52996;
        double dG = 1.76474;
        double dH = 3.89411;

	double Mab, sigma, EK, TS, BD, DD;
	double num, den; 


	for (int i = 0; i < m; i++){ for (int j = (i + 1); j < m; j++){
	   if ((abs(delta[i]) < 1e-12) && (abs(delta[j]) < 1e-12)) { sigma = (sig[i] + sig[j])/2. ;}
	   else { sigma = pow(sig[i]*sig[j],0.5); }
	   Mab = 2./((1/MW[i]) + (1/MW[j]));
	   EK = sqrt(ek[i]*ek[j]);
	   DD = sqrt(delta[i]*delta[j]);
	   TS = T/EK;
	
	   BD = dA/pow(TS,dB) + dC/exp(dD*TS) + dE/(exp(dF*TS)) + dG/exp(dH*TS) + 0.19*pow(DD,2)/TS;


	   num = (3.03 - (0.98/sqrt(Mab)))*pow(10.,-3.)*pow(T,1.5);
           den = (P*sqrt(Mab)*pow(sigma,2.)*BD)*10000;



           M_Dab[i][j] = num/den; // units of m^2/sec;
	   M_Dab[j][i] = num/den; M_Dab[i][i] = 0;}}



}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::MTCoefficient: 
 The following method calculates low flux mass transfer coefficients for the pool model and generates the low-flux
 mass transfer coefficient matrix.  The inputs include film Temperature (kelvins), system pressure (bar), 
 pool diameter (meters), windspeed (m/s). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::MTCoefficient(double T, double Tinf, double P, double PD, double U)
{


	int n = M_NOC;
	int m = n + 1; //fuels + air.
	double Y[m], M[m];

	// compositions:
	for (int i = 0; i < n; i++){
           Y[i] = getY(i);
           M[i] = getM(i);}
     
        Y[n] = 1 - getSumY();
        M[n] = 1 - getSumM();


	//assume a characterisic flame temperature:
	double Tfilm = (T + Tinf)/2; //film temperature for film properties
	double R = 8.314e-5; //gas constant (Bar-m^3/k-mol). 
	double g = 9.81; //gravitational constant (m/s/s).
	double MW[m], rho[m];

	double Ct = P/R/T;  //Total molar Concentration in film (mol/m^3).
//        double Ct = P/R/Tfilm;  //Total film Conc. with averaged temperature (mol/m^3).
	double L = PD/4*3.14159265358979; //pool characteristic length (meters)

	//calculate the various binary gas diffusivities:
	gasDiffusivity(T,P);



	for (int i = 0; i < n; i++){
	 MW[i] = M_Component[i]->getMW();
	 rho[i] = Ct*MW[i]/1000;} //species densities (kg/m^3).

	//air properties:
	MW[n] = 29.1;
	rho[n] = Ct*MW[n]/1000;



	//film properties:
	double Beta = 1/T;
//        double Beta = 1/Tfilm;
	double vis = film_Viscosity(Tfilm);
	double FD = film_Density(Tfilm,P);
	double FTK = film_thermalConductivity(Tfilm);
	double FCP = film_heatCapacity(Tfilm);
	double kv = vis/FD; // kinematic viscosity.
	double alpha = FTK/FD/FCP; // thermal diffusivity.
	double Pr = kv/alpha; // Prandtl number
	double Re = U*L/kv; // Reynolds number.
	double eta, Sc, Gr, Grw, Shf, A, Shb, Sh, ft, mf; 
        double Nu,Nub, Nuf; //Nusselt Number for Convective Heat Transfer.


	for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
	   if (i != j){

	   mf = M[i];
	   eta = abs(((rho[j] - rho[i])/(mf*(rho[j] - rho[i]) + rho[i]))); // 1/p*(dp/dmf), used for diffusional Grashoff number.
           Sc = kv/M_Dab[i][j];  //film Schmidt number (dimensionless)

	   //Forced Convection Sherwood Number (see Incropera and Dewitt 5th ed. ppg. 394-396)
           

           if (Re < (5e5))
           { 
             Shf = 0.664*sqrt(Re)*pow(Sc,(1./3.));
           }
           else if (Re > (5e5)) 
           {
             A = 871;
             Shf = (0.037*pow(Re,0.8) - A)*pow(Sc,(1./3.));
           }


	   //Buoyant Convection Sherwood Number:

	      //Calculate the thermal and diffusional Grashof Numbers see pg. 600 (Bird,Stewart, and lightfoot) as
              // well as pg. 698 of the same for model for combined heat and mass transer:
           Gr = g*Beta*(abs(T-Tinf))*pow(L,3)/kv/kv; //Thermal Grashof Number (dimensionless)
           Grw = g*eta*abs(mf)*pow(L,3)/kv/kv; // Diffusional Grashof Number (dimensionless)

           Shb = 0.27*pow(((Gr + Grw)*Sc),0.25);  //Buoyancy Sherwood Number (dimensionless)


	   // Combine forced and buoyant sherwood numbers as per pg. 567-568 of Incropera and Dewitt (2002):
	   Sh = pow((pow(Shb,3) + pow(Shf,3)),(1./3.)); // Combine Sherwood numbers for forced and buoyant effects.

           ft = L/Sh; //Boundary Layer film thickness (meters)

           M_KC[i][j] = M_Dab[i][j]/ft; //  Mass transfer coefficient;
		
	   }}}



	// Now as a first step to calculating the multi-component mass coefficient matrix, calculate the R-matrix from
	// the binary mass coefficients:



	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	   if (i == j) {
	      double sum = 0;
	      for (int k = 0; k < m; k++){ if (k != i){
	         sum = sum + Y[k]/M_KC[i][k];}}
	      M_RR[i][j] = Y[i]/M_KC[i][n] + sum;}
	   else {
	      M_RR[i][j] = -Y[i]*(1/M_KC[i][j] - 1/M_KC[i][n]);}}}

	//Now invert the R-matrix:

	RRinvert();

 
	//Calculate Heat Transfer Coefficient:       

        //Forced Convection Nusselt Number (see Incropera and Dewitt 5th ed. ppg. 394-396)
     

        if (Re < (5e5))
        {
          Nuf = 0.664*sqrt(Re)*pow(Pr,(1./3.));
        }
        else if (Re > (5e5)) 
        {
          A = 871; 
          Nuf = (0.037*pow(Re,0.8) - A)*pow(Pr,(1./3.));
        }

        //Buoyant Convection Nusselt Number:

//        eta = abs(((rho[j] - rho[i])/(mf*(rho[j] - rho[i]) + rho[i]))); // 1/p*(dp/dmf), used for diffusional Grashoff number.

        //Calculate the thermal and diffusional Grashof Numbers see pg. 600 (Bird,Stewart, and lightfoot) as
        // well as pg. 698 of the same for model for combined heat and mass transer:
        Gr = g*Beta*(abs(T-Tinf))*pow(L,3)/kv/kv; //Thermal Grashof Number (dimensionless)
//        Grw = g*eta*abs(mf)*pow(L,3)/kv/kv; // Diffusional Grashof Number (dimensionless)


        //For now the Nusselt number is calculated w/o taking into account the effects of mass transfer
//        Nub = 0.27*pow((Gr*Pr),0.25);  //Buoyancy Driven Nusselt Number (d-less)
        double RA = Gr*Pr;
        if (RA <= 1.00e7) {Nub = 0.54*pow(RA,0.25);}
        if (RA > 1.00e7) {Nub = 0.15*pow(RA,(1./3.));}


        // Combine forced and buoyant Nusselt numbers as per pg. 567-568 of Incropera and Dewitt (2002):
        Nu = pow((pow(Nub,3) + pow(Nuf,3)),(1./3.)); // Combine Nusselt numbers for forced and buoyant effects.

 	M_HeatTransferCoeff = Nu*FTK/L;  //Convective Heat transfer coefficent (W/m/m/K)

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::RRinvert: 
 This method is another version of the matrix inversion algorithm for the R- matrix */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::RRinvert()
{

	int m = M_NOC; 
        double I[m][m], l;
        double S[m][m+m];
     
        for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
         S[i][j] = M_RR[i][j];
         if (i == j) {I[i][j] = 1.0; }
         else { I[i][j] = 0.0; }}
         for (int j = m; j < (m+m); j++) { S[i][j] = I[i][(j - m)]; }}


        for (int i = 0; i < (m - 1); i++){ 
           double div = S[i][i];
           for (int j = 0; j < (m + m); j++){
             S[i][j] = S[i][j]/div;}
           for (int j = (i + 1); j < (m); j++){
              l = S[j][i];
              for (int k = 0; k < (m + m); k++){
                 S[j][k] = S[j][k] - l*S[i][k];}}}
     

        double div1 = S[m-1][m-1];
        for (int i = 0; i < (m + m); i++){
         S[m - 1][i] = S[m - 1][i]/div1;}


        for (int i = (m-1); i > 0; i--){
           for (int j = (i - 1); j >= 0; j--){
              l = S[j][i];
              for (int k = 0; k < (m+m); k++){
                 S[j][k] = S[j][k] - l*S[i][k];}}}


        for (int i = 0; i < m; i++){ for (int j = 0; j < m; j++){
           M_K[i][j] = S[i][(j + m)];}}


}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::MassFraction: 
 This simple method converts molefractions to mass fractions: */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::MassFraction()
{
// this version of the method is for mass fraction in the gas film above the pool
	M_MassFilmComp.clear();

	int n = M_NOC;

	double M[n];
	double sum = 0;

	for (int i = 0; i < n; i++){
	 double MW = M_Component[i]->getMW();
	 M[i] = getY(i)*MW;
	 sum = sum + M[i];}
	 sum = sum + 29.1*(1 - getSumY());

	for (int i = 0; i < n; i++){
	   M_MassFilmComp.push_back(M[i]/sum);}

}

//----------------------------------------------------------------------
void 
RBMixerProperties::MassFraction(vector<double> x)
{
// this version of the method is for the liquid phase mass fractions

	M_LiquidMassComp.clear();

	int n = M_NOC;

	double M[n];
	double sum = 0;

	for (int i = 0; i < n; i++){
	  double MW = M_Component[i]->getMW();
	  M[i] = x.at(i)*MW;
	  sum = sum + M[i];}

	for (int i = 0; i < n; i++){
	  M_LiquidMassComp.push_back(M[i]/sum);}


}

//----------------------------------------------------------------------
vector<double>
RBMixerProperties::MoleFraction(vector<double> m)
{
//This version is more generic and it converts a mass fraction to a mole fraction composition.


        int n = M_NOC;
        double sum = 0;
        double X[n+1];
        double MW[n+1],MF[n+1];
        vector<double> mole;

        for (int ii = 0; ii < n; ii++){ MW[ii] = M_Component[ii]->getMW(); MF[ii] = m[ii];}
        MW[n] = 29.1;  //Molar Mass of Air.
        for (int ii = 0; ii < n; ii++){sum = sum + m[ii];}
        MF[n] = 1 - sum; //Mass Fraction of Air.
        sum = 0;

        for (int ii = 0; ii < (n+1); ii++)
        {
          X[ii]=MF[ii]/MW[ii];
          sum = sum + X[ii];
        }

        for (int ii = 0; ii < n; ii++) {mole.push_back(X[ii]/sum);}

        return mole;

}
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::film_Flux_Main:
  This method controls the film_Fluxes methods for fluxes when the liquid temperature is below the bubble point.
  Certain Temperature inputs are found to be unstable when applied to the film_Fluxes methods.  This method
  will run the film_Fluxes method and check for its convergence.  If convergence fails the input temperature
  will be perturbed by a small quanitity (to minimize error) and the film_Fluxes routine will run again with the
  new temperature. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::film_Flux_Main(double T,double Tinf,double P,vector<double> x,vector<double> Yb,double PD,double U,double RT)
{

        M_convergeFluxProblem = 0;
        int counter = 0;
        int Limit = 10;
        double perturb = 2e-11;
      
        while (counter <= Limit)
        {
          film_Fluxes(T,Tinf,P,x,Yb,PD,U,RT);
          if (M_convergeFluxProblem)
          {
            cout << "Flux failed to converge, try again." << endl;
            cout << "Attempt: " << counter << endl;
            T = T - perturb;
            counter++;
            M_convergeFluxProblem = 0;
          }
          else
          {
            M_convergeFluxProblem = 0;
            break;
          }
        }


}
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::film_Fluxes:
 This version of film_Fluxes is used to compute pool boiling rates where the liquid temperature has reached 
 its boiling point */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::film_Fluxes(double T, double RT, vector<double> x, double P, double Energy, double Tinf, double PD, double U)
{

        int n = M_NOC;
        vector<double> m;
        double LiquidEvap = 0;
        double MW[n];


        film_VLE(T,P,x);
        
        for (int ii = 0; ii < n; ii++) { m.push_back(getM(ii)/getSumM()); }


        M_HEVAP.clear();
        M_NFLUX.clear();
        M_MFLUX.clear();

        double SH[n], LH[n];
        for (int ii = 0; ii < n; ii++) 
        {
          SH[ii] = M_Component[ii]->liquidSensibleHeat(T, RT);
          double dummy1 = M_Component[ii]->latentHeat(T);
          LH[ii] = M_Component[ii]->getClat();
          M_HEVAP.push_back((SH[ii] + LH[ii]));
        }

 
        for (int ii = 0; ii < n; ii++){ LiquidEvap = LiquidEvap + m[ii]*getHevap(ii); }
        
        double TotalMassBurn = Energy/LiquidEvap;

        for (int ii = 0; ii < n; ii++) 
        { 
          MW[ii] = M_Component[ii]->getMW();
          M_MFLUX.push_back(m[ii]*TotalMassBurn); 
          M_NFLUX.push_back(M_MFLUX.at(ii)*1000/MW[ii]);
        }
       

        //Although mass transfer coefficient aren't needed, the heat transfer coefficient is.
       	MTCoefficient(T, Tinf, P, PD, U);

cout << "Boiling Going on Here!!!" << endl;
cout << "Flux: " << getSumMflux() << endl;
cout << "Surface Temperature: " << T << endl;
cout << "Energy: " << Energy << endl;
}
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::film_Fluxes: 
 This method calculates the fluxes from the film.  For details of the method see pg. 169 of Taylor 
 and Krishna (1993). Inputs are system Temperature (Kelvins), system Pressure (Bar), liquid phase mole fractions,
 pool diameter (meters), and windspeed (m/s).  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::film_Fluxes(double T, double Tinf, double P, vector<double> x, vector<double> Yb, double PD, double U, double RT)
{


	int n = M_NOC;

	double J[n]; // Diffusive Flux Array
	double N[n], Nnew[n], Error;
	double NM[n], BK[n][n];

	
	//compute film Composition:

	film_VLE(T, P, x);

	double Yn = 1 - getSumY();

for (unsigned int ii = 0; ii < Yb.size(); ii++) 
{
}

	//Compute total molar film concentration (mol/m^3):

	double R = 8.314e-5; // gas constant (Bar-m^3/K-mol)
	double Ct = P/R/T; // mol/m^3
//        double FilmTemperature = Tinf/2 + T/2;
//        double Ct = P/R/FilmTemperature;
	M_filmConc = Ct;



	// Compute BETA:

	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	     M_BETA[i][j] = KD(i,j) + getY(i)/Yn;}}




	// Compute K-matrix:

	MTCoefficient(T, Tinf, P, PD, U);


	// First Estimate of the fluxes:
	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	   double sum = 0;
	   for (int k = 0; k < n; k++){
	      sum = sum + getBETA(i,j)*M_K[k][j];}
	   BK[i][j] = sum;}}

	for (int i = 0; i < n; i++) {
	   double sum = 0;
	   for (int k = 0; k < n; k++){
	      sum = sum + BK[i][k]*(getY(k) - Yb[k]);}
	   N[i] = Ct*sum;}

	


	//Calculate total molar flux:
	double Nt = 0;
	for (int i = 0; i < n; i++){ Nt = Nt + N[i]; }
	



	//Now to establish the interative procedure:
	int sentinel = 0;
	int converge = 0;
	int count = 0;
	double F[n], dN[n];
	double dFdN[n][n + 1];

	while ( sentinel == 0 ) {
	 count++;


	//compute phi:

        for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
           if (i == j) {
              double sum = 0;
              for (int k = 0; k < n; k++){
                 if (i != k) {sum = sum + N[k]/Ct/M_KC[i][k];}}
               M_phi[i][j] = N[i]/Ct/M_KC[i][n] + sum;}
           else { M_phi[i][j] = -N[i]*(1/Ct/M_KC[i][j] - 1/Ct/M_KC[i][n]);}}}




	//compute theta:
	
	computeTheta(8);




	//compute Diffusive Flux:
	
	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	   double sum = 0;
	   for (int k = 0; k < n; k++){ 
	      sum = sum + getK(i,k)*getTHETA(k,j);}
	   BK[i][j] = sum;}}

	for (int i = 0; i < n; i++){ 
	   double sum = 0;
	   for (int j = 0; j < n; j++){	
	      sum = sum + BK[i][j]*(getY(j) - Yb[j]);}
	   J[i] = Ct*sum;}

	

	//Calculate Molar Fluxes:

	for (int i = 0; i < n; i++){ 
	   double sum = 0;
	   for (int k = 0; k < n; k++){
	      sum = sum + M_BETA[i][k]*J[k];}
	   Nnew[i] = sum;}


	  Error = 0;
	for (int i = 0; i < n; i++){ Error = Error + abs(N[i] - Nnew[i]); }

	if (Error < 1e-12) { sentinel = 1; converge = 1; }
	if (count > 50) {sentinel = 1; }

	for (int i = 0; i < n; i++) { NM[i] = N[i]/2 + Nnew[i]/2; }
	for (int i = 0; i < n; i++) { N[i] = Nnew[i];} 

} // end while loop

	M_NFLUX.clear();
	M_MFLUX.clear();
	M_EFLUX.clear();
        M_HEVAP.clear();

	if ( converge == 1)
        {
	  for (int i = 0; i < n; i++) { M_NFLUX.push_back(Nnew[i]);}
	   
	//compute mass flux

	  double MW[n];

	  for (int i = 0; i < n; i++) { MW[i] = M_Component[i]->getMW(); 
	     M_MFLUX.push_back(M_NFLUX.at(i)*MW[i]/1000);}
//	  cout << "Nt: " << getSumNflux() << "\n";
//	  cout << "Mt: " << getSumMflux() << "\n";
	}

	if (converge == 0 )
        {


	  for (int i = 0; i < n; i++){ M_NFLUX.push_back(NM[i]); }

	  film_Fluxes(Yb);
        }


        //Eliminate Negative Fluxes:
        for (int ii = 0; ii < n; ii++)
        {
          if (M_NFLUX[ii] < 0)
          {
            M_NFLUX[ii] = 0;
            M_MFLUX[ii] = 0;
          }
        }


	//compute energy flux:
	double SH[n], LH[n];
	for (int i = 0; i < n; i++) 
          {
	  SH[i] = M_Component[i]->liquidSensibleHeat(T, RT);
	  double dummy1 = M_Component[i]->latentHeat(T);
	  LH[i] = M_Component[i]->getClat();
          M_HEVAP.push_back((SH[i] + LH[i]));
	  M_EFLUX.push_back(M_MFLUX.at(i)*(SH[i] + LH[i]));
          }
}



//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::film_Fluxes:
 This method solves for the evaporative fluxes from the pool for cases where the standard method 
 (film_Fluxes) fails to converge (near the bubble point.)  It uses the Newton-Raphon method.  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::film_Fluxes(vector<double> Yb)
{

	int n = M_NOC;
	double F[n],J[n][n], DN[n];
	vector<double> N;
	vector<double> JNP;
	vector<double> JNM;
	double Error;
	int count = 0;
        bool testInput = 0;
	//Establish the initial guess:

	for (int i = 0; i < n; i++){ N.push_back(M_NFLUX.at(i)); }



	//Main Loop:

	int sentinel = 0;

	while (sentinel == 0){
	
	 count++;

	   JNP.clear();
	   JNM.clear();

	for (int i = 0; i < n; i++) {
	   JNP.push_back(N.at(i));
	   JNM.push_back(N.at(i));}


	//calculate the components of the residual vector:

	
	Residual(N,Yb);
	for (int i = 0; i < n; i++) { F[i] = getRes(i); }

	Error = getResNorm();

	//Now create the Jacobian (Approximate using finite differences)

	for (int i = 0; i < n; i++) {
	
	   double NP[n], NM[n];	

	   JNP.at(i) = N.at(i)*(1.01);
	   JNM.at(i) = N.at(i)*(0.99);

	   Residual(JNP,Yb);
	
	   for (int j = 0; j < n; j++){ NP[j] = getRes(j); }

	   Residual(JNM,Yb);

	   for (int j = 0; j < n; j++){ NM[j] = getRes(j); }

	   for (int j = 0; j < n; j++){
	      J[j][i] = (NP[j] - NM[j])/(N.at(i)*(0.02));}

	   JNP.at(i) = N.at(i);
	   JNM.at(i) = N.at(i);}
	







	//Now Solve the Jacobian System:

	double S[n][n+1], ll;

         for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
          S[i][j] = J[i][j];
          for (int j = n; j < (n+1); j++) { S[i][j] = F[i]; }}}


         for (int i = 0; i < (n - 1); i++){
            double div = S[i][i];
            for (int j = 0; j < (n + 1); j++){
              S[i][j] = S[i][j]/div;}
            for (int j = (i + 1); j < (n); j++){
               ll = S[j][i];
               for (int kk = 0; kk < (n + 1); kk++){
                  S[j][kk] = S[j][kk] - ll*S[i][kk];}}}


         double div1 = S[n-1][n-1];
         for (int i = 0; i < (n + 1); i++){
          S[n - 1][i] = S[n - 1][i]/div1;}


         for (int i = (n-1); i > 0; i--){
            for (int j = (i - 1); j >= 0; j--){
               ll = S[j][i];
               for (int kk = 0; kk < (n+1); kk++){
                  S[j][kk] = S[j][kk] - ll*S[i][kk];}}}

         for (int i = 0; i < n; i++){ 
            DN[i] = S[i][n];}



	//Now calculate the new value for N

	for (int i = 0; i < n; i++){ N.at(i) = N.at(i) - DN[i];}

	
	if (Error < 1e-12) 
        { 
          sentinel = 1;
        }
	if (count > 35) 
        {
          
          sentinel = 1;  
          cout << "counted out: " << Error << endl;
          M_convergeFluxProblem = 1;
        }
	

	} // end while loop


        


	//Assign the fluxes:

	M_NFLUX.clear();
        for (int i = 0; i < n; i++) { M_NFLUX.push_back(N[i]); }

        //compute mass flux

        double MW[n];

	M_MFLUX.clear();
        for (int i = 0; i < n; i++) { MW[i] = M_Component[i]->getMW();
           M_MFLUX.push_back(M_NFLUX.at(i)*MW[i]/1000);}

//	cout << "Nt: " << getSumNflux() << "\n";
//	cout << "Mt: " << getSumMflux() << "\n";
//	cout << "NRCount: " << count << "\n";
//	cout << "NRError: " << Error << "\n";
//	for (int i = 0; i < n; i++){cout << "N[" << i << "]: " << N[i] << "\n";}
        
}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of RBMixerProperties::Residual
 This is a support function for calculating the evaporation fluxes.  It takes the vector of fluxes and calculates
 a vector of residuals. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
RBMixerProperties::Residual(vector<double> fluxes,vector<double> Yb){

	//clear the residual vector so we don't keep adding unwanted terms:
	M_Res.clear();


	int n = M_NOC;
	double N[n], Nnew[n], J[n], BK[n][n];
	double Ct = M_filmConc;

	//Transfer the fluxes to the N vector

	for (int i = 0; i < n; i++){ N[i] = fluxes.at(i); }


	//compute phi:

        for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
           if (i == j) { 
              double sum = 0; 
              for (int k = 0; k < n; k++){
                 if (i != k) {sum = sum + N[k]/Ct/M_KC[i][k];}}
               M_phi[i][j] = N[i]/Ct/M_KC[i][n] + sum;}
           else { M_phi[i][j] = -N[i]*(1/Ct/M_KC[i][j] - 1/Ct/M_KC[i][n]);}}}




        //compute theta:
     
        computeTheta(8);

	

	//compute Diffusive Flux:

        for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
           double sum = 0;
           for (int k = 0; k < n; k++){
              sum = sum + getK(i,k)*getTHETA(k,j);}
           BK[i][j] = sum;}}

        for (int i = 0; i < n; i++){
           double sum = 0;
           for (int j = 0; j < n; j++){
              sum = sum + BK[i][j]*(getY(j) - Yb[j]);}
           J[i] = Ct*sum;}



        //Calculate Molar Fluxes:

        for (int i = 0; i < n; i++){
           double sum = 0;
           for (int k = 0; k < n; k++){
              sum = sum + M_BETA[i][k]*J[k];}
           Nnew[i] = sum;}

	//Calculate the Residual:

	for (int i = 0; i < n; i++){ M_Res.push_back(N[i] - Nnew[i]);}




}


//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Descripton of RBMixerProperties::computeTheta:
 This method computes the theta matrix for high flux correction to multi- component mass transfer
 coefficients.  Uses the method of Taylor and Webb (Comp. Chem. Eng. V.5 pg. 61-73, 1981). */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
RBMixerProperties::computeTheta(int k)
{


	int n = M_NOC;
	

	double I[n][n];
	double C[k+1];

	//Define the coefficients:
	for (int i = 1; i < (k+1); i++){
	   C[i] = 1/Fact(1.0*i + 1);}


	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++) {
	   if (i == j) { I[i][j] = 1; }
	   else { I[i][j] = 0; }}}


	//Compute the power series for k terms:
	double P[n][n], TI[n][n], Temp[n][n];

	for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	   P[i][j] = M_phi[i][j];
	   TI[i][j] = I[i][j] + C[1]*P[i][j];}}

	for (int kk = 2; kk <= k; kk++){
	 
	   for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	      double sum = 0;
	      for (int l = 0; l < n; l++){
	         sum = sum + P[i][l]*M_phi[l][j];}
	      Temp[i][j] = sum;}}

	   for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
	      P[i][j] = Temp[i][j];
	      TI[i][j] = TI[i][j] + C[kk]*P[i][j];}}}


	//Now invert TI to get Theta:


         double S[n][n+n], ll;
 
         for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
          S[i][j] = TI[i][j];
          if (i == j) {I[i][j] = 1.0; }
          else { I[i][j] = 0.0; }}
	  for (int j = n; j < (n+n); j++) { S[i][j] = I[i][(j - n)]; }}
 
 
         for (int i = 0; i < (n - 1); i++){
            double div = S[i][i];
            for (int j = 0; j < (n + n); j++){
              S[i][j] = S[i][j]/div;}
            for (int j = (i + 1); j < (n); j++){
               ll = S[j][i];
               for (int kk = 0; kk < (n + n); kk++){
                  S[j][kk] = S[j][kk] - ll*S[i][kk];}}}
 
 
         double div1 = S[n-1][n-1];
         for (int i = 0; i < (n + n); i++){
          S[n - 1][i] = S[n - 1][i]/div1;}
 
 
         for (int i = (n-1); i > 0; i--){
            for (int j = (i - 1); j >= 0; j--){
               ll = S[j][i];
               for (int kk = 0; kk < (n+n); kk++){
                  S[j][kk] = S[j][kk] - ll*S[i][kk];}}}
 
 
         for (int i = 0; i < n; i++){ for (int j = 0; j < n; j++){
            M_theta[i][j] = S[i][(j + n)];}}
	


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************


