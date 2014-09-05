// This file defines the different methods for the component class


#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include<CCA/Components/Arches/RBComponentProperties.h>
//#include"RBComponentProperties.h"

using namespace Uintah;
using namespace std;


//Standard Constructor:

RBComponentProperties::RBComponentProperties(fuel f):
			C_ID(f)

{

	switch (C_ID) {

		case (n_Octane):
                        C_name = "n_Octane";
                        C_type = "Paraffin";
			C_crit_T = 568.70;
			C_crit_P = 24.90;
			C_crit_V = 492.00;
			C_crit_Z = 0.259;
			C_MW = 114.231;
			C_omega = 0.399;
			C_Hvap = 34410; //KJ/mol
			C_NBP = 398.82;
			C_dipole = 0.0;
                        C_lcp_A = 24.4736; 
                        C_lcp_B = -1.009742; 
                        C_lcp_D = 1.03052; 
                        C_gcp_A0 = 10.824; // T range: 200 - 1000 K
                        C_gcp_A1 = 4.983e-3;
                        C_gcp_A2 = 17.751e-5;
                        C_gcp_A3 = -23.137e-8;
                        C_gcp_A4 = 8.980e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -8.04937;
                        C_vp3_B = 2.03865;
                        C_vp3_C = -3.3120;
                        C_vp3_D = -3.6480;
                        C_visnu = 0.210;
                        C_visN = 0.2;
                        break; 

		case(n_Dodecane):
                        C_name = "n_Dodecane";
                        C_type = "Paraffin";
			C_crit_T = 658.65;
                        C_crit_P = 18.20;
                        C_crit_V = 754.00;
                        C_crit_Z = 0.251;
                        C_MW = 170.338;
                        C_omega = 0.576;
                        C_Hvap = 43400;
                        C_NBP = 489.48;
			C_dipole = 0.0;
                        C_lcp_A = 35.6624; 
                        C_lcp_B = -1.22961; 
                        C_lcp_D = 1.45768; 
                        C_gcp_A0 = 17.229; // T range: 200 - 1000 K
                        C_gcp_A1 = -7.242e-3;
                        C_gcp_A2 = 31.922e-5;
                        C_gcp_A3 = -42.322e-8;
                        C_gcp_A4 = 17.022e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -9.08593;
                        C_vp3_B = 2.77846;
                        C_vp3_C = -5.1985;
                        C_vp3_D = -4.1730;
                        C_visnu = 0.210;
                        C_visN = 0.25;
                        break;
                        
		case(n_Hexadecane):
                        C_name = "n_Hexadecane";
                        C_type = "Paraffin";
			C_crit_T = 723.00;
                        C_crit_P = 14.00;
                        C_crit_V = 1034.00;
                        C_crit_Z = 0.241;
                        C_MW = 226.446;
                        C_omega = 0.718;
                        C_Hvap = 51210;
                        C_NBP = 559.98;
			C_dipole = 0.0;
                        C_lcp_A = 46.8512; 
                        C_lcp_B = -1.449478; 
                        C_lcp_D = 1.88484; 
                        C_gcp_A0 = 39.747; // T range: 200 - 1000 K
                        C_gcp_A1 = -206.152e-3;
                        C_gcp_A2 = 114.814e-5;
                        C_gcp_A3 = -155.548e-8;
                        C_gcp_A4 = 67.534e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -10.03664;
                        C_vp3_B = 3.41426;
                        C_vp3_C = -6.8627;
                        C_vp3_D = -4.8630;
                        C_visnu = 0.210;
                        C_visN = 0.25;
                        break;

		case(O_Xylene):
                        C_name = "O_Xylene";
                        C_type = "Aromatic";
			C_crit_T = 630.30;
                        C_crit_P = 37.32;
                        C_crit_V = 370.00;
                        C_crit_Z = 0.263;
                        C_MW = 106.167;
                        C_omega = 0.312;
                        C_Hvap = 36240;
                        C_NBP = 417.59;
                        C_dipole = 0.5;
                        C_lcp_A = 19.748; 
                        C_lcp_B = -1.94726; 
                        C_lcp_D = 0.917058; 
                        C_gcp_A0 = 3.289; // T range: 50 - 1000 K
                        C_gcp_A1 = 34.144e-3;
                        C_gcp_A2 = 4.989e-5;
                        C_gcp_A3 = -8.335e-8;
                        C_gcp_A4 = 3.338e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.60491;
                        C_vp3_B = 1.75383;
                        C_vp3_C = -2.27531;
                        C_vp3_D = -3.73771;
                        C_visnu = 0.240;
                        C_visN = 0.25;
                        break;

		case(M_Xylene):
                        C_name = "M_Xylene";
                        C_type = "Aromatic";
			C_crit_T = 617.00;
                        C_crit_P = 35.41;
                        C_crit_V = 375.00;
                        C_crit_Z = 0.259;
                        C_MW = 106.167;
                        C_omega = 0.327;
                        C_Hvap = 35660;
                        C_NBP = 412.34;
			C_dipole = 0.3;
                        C_lcp_A = 19.748; 
                        C_lcp_B = -1.94726; 
                        C_lcp_D = 0.917058; 
                        C_gcp_A0 = 4.002; // T range: 50 - 1000 K
                        C_gcp_A1 = 17.537e-3;
                        C_gcp_A2 = 10.590e-5;
                        C_gcp_A3 = -15.037e-8;
                        C_gcp_A4 = 6.008e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.67717;
                        C_vp3_B = 1.80240;
                        C_vp3_C = -2.47745;
                        C_vp3_D = -3.66068;
                        C_visnu = 0.240;
                        C_visN = 0.2;
                        break;

		case(P_Xylene):
                        C_name = "P_Xylene";
                        C_type = "Aromatic";
			C_crit_T = 616.20;
                        C_crit_P = 35.11;
                        C_crit_V = 378.00;
                        C_crit_Z = 0.259;
                        C_MW = 106.167;
                        C_omega = 0.322;
                        C_Hvap = 35670;
                        C_NBP = 411.53;
                        C_dipole = 0.1;
                        C_lcp_A = 19.748; 
                        C_lcp_B = -1.94726; 
                        C_lcp_D = 0.917058; 
                        C_gcp_A0 = 4.113; // T range: 50 - 1000 K
                        C_gcp_A1 = 14.909e-3;
                        C_gcp_A2 = 11.810e-5;
                        C_gcp_A3 = -16.724e-8;
                        C_gcp_A4 = 6.736e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.71694;
                        C_vp3_B = 1.89119;
                        C_vp3_C = -2.39695;
                        C_vp3_D = -3.63026;
                        C_visnu = 0.240;
                        C_visN = 0.2;
                        break;

		case(Tetralin):
                        C_name = "Tetralin";
                        C_type = "Aromatic";
			C_crit_T = 720.00;
                        C_crit_P = 36.50;
                        C_crit_V = 408.00;
                        C_crit_Z = 0.249;
                        C_MW = 132.205;
                        C_omega = 0.3345;
                        C_Hvap = 58600;
                        C_NBP = 480.75;
                        C_dipole = 0.1; //???????? check on this later.
                        C_lcp_A = 14.0943; 
                        C_lcp_B = 2.386826; 
                        C_lcp_D = 0.555771; 
                        C_gcp_A0 = 3.4435; // T range: 50 - 1000 K
                        C_gcp_A1 = 16.946e-3;
                        C_gcp_A2 = 17.415e-5 ;
                        C_gcp_A3 = -25.220e-8;
                        C_gcp_A4 = 10.515e-11;
                        C_vp_equation_type = 1;
                        C_vp1_A = 4.12671;
                        C_vp1_B = 1690.912;
                        C_vp1_C = -70.229;
                        C_visnu = 0.270;
                        C_visN = 0.25;
                        break;

		case(C_Decalin):
                        C_name = "C_Decalin";
                        C_type = "Cycloparaffin";
			C_crit_T = 703.60;
                        C_crit_P = 32.00;
                        C_crit_V = 480.00;
                        C_crit_Z = 0.265;
                        C_MW = 138.253;
                        C_omega = 0.276;
                        C_Hvap = 41000;
                        C_NBP = 468.92;
                        C_dipole = 0.0;
                        C_lcp_A = 14.62186;
                        C_lcp_B = 2.104824;
                        C_lcp_D = 0.7902076;
                        C_gcp_A0 = -5.445; // T range: 298 - 1000 K
                        C_gcp_A1 = 80.068e-3;
                        C_gcp_A2 = 5.065e-5;
                        C_gcp_A3 = -11.756e-8;
                        C_gcp_A4 = 5.088e-11;
                        C_vp_equation_type = 1; // T range: 349.53 - 500.79 K
                        C_vp1_A = 4.00019;
                        C_vp1_B = 1594.460;
                        C_vp1_C = 203.392 - 273.15;
                        C_visnu = 0.310;
                        C_visN = 0.3;
                        break;

		case(T_Decalin):
                        C_name = "T_Decalin";
                        C_type = "Cycloparaffin";
			C_crit_T = 687.00;
                        C_crit_P = 32.00;
                        C_crit_V = 480.00;
                        C_crit_Z = 0.272;
                        C_MW = 138.253;
                        C_omega = 0.303;
                        C_Hvap = 40200;
                        C_NBP = 460.42;
                        C_dipole = 0.0;
                        C_lcp_A = 14.62186;
                        C_lcp_B = 2.104824;
                        C_lcp_D = 0.7902076;
                        C_gcp_A0 = -2.155; // T range: 298 - 1000 K
                        C_gcp_A1 = 53.852e-3;
                        C_gcp_A2 = 12.610e-5;
                        C_gcp_A3 = -20.981e-8;
                        C_gcp_A4 = 9.066e-11;
                        C_vp_equation_type = 1; // T range: 342.33 - 492 K
                        C_vp1_A = 3.98171;
                        C_vp1_B = 1564.683;
                        C_vp1_C = 206.259 - 273.15;
                        C_visnu = 0.310;
                        C_visN = 0.3;
                        break;

		case(Benzene):
                        C_name = "Benzene";
                        C_type = "Aromatic";
			C_crit_T = 562.05;
                        C_crit_P = 48.95;
                        C_crit_V = 256.00;
                        C_crit_Z = 0.268;
                        C_MW = 78.114;
                        C_omega = 0.210;
                        C_Hvap = 30720;
                        C_NBP = 353.24;
                        C_dipole = 0.0;
                        C_lcp_A = 13.5654; 
                        C_lcp_B = -1.5; 
                        C_lcp_D = 0.75552; 
                        C_gcp_A0 = 3.551; // T range: 50 - 1000 K
                        C_gcp_A1 = -6.184e-3;
                        C_gcp_A2 = 14.365e-5;
                        C_gcp_A3 = -19.807e-8;
                        C_gcp_A4 = 8.234e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.01433;
                        C_vp3_B = 1.55256;
                        C_vp3_C = -1.8479;
                        C_vp3_D = -3.7130;
                        C_visnu = 0.30;
                        C_visN = 0.3;
                        break;

		case(Toluene):
                        C_name = "Toluene";
                        C_type = "Aromatic";
			C_crit_T = 591.75;
                        C_crit_P = 41.08;
                        C_crit_V = 316.00;
                        C_crit_Z = 0.264;
                        C_MW = 92.141;
                        C_omega = 0.264;
                        C_Hvap = 33180;
                        C_NBP = 383.79;
                        C_dipole = 0.4;
                        C_lcp_A = 16.6567; 
                        C_lcp_B = -1.72363; 
                        C_lcp_D = 0.836289; 
                        C_gcp_A0 = 3.866; // T range: 50 - 1000 K
                        C_gcp_A1 = 3.558e-3;
                        C_gcp_A2 = 13.356e-5;
                        C_gcp_A3 = -18.659e-8;
                        C_gcp_A4 = 7.690e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.31600;
                        C_vp3_B = 1.59425;
                        C_vp3_C = -1.93165;
                        C_vp3_D = -3.72220;
                        C_visnu = 0.235;
                        C_visN = 0.25;
                        break;

		case(n_Pentane):
                        C_name = "n_Pentane";
                        C_type = "Paraffin";
			C_crit_T = 469.70;
                        C_crit_P = 33.70;
                        C_crit_V = 311.00;
                        C_crit_Z = 0.268;
                        C_MW = 72.150;
                        C_omega = 0.252;
                        C_Hvap = 25790;
                        C_NBP = 309.22;
                        C_dipole = 0.0;
                        C_lcp_A = 16.082; 
                        C_lcp_B = -0.844841; 
                        C_lcp_D = 0.71015; 
                        C_gcp_A0 = 7.554;  // T range: 200 - 1000 K
                        C_gcp_A1 = -0.368e-3;
                        C_gcp_A2 = 11.846e-5;
                        C_gcp_A3 = -14.939e-8;
                        C_gcp_A4 = 5.753e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.30698;
                        C_vp3_B = 1.75845;
                        C_vp3_C = -2.1629;
                        C_vp3_D = -2.9130;
                        C_visnu = 0.210;
                        C_visN = 0.2;
                        break;

		case(n_Hexane):
                        C_name = "n_Hexane";
                        C_type = "Paraffin";
			C_crit_T = 507.60;
                        C_crit_P = 30.25;
                        C_crit_V = 368.00;
                        C_crit_Z = 0.264;
                        C_MW = 86.117;
                        C_omega = 0.300;
                        C_Hvap = 28850;
                        C_NBP = 341.88;
                        C_dipole = 0.0;
                        C_lcp_A = 18.8792; 
                        C_lcp_B = -0.899808; 
                        C_lcp_D = 0.81694; 
                        C_gcp_A0 = 8.831; // T range: 200 - 1000 K
                        C_gcp_A1 = -0.166e-3;
                        C_gcp_A2 = 14.302e-5;
                        C_gcp_A3 = -18.314e-8;
                        C_gcp_A4 = 7.124e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.53998;
                        C_vp3_B = 1.83759;
                        C_vp3_C = -2.5438;
                        C_vp3_D = -3.1630;
                        C_visnu = 0.210;
                        C_visN = 0.2;
                        break;

		case(n_Heptane):
                        C_name = "n_Heptane";
                        C_type = "Paraffin";
			C_crit_T = 540.20;
                        C_crit_P = 27.40;
                        C_crit_V = 428.00;
                        C_crit_Z = 0.261;
                        C_MW = 100.204;
                        C_omega = 0.350;
                        C_Hvap = 31770;
                        C_NBP = 371.57;
                        C_dipole = 0.0;
                        C_lcp_A = 21.6764; 
                        C_lcp_B = -0.954775; 
                        C_lcp_D = 0.92373; 
                        C_gcp_A0 = 9.634; // T range: 200 - 1000 K
                        C_gcp_A1 = 4.156e-3;
                        C_gcp_A2 = 15.494e-5;
                        C_gcp_A3 = -20.066e-8;
                        C_gcp_A4 = 7.770e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -7.77404;
                        C_vp3_B = 1.85614;
                        C_vp3_C = -2.8298;
                        C_vp3_D = -3.5070;
                        C_visnu = 0.210;
                        C_visN = 0.2;
                        break;

		case(Methanol):
                        C_name = "Methanol";
                        C_type = "Alcohol";
			C_crit_T = 512.64;
                        C_crit_P = 80.97;
                        C_crit_V = 118.00;
                        C_crit_Z = 0.224;
                        C_MW = 32.042;
                        C_omega = 0.565;
                        C_Hvap = 35210;
                        C_NBP = 337.69;
                        C_dipole = 1.7;
                        C_lcp_A = 15.1729; //
                        C_lcp_B = -11.58; // Had to approximate these ones, probably want to find better coefficients in the future
                        C_lcp_D = 3.32118; //
                        C_gcp_A0 = 4.714; //Trange 50-1000 K
                        C_gcp_A1 = -6.986e-3;
                        C_gcp_A2 = 4.211e-5;
                        C_gcp_A3 = -4.443e-8;
                        C_gcp_A4 = 1.535e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -8.63571;
                        C_vp3_B = 1.17982;
                        C_vp3_C = -2.4790;
                        C_vp3_D = -1.0240;
                        C_visnu = 0.4828;
                        C_visN = 0.35;
                        break;

		case(Ethanol):
                        C_name = "Ethanol";
                        C_type = "Alcohol";
			C_crit_T = 513.92;
                        C_crit_P = 61.48;
                        C_crit_V = 167.00;
                        C_crit_Z = 0.240;
                        C_MW = 46.069;
                        C_omega = 0.649;
                        C_Hvap = 38560;
                        C_NBP = 351.80;
                        C_dipole = 1.7;
                        C_lcp_A = 18.2568; 
                        C_lcp_B = -9.01927; 
                        C_lcp_D = 2.54959; 
                        C_gcp_A0 = 4.396; //T range: 50 - 1000 K
                        C_gcp_A1 = 0.628e-3;
                        C_gcp_A2 = 5.546e-5;
                        C_gcp_A3 = -7.024e-8;
                        C_gcp_A4 = 2.685e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -8.68587;
                        C_vp3_B = 1.17831;
                        C_vp3_C = -4.8762;
                        C_vp3_D = 1.5880;
                        C_visnu = 0.4828;
                        C_visN = 0.35;
                        break;

		case(Isopropanol):
                        C_name = "Isopropanol";
                        C_type = "Alcohol";
			C_crit_T = 508.30;
                        C_crit_P = 47.62;
                        C_crit_V = 220.00;
                        C_crit_Z = 0.248;
                        C_MW = 60.096;
                        C_omega = 0.665;
                        C_Hvap = 39850;
                        C_NBP = 355.39;
                        C_dipole = 1.7;
                        C_lcp_A = 22.8633; 
                        C_lcp_B = -12.25994; 
                        C_lcp_D = 3.71096; 
                        C_gcp_A0 = 3.334; // T range: 50 - 1000 K
                        C_gcp_A1 = 18.853e-3;
                        C_gcp_A2 = 3.644e-5;
                        C_gcp_A3 = -6.115e-8;
                        C_gcp_A4 = 2.543e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -8.73656;
                        C_vp3_B = 2.16240;
                        C_vp3_C = -8.70785;
                        C_vp3_D = 4.77927;
                        C_visnu = 0.4568;
                        C_visN = 0.7;
                        break;                        
	


		case(JP8):
                        C_name = "JP8";
                        C_type = "Paraffin";
                        C_crit_T = 658.65;
                        C_crit_P = 18.20;
                        C_crit_V = 754.00;
                        C_crit_Z = 0.251;
                        C_MW = 170.338;
                        C_omega = 0.576;
                        C_Hvap = 43368;
                        C_NBP = 489.48;
                        C_dipole = 0.0;
                        C_lcp_A = 35.6624;
                        C_lcp_B = -1.22961;
                        C_lcp_D = 1.45768;
                        C_gcp_A0 = 17.229; // T range: 200 - 1000 K
                        C_gcp_A1 = -7.242e-3;
                        C_gcp_A2 = 31.922e-5;
                        C_gcp_A3 = -42.322e-8;
                        C_gcp_A4 = 17.022e-11;
                        C_vp_equation_type = 3;
                        C_vp3_A = -9.08593;
                        C_vp3_B = 2.77846;
                        C_vp3_C = -5.1985;
                        C_vp3_D = -4.1730;
                        C_visnu = 0.210;
                        C_visN = 0.25;
			break;

		default:
			cout << "error, unrecognized component" << "\n";


}}

// Destructor:
RBComponentProperties::~RBComponentProperties(){}



// Methods:



//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentPropertiesliquidHeatCapacity computes the liquid heat capacity of a 
 pure component using the group contribution method of
 Ruzicka and Domalski (1993) as described on pg. 6.19 of the 5th edition of Poling et al.  The input is 
 system Temperature (in Kelvins) the output is the liquid heat capacity (J/kg-K). */
//*********************************************************************************************************************
//*********************************************************************************************************************

double
RBComponentProperties::liquidHeatCapacity(double T)
{

        double A = C_lcp_A;
        double B = C_lcp_B;
        double D = C_lcp_D;
        double MW = C_MW;
        double R = 8.314; //gas constant (J/mol-K)

        double cpmolar, cp;

        cpmolar = R*(A + B*(T/100) + D*pow((T/100),2)); //in units of J/mol - K

        cp = cpmolar/MW*1000; //converts to units of J/kg - K

	C_LCP = cp;

        return cp;
}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentProperties::liquidSensibleHeat uses a liquid heat 
 capacity expression to calculate the sensible heat (J/kg) for a pure liqid
 between a standard temperature (Ts in Kelvins) and another temperature (T in Kelvins). */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::liquidSensibleHeat(double T,double Ts)
{
        //note: Ts should be the initial liquid temperature of the pool system.

        double A = C_lcp_A;
        double B = C_lcp_B;
        double D = C_lcp_D;
        double MW = C_MW;
        double R = 8.314; //Gas Constant (J/K-mol)

        A = A*R*1000/MW;
        B = B*R*1000/MW;
        D = D*R*1000/MW;


         double Hs = A*(T - Ts) + (B/200)*(T*T - Ts*Ts) + (D/30000.)*(pow(T,3)-pow(Ts,3));

	C_SensM = Hs*MW/1000; // in J/mol
        C_Sens = Hs; // in J/kg
        return Hs;

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentProperties::liquidThermalConductivity calculates the 
  liquid thermal conductivity (in W/m/K) of a pure component.  The only
  input is the liquid Temperature (in Kelvins).  The method is that of Latini et al. (1977).  Details of this
  estimation method are found on pg. 10.44 of the 5th edition of Poling et al. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::liquidThermalConductivity(double T)
{

	double Astar, beta;

        if (C_type == "Paraffin"){
		Astar = 0.00350;
		beta = 0.5;}

	if (C_type == "Aromatic"){
		Astar = 0.0346;
		beta = 1.0;}

	if (C_type == "Cycloparaffin"){
		Astar = 0.0310;
		beta = 1.0;}
	if (C_type == "Alcohol"){
                Astar = 0.00339;
		beta = 0.5;}

	double alpha = 1.2;
	double gamma = 0.167;

	double Tc = C_crit_T;
	double Tr = T/Tc;
	double Tb = C_NBP;
	double MW = C_MW;
	double A,kl;

	A = Astar*pow(Tb, alpha)/pow(MW,beta)/pow(Tc, gamma);

	kl = A*pow((1 - Tr), 0.38)/pow(Tr,1./6.); // Thermal Conductivity in (W/m-K)

	C_LTHK = kl;

	return kl;
}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentProperties::liquidDensity calculates the density of a pure liquid 
  as a function of Temperature (Kelvins) and Pressure (Bar), using the method of Hankinson and Thomson 
  (1979.)  See Phase Equilibria in Chemical Engineering (Walas, 1985),  pg. 77.  Density is given in kg/m^3. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double 
RBComponentProperties::liquidDensity(double T, double P)
{

	//Preliminary Constants:
	double Tc = C_crit_T;
	double Pc = C_crit_P;
	double Vc = C_crit_V;
	double w = C_omega;
	double MW = C_MW;
	double Tb = C_NBP;
	double R = 83.14; //gas constant (cm^3-Bar/K-mol)

	//Calculate Vapor Pressure at System Temperature:
	double vp = vaporPressure(T);

	// Estimate srk eccentic factor
        double Pbr = 1./(Pc*1.01325);
        double Tbr = Tb/Tc;

        double num, den, wsrk;

        num = log(Pbr) - 5.92714 + 6.09648/Tbr + 1.28862*log(Tbr) - 0.169347*pow(Tbr,6);
        den = 15.2518 - 15.6875/Tbr - 13.4721*log(Tbr) + 0.43577*pow(Tbr,6.0);

        wsrk = num/den; //srk eccentric factor


	//Estimate the saturation molar volume using the method of Harkison and Thompson (1979).  Described on
        // pg. 77 of Walas (1985).

        double a = -1.52816;
        double b = 1.43907;
        double c = -0.81446;
        double d = 0.190454;
        double e = -0.296123;
        double f = 0.386914;
        double g = -0.0427258;
        double h = -0.0480645;

        double Tr = T/Tc;
        double tau = 1 - Tr;
        double VRO, VRD, Vstar, vs;
	double aa, bb, cc;

	//The following coefficients depend on the type of species
	if (C_type == "Paraffin")
	{
	 aa = 0.2905331;
	 bb = -0.08057958;
	 cc = 0.02276965;} // avg error: 1.23%

	if (C_type == "Cycloparaffin")
	{
	 aa = 0.6564296;
	 bb = -3.391715;
	 cc = 7.442388;} // avg error: 1.00%

	if (C_type == "Aromatic")
	{
	 aa = 0.2717636;
	 bb = -0.05759377;
	 cc = 0.05527757;} // avg error: 0.58%

	if (C_type == "Alcohol") //note: the literature does not provide coefficients for alcohols specifically
	// therefore, for alcohols, the category "all Hydrocarbons" is used instead.
	{
	 aa = 0.2851686;
	 bb = -0.06379110;
	 cc = 0.01379173;} // avg error: 1.89%



	VRO = 1 + a*pow(tau,(1./3.)) + b*pow(tau,(2./3.)) + c*tau + d*pow(tau,(4./3.));
        VRD = (e + f*Tr + g*pow(Tr,2) + h*pow(Tr,3))/(Tr - 1.00001);

        Vstar = R*Tc/Pc*(aa + bb*wsrk + cc*pow(wsrk,2));
        vs = Vstar*VRO*(1 - wsrk*VRD); // The saturated volume.




	// Now calculate the compressed volume

        double ca = -9.070217;
        double cb = 62.45326;
        double cd = -135.1102;
        double cf = 4.79594;
        double cg = 0.250047;
        double ch = 1.14188;
        double cj = 0.0861488;
        double ck = 0.0344483;

        double C = cj +ck*wsrk;
        double eee,B,v,p;

        eee = exp(cf +cg*wsrk + ch*pow(wsrk,2.));

        B = (Pc)*(-1. + ca*pow(tau,1./3.) + cb*pow(tau,2./3.) + cd*tau + eee*pow(tau,4./3.));
        v = vs*(1. - C*log((B + P)/(B + vp)));

        p = MW/v*1000.; // Convert to kg/m^3.

	C_Vm = v;
	C_Vstar = Vstar;
	C_wsrk = wsrk;

	C_Lden = p;

	return p;

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponents::liquidViscosity estimates the viscosity of a pure liquid compoenent 
 using the estimation method Sastri and Rao (1992).
 Details of the method are found on pg. 9.61 of the 5th edition of Poling et al.  The input is the system 
 Temperature (Kelvins). Liquid Viscosity is given in units of mPa-s (cP). */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::liquidViscosity(double T)
{

	double Tb = C_NBP;
	double nu = C_visnu;
	double N = C_visN;
	double Pv, Lvis;


// The following vapor pressure correlation is not the most accurate, but is the one recommended for use with this method:

	double TT = T/Tb;
	double A = 4.5398 + 1.0309*log(Tb);
	double a = pow((3 - 2*TT),0.19);
	double b = log(TT);
	double B = 1 - a/TT - 0.38*a*b;

	Pv = exp(A*B);

	Lvis = nu*pow(Pv,(-1*N)); //units of mPa-s (cP)

	C_Lvis = Lvis;

	return Lvis;


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponents::latentHeat calculates the latent heat of vaporization 
 of a pure liquid (in J/kg) as a function of liquid temperature(Kelvins). 
  The estimation method is that of Watson, for details see pg. 7.24 of the 5th edition
 of Poling et al. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::latentHeat(double T)
{

	double hvb = C_Hvap; // This should be in J/mol
	double Tc = C_crit_T;
	double Tb = C_NBP;
	double MW = C_MW;

//	double hv = hvb*pow(((1 - T/Tc)/(1 - Tb/Tc)),0.375);
	
//	C_Latent = hv/MW*1000; //convert to J/Kg


        double hv = hvb/MW*1000; //convert the boiling point latent heat to J/kg
        double hvs = liquidSensibleHeat(Tb,T);

        C_Latent = hv + hvs;  //in J/kg

        hv = C_Latent*MW/1000; // in J/mol

 	return hv;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentProperties::vaporPressure calculates the vapor pressure of a pure species (in Bar) as
 a function of Temperature (Kelvins). This incorporates two equations.  Most fuels use Wagner's Equation (1977). 
 This equation applies to Temperatures from the triple point to the critical point.  Where Wagner's coefficients are 
 not available, the Antoinne equation is used with a more limited Temperature Range.  
 See pg. 7.4 and 7.5 in the 5th edition of Poling et al. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::vaporPressure(double T)
{

	double vp;

	if (C_vp_equation_type == 3) //Wagner's Equation
{
	double A = C_vp3_A;
	double B = C_vp3_B;
	double C = C_vp3_C;
	double D = C_vp3_D;

	double Tc = C_crit_T;
	double Pc = C_crit_P;

	double tau, ps1, ps2;

	tau = (1 - T/Tc);
        ps1 = A*tau + B*pow(tau,1.5) + C*pow(tau,2.5) + D*pow(tau,5.0);
        ps2 = log(Pc) + (Tc/T)*ps1;
        vp = exp(ps2);

}

	if (C_vp_equation_type == 1)  //Antoinne Equation
{
	double A = C_vp1_A;
	double B = C_vp1_B;
	double C = C_vp1_C;

	double lvp;

	lvp = A - (B/(T + C));

	vp = pow(10,lvp);
}

	C_VP = vp;
	return vp;	

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief  The method RBComponents:: gasViscosity calculates the viscosity of a pure gas species (in Kg/m/s) 
 using the method of Chung et al. (1988) as described on pg. 9.7 of the 5th edition of Poling et al. 
 The input is gas temperature (in Kelvins). */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::gasViscosity(double T)
{

// Preliminary Constants
        double Tc = C_crit_T;
        double Vc = C_crit_V;
        double w = C_omega;
        double MW = C_MW;
	double Dm = C_dipole;

        double Tr = T/Tc;
        double Tstar = 1.2593*Tr;

        double vA = 1.16145;
        double vB = 0.14874;
        double vC = 0.52487;
        double vD = 0.77320;
        double vE = 2.16178;
        double vF = 2.43787;


	double CI, Fc, mur, k, muf;

	//calculate dimensionless dipole moment:
	mur = 131.3*Dm/pow(Vc*Tc,0.5);

	//Association Factor, non-zero only for associating components (alcohols).
	if (C_type != "Alcohol") { k = 0.0;}

	if (C_type == "Alcohol"){
	 if (C_name == "Methanol") { k = 0.215; }
	 if (C_name == "Ethanol") { k = 0.175; }
	 if (C_name == "Isopropanol") { k = 0.143; }}

	CI = (vA*pow(Tstar,-vB)) + vC*exp(-vD*Tstar) + vE*exp(-vF*Tstar);
        Fc = 1.0 -0.2756*w + 0.059035*pow(mur,4) + k;


	muf = 40.785*Fc*sqrt(MW*T)/pow(Vc,(2./3.))/CI; // in micropoise
        muf = muf*1e-7; // in kg/m-s;

	C_Gvis = muf;

	return muf;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponentProperties:: gasHeatCapacity calculates the gas phase heat capacity of a pure component 
 in (J/kg-K) using a fourth order polynomial fit for Temperature (in Kelvins).  
 See appendix A (section C) of the 5th edition of Poling et al. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::gasHeatCapacity(double T)
{

	double MW = C_MW;

	//coefficients for gas phase heat capacity polynomial fit

        double a0 = C_gcp_A0;
        double a1 = C_gcp_A1;
        double a2 = C_gcp_A2;
        double a3 = C_gcp_A3;
        double a4 = C_gcp_A4;

        double cppm = a0 + a1*T + a2*T*T + a3*T*T*T + a4*T*T*T*T; //dimensionless

        double R = 8.314; //gas constant (J/K-mol)

        double cp = cppm*R*1000./MW; // (J/kg-K)

	C_GCP = cp;

 	return cp;

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief The method RBComponents::gasThermalConductivity calculate the thermal conductivity of a low pressure 
 pure gas (W/m/K) as a function of Temperature (Kelvins). The method of Chung et al. (1984) is used.  
 For details of the method see pg.10.12 of the 5th edition of Poling et al. */
//*********************************************************************************************************************
//*********************************************************************************************************************
double
RBComponentProperties::gasThermalConductivity(double T)
{
	

	double MW = C_MW;
        double muf = gasViscosity(T);
        double Cv = gasHeatCapacity(T)*MW/1000-8.314; // J/K-mol
        double Tr = T/C_crit_T;
        double w = C_omega;

        
        double Z = 2.0 + 10.5*Tr*Tr;
	
	double Beta;
	if (C_type != "Alcohol") { Beta = 0.7862 - 0.7109*w + 1.3168*w*w;}
	if (C_type == "Alcohol") {
	 if (C_name == "Methanol") { Beta = 1/1.31; }
	 if (C_name == "Ethanol" ) { Beta = 1/1.38; }
	 if (C_name == "Isopropanol") { Beta = 1/1.32; }}

        double Alpha = (Cv/8.314) - (3./2.);
        double num, den, psi;

        num = (0.215 + 0.28288*Alpha - 1.061*Beta + 0.266652*Z);
        den = (0.6366 + Beta*Z + 1.061*Alpha*Beta);
        psi = 1.0 + Alpha*num/den;

        double tkf = 3750*psi*muf*8.314/MW; //(W/m-K)

	C_GTHK = tkf;

	return tkf;


}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************













