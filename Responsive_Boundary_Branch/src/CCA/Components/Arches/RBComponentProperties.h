/* This code defines the RBComponentProperties class for use in the multicomponent version
of the liquid pool model.  
*/


#ifndef Uintah_Components_Arches_RBComponentProperties_h
#define Uintah_Components_Arches_RBComponentProperties_h


#include<iostream>
#include<cmath>
#include<vector>
#include<string>

using namespace std;
namespace Uintah{



//-------------------------------------------------------

/** 
* @class RBComponentProperties
* @author Weston Eldredge
* @date February 2009
*
* @brief This class supports the Responsive Boundary Class, it contains several physical properties
* of a pure fuel chemical species (i.e. critical properties, vapor pressure coefficients, etc.) and 
* the methods to compute physical properties as functions of temperature and Pressure. 
*/

//-------------------------------------------------------



//enumerate the fuels:
enum fuel { n_Octane = 0, n_Dodecane = 1, n_Hexadecane, O_Xylene, M_Xylene, P_Xylene,
            Tetralin, C_Decalin, T_Decalin, Benzene, Toluene, n_Pentane, n_Hexane,
            n_Heptane, Methanol, Ethanol, Isopropanol, JP8};


class RBComponentProperties {

	public:

//Default Constructor:
	RBComponentProperties();

//Standard Constructor:
 	RBComponentProperties(fuel);


//Destructor:
	~RBComponentProperties();

//Getter Methods:
	double getTC() { return C_crit_T; }
	double getPC() { return C_crit_P; }
 	double getVC() { return C_crit_V; }
	double getZC() { return C_crit_Z; }
	double getMW() { return C_MW; } 
	double getOmega() { return C_omega; }
	double getHV() { return C_Hvap; }
	double getNBP() { return C_NBP; }
	double getDP() { return C_dipole; }
	fuel   getID() { return C_ID; }
	string getName() { return C_name; }
	string getType() { return C_type; }
	double getLMV() { return C_Vm; }
	double getVstar() { return C_Vstar; }
	double getWSRK() { return C_wsrk; }
	double getSensM() { return C_SensM; } //accesses molar liquid Enthalpy
	double getSens() { return C_Sens; }  //accesses mass liquid Enthalpy
	double getClat() { return C_Latent; } // Access specific latent heat

//Printer Methods:
	void printProperties() {
		cout << "The fuel is: " << getName() << "\n";
		cout << "Its pure component properties are: " << "\n";
		cout << "Tc: " <<  getTC() << "\n";
		cout << "Pc: " <<  getPC() << "\n";
		cout << "Vc: " <<  getVC() << "\n";
		cout << "Zc: " <<  getZC() << "\n";
		cout << "MW: " <<  getMW() << "\n";
		cout << "w: " <<   getOmega() << "\n";
		cout << "Hvap: " << getHV() << "\n";
		cout << "Boiling Point: " <<  getNBP() << "\n";
		cout << "Dipole: " <<  getDP() << "\n";
	}

//Property Calculation Methods:
        double liquidHeatCapacity(double);
        double liquidSensibleHeat(double, double);
	double liquidThermalConductivity(double);
	double liquidDensity(double, double);
        double liquidViscosity(double);
	double latentHeat(double);
	double vaporPressure(double);
	double gasViscosity(double);
	double gasHeatCapacity(double);
	double gasThermalConductivity(double);

//Members:

	private:

	//componenet identity:
        fuel C_ID;
	string C_name;
        string C_type;

	//Component properties:
	double C_crit_T; //critical Temperature (kelvins)
	double C_crit_P; //crtical Pressure (Bar)
	double C_crit_V; //crtical Volume (cm^3/mol)
	double C_crit_Z; //crtical Compressibility (d-less)
	double C_MW; //molar mass (g/mol)
	double C_omega; //accentric factor (d-less)
	double C_Hvap; //heat of vaporization at normal boiling point (J/mol)
	double C_NBP; //normal boiling point (kelvins)
	double C_dipole; //molecule dipole (debye)

	//liquid heat capacity parameters:
	double C_lcp_A;
	double C_lcp_B;
	double C_lcp_D;

	//gas phase heat capacity parameters:
	double C_gcp_A0;
	double C_gcp_A1;
	double C_gcp_A2;
	double C_gcp_A3;
	double C_gcp_A4;

	//vapor pressure parameters:
	int C_vp_equation_type;
        //for equation type1:
	double C_vp1_A;
	double C_vp1_B;
	double C_vp1_C;
	//for equation type3:
	double C_vp3_A;
	double C_vp3_B;
	double C_vp3_C;
	double C_vp3_D;

        //miscellaneous parameters
        // for gas viscosity:
        double C_Astar;
        double C_Beta;
	// for liquid viscosity:
	double C_visnu; // mPa-s
	double C_visN;

        //Computed Physcial Pure Component Properties:
        double C_LCP; //Liquid Heat Capacity (J/kg-K)
        double C_Latent;  //Liquid Latent Heat of Vaporization (J/kg)
        double C_VP; //Component Vapor Pressure (Bar)
        double C_LTHK; //Liquid Thermal Conductivity (W/m-K)
        double C_Sens; //Liquid Sensible Heat (J/kg)
	double C_SensM; //lqiuid Sensible Heat (J/mol)
        double C_Lden; // Liquid Density (kg/m^3)
	       double C_Vm; // liquid Molar Volume (cm^3/mol)
	       double C_Vstar; // Special Volume for density calculations;
	       double C_wsrk; //
	double C_Lvis; // Liquid Viscosity (Pa-s)
        double C_Gvis; // Gas Viscosity (kg/m/s)
        double C_GCP; // Gas Heat Capacity (J/kg-K)
        double C_GTHK; // Gas Thermal Conductivity (W/m-K)


};


} //namespace Uintah

#endif
