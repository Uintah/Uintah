// The following header file defines the ResponsiveBoundary Class for 
// both single and multi-component systems:


#ifndef Uintah_Components_Arches_ResponsiveBoundary_h
#define Uintah_Components_Arches_ResponsiveBoundary_h

#include <Core/Containers/Array1.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/ResponsiveBoundary.h>
#include <CCA/Ports/SchedulerP.h>



#include<iostream>
#include<cmath>
#include<vector>
#include<string>
#include<CCA/Components/Arches/RBMixerProperties.h>
//#include"RBMixerProperties.h"


using namespace std;

namespace Uintah{

using namespace SCIRun;

 class ArchesVariables;
 class ArchesConstVariables;
 class CellInformation;
 class VarLabel;
 class PhysicalConstants;
 class Properties;
 class Stream;
 class InletStream;
 class ArchesLabel;
 class MPMArchesLabel;
 class ProcessorGroup;
 class DataWarehouse;
 class TimeIntegratorLabel;
 class ExtraScalarSolver;
 class Output;


//-------------------------------------------------------

/** 
* @class ResponsiveBoundary
* @author Weston Eldredge (weseldredge@yahoo.com)
* @date Febuary 2009
*
* @brief This class predicts burn rates for liquid pool systems in responsive to heat input from a flame.
*
*/

//-------------------------------------------------------




class ResponsiveBoundary {

        public:

        //Default Constructor:

        ResponsiveBoundary();

        //Standard Constructor:

        ResponsiveBoundary(vector<bool>,vector<double>,vector<double>,double,double,double,double,double,double,double,double,double,int);

        //Standard Destructor:

        ~ResponsiveBoundary();

	//Setter/Assigner methods:

	void setDataSaveTime(double dst) { Data_SaveTime = dst; }
        void setTimeStep(double dt) { RB_dt = dt; cout << "TimeStep: " << RB_dt << endl;  }
        void setRefTemp(double T) { RB_RefTemp = T; }
        void setReflectivity(double R) { RB_Reflectivity = R; }
	void AssignHeight(double H) { RB_LiquidHeight = H; }
	void AssignT(vector<double> T) { RB_Tprofile.clear(); for (int i = 0; i < RB_NON; i++){RB_Tprofile.push_back(T[i]);}}
        void AssignYb(vector<double> Yb){ RB_BulkComposition.clear(); for (int ii=0;ii<RB_NOC;ii++){RB_BulkComposition.push_back(Yb[ii]);}}
//	void AssignX(vector<double> X) { 
//                        for (unsigned int ii = 0; ii < RB_Xprofile.size(); ii++){ delete RB_Xprofile[ii]; }
//                        RB_Xprofile.clear(); 
//                        for (int j = 0; j < RB_NOC; j++){ RB_Xprofile.push_back(new vector<double>);
//	   		for (int i = 0; i < RB_NON; i++){
//				RB_Xprofile[j]->push_back(X[j]);}}}

        void AssignX(vector<double> X) 
        {
          RB_Xprofile.clear();
          for (unsigned int ii = 0; ii < X.size();  ii++) { RB_Xprofile.push_back(X[ii]); }
        }

	//getter/accessor methods:


	double getT(int i){ return RB_Tprofile.at(i); }
        double getST() { return RB_SurfaceTemperature; }
        double getTinfinity() { return RB_Tinfinity; }
	double getDTdt(int i) { return RK_DTdt.at(i); }
	double getHeight() { return RB_LiquidHeight; }
	double getRadiation() { return RB_RadiationInput; }
	double getPoolDiameter() {return RB_PoolDiameter; }
	double getVaporDensity() { return RB_VaporDensity; }
	double getEndTime() { return RB_EndTime; }
	double getTime() { return RB_Time; }
	double getWindSpeed() { return RB_WindSpeed; }
	double getSystemPressure() { return RB_SystemPressure; }
	double getRefTemp() { return RB_RefTemp; }
        double getReflectivity() { return RB_Reflectivity; }
//	double getX(int i, int j) { return RB_Xprofile[i]->at(j); }
//	double getC(int i, int j) { return RB_Cprofile[i]->at(j); }
        double getX(int j) { return RB_Xprofile.at(j); }
        double getC(int j) { return RB_Cprofile.at(j); }
//	double getDCdt(int i, int j) { return RK_DCdt[i]->at(j); }
        double getDCdt(int ii) { return RK_DCdt.at(ii); }
        double getDHdt() { return RK_DHdt; }
//	double getSumC() {double sum = 0; for (int j=0;j<(RB_NOC);j++){sum = sum + getC(j);} return sum;}
	double getSumX() {double sum = 0; for (int j=0;j<(RB_NOC);j++){sum = sum + getX(j);} return sum;}
	double getJ(int i, int j) { return RB_J[i]->at(j); }
	double getH(int i, int j) { return RB_H[i]->at(j); }
	double getRFA(int i) { return RB_RFA.at(i); }
	double getDA(int i) { return RB_DA.at(i); }
//	double getDensity(int i) { return RB_Density.at(i); }
        double getDensity() { return RB_Density; }
//	double getMDensity(int i) { return RB_MolarDensity.at(i); }
        double getMolarDensity() { return RB_MolarDensity; }
	double getTHK(int i) { return RB_THK.at(i); }
	double getHCP(int i) { return RB_HCP.at(i); }
	double getNflux(int i) { return RB_mix->getNflux(i); }
//	double getCT(int i) {double sum = 0; for(int j=0;j<RB_NOC;j++){sum = sum + getC(j,i);} return sum;}
        double getCT() {double sum=0; for(int j=0;j<RB_NOC;j++){sum = sum + getC(j);} return sum;}
	double getBP() { return RB_BPT; }
	double getFV() { return RB_FV; }
        double getMassFlux() { return RB_MF; }
        double getConvectiveHeatFlux() { return RB_ConvectiveHeatFlux; }
        double getSumEnergy() { double sum = 0; for (unsigned int i = 0; i < RB_EnergySave.size(); i++) {sum = sum + RB_EnergySave[i];} return sum;}


        //Methods related to model verfication:
        void verifyOn(int);
        void verifyOff() {RB_Verification_Energy=0; RB_Verification_Height = 0; RB_Verification_Height_Energy = 0;}
        double verifyMMS(double,double,int,int);
        void compareMMS();
        double getErrorMMS_Energy() { return RB_normErrorMMS_Energy; }
        double getcompareMMS_Energy(int ii) { return RB_compareMMS_Energy.at(ii);}
        double getcompareMMS_DTdz2(int ii) { return RB_compareMMS_DTdz2.at(ii); }
        double getcompareMMS_DTdt(int ii) { return RB_compareMMS_DTdt.at(ii); }
        double getcompareMMS_DHdt() { return RB_compareMMS_DHdt;}
        double getcompareMMS_Height() { return RB_compareMMS_Height; }
        double getEnergyVerify(int i, int j) { return Data_Energy_Verify[i]->at(j); }
        double getDTdz2Verify(int i, int j) { return Data_DTdz2_Verify[i]->at(j); }
        double getDTdtVerify(int i, int j) { return Data_DTdt_Verify[i]->at(j); }
        double getDHdtVerify(int i) { return Data_DHdt_Verify.at(i); }

//	double getMassBalance() {double sum;
//		for (int i = 0; i < RB_NON; i++){ for (int j = 0; j < RB_NOC; j++){
//			sum = sum + getC(j,i)*dz.at(i);}}
//		return sum;}

	int getNON() {return RB_NON;}
	int getNOC() {return RB_NOC;}

	//accessor methods for the print data:
	double getD_Time(int i) { return Data_Time.at(i); }
	double getD_T(int i) { return Data_STemp.at(i); }
	double getD_X(int i, int j){ return Data_XComp[i]->at(j); }
	double getD_C(int i, int j) { return Data_SComp[i]->at(j); }
	double getD_MF(int i) { return Data_MBR.at(i); }
	double getD_NF(int i) { return Data_NBR.at(i); }
	double getD_H(int i) { return Data_Height.at(i); }
	double getD_BPT(int i) { return Data_BPT.at(i); }
        double getD_ConvectiveHeatFlux(int i) { return Data_ConvectiveHeatFlux.at(i); }
        double getD_HV(int i) { return Data_Height_Verify.at(i); }
        double getD_TP(int ii, int jj) { return Data_Temperature_Profile[ii]->at(jj); }
        int getD_size() { return Data_Time.size(); }

	string getName(int i) { return RB_mix->getName(i); }



	//print out methods:
//	void printC(int l){ 
//	   cout.precision(7);
//	   cout << "Component: " << "\t";
//	   for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
//	   cout << "\n\n\n";
//	   for (int i = 0; i < l; i++){ 
//	   cout << "C[" << i << "]: " << "\t\t";
//	   for (int j = 0; j < RB_NOC; j++){
//	   cout << getC(j,i) << "    ";} cout << "\n";}}


//	void printDXdt(int l){
//           cout.precision(4);
//           cout << "Component: " << "\t";
//           for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
//           cout << "\n\n\n";
//           for (int i = 0; i < l; i++){
//           cout << "DXdt[" << i << "]: " << "\t\t";
//           for (int j = 0; j < RB_NOC; j++){
//           cout << getDXdt(j,i) << "    ";} cout << "\n";}}


	
//	void printX(int l){
//           cout.precision(5);
//          cout << "Component: " << "\t";
//           for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
//           cout << "\n\n\n";
//           for (int i = 0; i < l; i++){
//           cout << "X[" << i << "]: " << "\t";
//           for (int j = 0; j < RB_NOC; j++){
//           cout << getX(j,i) << "    ";} cout << "\n";}}


	void printT(int l){
	  for (int i = 0; i < l; i++){ cout << "T[" << i << "]: " << getT(i) << "\n";}}

        void printMMS(int);

        void printCompareMMS_Energy() 
        {
          cout.precision(25); 
          for (unsigned int jj = 0; jj < Data_Time.size(); jj++)
          {
            cout << "For Time = " << getD_Time(jj) << "\n\n";
            double H = Data_Height[jj]; 
            double dz = H/RB_NON;
            for (int ii = 0; ii < RB_NON; ii++) 
              {cout << "Height: " << (H - (ii + 0.5)*dz) << "     " << "Temperature: " << Data_Temperature_Profile[jj]->at(ii) << "     " << "Error: " << getEnergyVerify(jj,ii) << endl;}
          }
        }     

        void printCompareMMS_DTdz2()
        {
          cout.precision(25);
          for (unsigned int jj = 0; jj < Data_Time.size(); jj++)
          {
            cout << "For Time = " << getD_Time(jj) << "\n\n";
            double H = Data_Height[jj];
            double dz = H/RB_NON;
            for (int ii = 0; ii < RB_NON; ii++)
            {
              cout << "Height: " << (H - (ii+0.5)*dz) << "   Error in DTdz2: " << getDTdz2Verify(jj,ii) << endl;
            } 
          } 
        }



        void printCompareMMS_DTdt()
        {
          cout.precision(25);
          for (unsigned int jj = 0; jj < Data_Time.size(); jj++)
          {
            cout << "For Time = " << getD_Time(jj) << "\n\n";
            double H = Data_Height[jj];
            double dz = H/RB_NON;
            for (int ii = 0; ii < RB_NON; ii++)
            {
              cout << "Height: " << (H - (ii+0.5)*dz) << "   Error in DTdt: " << getDTdtVerify(jj,ii) << endl;
            }
          }
        }

        void printCompareMMS_DHdt()
        {
          cout.precision(17);
          for (unsigned int jj = 0; jj < Data_Time.size(); jj++)
          {
            cout << "Time: " << getD_Time(jj) << "   Error in DHdt: " << getDHdtVerify(jj) << endl;
          }
        }



        void printErrorMMS_Energy() { cout << "ErrorNorm: " << getErrorMMS_Energy() << endl; }


	void printDTdt(int l){
          for (int i = 0; i < l; i++){ cout << "DTdt[" << i << "]: " << getDTdt(i) << "\n";}}

	void printFlux() {
	   cout.precision(7);
	   cout << "Component: " << "\t";
	   for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
	   cout << "\n\n";
	   for (int j = 0; j < RB_NOC; j++){
	   cout << getNflux(j) << "    "; } cout << "\n";}

	void printJ(int l){
           cout << "Component: " << "\t";
           for (int i = 0; i < (RB_NOC); i++){ cout << getName(i) << "\t";}
           cout << "\n\n\n";
           for (int i = 0; i < l; i++){ 
           cout << "J[" << i << "]: " << "\t\t";
           for (int j = 0; j < (RB_NOC); j++){
           cout << getJ(j,i) << "    ";} cout << "\n";}}

	void PrintD_T() {
           cout.precision(19);
	   cout << "Surface Temperature: " << "\n\n";
	   cout << "Time: " << "\t\t" << "Surface Temp: " << "\n";
	   for( unsigned int i = 0; i < Data_STemp.size(); i++ ) {
	     cout << getD_Time(i) << "\t\t" << getD_T(i) << "\n";
	   }
	}

        void PrintD_ConvectiveHeatFlux()
           {
             cout << "Convective Heat Fluxes: " << "\n\n";
             cout << "Time: " << "\t\t" << "Conv. Heat Flux: " << "\n\n";
             for (unsigned int ii = 0; ii < Data_ConvectiveHeatFlux.size(); ii++) 
             {
               cout << getD_Time(ii) << "\t\t" << getD_ConvectiveHeatFlux(ii) << endl;
             }
           }

	void PrintD_MASS() {
           cout.precision(19);
           cout << "Burn Rates: " << "\n\n";
           cout << "Time: " << "\t\t" << "Mass Flux: " << "\t\t" << "Mole Flux: " << "\n";
           for (unsigned int i = 0; i < Data_STemp.size(); i++) {
              cout << getD_Time(i) << "\t\t" << getD_MF(i) << "\t\t" << getD_NF(i) << "\n";}}

	void PrintD_HEIGHT() {
           cout.precision(16);
	   cout << "Pool Height: " << "\n\n";
           cout << "Time: " << "\t\t" << "Height: " << "\n";
           for (unsigned int i = 0; i < Data_STemp.size(); i++) {
              cout << getD_Time(i) << "\t\t" << getD_H(i) << "\n";}}

	void PrintD_C() {	
	   cout << "Surface Composition (concentration) for: " << "\n\n" << "Time: " << "\t\t";
	   for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
	   cout << "\n\n";
	   for (unsigned int i = 0; i <Data_STemp.size(); i++){
	      cout << getD_Time(i);
	     for (int j = 0; j < RB_NOC; j++){
	       cout << "\t\t" << getD_C(j,i);} cout << "\n";}}

	void PrintD_X() {
	   cout.precision(5);
           cout << "Surface Composition (mole fraction) for: " << "\n\n" << "Time: " << "\t\t";
           for (int i = 0; i < RB_NOC; i++){ cout << getName(i) << "\t";}
           cout << "\n\n";
           for (unsigned int i = 0; i <Data_STemp.size(); i++){
              cout << getD_Time(i);
             for (int j = 0; j < RB_NOC; j++){
               cout << "    " << getD_X(j,i)*100;} cout << "\n";}}


	void PrintD_BUBBLE() {
           cout << "Bubble Point Temperature: " << "\n\n";
           cout << "Time: " << "\t\t" << "Height: " << "\n";
           for (unsigned int i = 0; i < Data_BPT.size(); i++) {
              cout << getD_Time(i) << "\t\t" << getD_BPT(i) << "\n";}}
	   

	void PrintD_HVerify() {
          cout.precision(17);
          cout << "Height Verification: " << "\n\n";
          cout << "Time: " << "\t\t" << "Height Error: " << "\n";
          for (unsigned int ii = 0; ii < Data_Height_Verify.size(); ii++) {
            cout << getD_Time(ii) << "\t\t" << getD_HV(ii) << "\n";}}


        void PrintD_TProfile(int n) 
        {
          cout.precision(19);
          for (unsigned int ii = 0; ii < Data_Time.size(); ii++)
          {
            cout << "For Time = " << Data_Time.at(ii) << "\n\n";
            double H = getD_H(ii);
            double dz = H/RB_NON;
            for (int jj = 0; jj < n; jj++)
            { 
              cout << "Height: " << (H - (jj + 0.5)*dz) << "   Temperature: " << getD_TP(ii,jj) << endl;
            }
          }
        }
               

//Primary Methods:
  void TimeStep();
  void CalculateTimeStep();
  void RungaKutta4();
  void RungaKutta2();
  void LinearLevel();
  void Calculate_Flux();
  void Calculate_Enthalpy();
  void EnergyBalance(int);
  void SurfaceTemperature();
  void MoleBalance();
  void RadiationDistribution(double);
  void Reset(vector<bool>,vector<double>, vector<double>,vector<double>,double,double,double,
             double,double,double,double,double,int);
  void Reset(vector<bool> fv);




//problemSetup:
  void problemSetup(ProblemSpecP& params,
                    const ProblemSpecP& restart_ps);



//Methods called from Arches directly:
  void sched_readResponsiveBoundaryData( SchedulerP& sched,             
                     const PatchSet* patches,                  
                     const MaterialSet* matls,                 
                     Output* DataArchive,
                     int direction);   

  void sched_saveResponsiveBoundaryData( SchedulerP& sched,          
                      const PatchSet* patches,         
                      const MaterialSet* matls,
                      const ArchesLabel* d_lab,       
                      Output * DataArchive,
                      int initialStep,
                      int direction);        

  void sched_updateResponsiveBoundaryProfile( SchedulerP& sched,         
                            const PatchSet* patches,     
                            const MaterialSet* matls,
                            const ArchesLabel* d_lab,
                            int cellTypeID,
                            bool initialStep,
                            int direction,
                            double windspeed);   

       
  void saveResponsiveBoundaryData( const ProcessorGroup*,           
                 const PatchSubset*,                  
                 const MaterialSubset*,                 
                 DataWarehouse*,                           
                 DataWarehouse*,  
                 const ArchesLabel* d_lab,          
                 Output * DataArchive,
                 int cellTypeID,
                 int direction);        


  void readResponsiveBoundaryData( const ProcessorGroup*,          
                 const PatchSubset*,                 
                 const MaterialSubset*,                  
                 DataWarehouse*,                             
                 DataWarehouse*,                              
                 Output * DataArchive,
                 int direction); 

  void updateResponsiveBoundaryProfile(const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     const ArchesLabel* d_lab,
                     int cellTypeID,
                     bool initialStep,
                     int direction,
                     double windspeed);
     
 

	private:

	//members:

	int RB_NOC;
	int RB_NON;
        vector<bool> RB_tempfuelvec;
 	vector<bool> RB_fuelvec;
        vector<double> RB_fuelX; 
        RBMixerProperties* RB_mix;
//	vector<vector<double>*> RB_Xprofile; // tracks the mole fractions
//	vector<vector<double>*> RB_Cprofile; // tracks the number of moles
        vector<double> RB_Xprofile;  //tracks the mole fractions of liquid species
        vector<double> RB_Cprofile;  //tracks the molar concentrations of liquid species
        vector<double> RB_BulkComposition; //contains the molefractions of fuel species in the bulk gas phase above the pool.


	double RB_LiquidHeight;
	double RB_RadiationInput;
	double RB_PoolDiameter;
	double RB_VaporDensity;
	double RB_WindSpeed;
	double RB_EndTime;
	double RB_SystemPressure;
	double RB_RefTemp;
	double RB_dt;
	double RB_Time;
	vector<double> RB_Tprofile;
        double RB_SurfaceTemperature;
        double RB_Tinfinity;
        double RB_Reflectivity;
//	vector<double> dz;
        bool RB_Accelerate;
        double RB_AccelTime; 	
        double RB_Kfactor; //adjustment factor for mass transfer coefficients (for validation purposes)
	
	//computed members
	vector<vector<double>*> RB_J;  // diffusive fluxes.
	vector<vector<double>*> RB_H;  // species Enthalpies.
	vector<double> RB_RFA; //Radiation Fraction Array.
	vector<double> RB_DA; // Derivative Array for Runga-Kutta Method.
	   //liquid properties:
//	vector<double> RB_Density;
        double RB_Density;
//	vector<double> RB_MolarDensity;
        double RB_MolarDensity;
	vector<double> RB_THK;
	vector<double> RB_HCP;
	double RB_FV;  //fuel vapor velocity (m/s)
        double RB_MF;  //total mass flux (kg/m^2/s)
        double RB_ConvectiveHeatFlux; //Convection Heat Flux (W/m/m) 
        vector<double> RB_EnergySave;  //used to calculate burn rate in boiling phase

	//members associated with RungaKutta:
	vector<double> k1;
	vector<double> k2;
	vector<double> k3;
	vector<double> k4;
        double RK_DHdt;
	vector<double> RK_DTdt;
//	vector<vector<double>*> RK_DCdt;
	vector<double> RK_DCdt;
	double RB_BPT;

	//Data Saving Vectors:
	vector<double> Data_Time; // Data Times
	vector<double> Data_Height;  // Liquid Height
	vector<double> Data_STemp;   // Surface Temperature
	vector<vector<double>*> Data_SComp;  //Surface Composition
	vector<vector<double>*> Data_XComp; //surface mole fraction
	vector<double> Data_MBR; //Mass Burn Rate
	vector<double> Data_NBR; // Molar Burn Rate
	vector<double> Data_BPT; // Bubble Point Temperature
        vector<double> Data_ConvectiveHeatFlux; //Convective Heat Flux (W/m/m)
	double Data_SaveTime;
        vector<vector<double>*> Data_Temperature_Profile;
        //Verification Related Saving Vectors:
        vector<double> Data_Height_Verify;
        vector<vector<double>*> Data_Energy_Verify;
        vector<vector<double>*> Data_DTdz2_Verify;
        vector<vector<double>*> Data_DTdt_Verify;
        vector<double> Data_DHdt_Verify;


	//Member associated with Time Stepping:
	double Old_Height;
	double New_Height;

        //Data Storage Maps:
        map<IntVector, double> RB_mapHeight;
	map<IntVector, vector<double> > RB_mapX;
        map<IntVector, vector<double> > RB_mapTemperature;

        //FilePath:
        string RB_filePath;  
        
        //Members Related To Model Verification:
        bool RB_Verification_Energy;
        bool RB_Verification_Height;
        bool RB_Verification_Height_Energy;
        double RB_normErrorMMS_Energy;
        vector<double> RB_compareMMS_Energy;
        vector<double> RB_compareMMS_DTdz2;
        vector<double> RB_compareMMS_DTdt;
        double RB_compareMMS_DHdt;
        double RB_compareMMS_Height;

};

}  //namespace Uintah

#endif
