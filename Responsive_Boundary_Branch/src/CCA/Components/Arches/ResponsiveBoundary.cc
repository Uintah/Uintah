//--------------------------ResponsiveBoundary.cc----------------------------------------------------------------------


#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ResponsiveBoundary.h>
//#include "ResponsiveBoundary.h"
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>   
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Disclosure/TypeUtils.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/OS/Dir.h> //WME 
#include <sys/stat.h> //WME
#include <errno.h> //WME


#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<string>
#include<vector>



using namespace std;
using namespace Uintah;
using namespace SCIRun;


//*********************************************************************************************************************
//*********************************************************************************************************************
// Destructor and Constructors for ResponsiveBoundary:
//*********************************************************************************************************************
//*********************************************************************************************************************
//Default Constructor:
ResponsiveBoundary::ResponsiveBoundary(){}

//Standard Constructor:
ResponsiveBoundary::ResponsiveBoundary(vector<bool> fv, vector<double> x, vector<double> Yb, double T, double P,
					double h, double rad, double U, double dt, double PD,
					double VD, double Tinf, int NON):RB_fuelvec(fv)

{


	RB_NON = NON;
	RB_mix = 0;
	RB_mix = new RBMixerProperties(fv);

	RB_NOC = RB_mix->getNOC();

	RB_LiquidHeight = h;
        RB_RadiationInput = rad;
        RB_PoolDiameter = PD;
        RB_VaporDensity = VD;
        RB_WindSpeed = U;
	RB_EndTime = dt;
	RB_SystemPressure = P;
	RB_RefTemp = T;
	RB_Time = 0.0;
	Data_SaveTime = 1.0; 
        RB_Tinfinity = Tinf;

	RB_Tprofile.clear();
	for (int i = 0; i < RB_NON; i++){ RB_Tprofile.push_back(T);}

	RB_Density = RB_mix->mix_liquidDensity(T,P,x);
	RB_MolarDensity = RB_mix->getLMCon();

	RB_Xprofile.clear();
	RB_Cprofile.clear();

//	for (unsigned int j = 0; j < x.size(); j++){ RB_Xprofile.push_back(new vector<double>);  RB_Cprofile.push_back(new vector<double>);
//	   for (int i = 0; i < RB_NON; i++){
//	   	RB_Xprofile[j]->push_back(x.at(j));
//	        RB_Cprofile[j]->push_back(Ct*x.at(j));}}

        for (int jj = 0; jj < RB_NON;  jj++)
        {
          RB_Xprofile.push_back(x.at(jj));
          RB_Cprofile.push_back(RB_MolarDensity*x.at(jj));
        }

        vector<double> MoleCompBulk = RB_mix->MoleFraction(Yb);
        AssignYb(MoleCompBulk);

        SurfaceTemperature();
}


ResponsiveBoundary::~ResponsiveBoundary(){}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************



//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description  of ResponsiveBounary::TimeStep:
 This method executes the timestep calculations for the liquid pool model.  It is the main method of the 
 Responsive Boundary class. */ 
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::TimeStep()
{

cout << "Start TimeStep." << endl;
	int c = RB_NOC;


//  Create Data Saving Vectors
	Data_Time.clear();	Data_Time.push_back(getTime());
	Data_Height.clear();	Data_Height.push_back(getHeight());
	Data_STemp.clear();	Data_STemp.push_back(getST());
        for (unsigned int ii = 0; ii < Data_SComp.size(); ii++){ delete Data_SComp[ii];}
        for (unsigned int ii = 0; ii < Data_XComp.size(); ii++){ delete Data_XComp[ii];}
	Data_SComp.clear();     for (int i = 0; i < c; i++){ Data_SComp.push_back(new vector<double>); Data_SComp[i]->push_back(getC(i)); }
	Data_XComp.clear(); 	for (int i = 0; i < c; i++){ Data_XComp.push_back(new vector<double>); Data_XComp[i]->push_back(getX(i)); }
        

	Data_BPT.clear();	Data_BPT.push_back(0);
        Data_Height_Verify.clear();  Data_Height_Verify.push_back(0.0);
       
        
        for (unsigned int ii = 0; ii < Data_Temperature_Profile.size(); ii++) { delete Data_Temperature_Profile[ii];} 
        Data_Temperature_Profile.clear();  Data_Temperature_Profile.push_back(new vector<double>);
        for (int ii = 0; ii < RB_NON; ii++) {Data_Temperature_Profile[0]->push_back(getT(ii));}

        if (RB_Verification_Energy | RB_Verification_Height | RB_Verification_Height_Energy)
        {
          for (unsigned int ii = 0; ii < Data_Energy_Verify.size(); ii++) {delete Data_Energy_Verify[ii]; delete Data_DTdz2_Verify[ii];}
          Data_Energy_Verify.clear(); Data_Energy_Verify.push_back(new vector<double>);
          for (int ii = 0; ii < RB_NON; ii++) {Data_Energy_Verify[0]->push_back(0.0);}
          Data_DTdz2_Verify.clear(); Data_DTdz2_Verify.push_back(new vector<double>);
          for (int ii = 0; ii < RB_NON; ii++) {Data_DTdz2_Verify[0]->push_back(0.0);}


          for (unsigned int ii = 0; ii < Data_DTdt_Verify.size(); ii++) {delete Data_DTdt_Verify[ii];}
          Data_DTdt_Verify.clear();  Data_DHdt_Verify.clear(); Data_DTdt_Verify.push_back(new vector<double>);
          for (int ii = 0; ii < RB_NON; ii++) {Data_DTdt_Verify[0]->push_back(0.0000);}  Data_DHdt_Verify.push_back(0.0000);
        }

        

	vector<double> x(c);
	for (int i = 0; i < c; i++){ x[i] = getX(i); }
	double PD = RB_PoolDiameter;
	double U = RB_WindSpeed;
	double RT = RB_RefTemp;
        double P = getSystemPressure();
        double BPT = RB_mix->BubblePointT(P,x);


	RB_mix->film_Flux_Main(getST(), getTinfinity(),P,x,RB_BulkComposition,PD,U,RT);
        double HTC = RB_mix->getHeatTransferCoeff(); 
        RB_ConvectiveHeatFlux = HTC*(RB_Tinfinity - getST());
         

	Data_MBR.clear();   	Data_MBR.push_back(RB_mix->getSumMflux());
	Data_NBR.clear();	Data_NBR.push_back(RB_mix->getSumNflux());
        Data_ConvectiveHeatFlux.clear();     Data_ConvectiveHeatFlux.push_back(RB_ConvectiveHeatFlux);


// Actual TimeStepping Calculations	
	double counter = Data_SaveTime;
        while ( RB_Time < RB_EndTime ){

//		cout << RB_Time << " / " << RB_EndTime << "\n";

                Old_Height = RB_LiquidHeight;	
                RungaKutta4();  
                New_Height = RB_LiquidHeight;

                if (!RB_Verification_Height)
                {
                  LinearLevel();
                } 


                SurfaceTemperature();
	        for (int i = 0; i < c; i++) { x.at(i) = getX(i); }
                if (getST() < RB_BPT) { RB_mix->film_Flux_Main(getST(), getTinfinity(), getSystemPressure(),x,RB_BulkComposition,PD,U,RT);}
                else {RB_mix->film_Fluxes(getST(),RT,x,P,getSumEnergy(),RB_Tinfinity,PD,U);}

		RB_FV = RB_mix->getSumMflux()/RB_VaporDensity;
                RB_MF = RB_mix->getSumMflux();
                RB_ConvectiveHeatFlux = RB_mix->getHeatTransferCoeff()*(RB_Tinfinity - getST());
                

                if (abs(RB_Time - counter) < RB_dt/100)
                {
                        counter = counter + Data_SaveTime;
	                Data_Time.push_back(getTime());
	                Data_Height.push_back(getHeight());
	                Data_STemp.push_back(getST());
			Data_BPT.push_back(getBP());
	                for (int i = 0; i < c; i++){ Data_SComp[i]->push_back(getC(i)); }
			for (int i = 0; i < c; i++){ Data_XComp[i]->push_back(getX(i)); }
                        Data_MBR.push_back(RB_mix->getSumMflux());
                        Data_NBR.push_back(RB_mix->getSumNflux());
                        Data_ConvectiveHeatFlux.push_back(RB_ConvectiveHeatFlux);
                        Data_Temperature_Profile.push_back(new vector<double>);
                        int dv = Data_Temperature_Profile.size() - 1;
                        for (int ii = 0; ii < RB_NON; ii++){Data_Temperature_Profile[dv]->push_back(getT(ii));}
                        if (RB_Verification_Height | RB_Verification_Energy | RB_Verification_Height_Energy) {compareMMS();}
                        if (RB_Verification_Height | RB_Verification_Height_Energy)
                        {  
                          Data_Height_Verify.push_back(getcompareMMS_Height());
                          Data_DHdt_Verify.push_back(getcompareMMS_DHdt());
                        }
                        if (RB_Verification_Energy | RB_Verification_Height_Energy)
                        {
                          Data_Energy_Verify.push_back(new vector<double>);
                          Data_DTdz2_Verify.push_back(new vector<double>);
                          Data_DTdt_Verify.push_back(new vector<double>);
                          int ve = Data_Energy_Verify.size() - 1;
                          for (int ii = 0; ii < RB_NON; ii++)
                          { 
                            Data_Energy_Verify[ve]->push_back(getcompareMMS_Energy(ii));
                            Data_DTdz2_Verify[ve]->push_back(getcompareMMS_DTdz2(ii));
                            Data_DTdt_Verify[ve]->push_back(getcompareMMS_DTdt(ii));
                          }
                        }
                }
        }

        delete RB_mix;

      
} // end ResponsiveBoundary::TimeStep()


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/* @brief Description of ResponsiveBoundary::CalculateTimeStep: 
  This method calculate the proper time step size for the liquid pool simulation. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::CalculateTimeStep()
{

	double tend = RB_EndTime;
        int iteration = 1;

	double maxDT = 0.005;
	

        if (tend <= maxDT) RB_dt = tend;
        if (tend > maxDT) 
        {
         RB_dt = tend;
         while (RB_dt > maxDT) 
         {
           RB_dt = RB_dt/2;
           iteration = iteration*2;
         }
        }
cout << "Time Step: " << RB_dt << "\n";	



} // end ResponsiveBoundary::CalculateTimeStep()
 
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::RungaKutta4
  The following method applies the 4th order explicit Runga-Kutta Method to advance time and calculate new
  values for liquid height, liquid temperature, and liquid composiiton.  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::RungaKutta4()
{

        int n = RB_NON;
	int c = RB_NOC;
//	int p = (c + 1)*n + 1;
        int p = n + c + 1;
        double dt = RB_dt;
        double u[p];
        double dz = RB_LiquidHeight/n;
    

	k1.clear(); k2.clear(); k3.clear(); k4.clear();	

        u[0] = RB_LiquidHeight;

        for (int ii = 0; ii < n; ii++) { u[ii+1] = getT(ii); }
        for (int ii = 0; ii < c; ii++) { u[n+ii+1] = getC(ii); }


        EnergyBalance(1);
	MoleBalance();




	for (unsigned int ii = 0; ii < RB_DA.size(); ii++) { k1.push_back(RB_DA[ii]); }

        RB_LiquidHeight = u[0] + k1[0]*dt/2.0;
        for (int ii = 0; ii < n; ii++) { RB_Tprofile.at(ii) = u[ii+1] + k1[ii+1]*dt/2.0; }
        for (int ii = 0; ii < c; ii++) { RB_Cprofile.at(ii) = u[n+ii+1] + k1[n+ii+1]*dt/2.0; }
        for (int ii = 0; ii < c; ii++) { RB_Xprofile.at(ii) = RB_Cprofile.at(ii)/getCT(); }

//Ensure that the boiling point is not exceeded:
        for (int ii = 0; ii < n; ii++)
        {
          if (RB_Tprofile[ii] > RB_BPT) {RB_Tprofile[ii] = RB_BPT;}
        }


        EnergyBalance(2);
	MoleBalance();


	for (unsigned int i = 0; i < RB_DA.size(); i++) { k2.push_back(RB_DA[i]); }


        RB_LiquidHeight = u[0] + k2[0]*dt/2.0;
        for (int ii = 0; ii < n; ii++) { RB_Tprofile.at(ii) = u[ii+1] + k2[ii+1]*dt/2.0; }
        for (int ii = 0; ii < c; ii++) { RB_Cprofile.at(ii) = u[n+ii+1] + k2[n+ii+1]*dt/2.0; }
        for (int ii = 0; ii < c; ii++) { RB_Xprofile.at(ii) = RB_Cprofile.at(ii)/getCT(); }

//Ensure that the boiling point is not exceeded:
        for (int ii = 0; ii < n; ii++)
        {
          if (RB_Tprofile[ii] > RB_BPT) {RB_Tprofile[ii] = RB_BPT;}
        }


        EnergyBalance(3);
	MoleBalance();



	for (unsigned int i = 0; i < RB_DA.size(); i++){ k3.push_back(RB_DA[i]); }	

        RB_LiquidHeight = u[0] + k3[0]*dt;
        for (int ii = 0; ii < n; ii++) { RB_Tprofile.at(ii) = u[ii+1] + k3[ii+1]*dt; }
        for (int ii = 0; ii < c; ii++) { RB_Cprofile.at(ii) = u[n+ii+1] + k3[n+ii+1]*dt; }
        for (int ii = 0; ii < c; ii++) { RB_Xprofile.at(ii) = RB_Cprofile.at(ii)/getCT(); }

//Ensure that the boiling point is not exceeded:
        for (int ii = 0; ii < n; ii++)
        {
          if (RB_Tprofile[ii] > RB_BPT) {RB_Tprofile[ii] = RB_BPT;}
        }


        EnergyBalance(4);
	MoleBalance();



//        for (unsigned int ii = 0; ii < RK_DCdt.size(); ii++){delete RK_DCdt[ii];} 
//	RK_DTdt.clear();  RK_DCdt.clear(); for (int i = 0; i < c; i++) {RK_DCdt.push_back(new vector<double>);}
        RK_DTdt.clear(); RK_DCdt.clear(); 
        

	for (unsigned int i = 0; i < RB_DA.size(); i++){ k4.push_back(RB_DA[i]); }


        RB_LiquidHeight = u[0] + (dt/6.)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        for (int ii = 0; ii < n; ii++)
        {
          RB_Tprofile[ii] = u[ii+1] + (dt/6.)*(k1[ii+1] + 2*k2[ii+1] + 2*k3[ii+1] + k4[ii+1]);
          RK_DTdt.push_back((k1[ii+1] + 2*k2[ii+1] + 2*k3[ii+1] + k4[ii+1])/6);
        }


        for (int ii = 0; ii < c; ii++)
        {
          int k = n + ii + 1;
          RB_Cprofile[ii] = u[k] + (dt/6.)*(k1[k] + 2*k2[k] + 2*k3[k] + k4[k]);
          RK_DCdt.push_back((k1[k] + 2*k2[k] + 2*k3[k] + k4[k])/6);
        }

	RK_DHdt = ((k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.);


//Ensure that the boiling point is not exceeded:
        SurfaceTemperature();
        for (int ii = 0; ii < n; ii++)
        {
          if (RB_Tprofile[ii] > RB_BPT) {RB_Tprofile[ii] = RB_BPT;}
        }


//This segment of code compares the values of the estimate of the time derivative for energy from the
// Runga Kutta routine with the value for the Manufactured Solution.  This is done in order to determine
// if the order of error caused by the runga kutta routine is what it should be.
        if (RB_Verification_Energy) 
        {
          RB_compareMMS_DTdt.clear();
          double t = RB_Time;
          for (int ii = 0; ii < n; ii++)
          {
            double h = RB_LiquidHeight - (ii + 0.5)*dz;
            double DT_Analytic = abs(verifyMMS(t+dt,h,0,0) - verifyMMS(t,h,0,0));
            RB_compareMMS_DTdt.push_back(abs((dt*RK_DTdt[ii] - DT_Analytic)/DT_Analytic));
          }
        }

        if (RB_Verification_Height)
        {
          double t = RB_Time;
          double DH_Analytic = (verifyMMS(t+dt,0,1,0) - verifyMMS(t,0,1,0));
          RB_compareMMS_DHdt = abs((dt*RK_DHdt - DH_Analytic)/DH_Analytic);
//          RB_compareMMS_DHdt = RK_DHdt*dt;

        }

 
        if (RB_Verification_Height_Energy)
        {
          RB_compareMMS_DTdt.clear();
          double t = RB_Time;
          for (int ii=0; ii < n; ii++)
          {
            double h = RB_LiquidHeight - (ii+0.5)*dz;
            double DT_Analytic = (verifyMMS(t+dt,h,2,0) - verifyMMS(t,h,2,0));
            RB_compareMMS_DTdt.push_back(abs((dt*RK_DTdt[ii] - DT_Analytic)));
          } 
          double DH_Analytic = (verifyMMS(t+dt,0,1,0) - verifyMMS(t,0,1,0));
          RB_compareMMS_DHdt = abs((dt*RK_DHdt - DH_Analytic));
        }

// END of Verification Code Segment ***************************************************************************




        RB_Time = RB_Time + dt; 

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::RungaKutta2
  The following method applies the 2nd order explicit Runga-Kutta Method to advance time and calculate new
  values for liquid height, liquid temperature, and liquid composiiton.  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::RungaKutta2()
{

        int n = RB_NON;
        int c = RB_NOC;
//        int p = (c + 1)*n + 1;
        int p = n + c + 1;
        double dt = RB_dt;
        double u[p];

        k1.clear(); k2.clear(); 

        u[0] = RB_LiquidHeight;
//        for (int i = 0; i < n; i++){
//           u[i+1] = getT(i);
//           for (int j = 0; j < c; j++){
//              int k = n + j*n + i + 1;
//              u[k] = getC(j,i);}}
        for (int ii = 0; ii < n; ii++) { u[ii+1] = getT(ii); }
        for (int ii = 0; ii < c; ii++) { u[n+ii+1] = getC(ii); }




        EnergyBalance(1);
        MoleBalance();


        for (unsigned int ii = 0; ii < RB_DA.size(); ii++) { k1.push_back(RB_DA[ii]); }

//        for (int i = 0; i < n; i++) {
//             if (i == 0) RB_LiquidHeight = u[0] + k1[0]*dt;
//             RB_Tprofile.at(i) = u[i+1] + k1[i+1]*dt;
//           for (int j = 0; j < c; j++){
//              int k = (n + j*n + i + 1);
//              RB_Cprofile[j]->at(i) = u[k] + k1[k]*dt;}}
        RB_LiquidHeight = u[0] + k1[0]*dt;
        for (int ii = 0; ii < n; ii++) { RB_Tprofile.at(ii) = u[ii+1] + k1[ii+1]*dt; }
        for (int ii = 0; ii < c; ii++) { RB_Cprofile.at(ii) = u[n+ii+1] + k1[n+ii+1]*dt; }
        for (int ii = 0; ii < c; ii++) { RB_Xprofile.at(ii) = RB_Cprofile.at(ii)/getCT(); }



        EnergyBalance(4);
        MoleBalance();

        for (unsigned int ii = 0; ii < RB_DA.size(); ii++) { k2.push_back(RB_DA[ii]); }



//        for (unsigned int ii = 0; ii < RK_DCdt.size(); ii++){delete RK_DCdt[ii];}
//        RK_DTdt.clear();  RK_DCdt.clear(); for (int i = 0; i < c; i++) {RK_DCdt.push_back(new vector<double>);}
          RK_DTdt.clear(); RK_DCdt.clear();


//        for (int i = 0; i < n; i++) {
//             if (i == 0) RB_LiquidHeight = u[0] + (dt/2.)*(k1[0] + k2[0]);
//             RB_Tprofile[i] = u[i+1] + (dt/2.)*(k1[i+1] + k2[i+1]);
//             RK_DTdt.push_back((k1[i+1] + k2[i+1])/2);
//             if ((i == 0) && (RB_Tprofile[0] > RB_BPT)) {RB_Tprofile[0] = RB_BPT;}
//           for (int j = 0; j < c; j++){
//              int k = (n + j*n + i + 1);
//              RB_Cprofile[j]->at(i) = u[k] + (dt/2)*(k1[k] + k2[k]);
//              RK_DCdt[j]->push_back((k1[k] + k2[k])/2);}}

        RB_LiquidHeight = u[0] + (dt/2.)*(k1[0] + k2[0]);
        RK_DHdt = (k1[0] + k2[0] )/2;
        for (int ii = 0; ii < n; ii++)
        {
          RB_Tprofile[ii] = u[ii+1] + (dt/2.)*(k1[ii+1] + k2[ii+1]);
          RK_DTdt.push_back((k1[ii+1] + k2[ii+1])/2);
        }
        for (int ii = 0; ii < c; ii++)
        {
          int k = n + ii + 1;
          RB_Cprofile[ii] = u[k] + (dt/2.)*(k1[k] + k2[k]);
          RK_DCdt.push_back((k1[k] + k2[k])/2);
        }



        RB_Time = RB_Time + dt;


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::LinearLevel:
 This method uses linear interpolation to give temperature and Compositon values at newly calculated spatial nodes
 which have changed due to liquid level drop.  The same number of nodes are used regardless of how
 low the liquid level get.  This means that the spatial step size will shrink with loss of liquid
 level. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::LinearLevel()
{

	int n = RB_NON;     
//	int c = (RB_NOC - 1);
	int c = RB_NOC;
        double zO[n], z[n], TT[n], CTOT[n], TC[n];
        double T[n];
	double PenDep = 0.03;
        double dz = New_Height/n;
	double dzo = Old_Height/n;




        for (int ii = 0; ii < n; ii++) { T[ii] = getT(ii); } 


//**** Verification of Interpolation Code starts here: *******
//double dz = 0.005;
//double dz2 = 0.0001;
//int n = 1/dz2 + 1;
//int nn = 1/dz + 1;
//double zO[nn], z[n], T[nn], TT[n], TC[n];
//for (int ii = 0; ii < nn; ii++) { zO[ii] = 1 - (dz*ii); T[ii] = verifyMMS(0,zO[ii],2,0); }
//for (int ii = 0; ii < n; ii++) {z[ii] = 1 - (dz2*ii);}
//******************************************************End of Block**********************


        zO[n-1] = dzo/2.0;
        z[n-1] = dz/2.0;

        for (int ii = (n-2); ii >=0; ii--)
        {
          zO[ii] = zO[ii+1] + (dzo);
          z[ii] = z[ii+1] + (dz); 
       }


/*
//Implementation of Linear Interpolation:
        for (int ii = 0; ii < n; ii++) {
               int s,j;
                double dtdz, DZ;
                s = 0;
                j = 1;
                while ( s == 0 ) {
                if ( (z[ii] >= zO[j]) && (z[ii] <= zO[j-1]) ){
                TC[ii] = T[j-1] + ((T[j] - T[j-1])/(zO[j] - zO[j-1]))*(z[ii] - zO[j-1]);
                s = 1; }

                else if ( z[ii] < zO[n-1]) {
                dtdz = (T[n-1] - T[n-2])/(zO[n-1] - zO[n-2]);
                DZ = (z[n-1] - zO[n-1]);
                TC[ii] = T[n-1] + dtdz*DZ;
                s = 1; }

                else j++; 
                }}

*/


/*
//Implementation of Cubic Interpolation:
        int m = n-2; 
//        int m = nn -2;  //VERIFICATION
        double D[m], U[m-1], Tpp[n], YV[m]; 
//        double D[m], U[m-1], Tpp[nn], YV[m];  //VERIFICATION




        for (int ii = 0; ii < (m); ii++)
        {
          D[ii] = 2*(zO[ii+2] - zO[ii]);
          YV[ii] = 6*( (T[ii+2]-T[ii+1])/(zO[ii+2]-zO[ii+1]) - (T[ii+1]-T[ii])/(zO[ii+1]-zO[ii]) );
          if (ii != (m-1))
          {
            U[ii] = (zO[ii+2] - zO[ii+1]);
          }
        }


//Implement Thomas Algorithm for Tridiagonal System:
        double Dstar[m], Cstar[m-1], XX[m];
        for (int ii = 0; ii < (m-1); ii++)
        {
          if (ii == 0) {Cstar[ii] = U[ii]/D[ii];}
          else {Cstar[ii] = U[ii]/(D[ii] - Cstar[ii-1]*U[ii-1]);}
        } 
       
        for (int ii = 0; ii < m; ii++)
        {
          if (ii == 0) {Dstar[ii] = YV[ii]/D[ii];}
          else { Dstar[ii] = (YV[ii] - Dstar[ii-1]*U[ii-1])/(D[ii] - Cstar[ii-1]*U[ii-1]); }
        }

        XX[m-1] = Dstar[m-1];
        for (int ii = (m-2); ii >= 0; ii--)
        {
          XX[ii] = Dstar[ii] - Cstar[ii]*XX[ii+1];
        }




        for (int ii = 0; ii < m; ii++) {Tpp[ii+1] = XX[ii];}


// These values of the endpoint 2nd derivatives are for the "Not-a-Knot" implementation of cubic spline interpolation.
//        Tpp[0] = (Tpp[1] - Tpp[2])*(zO[0] - zO[1])/(zO[1] - zO[2]) + Tpp[1]; 
//        Tpp[n-1] = Tpp[n-2] - (Tpp[n-3] - Tpp[n-2])*(zO[n-2] - zO[n-1])/(zO[n-3] - zO[n-2]); //VERIFICATION (nn -> n for real runs)


//These values of the endpoint 2nd derivative are for the "natural" implementation of cubic spline interpolation
        Tpp[0] = 0;
        Tpp[n-1] = 0;

*/

        for (int ii = 0; ii < n; ii++) 
        {
          int s,j; 
          double dtdz, DZ;
          s = 0; 
          j = 1; 
          while ( s == 0 ) 
          {
            int order = 4; //use 4th degree langrange interpolating polynomial
            if ( (z[ii] >= zO[j]) && (z[ii] <= zO[j-1]) ) 
            {
              if (j < order)
              { 
                double sum = 0;
                for (int kk = 0; kk < order; kk++)
                {
                  double prod = T[kk];
                  for (int ll = 0; ll < order; ll++)
                  {
                     if (kk != ll) {prod = prod*(z[ii] - zO[ll])/(zO[kk] - zO[ll]);}
                  }
                  sum = sum + prod;
                }
                TC[ii] = sum;
              }
              else if (j > (n-order))  
//              else if (j > (nn - order))//VERIFICATION
              {
                double sum = 0;
                for (int kk = 0; kk < order; kk++)
                {
                  double prod = T[j - kk];
                  for (int ll = 0; ll < order; ll++)
                  {
                    if (kk != ll) {prod = prod*(z[ii] - zO[j - ll])/(zO[j - kk] - zO[j - ll]);}
                  }
                  sum = sum + prod;
                }
                TC[ii] = sum;
              }
              else
              {
//                double Term11 = pow((z[ii] - zO[j]),3)/(zO[j-1] - zO[j]);
//                double Term12 = (zO[j-1] - zO[j])*(z[ii] - zO[j]);
//                double Term1 = (Term11 - Term12)*Tpp[j-1]/6.0;
                  

//                double Term21 = pow((z[ii]-zO[j-1]),3)/(zO[j]-zO[j-1]); 
//                double Term22 = (zO[j]-zO[j-1])*(z[ii]-zO[j-1]); 
//                double Term2 = (Term21 - Term22)*Tpp[j]/6.0; 


//               double Term3 = ((z[ii]-zO[j])/(zO[j-1]-zO[j]))*T[j-1];
//                double Term4 = ((z[ii]-zO[j-1])/(zO[j]-zO[j-1]))*T[j];

//                TC[ii] = Term1 + Term2 + Term3 + Term4;
              
                double sum = 0;
                for (int kk = 0; kk < order; kk++)
                {
                  double prod = T[j+1-kk];
                  for (int ll = 0; ll < order; ll++)
                  {
                    if (kk != ll) {prod = prod*(z[ii]-zO[j+1-ll])/(zO[j+1-kk]-zO[j+1-ll]);}
                  }
                  sum = sum + prod;
                }
                TC[ii] = sum;  
              }
              s = 1; 
            }
 
            else if (z[ii] < zO[n-1])
//            else if (z[ii] < zO[nn-1]) //VERIFICATION
            {
//              dtdz = (T[n-1] - T[n-2])/(zO[n-1] - zO[n-2]);
//              DZ = (z[n-1] - zO[n-1]);
//              TC[ii] = T[ii] + dtdz*DZ;
              double sum = 0;
              for (int kk = 0; kk < order; kk++)
              {
                double prod = T[n - 1 - kk];
                for (int ll = 0; ll < order; ll++)
                {
                  if (kk != ll) {prod = prod*(z[ii] - zO[n-1-ll])/(zO[n-1-kk] - zO[n-1-ll]);} 
//                  if (kk != ll) {prod = prod*(z[ii] - zO[nn-1-ll])/(zO[nn-1-kk] - zO[nn-1-ll]);} //VERIFICATION
                }
                sum = sum + prod;
              }
              TC[ii] = sum;
              s = 1; 
            } 

            else j++; 
          }
        }  



//End of Interpolation




        for (int ii = 0; ii < n; ii++){ RB_Tprofile[ii] = TC[ii]; }


}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief
 The following method calculates the diffusive fluxes of the species in the liquid pool, this model for
 the fluxes assumes equimolar counterdiffusion in the liquid phase. */
//*********************************************************************************************************************
//*********************************************************************************************************************

/*
void
ResponsiveBoundary::Calculate_Flux()
{

	int c = RB_NOC;
	int n = RB_NON;
	double dz = RB_LiquidHeight/n;	

        for (unsigned int ii = 0; ii < RB_J.size(); ii++){delete RB_J[ii];}
        RB_J.clear();
	for (int i = 0; i < c; i++){ RB_J.push_back(new vector<double>); }

	
	for (int i = 0; i < (n-1); i++){

	   double T = (getT(i)/2 + getT(i+1)/2);  //average temperature for boundary between nodes
	   double P = getSystemPressure();
	   vector<double> x(c);
	   double Ct, dummy;


	   for (int j = 0; j < c; j++){ x.at(j) = (getX(j,i)/2 + getX(j,i+1)/2); } //average composition between nodes

	   dummy = RB_mix->mix_liquidDensity(T, P, x);
//	   Ct = RB_mix->getLMCon(); // average molar concentration between nodes
	   Ct = (getCT(i)/2 + getCT(i+1)/2); //average molar concentration between nodes.


	   RB_mix->Fickian(T, P, x);


	   for (int k = 0; k < (c-1); k++){
	      double sum = 0;
	      for (int j = 0; j < (c-1); j++){
	         sum = sum + RB_mix->getFD(k,j)*(getX(j,i+1) - getX(j,i))/dz;}
	      RB_J[k]->push_back(-Ct*sum);}

	     double sum = 0;
	   for (int k = 0; k < (c-1); k++){ 
 	      sum = sum + getJ(k,i);}
	   RB_J[c-1]->push_back(-sum);}


}

*/
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************








//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::Calculate_Enthalpy:
 This method calculates the enthalpies of each component at each node for use in the energy balance: */
//*********************************************************************************************************************
//*********************************************************************************************************************
/*
void 
ResponsiveBoundary::Calculate_Enthalpy()
{

	int c = RB_NOC;
	int n = RB_NON;
	double Ts = getRefTemp();

        for (unsigned int ii = 0; ii < RB_H.size(); ii++){delete RB_H[ii];}
	RB_H.clear();
	for (int i = 0; i < c; i++){ RB_H.push_back(new vector<double>); }


	for (int i = 0; i < (n-1); i++){

	double T = getT(i)/2 + getT(i+1)/2;
	   for (int j = 0; j < c; j++){
	      RB_H[j]->push_back(RB_mix->getEnthalpy(T,Ts,j));}}

}

*/

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::SurfaceTemperature:
  The following method uses extrapolation based on langrange polynomials to 
  calculate the surface liquid temperature based on the nearest temperature node values.
  The only input to this method is the number of node values to use:  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void ResponsiveBoundary::SurfaceTemperature()
{

        int nnn = 3;
        vector<double> Z,T,x;
        double dz = RB_LiquidHeight/RB_NON;
        double P = RB_SystemPressure;    

        for (int ii = 0; ii < RB_NOC; ii++) {x.push_back(getX(ii));}
        RB_BPT = RB_mix->BubblePointT(P,x);

        for (int ii = 0; ii < nnn; ii++)
        {
          double h = RB_LiquidHeight - (ii + 0.5)*dz;
          double TT = getT(ii);
          Z.push_back(h);
          T.push_back(TT);
        } 

        double S = RB_LiquidHeight;
        double sum = 0;


        //Method1: Lagrange Polynomial
        for (int ii = 0; ii < nnn; ii++)
        {
          double prod = T[ii];
          for (int jj = 0; jj < nnn; jj++)
          {
            if (jj != ii)
            {
              prod = prod*(S - Z[jj])/(Z[ii] - Z[jj]);
            }
          }
          sum = sum + prod;
        }
/*
        //Method2: exponential function:
        double T0 = getT(0);
        double T1 = getT(1);
        double z0 = Z.at(0);
        double z1 = Z.at(1);
        double K = log(T0/T1)/(z0 - z1);
        double A = T0/exp(K*z0);
        sum = A*exp(K*RB_LiquidHeight);
*/
 
        if (sum <= RB_BPT) {RB_SurfaceTemperature = sum;}
        if (sum > RB_BPT) {RB_SurfaceTemperature = RB_BPT;}

        double temptest = (RB_BPT - getT(0));
        if (temptest < 1e-3) {RB_SurfaceTemperature = RB_BPT;}
}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief  Description of ResponsiveBoundary::EnergyBalance:
  The following method calculates the Energy Balance equation for the liquid pool model:  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void ResponsiveBoundary::EnergyBalance(int rkcycle)
{

	int n = RB_NON;
	int c = RB_NOC;
	double Alpha[n], FF[n];
	double Conduction[n], qrad[n], Mflux[n], Eflux[n], Convection[n], ReRad[n];
        double qmms[n], qmmsh; // source term for verification through MMS
	double P = getSystemPressure();
	double PD = RB_PoolDiameter;
	double U = RB_WindSpeed;
	double RT = RB_RefTemp;
	double PenDep = 0.03;
	double dz = RB_LiquidHeight/n;
        double t; // time variable to be used with Method of Manufactured Solutions
        double HTC;  //convective heat transfer coefficient;

        if (RB_Verification_Energy | RB_Verification_Height | RB_Verification_Height_Energy)  //Only used for verification simulations 
                                                                                               //for the energy balance equation
        {
          if (rkcycle == 1) { t = RB_Time; }
          if (rkcycle == 2) { t = RB_Time + RB_dt/2; }
          if (rkcycle == 3) { t = RB_Time + RB_dt/2; }
          if (rkcycle == 4) { t = RB_Time + RB_dt; }
        }



	/*RB_Density.clear();*/   RB_THK.clear();   RB_HCP.clear();   /*RB_MolarDensity.clear();*/
	//collect density, thermal conductivity, and heat capacity of each layer:
	for (int i = 0; i < n; i++)
        {
           vector<double> x(c); vector<double> m(c);
	   for (int j = 0; j < c; j++){ x.at(j) = RB_Xprofile[j]; }
	   RB_mix->MassFraction(x);
	   for (int j = 0; j < c; j++){ m.at(j) = RB_mix->getLMC(j); }
	   double T = getT(i);
//	   RB_Density.push_back(RB_mix->mix_liquidDensity(T,P,x));
	   RB_THK.push_back(RB_mix->mix_liquidThermalConductivity(T,m));
	   RB_HCP.push_back(RB_mix->mix_liquidHeatCapacity(T,m));
//	   RB_MolarDensity.push_back(getCT(i));
	   Alpha[i] = getTHK(i)/getDensity()/getHCP(i);
         }


        //Extrapolate the Surface Temperature:
        SurfaceTemperature();


	//Calculate the conduction terms

	for (int i = 0; i < n; i++)
        {
	   if (i == 0){ Conduction[i] = Alpha[i]*(0 - (getT(i) - getT(i+1)))/dz/dz;}
//           if (i == 0){ Conduction[i] = Alpha[i]*(2*getT(i) - 5*getT(i+1) + 4*getT(i+2) - getT(i+3))/dz/dz;} 
	   else if (i == (n - 1)){ Conduction[i] = Alpha[i]*((getT(i-1) - getT(i)) - 0)/dz/dz;} 
//           else if (i == (n - 1)){ Conduction[i] = Alpha[i]*(2*getT(i) - 5*getT(i-1) + 4*getT(i-2) - getT(i-3))/dz/dz;}
	   else { Conduction[i] = Alpha[i]*((getT(i-1) - getT(i))/(dz) - (getT(i) - getT(i+1))/(dz))/dz;}
        } 



	//Now to Calculate Radiation Terms:
	
	double rd = PenDep; //radiation penetration depth (~ 3 centimeters) (in meters)
        if (rd < RB_LiquidHeight) rd = rd;
        else { rd = RB_LiquidHeight;} //Probably want to change this, but since most simulations do not go long
                       // enough to drain the pool, this likely won't come into play for some time.




	// Radiation distribution in penetration zone:

        double rn = rd/dz;
        double nr = ceil(rn);
        RadiationDistribution(rn);



        // Reflection of radiation at pool surface;

        double rr = RB_Reflectivity; //reflectivity of pool surface (would like to compute this in the future)
        double Qr = RB_RadiationInput;
        double sigma = 5.67e-8;  //Stefan-Boltzman Constant (W/m/m/K/K/K/K)
        double RERAD = sigma*(1-rr)*pow(getST(),4);
        Qr = Qr*(1-rr) - RERAD;



	//Calculate Evaporation Energy Loss (only for mass-transfer control regime):  The heat loss from the 
	// surface is calculated and removed from the original radiation absorbed.  This method eliminates 
	// surface cooling and the need to calculate re-circulation in the liquid due to density differences.



	//Bubblepoint Temperature:
	  vector<double> x;
	  x.clear();
	  for (int i = 0; i < c; i++){ x.push_back(getX(i));}
	  double BPT = RB_mix->BubblePointT(P,x);



	   RB_BPT = BPT;


        if (getST() < BPT) //Less than the Boiling Point
        {
          RB_mix->film_Flux_Main(getST(),getTinfinity(),P,x,RB_BulkComposition,PD,U,RT);
          double evapload = RB_mix->getSumEflux();
          Qr = Qr - evapload;
        }

        //obtain the convective heat transfer coefficient
        HTC = RB_mix->getHeatTransferCoeff();

        // Calculate the convective heat flux:
       
        double CHF = HTC*(RB_Tinfinity - getST());
        
        for (int ii = 0; ii < n; ii++)
        {
          if (ii == 0)
          {
            Convection[ii] = CHF/dz/RB_Density/RB_HCP[ii];
          }
          else
          {
            Convection[ii] = 0;
          }
        }

        Qr = Qr + CHF; 


	// After all that prep. we can compute the radiation terms:
	for (int i = 0; i < n; i++)
        {
//          qrad[i] = 0;
//	  if (i < nr) qrad[i] = RB_RFA[i]*Qr/dz/RB_Density/RB_HCP[i];
//          else qrad[i] = 0.0;
          qrad[i] = RB_RFA[i]*Qr/dz/RB_Density/RB_HCP[i];
        }




//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// This set of code is for Verification Runs Only
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //Source Term for Verification Simulations through Method of Manufactured Solutions:
        if (RB_Verification_Energy)
        {
          for (int ii = 0; ii < n; ii++)
          {
            double h = RB_LiquidHeight - (ii + 0.5)*dz;
            double Tt = verifyMMS(t, h, 0, 1);
            double Th = Alpha[ii]*verifyMMS(t, h, 0, 2);
            qmms[ii] = Tt - Th - qrad[ii];
          }
        }

        if (RB_Verification_Height)
        {
          double Ht = verifyMMS(t,0,1,1); //Derivative of Height with respect to time.
          qmmsh = Ht + RB_mix->getSumNflux()/RB_MolarDensity;
        }
        
        if (RB_Verification_Height_Energy)
        {
          double Ht = verifyMMS(t,0,1,1);
          qmmsh = Ht + RB_mix->getSumNflux()/RB_MolarDensity;
          for (int ii = 0; ii < n; ii++)
          {
            double h = RB_LiquidHeight - (ii + 0.5)*dz;
            double Tt = verifyMMS(t,h,2,1);
            double Th = Alpha[ii]*verifyMMS(t,h,2,2);
            qmms[ii] = Tt - Th - qrad[ii];
          }  
        }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




	// Energy Balance:

        double Energy[n];
        for (int ii = 0; ii < n; ii++) {Energy[ii] = Conduction[ii] + qrad[ii];}// + Convection[ii];}



        RB_EnergySave.clear();
        for (int ii = 0; ii < n; ii++)
        {
          double temptest = (BPT - getT(ii));
          if (temptest < 1e-3)
          {
            RB_EnergySave.push_back((Energy[ii])*dz*RB_Density*RB_HCP[ii]);
          }
        }
for (unsigned int ii = 0; ii < RB_EnergySave.size(); ii++) {cout << "EnergySave[" << ii << "]: " << RB_EnergySave.at(ii) << endl;}


	if (getT(0) < BPT)
        {
	   for (int ii = 0; ii < n; ii++)
           {
             if ((!RB_Verification_Energy) && (!RB_Verification_Height))
             {
	     // FF[ii] = Conduction[ii] + qrad[ii] + Mflux[ii];}}
               FF[ii] = Energy[ii];
             }
             if (RB_Verification_Energy | RB_Verification_Height_Energy)
             {
               FF[ii] = Energy[ii] + qmms[ii];
             }
             if (RB_Verification_Height)
             {
               FF[ii] = 0.0;
             }
           }
         } 



	// As presently written, this energy balance assumes that the temperature of the pool will
	// not reach the Bubble Point.  It is possible for boil over to occur, but this code will
	// need to be altered to deal with such an event.
//	if (getST() >= BPT) cout << "Bubble Point Reached, Run For Your Lives!!!!!" << "\n";


        //code is now altered to deal with boiling, but it probably needs more testing.
        bool boil = 0;
        for (int ii = 0; ii < n; ii++)
        {
          double temptest = (BPT - getT(ii));
          if (temptest < 1e-6) {boil = 1;}
        }

        if (boil)    
        {
          if ((!RB_Verification_Energy) && (!RB_Verification_Height))
          {
            for (int ii = 0; ii < n; ii++)
            {
              double temptest = abs(getT(ii) - BPT);
              double energytest = Conduction[ii] + qrad[ii];// + Convection[ii];
              if ((temptest < 1e-6) && (energytest > 0)) {FF[ii] = 0;}
              else {FF[ii] = energytest;}
            }
            RB_mix->film_Fluxes(getST(),RT,x,P,getSumEnergy(),getTinfinity(),PD,U);
          }
        }


        
	// Drop in Height:
        double dh;
        if (!RB_Verification_Energy)
        {
          if (!RB_Verification_Height)
          {
            dh = -1.0*RB_mix->getSumNflux()/getMolarDensity();
          }
          if (RB_Verification_Height | RB_Verification_Height_Energy)
          {
            dh = -1*RB_mix->getSumNflux()/RB_MolarDensity + qmmsh;
          }
        }  

        if (RB_Verification_Energy)
        {
          dh = 0;
        }

       //Prepare derivative arrays for runga kutta method:
	RB_DA.clear();	
	RB_DA.push_back(dh);


	for (int i = 0; i < n; i++){
	  RB_DA.push_back(FF[i]);}

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::MoleBalance
 This method computes the mole balances in the liquid pool model. Note that the Energy balance must be run first
 in order for this method to work properly.  It is done this way so that certain processes common to both Energy 
 Balance and mole balance do not need to be calculated again, saving computational time. (Like the liquid
 properties and the fluxes at all nodes. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::MoleBalance()
{

	int n = RB_NON;
	int c = RB_NOC;
	double dz = RB_LiquidHeight/n;

//	double FN[c][n], Flux[c][n], Evap[c][n];
        double FN[c], Evap[c];
	

//	//Calculate the Liquid flux Terms
//	for (int i = 0; i < n; i++){
//	  for (int j = 0; j < c; j++){
//	    if ( i == 0 ){ Flux[j][i] = (0 - getJ(j,i))/dz; }
//	    else if (i == (n-1)) { Flux[j][i] = (getJ(j,i-1) - 0)/dz; }
//	    else { Flux[j][i] = (getJ(j,i-1) - getJ(j,i))/dz; }}}


	//Calculate the Evaporation flux Terms:

//	for (int i = 0; i < n; i++){ for (int j = 0; j < c; j++){
//	   if ( i == 0 ) { Evap[j][i] = RB_mix->getNflux(j)/dz; }
//	   else { Evap[j][i] = 0; }}}
        for (int ii = 0; ii < c; ii++) { Evap[ii] = RB_mix->getNflux(ii)/RB_LiquidHeight; }



	//Calculate the mole balance:

//	for (int i = 0; i < n; i++){ for (int j = 0; j < c; j++){
//	  FN[j][i] = Flux[j][i] - Evap[j][i];}}
////	  FN[j][i] = -Evap[j][i];}}
        for (int ii = 0; ii < c; ii++) { FN[ii] = -1*Evap[ii]; }


	//Assign the values to the Derivative Array:

//	for (int j = 0; j < c; j++){ for (int i = 0; i < n; i++){ 
//	   RB_DA.push_back(FN[j][i]);}}

        for (int ii = 0; ii < c; ii++) { RB_DA.push_back(FN[ii]); }

}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************







//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::RadiationDistribution:
 This method calcuates the distribution of radiant heat in the top layers of the the pool using
 exponentially decaying function, as opposed to assuming that the radiant energy divides
 equally among the top layers.
 This distribution function is an approxiation and could benefit from replacing with a
 radiation absorption/transmission model.
 Input: "number" is the number of nodes which take radiative input (a double precision number as
 opposed to an integer.) */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::RadiationDistribution(double number)
{

        int nn = RB_NON;
        int d;
	RB_RFA.clear();

        if(abs(number - floor(number)) < 1e-14) d = 1;
        else if (abs(number - floor(number)) > 1e-14) d = 0;


        double step = 6/number;
        int m = (int) floor(number);
        vector <double> s(m+1);
/*
        s[0] = 0;
        for (int i = 1; i < (m+1); i++) { s[i] = s[i-1] + step; }

        int p = m+1;

        if ( d == 0 ){
                s.push_back(6.0);
                p = m+2;}

        for (int i = 0; i < nn; i++) { RB_RFA.push_back(0); }

        double den = (1 - exp(-6));

        for (int i = 0; i < (p-1); i++) { RB_RFA[i] = (exp(-s[i]) - exp(-s[i+1]))/den; }
*/

        double step2 = 6/number;
        vector<double> s2(nn+1,0);
        s2[0] = 0;
        for (int ii = 0; ii < nn; ii++) { s2[ii+1] = s2[ii] + step2; }
        
        for (int ii = 0; ii < nn; ii++) { RB_RFA.push_back(exp(-s2[ii]) - exp(-s2[ii+1])); }


/*
for (unsigned int ii = 0; ii < RB_RFA.size(); ii++) {cout << "OLD[" << ii << "]: " << RB_RFA.at(ii) << endl;}
cout << "\n\n";
for (unsigned int ii = 0; ii < RFA2.size(); ii++) {cout << "NEW[" << ii << "]: " << RFA2.at(ii) << endl;}

double cow;
cout << "STOP!!!" << endl;
cin >> cow;
*/
}


//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::Reset
 This method resets the Responsive Boundary code with new values of Temperature, Composition, and liquid height
 in order to continue time stepping. */
//*********************************************************************************************************************
//*********************************************************************************************************************
void 
ResponsiveBoundary::Reset(vector<bool> fv, vector<double> T, vector<double> X, vector<double> Yb, double H, double dt, double U, double P, double rad,
                             double PD, double VD, double Tinf, int NON)
{

	RB_NON = NON;


	RB_fuelvec.clear();
	for (unsigned int i = 0; i < fv.size(); i++){ RB_fuelvec.push_back(fv.at(i));}


        RB_mix = 0;
        RB_mix = new RBMixerProperties(fv);

        RB_NOC = RB_mix->getNOC();


	RB_Time = 0.0;
	AssignT(T);
	AssignX(X);
	AssignHeight(H);
	RB_EndTime = dt;
        RB_RadiationInput = rad;
        RB_PoolDiameter = PD; 
        if (abs(VD) < 1e-2) { RB_VaporDensity = 6.900; } 
        else RB_VaporDensity = VD; 
        RB_WindSpeed = U;
        RB_SystemPressure = P;
	RB_RefTemp = 298.15;
	Data_SaveTime = 1.0;
        RB_Tinfinity = Tinf;
	RB_Verification_Energy = 0; //use the "void verifyOn()" method to do verification simulatons.
        RB_Verification_Height = 0;
        RB_Verification_Height_Energy = 0;

        SurfaceTemperature();
	RB_Density = RB_mix->mix_liquidDensity(RB_RefTemp,P,X);
        RB_MolarDensity = RB_mix->getLMCon();

//        double TT[41]; 
//        for (int ii = 0; ii < 41; ii++)
//        { 
//          TT[ii] = 485.00 + (ii - 1)*0.1;
//          RB_mix->film_Fluxes(TT[ii],P,X,7.93,0.85,298.15);
//          cout << TT[ii] << "     " << RB_mix->getSumMflux() << endl;
//        }
          






//        for (unsigned int ii = 0; ii < RB_Cprofile.size(); ii++){delete RB_Cprofile[ii];}
//	RB_Cprofile.clear();
//        for (int j = 0; j < RB_NOC; j++){  RB_Cprofile.push_back(new vector<double>);
//           for (int i = 0; i < RB_NON; i++){
//                RB_Cprofile[j]->push_back(RB_MolarDensity*getX(j,i));}}

        RB_Cprofile.clear();
        for (int ii = 0; ii < RB_NOC; ii++) { RB_Cprofile.push_back(RB_MolarDensity*getX(ii)); }

        vector<double> MoleCompBulk = RB_mix->MoleFraction(Yb);
        AssignYb(MoleCompBulk);

        CalculateTimeStep();

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************






//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary::Reset (version 2)
 This version of the Reset method sets the values for the fuel vector so that the number of components and the number of nodes
 are available for use with the readResponsiveBoundaryData method.  */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::Reset(vector<bool> fv)
{

        RB_NON = 200;



        RB_fuelvec.clear();
        for (unsigned int i = 0; i < fv.size(); i++){ RB_fuelvec.push_back(fv.at(i));}
     
//        delete RB_mix;  
        RB_mix = 0;
        RB_mix = new RBMixerProperties(fv);

        RB_NOC = RB_mix->getNOC();
}
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Problem Setup for ResponsiveBoundary */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::problemSetup(ProblemSpecP& params,
                                 const ProblemSpecP& restart_ps) 
{
  params->getWithDefault("LiquidHeight", RB_LiquidHeight, 1.0);  
  params->getWithDefault("PoolDiameter", RB_PoolDiameter, 1.0);  
  params->getWithDefault("SurfaceReflectivity", RB_Reflectivity, 0.07);
  params->getWithDefault("ReferenceTemperature", RB_RefTemp, 298.15);
  params->getWithDefault("Accelerate", RB_Accelerate, false);

//Obtain the liquid pool Composition 
  ProblemSpecP rb_db = params->findBlock("FuelComposition");                                     
  double xxx;  bool yyy;                                                                         
  RB_tempfuelvec.clear();  
  RB_fuelX.clear();                                                                          
  for (int i = 0; i < 18; i++) {RB_tempfuelvec.push_back(0);}                                         
  rb_db->getWithDefault("N_OCTANE", yyy, false);   RB_tempfuelvec.at(0) = yyy;                        
    if (RB_tempfuelvec[0]) { rb_db->getWithDefault("N_OCTANEX", xxx, 0.0); RB_fuelX.push_back(xxx);}   
  rb_db->getWithDefault("N_DODECANE", yyy, false);   RB_tempfuelvec.at(1) = yyy;                      
    if (RB_tempfuelvec[1]) { rb_db->getWithDefault("N_DODECANEX", xxx, 0.0); RB_fuelX.push_back(xxx);}  
  rb_db->getWithDefault("N_HEXDECANE", yyy, false);  RB_tempfuelvec.at(2) = yyy;                       
    if (RB_tempfuelvec[2]) { rb_db->getWithDefault("N_HEXDECANEX", xxx, 0.0); RB_fuelX.push_back(xxx);} 
  rb_db->getWithDefault("O_XYLENE", yyy, false); RB_tempfuelvec.at(3) = yyy;                           
    if (RB_tempfuelvec[3]) { rb_db->getWithDefault("O_XYLENEX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("M_XYLENE", yyy, false);  RB_tempfuelvec.at(4) = yyy;                          
    if (RB_tempfuelvec[4]) { rb_db->getWithDefault("M_XYLENEX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("P_XYLENE", yyy, false); RB_tempfuelvec.at(5) = yyy;                           
    if (RB_tempfuelvec[5]) { rb_db->getWithDefault("P_XYLENEX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("TETRALIN", yyy, false);  RB_tempfuelvec.at(6) = yyy;                          
    if (RB_tempfuelvec[6]) { rb_db->getWithDefault("TETRALINX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("C_DECALIN", yyy, false); RB_tempfuelvec.at(7) = yyy;                          
    if (RB_tempfuelvec[7]) { rb_db->getWithDefault("C_DECALINX", xxx, 0.0); RB_fuelX.push_back(xxx);}   
  rb_db->getWithDefault("T_DECALIN", yyy, false); RB_tempfuelvec.at(8) = yyy;                          
    if (RB_tempfuelvec[8]) { rb_db->getWithDefault("T_DECALINX", xxx, 0.0); RB_fuelX.push_back(xxx);}   
  rb_db->getWithDefault("BENZENE", yyy, false); RB_tempfuelvec.at(9) = yyy;                            
    if (RB_tempfuelvec[9]) { rb_db->getWithDefault("BENZENEX", xxx, 0.0); RB_fuelX.push_back(xxx);}      
  rb_db->getWithDefault("TOLUENE", yyy, false); RB_tempfuelvec.at(10) = yyy;                           
    if (RB_tempfuelvec[10]) { rb_db->getWithDefault("TOLUENEX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("N_PENTANE", yyy, false); RB_tempfuelvec.at(11) = yyy;                         
    if (RB_tempfuelvec[11]) { rb_db->getWithDefault("N_PENTANEX", xxx, 0.0); RB_fuelX.push_back(xxx);}  
  rb_db->getWithDefault("N_HEXANE", yyy, false); RB_tempfuelvec.at(12) = yyy;                          
    if (RB_tempfuelvec[12]) { rb_db->getWithDefault("N_HEXANEX", xxx, 0.0); RB_fuelX.push_back(xxx);}   
  rb_db->getWithDefault("N_HEPTANE", yyy, false); RB_tempfuelvec.at(13) = yyy;                          
    if (RB_tempfuelvec[13]) { rb_db->getWithDefault("N_HEPTANEX", xxx, 0.0); RB_fuelX.push_back(xxx);}  
  rb_db->getWithDefault("METHANOL", yyy, false); RB_tempfuelvec.at(14) = yyy;                          
    if (RB_tempfuelvec[14]) { rb_db->getWithDefault("METHANOLX", xxx, 0.0); RB_fuelX.push_back(xxx);}   
  rb_db->getWithDefault("ETHANOL", yyy, false); RB_tempfuelvec.at(15) = yyy;                           
    if (RB_tempfuelvec[15]) { rb_db->getWithDefault("ETHANOLX", xxx, 0.0); RB_fuelX.push_back(xxx);}    
  rb_db->getWithDefault("ISOPROPANOL", yyy, false); RB_tempfuelvec.at(16) = yyy;                       
    if (RB_tempfuelvec[16]) { rb_db->getWithDefault("ISPROPANOLX", xxx, 0.0); RB_fuelX.push_back(xxx);} 
  rb_db->getWithDefault("JP8", yyy, false); RB_tempfuelvec.at(17) = yyy;                               
    if (RB_tempfuelvec[17]) { rb_db->getWithDefault("JP8X", xxx, 0.0); RB_fuelX.push_back(xxx);}        

  if (restart_ps)
  {
    RB_filePath = restart_ps->getFile();
  }

}//end ResponsiveBoundary::problemSetup   
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary:: sched_updateResponsiveBoundaryProfile and updateResponsiveBoundaryProfile
 Schedule update profile
 Since Responsive Boundaries must be updated every time step
 this routine schedules the update */
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::sched_updateResponsiveBoundaryProfile(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const ArchesLabel* d_lab,
                                    int cellTypeID,
                                    bool initialStep,
                                    int direction,
                                    double windspeed)
{
  Task* tsk = scinew Task("ResponsiveBoundary::updateResponsiveBoundaryProfile",
                          this,
                          &ResponsiveBoundary::updateResponsiveBoundaryProfile,
                          d_lab,cellTypeID, initialStep,direction,windspeed);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN

// since I've passed cellTypeID to the method
// I'm thinking we don't need to require this anymore:    
   tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0);

   tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label(), Ghost::None, 0);
//  for (int ii = 0; ii < d_numInlets; ii++) {
//    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_area_label);
//  }
//  if (d_enthalpySolve) {
//    tsk->modifies(d_lab->d_enthalpySPLabel);
//  }
//  if (d_reactingScalarSolve) {
//    tsk->modifies(d_lab->d_reactscalarSPLabel);
//  }

  // This task computes new density, uVelocity.
  tsk->modifies(d_lab->d_tempINLabel);
  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_radiationFluxWINLabel);
  tsk->modifies(d_lab->d_poolConvectiveHeatFluxLabel);
  tsk->modifies(d_lab->d_poolMassFluxLabel);
  tsk->modifies(d_lab->d_poolSurfaceTemperatureLabel);

  if (RB_tempfuelvec[13])
  {
    tsk->modifies(d_lab->d_c7h16INLabel);
  }
  if (RB_tempfuelvec[14])
  {
    tsk->modifies(d_lab->d_ch3ohINLabel);
  }


//  tsk->modifies(d_lab->d_scalarSPLabel);

//  for (int ii = 0; ii < d_numInlets; ii++){
//    tsk->computes(d_flowInlets[ii]->d_flowRate_label);        //Had to remove this since FlowInlets objects can't be accessed
//  }                                                           //From This Class.   

//  if (d_calcExtraScalars){
//    for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++){
//      tsk->modifies(d_extraScalars->at(i)->getScalarLabel());
//    }
//  }

  sched->addTask(tsk, patches, matls);
} //end sched_updateResponsiveBoundaryProfile
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
// This method takes data from arches and updates the inlet velocities for ResponsiveBoundaries
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::updateResponsiveBoundaryProfile(const ProcessorGroup* /*pc*/,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              const ArchesLabel* d_lab,
                              int cellTypeID,
                              bool initialStep,
                              int direction,
                              double windspeed)
{
    for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    constCCVariable<int> cellType;
    CCVariable<double> density;
    CCVariable<double> flame_temperature;
    SFCXVariable<double> Velocity;
    SFCXVariable<double> VelRhoHat;
    CCVariable<double> radiationFlux;  
    CCVariable<double> poolConvectiveHeatFlux;
    CCVariable<double> poolMassFlux; 
    CCVariable<double> poolSurfaceTemperature;
    CCVariable<double> Fuel;

    if (direction == 1) //(x-direction is vertical)
    {
      new_dw->getModifiable(Velocity, d_lab->d_uVelocitySPBCLabel, index, patch);
      new_dw->getModifiable(VelRhoHat, d_lab->d_uVelRhoHatLabel, index, patch);
      new_dw->getModifiable(radiationFlux, d_lab->d_radiationFluxWINLabel, index, patch);  
    }

    if (direction == 2) //(y-direction is vertical)
    {
      new_dw->getModifiable(Velocity, d_lab->d_vVelocitySPBCLabel, index, patch);
      new_dw->getModifiable(VelRhoHat, d_lab->d_vVelRhoHatLabel, index, patch);
      new_dw->getModifiable(radiationFlux, d_lab->d_radiationFluxNINLabel, index, patch);
    }

    if (direction == 3) //(z-direction is vertical)
    {
      new_dw->getModifiable(Velocity, d_lab->d_wVelocitySPBCLabel, index, patch);
      new_dw->getModifiable(VelRhoHat, d_lab->d_wVelRhoHatLabel, index, patch);
      new_dw->getModifiable(radiationFlux, d_lab->d_radiationFluxTINLabel, index, patch);
    }

    //Aquire the time step size:
    double rdelta_t;
    double ElapsedTime = d_lab->d_sharedState->getElapsedTime();
    double AccelTime = 3.0;
    double TFACTOR = 10.0;
    if(initialStep)
    {
      rdelta_t = 1e-6;
    }
    else
    {  
      delt_vartype delT;   
      old_dw->get(delT, d_lab->d_sharedState->get_delt_label());  
      if (!RB_Accelerate) {rdelta_t = delT;}   
      if ((RB_Accelerate) && (ElapsedTime >= AccelTime)) {rdelta_t = delT;}
      if ((RB_Accelerate) && (ElapsedTime < AccelTime)) {rdelta_t = TFACTOR*delT;} 
    }
    
    // get cellType, density and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, index, patch, Ghost::None,
                Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(density,               d_lab->d_densityCPLabel,             index,patch);
    new_dw->getModifiable(flame_temperature,     d_lab->d_tempINLabel,                index,patch);
    new_dw->getModifiable(poolConvectiveHeatFlux,d_lab->d_poolConvectiveHeatFluxLabel,index,patch);
    new_dw->getModifiable(poolMassFlux,          d_lab->d_poolMassFluxLabel,          index,patch);    
    new_dw->getModifiable(poolSurfaceTemperature,d_lab->d_poolSurfaceTemperatureLabel,index,patch);  

    if (RB_tempfuelvec[13])
    {
      new_dw->getModifiable(Fuel,               d_lab->d_c7h16INLabel,               index,patch);
    }

    if (RB_tempfuelvec[14])
    {
      new_dw->getModifiable(Fuel,              d_lab->d_ch3ohINLabel,               index,patch);
    }


    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
//    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
//    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
//    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    bool minus;

    if (direction == 1) minus = xminus;
    if (direction == 2) minus = yminus;
    if (direction == 3) minus = zminus;
    
      // setup height and temperature and Composition map for the initialstep:
    if(initialStep)
    {
      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){  
        IntVector curr = *iter;
        IntVector mCurr;     
        if (direction == 1) mCurr = *iter - IntVector(1,0,0);    
        if (direction == 2) mCurr = *iter - IntVector(0,1,0);
        if (direction == 3) mCurr = *iter - IntVector(0,0,1);
        if (minus){    
          if (cellType[mCurr] == cellTypeID){    
            vector<double> TT;  
            vector<double> XX;  
            for (int i = 0; i < 200; i++) {TT.push_back(RB_RefTemp);}    
            for (unsigned int ii = 0; ii < RB_fuelX.size(); ii++) {XX.push_back(RB_fuelX.at(ii));}
            RB_mapHeight.insert(pair<IntVector,double> (curr,RB_LiquidHeight));
            RB_mapTemperature.insert(pair< IntVector,vector<double> > (curr,TT));   
            RB_mapX.insert(pair< IntVector, vector<double> > (curr, XX));        
          } // end if (cellType == d_cellTypeID)
        } // if (minus)
      }  // end loop(CellIterator)
    } // end if(initialStep);
          // end setup height and temperature map.




//          // obtain wind speed for pool calculations:
//          double windspeed;    //WME
//          if (d_numInlets <= 1) { windspeed = 0.0;}   //WME
//          if (d_numInlets == 2){    //WME
//            if (indx == 0) {windspeed = d_flowInlets[1]->inletVel;}   //WME
//            if (indx == 1) {windspeed = d_flowInlets[0]->inletVel;}}  //WME
//          if (d_numInlets == 3) {     //WME
//            double wind1;   //WME
//            double wind2;   //WME
//            if (indx == 0) {   //WME
//              wind1 = d_flowInlets[1]->inletVel;   //WME
//              wind2 = d_flowInlets[2]->inletVel;}   //WME
//            if (indx == 1) {       //WME
//              wind1 = d_flowInlets[0]->inletVel;   //WME
//              wind2 = d_flowInlets[2]->inletVel;}   //WME
//            if (indx == 2) {    //WME
//              wind1 = d_flowInlets[0]->inletVel;   //WME
//              wind2 = d_flowInlets[1]->inletVel;}  //WME
//            windspeed = pow(pow(wind1,2)+pow(wind2,2),0.5); }  //WME
//          //end wind speed.



/* Note this present configuration of code doesn't allow access to d_numInlet or d_flowInlets from this 
class.  Another method for calculating windspeed will be needed.  For now windspeed will be given a constant value
from here: */
    double windspeed = 0.0;


    if (minus){    //WME
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){    //WME

        //note that this responsiveboundary implementation assumes that the liquid pool
        // is placed only at the bottom of the domain.  For gravity resistant pools on walls
        // or ceilings this code must be modified.
        IntVector curCell = *iter;//WME
        IntVector mCell;
        if (direction == 1) mCell = *iter - IntVector(1,0,0); //WME
        if (direction == 2) mCell = *iter - IntVector(0,1,0); //WME
        if (direction == 3) mCell = *iter - IntVector(0,0,1); //WME
        if (cellType[mCell] == cellTypeID){    //WME
          vector<double> T = RB_mapTemperature.find(curCell)->second;   //WME
          for (int i = 0; i < 3; i++) { cout << "T[" << i << "]: " << T.at(i) << "\n";}
          vector<double> X = RB_mapX.find(curCell)->second;   //WME
          double height = RB_mapHeight.find(curCell)->second;//WME
          cout << "time step: " << rdelta_t << "\n";//WME
         // cout << "height:  " << height << "\n";  //WME
          double P = 1; // System Pressure (bar) //WME
          int NON = 200;  // number of liquid nodes for responsive boundary model (~200)
          vector<double> Y;

          Y.push_back(Fuel[curCell]);  // bulk fuel composition (mass fraction)

         Reset(RB_tempfuelvec,T,X,Y,height,rdelta_t,windspeed,P,
                radiationFlux[curCell],RB_PoolDiameter,density[curCell],flame_temperature[curCell],NON);//WME

  
          setReflectivity(RB_Reflectivity);  //WME


          TimeStep();//WME
//cout << "TIME: " << d_lab->d_sharedState->getElapsedTime() << "\n\n\n";

          Velocity[curCell] = getFV();//WME
          Velocity[mCell] = getFV();//WME
          
          poolConvectiveHeatFlux[curCell] = getConvectiveHeatFlux();
          poolMassFlux[curCell]           = getMassFlux();
          poolMassFlux[mCell]             = getMassFlux();
          poolSurfaceTemperature[curCell] = getST();
 
          RB_mapHeight.find(curCell)->second = getHeight();//WME
          T.clear();  X.clear(); Y.clear();//WME
          for (int i = 0; i < getNON(); i++) {T.push_back(getT(i));}//WME
          for (int i = 0; i < getNOC(); i++) {X.push_back(getX(i));} //WME
          RB_mapTemperature.find(curCell)->second = T;//WME
          T.clear();
          T = RB_mapTemperature.find(curCell)->second;//WME
          RB_mapX.find(curCell)->second = X;  //WME
          X = RB_mapX.find(curCell)->second;  //WME
cout << "node: " << curCell << " radiation: " << radiationFlux[curCell] << endl; //WME
cout << "Cell: " << curCell << "  Fuel: " << Fuel[curCell] << "  MassFlux: " << poolMassFlux[curCell] <<  endl;
        } // end if (cellType == cellTypeID)                      //WME
      } // end for  (cellIterator)                                                //WME
    } // end if minus                                                 //WME



    VelRhoHat.copyData(Velocity);




  }  // end loop for patches
} // end updateResponsiveBoundaryProfile

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************





//*********************************************************************************************************************
//*********************************************************************************************************************
/** @brief Description of ResponsiveBoundary:: sched_saveResponsiveBoundaryData and saveResponsiveBoundaryData: 
 The following "saveResponsiveBoundaryData" method (with Scheduler) , stores the data in mapTemperature, mapHeight, and mapX to a file
 to be read later at restart.  This enables Arches with Responsive Boundaries to do restarts without losing those data. */                        
//*********************************************************************************************************************
//********************************************************************************************************************* 
void
ResponsiveBoundary::sched_saveResponsiveBoundaryData( SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const ArchesLabel* d_lab,
                                   Output * dataArchive,
                                   int cellTypeID,
                                   int direction)
{
  Task* tsk = scinew Task("ResponsiveBoundary::saveResponsiveBoundaryData",
                          this,
                          &ResponsiveBoundary::saveResponsiveBoundaryData,
                          d_lab, dataArchive, cellTypeID,direction);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0);

  sched->addTask(tsk, patches, matls);

} //end ResponsiveBoundary::sched_saveResponsiveBoundaryData
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
// The following "saveResponsiveBoundaryData" method (with Scheduler) , stores the data in mapTemperature, mapHeight, and mapX to a file
// to be read later at restart.  This enables Arches with Responsive Boundaries to do restarts without losing that data. 
//*********************************************************************************************************************
//********************************************************************************************************************* 
void                                                     
ResponsiveBoundary::saveResponsiveBoundaryData( const ProcessorGroup*,      
                             const PatchSubset* patches, 
                             const MaterialSubset*,      
                             DataWarehouse*,            
                             DataWarehouse* new_dw,
                             const ArchesLabel* d_lab,    
                             Output* dataArchive,
                             int cellTypeID,
                             int direction)     
{


// Test if this is a Checkpoint time step: 
  if ( dataArchive->isCheckpointTimestep() ) {

  //Create the file directory to save the data for this time step
    const int & timestep = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    ostringstream data_dir;
    //data_dir should be: ".../simulation.uda.###/checkpoints/t####/mydata
    data_dir << dataArchive->getOutputLocation() << "/checkpoints" << "/t" << setw(5) << setfill('0') << timestep << "/mydata";

    int result = MKDIR( data_dir.str().c_str(), 0777);

//    //Test to see if the file already exists:
//    ifstream test;

//    test.open(data_dir.str().c_str(), ifstream::in);
//    test.close();

    //Loop Through the Patches:
    for ( int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);

      int archIndex = 0; //only 1 arches material.            
      int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
      constCCVariable<int> cellType;

      new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None,
                Arches::ZEROGHOSTCELLS);



     bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
     bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
     bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
     bool minus;

     if (direction == 1) minus = xminus;
     if (direction == 2) minus = yminus;
     if (direction == 3) minus = zminus;



     if (minus) {
       ostringstream file_name;
       file_name << data_dir.str() << "/p" << setw(5) << setfill('0') << Parallel::getMPIRank();

       ofstream output(  file_name.str().c_str() ,ios::out | ios::binary);


       if ( !output ) {
         throw InternalError ( string( "ResponsiveBoundary::saveResponsiveBoundaryData(): couldn't open file '") + file_name.str() + "'.",
                                __FILE__, __LINE__);
       } // end if (!output)


       for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){    
         IntVector Cell = *iter;
         IntVector mCell;  
         if (direction == 1) mCell = *iter - IntVector(1, 0, 0);    
         if (direction == 2) mCell = *iter - IntVector(0, 1, 0);
         if (direction == 3) mCell = *iter - IntVector(0, 0, 1); 
         if (cellType[mCell] == cellTypeID){    
           vector<double> & Tdata = RB_mapTemperature.find(Cell)->second;
           double & Hdata = RB_mapHeight.find(Cell)->second;
           vector<double> & Xdata = RB_mapX.find(Cell)->second;

           output << Cell[0] << " " << Cell[1] << " " << Cell[2] << " ";
           output << Hdata << " ";
           for (unsigned int ii = 0; ii < Xdata.size(); ii++){ output << Xdata[ii] << " "; }
           for (unsigned int ii = 0; ii < Tdata.size(); ii++){ output << Tdata[ii] << " "; }

          } // end if(cellType...)  
        } // end for loop (CellIterator ...)  

// close the file for this processor:
            output.close();


      }  // end if (minus)
    }  // end for loop (int p = 0; p < patches->size(); ...)  



    //Test to see if the file was succesfully created:
    ifstream test;

    test.open(data_dir.str().c_str(), ifstream::in);
    test.close();

    if (test.fail()) {
      ostringstream error;
      error << ("ResponsiveBoundary::saveResponsiveBoundaryData():  couldn't create directory '") + data_dir.str() + "; (" << errno << ").";
      throw InternalError( error.str(), __FILE__, __LINE__);
    } //end if (test.fail())


// Parallel::getMPIRank()  <- the MPI node number              
//    for( map< IntVector,vector<double> >::iterator iter = mapTemperature.begin(); iter != mapTemperature.end(); iter++ ) {}   

 } // end dataArchive-> isCheckpointsTimestep                                                                       
}  //end saveResponsiveBoundaryData  
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//********************************************************************************************************************* 
/** @brief Description of ResponsiveBoundary:: sched_readResponsiveBoundaryData and readResponsiveBoundaryData:
 The following method: "readResponsiveBoundaryData" uploads the data stored by saveResponsiveBoundaryData and repopulates mapTemperature, mapHeight,
 and mapX at the time of restart so that arches simulations with Responsive Boundary can resume at a restart. */                          
//*********************************************************************************************************************
//********************************************************************************************************************* 
void
ResponsiveBoundary::sched_readResponsiveBoundaryData( SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   Output* dataArchive,
                                   int direction)
{
  Task* tsk = scinew Task("ResponsiveBoundary::readResponsiveBoundaryData",
                          this,
                          &ResponsiveBoundary::readResponsiveBoundaryData,
                          dataArchive,direction);

//  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0);

  sched->addTask(tsk, patches, matls);

} //end ResponsiveBoundary::sched_readResponsiveBoundaryData                                     
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
// The following method: "readResponsiveBoundaryData" uploads the data stored by saveResponsiveBoundaryData and repopulates mapTemperature, mapHeight,
// and mapX at the time of restart so that arches simulations with Responsive Boundary can resume at a restart.   
//*********************************************************************************************************************
//*********************************************************************************************************************
void                                                     
ResponsiveBoundary::readResponsiveBoundaryData( const ProcessorGroup*,     
                             const PatchSubset* patches, 
                             const MaterialSubset*,      
                             DataWarehouse*,            
                             DataWarehouse* new_dw,    
                             Output* dataArchive,
                             int direction)     
{


  //ensure that the number of nodes and components are available:
  Reset(RB_tempfuelvec);

  string file0 = RB_filePath;
  string file1 = file0.substr(0,(file0.length() - 13));
  ostringstream data_dir;
  data_dir << file1 << "/mydata";  




 /*  ostringstream data_dir;
  string file0 = dataArchive->getOutputLocation();
  string filebase = file0.substr(0,(file0.length() - 4));
  string filenumber = file0.substr((file0.length() - 3),file0.length());
  int newfilenumber = atoi(filenumber.c_str()) - 1;
  data_dir << filebase << "." << setw(3) << setfill('0') << newfilenumber;
  string newfile = data_dir.str().c_str();
    // Note: As currently written this code assumes that the restart data is in the uda file numbered one level
    // Previous to the current Outputfile.  This may not always be the case, and a better system should be 
    // implemented.  WME                                      

  string index(newfile+"/checkpoints/index.xml");
 
//  ProblemSpecReader psr(index.c_str()); 
  ProblemSpecReader psr;
  ProblemSpecP indexDoc = psr.readInputFile(index.c_str());
  ProblemSpecP ts = indexDoc->findBlock("timesteps");
  if (ts == 0){                   
    throw InternalError("ResponsiveBoundary::readResponsiveBoundaryData: 'timesteps' node not found in index.xml",
                          __FILE__,__LINE__);
  }

  vector<int> tsIndex;

  for (ProblemSpecP t = ts->getFirstChild(); t != 0; t = t->getNextSibling())
  {
    if (t->getNodeType() == ProblemSpec::ELEMENT_NODE)
    {  
      map<string,string> attributes;
      t->getAttributes(attributes);
      string tsfile = attributes["href"];
      if(tsfile == "")
        throw InternalError("ResponsiveBoundary::readResponsiveBoundaryData: timesteps href not found",
                            __FILE__,__LINE__);
*/
//      int timestepNumber;
//
//      istringstream timeStepVal (t->getNodeValue());
//
//      timeStepVal >> timestepNumber;
//
//      if (!timeStepVal){
//        printf("Warning: ResponsiveBoundary::readResponsiveBoundaryData: stringstream failed...\n");
//      }
//
//      tsIndex.push_back(timestepNumber);
//    }
//  }

//  int timestepFileNumber = tsIndex.back();
//  ostringstream data_dir2;
//  data_dir2 << newfile<< "/checkpoints" << "/t" << setw(5) << setfill('0') << timestepFileNumber << "/mydata";





  //Loop Through the Patches:
  for ( int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);


    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool minus;

    if (direction == 1) minus = xminus;
    if (direction == 2) minus = yminus;
    if (direction == 3) minus = zminus;

    if (minus) {


       ostringstream data_file;
       data_file << data_dir.str() << "/p" << setw(5) << setfill('0') << Parallel::getMPIRank();


      fstream input ( data_file.str().c_str() , ios::in | ios::binary);
          
       if ( !input )
       {
         throw InternalError ( string( "ResponsiveBoundary::readResponsiveBoundaryData(): couldn't open file '") + data_file.str() + "'.",
                                __FILE__, __LINE__);
       } // end if (!input)


       RB_mapHeight.clear(); RB_mapX.clear(); RB_mapTemperature.clear();

       while (!input.eof())
       { 
         IntVector Cell;
         double H;
         vector<double> Tv;
         vector<double> Xv;

         Tv.clear(); Xv.clear();
         for (int ii = 0; ii < 3; ii++)
         {
          int Data;
          input >> Data;
          Cell[ii] = (Data);
         }  // end for loop

         input >> H;
         for (int ii = 0; ii < getNOC(); ii++)
         {
          double DataX;
          input >> DataX;
          Xv.push_back(DataX);
         }

         for (int ii = 0; ii < getNON(); ii++)
         {
          double DataT;
          input >> DataT;
          Tv.push_back(DataT);
         }



         if (!input.eof()){
           RB_mapHeight.insert(pair<IntVector, double> (Cell,H));
           RB_mapX.insert(pair<IntVector, vector<double> > (Cell,Xv));
           RB_mapTemperature.insert(pair<IntVector, vector<double> > (Cell,Tv));
         } // end if (!input.eof())

      } //end while loop
     
      input.close();

//for (map<IntVector, double>::iterator iter = RB_mapHeight.begin(); iter != RB_mapHeight.end(); iter++)
//{
// cout << "Map Check: " << endl;
//  cout << "Map Size: " << RB_mapHeight.size() << endl;
//  const IntVector node = (*iter).first;
//  double HH = (*iter).second;
//  cout << "Node: " << node << "\t" << "Height: " << HH << endl;
//}
 
    } // end if (minus) 
  } // end for loop (patches)
} // end ResponsiveBoundary::readResponsiveBoundaryData
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
// The following method "verifyMMS" generates the needed derivatives for use in the method of manufactured solutions for the 
// purposes of code verification.  The manufactured solution in this case is: T(x,t) = T0 + (1 + 5*exp(-t))*sin(2*pi*h).
//*********************************************************************************************************************
//*********************************************************************************************************************
double
ResponsiveBoundary::verifyMMS(double t, double h, int type, int derivative)
{

double RT = RB_RefTemp;
double pi = 3.14159265358979323846;
double e = exp(1);
double D;

if (type == 0)  //for the energy equation:
{
  if (derivative == 1) //temporal derivative
  { 
    D = (-5*exp(-t)*sin(2*pi*h));
  }

  if (derivative == 2) //second spatial derivative
  {
    D = -4*pi*pi*(1 + 5*exp(-t))*sin(2*pi*h);
  }

  if (derivative == 0) //function value (for inital values)
  {
    D = RT + (1 + 5*exp(-t))*sin(2*pi*h);
  }
}



if (type == 1) //for the height equation:
{
  double EndTime = 50;  // # of seconds required for liquid level to drop from 1 to 0.
  double alpha = (e - 1)/EndTime;
  if (derivative == 0) //function value
  {
    D = 1.0000 - log(1 + alpha*t); 
  }

  if (derivative == 1) //first derivative of Height with respect to time
  {
    D = -alpha/(1 + alpha*t);
  }
}


if (type == 2) // Energy Equation for coupled Energy-Height Manufactured Solution
{
  double H = verifyMMS(t,h,1,0);
  double HH = verifyMMS(t,h,1,1);
  double C1 = 0.5, C2 = -25.0, C3 = 10.0;


  if (derivative == 0) //function value
  {
//    D = RT + (1 + 5*exp(-t)*sin(2*pi*h/H)); 
    D = RT + C3*exp(C2*(1 - h));
  }

  if (derivative == 1) //first temporal derivative
  {
//    D = (1+5*exp(-t))*cos(2*pi*h/H)*(2*pi*HH/H/H) + sin(2*pi*h/H)*(-5*exp(-t));
    D = 0; 
  }

  if (derivative == 2) //second spatial derivative
  {
//    D = (1+5*exp(-t))*sin(2*pi*h/H)*(-4*pi*pi/H/H);
    D = (C2*C2)*C3*exp(C2*(1-h));
  }

}
 

return D;
}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
// The method "verifyOn" switches the verification of the responsive boundary model on, and sets the initial conditions
// for the verification simulation.
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::verifyOn(int type)
{

        vector<double> T;
        int n = RB_NON;
        double t = 0;
        double h;


        if (type == 0)  //for the verification of the energy equation
        {
          RB_Verification_Energy = 1;
          AssignHeight(0.5000);


	  double dz = RB_LiquidHeight/n;

          for (int ii = 0; ii < n; ii++)
          {
            h = RB_LiquidHeight - (ii + 0.5)*dz;
            T.push_back(verifyMMS(t,h,0,0));
          }  
        
          AssignT(T);
          SurfaceTemperature();
        }
    
        if (type == 1) //for the verification of the height equation
        {
          RB_Verification_Height = 1;
        
          AssignHeight(verifyMMS(t,0,1,0));
        } 

        if (type == 2) // for verification using a coupled energy/height solution
        {
          RB_Verification_Height_Energy = 1;
          AssignHeight(verifyMMS(t,0,1,0));

          double dz = RB_LiquidHeight/n;
          for (int ii = 0; ii < n; ii++)
          {
            h = RB_LiquidHeight - (ii + 0.5)*dz;
            T.push_back(verifyMMS(t,h,2,0));    
          }
          AssignT(T);
          SurfaceTemperature();
        }

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************



//*********************************************************************************************************************
//*********************************************************************************************************************
// The method "printMMS" prints on screen the values of the manufactured solution for comparision/verification purposes.
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::printMMS(int l)
{

        int n = RB_NON;
        double h;
        double t = RB_Time;
        double dz = RB_LiquidHeight/n;

        for (int ii = 0; ii < l; ii++)
        {
          h = RB_LiquidHeight - (ii + 0.5)*dz;
          cout << "MS[" << ii << "]: " << verifyMMS(t,h,0,0) << endl;
        }

}
//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************




//*********************************************************************************************************************
//*********************************************************************************************************************
// The method "compareMMS" evaluates the error in the computed solution by comparison with the manufactured solution 
// for the purposes of verification.
//*********************************************************************************************************************
//*********************************************************************************************************************
void
ResponsiveBoundary::compareMMS()
{

        if (RB_Verification_Energy | RB_Verification_Height_Energy)
        {
          RB_compareMMS_Energy.clear();
          RB_compareMMS_DTdz2.clear();
   
          int n = RB_NON;
          double h;
          double t = RB_Time;
          double dz = RB_LiquidHeight/n;

          if (RB_Verification_Energy)
          {


            double DTdz2[n];
            for (int ii = 0; ii < n; ii++)
            {
//              if (ii == 0) { DTdz2[0] = (0-(getT(0) - getT(1))/dz)/dz;}
//              else if (ii == n-1) {DTdz2[ii] = ((getT(ii-1) - getT(ii))/dz - 0)/dz;}
              if (ii == 0) { DTdz2[0] = (2*getT(0) - 5*getT(1) + 4*getT(2) - getT(3))/dz/dz;}
              else if (ii == (n-1)) { DTdz2[ii] = (2*getT(ii) - 5*getT(ii-1) + 4*getT(ii-2) - getT(ii-3))/dz/dz;}
              else {DTdz2[ii] = ((getT(ii-1) - getT(ii))/dz - (getT(ii) - getT(ii+1))/dz)/dz;}
            }


            for (int ii = 0; ii < n; ii++)
            {
              h = RB_LiquidHeight - (ii + 0.5)*dz;
              RB_compareMMS_Energy.push_back(abs(getT(ii) - verifyMMS(t,h,0,0))); 
              RB_compareMMS_DTdz2.push_back(abs(DTdz2[ii] - verifyMMS(t,h,0,2)/verifyMMS(t,h,0,2)));
            }
          } 


          if (RB_Verification_Height_Energy)
          {

            double DTdz2[n];
            for (int ii = 0; ii < n; ii++)
            {
//              if (ii == 0) { DTdz2[0] = (0-(getT(0) - getT(1))/dz)/dz;}
//              else if (ii == n-1) {DTdz2[ii] = ((getT(ii-1) - getT(ii))/dz - 0)/dz;}
              if (ii == 0) { DTdz2[0] = (2*getT(0) - 5*getT(1) + 4*getT(2) - getT(3))/dz/dz;}
              else if (ii == (n-1)) { DTdz2[ii] = (2*getT(ii) - 5*getT(ii-1) + 4*getT(ii-2) - getT(ii-3))/dz/dz;}
              else {DTdz2[ii] = ((getT(ii-1) - getT(ii))/dz - (getT(ii) - getT(ii+1))/dz)/dz;}
            }


            for (int ii = 0; ii < n; ii++)
            {
              h = RB_LiquidHeight - (ii + 0.5)*dz;
              double Tanalytic = verifyMMS(t,h,2,0);
              double DTanalytic = verifyMMS(t,h,2,2);
              RB_compareMMS_Energy.push_back((getT(ii) - Tanalytic));
              RB_compareMMS_DTdz2.push_back(abs(DTdz2[ii] - DTanalytic/DTanalytic));
            }
          }

          double sum = 0;

          for (int ii = 0; ii < n; ii++)
          {
            sum = sum + pow(RB_compareMMS_Energy[ii],2);
          }

          RB_normErrorMMS_Energy = pow(sum,0.5);
        }
        
        if (RB_Verification_Height | RB_Verification_Height_Energy)
        {
          double t = RB_Time;
          RB_compareMMS_Height = abs(RB_LiquidHeight - verifyMMS(t,0,1,0));
        }

}

//---------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************
