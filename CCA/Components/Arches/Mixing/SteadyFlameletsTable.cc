//----- SteadyFlameletsTable.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/SteadyFlameletsTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/VectorTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ILDMReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>
#include <math.h>
#include <Core/Math/MiscMath.h>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for SteadyFlamaletsTable
//****************************************************************************
SteadyFlameletsTable::SteadyFlameletsTable():MixingModel()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
SteadyFlameletsTable::~SteadyFlameletsTable()
{
}

//****************************************************************************
// Problem Setup for SteadyFlameletsTable
//****************************************************************************
void 
SteadyFlameletsTable::problemSetup(const ProblemSpecP& params)
{
  std::string d_inputfile;
  ProblemSpecP db = params->findBlock("SteadyFlameletsTable");
  db->require("adiabatic",d_adiabatic);
  db->require("rxnvars",d_numRxnVars);
  db->require("mixstatvars",d_numMixStatVars);
  db->require("inputfile",d_inputfile);
  // Set up for MixingModel table
  d_numMixingVars = 1;
  // Define mixing table, which includes call reaction model constructor
  d_tableDimension = d_numMixingVars + d_numRxnVars + d_numMixStatVars + !(d_adiabatic)+1;
  readMixingTable(d_inputfile);
  // Printing the indices of the species 
  cout<<"CO2 index is " << co2_index<<endl;
  cout<<"H2O index is " << h2o_index<<endl;
  cout<<"C2H2 index is " << c2h2_index<<endl;

}

      

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
SteadyFlameletsTable::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  double small=1.0e-10;
  // Extracting the independent variables from the in stream 
  double mixFrac = inStream.d_mixVars[0];
  double mixFracVars = 0.0;
  if (inStream.d_mixVarVariance.size() != 0)
    mixFracVars = inStream.d_mixVarVariance[0];
  // Raj advised to multiply by 2.0
  double scalDisp=2.0*inStream.d_scalarDisp;	
  if(mixFrac > 1.0)
	mixFrac=1.0;
  else if (mixFrac < small)
	mixFrac=0.0;
  if(mixFracVars < small)
	mixFracVars=0.0;
  double var_limit=mixFracVars/((mixFrac*(1.0-mixFrac))+small);
  if(var_limit > 0.9)
  	mixFracVars=(2.0/3.0)*mixFracVars;
  // Clipping the scalar disspation to minimum and maximum levels  
  if(scalDisp < scalarDisp[0])
	  scalDisp = scalarDisp[0];
  else if (scalDisp > scalarDisp[d_scaldispcount-1])
	  scalDisp = scalarDisp[d_scaldispcount-1];
  // Looking for the properties corresponding to the scalar dissipation 
  tableLookUp(mixFrac, mixFracVars, scalDisp, outStream); 
  // Debug print statements
  /*cout<<"Temperature for properties is:  "<<outStream.d_temperature<<endl;
  cout<<"Density for properties is:  "<<outStream.d_density<<endl;
  cout<<"Mixture fraction is :  "<<mixFrac<<endl;
  cout<<"Mixture fraction variance is :  "<<mixFracVars<<endl;
  cout<<"Scalar Dissipation is :  "<< scalDisp<<endl; */
}

//****************************************************************************
//Interpolating for the properties 
//****************************************************************************


void SteadyFlameletsTable::tableLookUp(double mixfrac, double mixfracVars, double scalDisp, Stream& outStream)
{
  double small = 1.0e-10;
  // nx - mixfrac, nh - Scalar dissipation 
  
  std::vector <double> fmg = vector <double>(d_varcount);
  std::vector <double> fpmg = vector <double>(d_varcount);
  std::vector <double> s1 = vector <double>(d_varcount);
  std::vector <double> s2 = vector <double>(d_varcount);

  // Scalar dissipation lookup
  int nhl_lo=0, nhl_hi=0;
  double dhl_lo=0.0, dhl_hi=0.0;
  if(scalDisp == 0.0){
	nhl_lo=0;
	nhl_hi=0;
        dhl_lo=0.0;
        dhl_hi=1.0;
  }
  else{
	for(int hl_index=0; hl_index < d_scaldispcount-1; hl_index++){
		dhl_lo = scalarDisp[hl_index]-scalDisp;
		dhl_hi = scalarDisp[hl_index+1]-scalDisp;
		if((dhl_lo*dhl_hi) == 0.0 && hl_index != 0){
			nhl_lo=hl_index+1;
			nhl_hi=nhl_lo;
                        break;
		}
		else if((dhl_lo*dhl_hi) <= 0.0){
			nhl_lo=hl_index;
			nhl_hi=nhl_lo+1;
                        break;
		}
	}

  }
  /*cout<< "nhl_lo is :" << nhl_lo<<endl;
  cout<< "nhl_hi is :" << nhl_hi<<endl;
  cout<< "dhl_lo is :" << dhl_lo<<endl;
  cout<< "dhl_hi is :" << dhl_hi<<endl;*/
 
  // Main loop - Scalar dissipation
  for(int m_index = nhl_lo; m_index <= nhl_hi; m_index++){ 
  int nx_lo, nx_hi;

  //Mixture fraction 
  double df1,df2; 
  for(int index=0; index < d_mixfraccount-1; index++){
	df1 = meanMix[index]-mixfrac;
	df2 = meanMix[index+1]-mixfrac;
        //cout<<"df1*df2 value is  : "<<df1*df2<<endl;
	if((df1*df2) == 0.0 && index != 0){
		nx_lo=index+1;
		nx_hi=nx_lo;
                break;
	}
	else if((df1*df2) <= 0.0){
		nx_lo=index;
		nx_hi=nx_lo+1;
                break;
	}
  }

  //cout<<"nx_lo index is  : "<<nx_lo<<endl;
  //cout<<"nx_hi index is  : "<<nx_hi<<endl;

  
  // variance is uniform table
  double max_curr_Zvar = mixfrac*(1.0-mixfrac);
  double max_Zvar = min(mixfracVars,max_curr_Zvar);
  double g, gi1, gp, gi2;
  if(meanMix[nx_lo]<=small || abs(meanMix[nx_lo]-1.0)<=small)
	g=0.0;
  else{
	gi1=max_Zvar*meanMix[nx_lo]*(1.0-meanMix[nx_lo])/(max_curr_Zvar+small);
        g=gi1*double(d_mixvarcount-1)/(meanMix[nx_lo]*(1.0-meanMix[nx_lo]));
  }

  if(meanMix[nx_hi]<=small || abs(meanMix[nx_hi]-1.0)<=small){
	gp=0.0;
  }
  else{
	gi2=max_Zvar*meanMix[nx_hi]*(1.0-meanMix[nx_hi])/(max_curr_Zvar+small);
        gp=gi2*double(d_mixvarcount-1)/(meanMix[nx_hi]*(1.0-meanMix[nx_hi]));

  }
  //cout<<" G value is :"<<g<<endl;
  //cout<<"GP value is :"<<gp<<endl;
  int k1,k2,k1p,k2p;
  k1=floor(g);
  if(k1 < 0)
	k1=0;
  if(k1 > d_mixvarcount-2)
	k1=d_mixvarcount-2;
  //cout<<"Index of K1 is: "<<k1<<endl;
  k1p=k1+1;
  k2=floor(gp);
  if(k2 < 0)
	k2=0;
  if(k2 > d_mixvarcount-2)
	k2=d_mixvarcount-2;
  //cout<<"Index of K2 is: "<<k2<<endl;
  k2p=k2+1;
  //Interpolating the values
  for(int ii=0; ii< d_varcount; ii++){
  	fmg[ii]=(g-double(k1))*table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k1p][ii]-(g-double(k1p))*table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k1][ii];
  //cout<< "FMG is :"<<fmg[2]<<endl;
  	fpmg[ii]=(gp-double(k2))*table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k2p][ii]-(gp-double(k2p))*table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k2][ii];
  //cout<< "FPMG is : "<<fpmg[2]<<endl;
  }

 if(nhl_lo==nhl_hi){
	  for(int ii=0; ii< d_varcount; ii++){
		s1[ii]=(df1*fpmg[ii]-df2*fmg[ii])/(df1-df2);
        	s2[ii]=s1[ii];
	  }
	break;
  }
  else if(m_index == nhl_lo){
	  for(int ii=0; ii< d_varcount; ii++)
  		s1[ii]=(df1*fpmg[ii]-df2*fmg[ii])/(df1-df2);
  }
  else{
	  for(int ii=0; ii< d_varcount; ii++)
  		s2[ii]=(df1*fpmg[ii]-df2*fmg[ii])/(df1-df2);
  }

  }	
  //cout<<"value of S1_temperature is "<<s1[2]<<endl;
  //cout<<"value of S2_temperature is "<<s2[2]<<endl;
  outStream.d_temperature= (dhl_lo*s2[2]-dhl_hi*s1[2])/(dhl_lo-dhl_hi);  
  outStream.d_density=(dhl_lo*s2[1]-dhl_hi*s1[1])/(dhl_lo-dhl_hi) *1000.0;  
  outStream.d_cp= 0.0; // Not in the table  
  outStream.d_enthalpy= 0.0; // Not in the table  
  outStream.d_co2= (dhl_lo*s2[co2_index]-dhl_hi*s1[co2_index])/(dhl_lo-dhl_hi);  
  outStream.d_h2o= (dhl_lo*s2[h2o_index]-dhl_hi*s1[h2o_index])/(dhl_lo-dhl_hi); 
  outStream.d_c2h2= (dhl_lo*s2[c2h2_index]-dhl_hi*s1[c2h2_index])/(dhl_lo-dhl_hi); 

}
//****************************************************************************
// Reading the flamelets mixing table
//****************************************************************************

void SteadyFlameletsTable::readMixingTable(std::string inputfile)
{
  int dummy;	
  cerr << "Preparing to read the inputfile:   " << inputfile << endl;
  	ifstream fd(inputfile.c_str());
  if(fd.fail()){
	cout<<" Unable to open the given input file" <<endl;
	exit(1);
  }
  // Total number of variables in the table
  fd >> d_varcount;
  cout<<"d_var count: "<< d_varcount << endl;
  // Increasing the count for mixfraction variance
  d_varcount=d_varcount+1;
  // Names of the variables 
  variables_list= vector<string>(d_varcount);
  variables_list[0]="Mixfracvariance";
  for(int i_vc = 1; i_vc < d_varcount ; i_vc++){
	  fd >> variables_list[i_vc];
          cout<< variables_list[i_vc]<< " ";
          if(variables_list[i_vc]== "CO2")
	  	  co2_index = i_vc;
    	  else if(variables_list[i_vc]== "H2O")
	  	  h2o_index = i_vc;
    	  else if(variables_list[i_vc]== "C2H2")
	  	  c2h2_index = i_vc;
  }
  cout<<endl;
  fd >> d_mixfraccount >>d_mixvarcount >> d_scaldispcount >> dummy;
  cout << d_mixfraccount << " " << d_mixvarcount << " " << d_scaldispcount << " " << dummy << endl;
  // meanMix stores the mixture fractions available in the table
     meanMix =  vector<double>(d_mixfraccount);
  // Scalar disspation values stored in the table
     scalarDisp=vector<double>(d_scaldispcount);
  // Allocating the table space 
  int size = d_mixfraccount*d_mixvarcount*d_scaldispcount;
  table = vector <vector <double> > (size);
  for (int ii = 0; ii < size; ii++)
    table[ii] = vector <double> (d_varcount);

  //Reading the data
  // Mixture fraction loop
  for (int mm=0; mm< d_mixfraccount; mm++){
	fd >> meanMix[mm];
        //cout<< meanMix[mm]<<endl;
  	// Scalar dissipation loop 
  	for (int ii=0; ii< d_scaldispcount; ii++){
 		fd >> scalarDisp[ii];
               // cout<< scalarDisp[ii]<<endl;
        	// Variance loop
  		for (int jj=0;jj<d_mixvarcount; jj++){
                	// Variables loop 
			for (int kk=0; kk< d_varcount; kk++){
					fd >> table[mm*(d_scaldispcount*d_mixvarcount)+ii*d_mixvarcount+jj][kk];
					//cout << table[mm*(d_scaldispcount*d_mixvarcount)+ii*d_mixvarcount+jj][kk] << " ";
			}
			//cout<<endl;
		} // End of variance loop
  	}// End of scalar dissipation loop
  }//End of mixture fraction loop
  // Closing the file pointer
  fd.close();
}





