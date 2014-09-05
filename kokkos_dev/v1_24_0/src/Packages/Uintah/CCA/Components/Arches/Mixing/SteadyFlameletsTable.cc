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
	d_calcthermalNOx=false;
}

//****************************************************************************
// Default constructor for SteadyFlamaletsTable
//****************************************************************************
SteadyFlameletsTable::SteadyFlameletsTable(bool d_thermalNOx):MixingModel()
{
	d_calcthermalNOx=d_thermalNOx;
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
  readChiTable();
  // Printing the indices of the species 
  cout<<"CO2 index is " << co2_index<<endl;
  cout<<"H2O index is " << h2o_index<<endl;
  cout<<"C2H2 index is " << c2h2_index<<endl;
  cout<<"NO index is " << NO_index<<endl;

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
  double scalDisp_filtered=2.0*inStream.d_scalarDisp;	
  if(mixFrac > 1.0)
	mixFrac=1.0;
  else if (mixFrac < small)
	mixFrac=0.0;
  if(mixFracVars < small)
	mixFracVars=0.0;
  double var_limit=mixFracVars/((mixFrac*(1.0-mixFrac))+small);
  if(var_limit > 0.9)
  	mixFracVars=(2.0/3.0)*mixFracVars;
  //Mapping the filtered khi to the flamelets khi
  double scalDisp=scalDisp_filtered/chitableLookUp(mixFrac,mixFracVars);   
  // Clipping the scalar disspation to minimum and maximum levels  
  if(scalDisp < scalarDisp[0])
	  scalDisp = scalarDisp[0];
  else if (scalDisp > scalarDisp[d_scaldispcount-1])
	  scalDisp = scalarDisp[d_scaldispcount-1];
  // Looking for the properties corresponding to the scalar dissipation 
  // Debug print 
  tableLookUp(mixFrac, mixFracVars, scalDisp, outStream); 
  // Debug print statements
  /*cout<<"Temperature for properties is:  "<<outStream.d_temperature<<endl;
  cout<<"Density for properties is:  "<<outStream.d_density<<endl;
  cout<<"Mixture fraction is :  "<<mixFrac<<endl;
  cout<<"Mixture fraction variance is :  "<<mixFracVars<<endl;
  cout<<"Filetered Scalar Dissipation is :  "<< scalDisp_filtered<<endl; 
  cout<<"Flamelets Scalar Dissipation is :  "<< scalDisp<<endl;*/ 
}

//****************************************************************************
//Interpolating for the properties 
//****************************************************************************


void SteadyFlameletsTable::tableLookUp(double mixfrac, double mixfracVars, double scalDisp, Stream& outStream)
{
  double small = 1.0e-10;
  // nx - mixfrac, sd - Scalar dissipation 
  
  std::vector <double> fmg = vector <double>(d_varcount);
  std::vector <double> fpmg = vector <double>(d_varcount);
  std::vector <double> s1 = vector <double>(d_varcount);
  std::vector <double> s2 = vector <double>(d_varcount);

  // Scalar dissipation lookup
  int nsd_lo=0, nsd_hi=0;
  double dsd_lo=0.0, dsd_hi=0.0;
  if(scalDisp == 0.0){
	nsd_lo=0;
	nsd_hi=0;
        dsd_lo=0.0;
        dsd_hi=1.0;
  }
  else{
	for(int sd_index=0; sd_index < d_scaldispcount-1; sd_index++){
		dsd_lo = scalarDisp[sd_index]-scalDisp;
		dsd_hi = scalarDisp[sd_index+1]-scalDisp;
		if((dsd_lo*dsd_hi) == 0.0 && sd_index != 0){
			nsd_lo=sd_index+1;
			nsd_hi=nsd_lo;
                        break;
		}
		else if((dsd_lo*dsd_hi) <= 0.0){
			nsd_lo=sd_index;
			nsd_hi=nsd_lo+1;
                        break;
		}
	}

  }
  /*cout<< "nsd_lo is :" << nsd_lo<<endl;
  cout<< "nsd_hi is :" << nsd_hi<<endl;
  cout<< "d_sdlo is :" << dsd_lo<<endl;
  cout<< "d_sdhi is :" << dsd_hi<<endl;*/
 
  // Main loop - Scalar dissipation
  for(int m_index = nsd_lo; m_index <= nsd_hi; m_index++){ 
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

  
  // variance is non uniform table
  double max_curr_Zvar = mixfrac*(1.0-mixfrac);
  double max_Zvar = min(mixfracVars,max_curr_Zvar);
  double dfv11, dfv12, dfv21, dfv22;
  int k1,k2,k1p,k2p;
  if(meanMix[nx_lo]<=small || fabs(meanMix[nx_lo]-1.0)<=small){
	k1=0;
	k1p=0;
	dfv11=0.0;
	dfv12=1.0;
  }
  else{
	  for (int v_ind=0; v_ind < d_mixvarcount-1; v_ind++)
	  {
		  // Varaince is stored as first entry in the table
		dfv11 = table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+v_ind][0]-max_Zvar;
		dfv12 = table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+v_ind+1][0]-max_Zvar;
		if((dfv11*dfv12) == 0.0 && v_ind != 0){
			k1=v_ind+1;
			k1p=k1;
                	break;
		}
		else if((dfv11*dfv12) <= 0.0){
			k1=v_ind;
			k1p=k1+1;
                	break;
		}
	  }
  }

  if(meanMix[nx_hi]<=small || fabs(meanMix[nx_hi]-1.0)<=small){
	k2=0;
	k2p=0;
	dfv21=0.0;
	dfv22=1.0;
  }
  else{
	  for (int v_ind=0; v_ind < d_mixvarcount-1; v_ind++)
	  {
		dfv21 = table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+v_ind][0]-max_Zvar;
		dfv22 = table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+v_ind+1][0]-max_Zvar;
		if((dfv21*dfv22) == 0.0 && v_ind != 0){
			k2=v_ind+1;
			k2p=k2;
                	break;
		}
		else if((dfv21*dfv22) <= 0.0){
			k2=v_ind;
			k2p=k2+1;
                	break;
		}
	  }
  }
  //Interpolating the values
  for(int ii=0; ii< d_varcount; ii++){
  	fmg[ii]=(dfv11*table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k1p][ii]-dfv12*table[nx_lo*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k1][ii])/(dfv11-dfv12);
  //cout<< "FMG is :"<<fmg[2]<<endl;
  	fpmg[ii]=(dfv21*table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k2p][ii]-dfv22*table[nx_hi*d_scaldispcount*d_mixvarcount+m_index*d_mixvarcount+k2][ii])/(dfv21-dfv22);
  //cout<< "FPMG is : "<<fpmg[2]<<endl;
  }

 if(nsd_lo==nsd_hi){
	  for(int ii=0; ii< d_varcount; ii++){
		s1[ii]=(df1*fpmg[ii]-df2*fmg[ii])/(df1-df2);
        	s2[ii]=s1[ii];
	  }
	break;
  }
  else if(m_index == nsd_lo){
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
  outStream.d_temperature= (dsd_lo*s2[2]-dsd_hi*s1[2])/(dsd_lo-dsd_hi);  
  outStream.d_density=(dsd_lo*s2[1]-dsd_hi*s1[1])/(dsd_lo-dsd_hi) *1000.0;  
  outStream.d_cp= 0.0; // Not in the table  
  outStream.d_enthalpy= 0.0; // Not in the table  
  if(co2_index!=-1)
  	outStream.d_co2= (dsd_lo*s2[co2_index]-dsd_hi*s1[co2_index])/(dsd_lo-dsd_hi);  
  if(h2o_index!=-1)
  	outStream.d_h2o= (dsd_lo*s2[h2o_index]-dsd_hi*s1[h2o_index])/(dsd_lo-dsd_hi); 
  if(c2h2_index!=-1)
  	outStream.d_c2h2= (dsd_lo*s2[c2h2_index]-dsd_hi*s1[c2h2_index])/(dsd_lo-dsd_hi); 
  if(NO_index!=-1)
  	outStream.d_noxrxnRate= (dsd_lo*s2[NO_index]-dsd_hi*s1[NO_index])/(dsd_lo-dsd_hi); 

}

//****************************************************************************
//Interpolating for the integral to map the LES chi to flamelets chi 
//****************************************************************************

double SteadyFlameletsTable::chitableLookUp(double mixfrac, double mixfracVars)
{

  // nx - mixfrac, ny - mixfracVars
  // Table is uniform in mixfrac and variance direction
  // Computing the index and weighing factors for mixfraction  
  int nx_lo, nx_hi;
  double w_nxlo, w_nxhi;
  nx_lo = (floor)(mixfrac/mixfrac_Div);
  if (nx_lo < 0) 
    nx_lo = 0;
  if (nx_lo > dc_mixfraccount-2)
    nx_lo = dc_mixfraccount-2;
  nx_hi = nx_lo + 1;
  w_nxlo = ((nx_hi*mixfrac_Div)-mixfrac)/mixfrac_Div;
  if (w_nxlo < 0.0)
    w_nxlo = 0.0;
  if (w_nxlo > 1.0)
    w_nxlo = 1.0;
  w_nxhi = 1.0 - w_nxlo;
  // Computing the index and weighing factors for mixfraction variance 
  int ny_lo, ny_hi;
  double w_nylo, w_nyhi;
  ny_lo = (floor)(mixfracVars/mixvar_Div);
  if (ny_lo < 0) 
    ny_lo = 0;
  if (ny_lo > dc_mixvarcount-2)
    ny_lo = dc_mixvarcount-2;
  ny_hi = ny_lo + 1;
  w_nylo = ((ny_hi*mixvar_Div)-mixfracVars)/mixvar_Div;
  if (w_nylo < 0.0)
    w_nylo = 0.0;
  if (w_nylo > 1.0)
    w_nylo = 1.0;
  w_nyhi = 1.0 - w_nylo;
  // Finding the values for interpolation
  double integral_11=chitable[nx_lo*dc_mixvarcount+ny_lo];
  double integral_12=chitable[nx_hi*dc_mixvarcount+ny_lo];
  double integral_21=chitable[nx_lo*dc_mixvarcount+ny_hi];
  double integral_22=chitable[nx_hi*dc_mixvarcount+ny_hi];
  // Interpolation
  double integral=w_nylo*(w_nxlo*integral_11+w_nxhi*integral_12)+w_nyhi*(w_nxlo*integral_21+w_nxhi*integral_22);
  // To fix the low integrals/zeros from the chi table
  if(integral < 1.0e-5)
	  integral=1.0;
  return integral;
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
	cout<<" Unable to open the given input file "<< inputfile <<endl;
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
// Index = -1 means this variable doesn't exist in the table
  co2_index = -1;
  h2o_index = -1;
  c2h2_index = -1;
  NO_index = -1;
  for(int i_vc = 1; i_vc < d_varcount ; i_vc++){
	  fd >> variables_list[i_vc];
          cout<< variables_list[i_vc]<< " ";
          if(variables_list[i_vc]== "CO2")
	  	  co2_index = i_vc;
    	  else if(variables_list[i_vc]== "H2O")
	  	  h2o_index = i_vc;
    	  else if(variables_list[i_vc]== "C2H2")
	  	  c2h2_index = i_vc;
    	  else if(variables_list[i_vc]== "NO")
	  	  NO_index = i_vc;
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
//****************************************************************************
// Reading the Chi table to relate  filtered chi to flamelets chi 
//****************************************************************************

void SteadyFlameletsTable::readChiTable()
{
  ifstream fd2("chiTbl.dat");
  if(fd2.fail()){
        cout<<" Unable to open the chi table file" <<endl;
        exit(1);
  }
  double dum_reader1,dum_reader2;
  //Read the number of mixture fractions and variances from the chi table 
  fd2 >> dc_mixfraccount >>dc_mixvarcount;
  cout << dc_mixfraccount << " " << dc_mixvarcount << endl;
  // Calculating the spacing for uniform mixture fraction and variance 
     mixfrac_Div=1.0/(dc_mixfraccount-1);
     mixvar_Div=1.0/(dc_mixvarcount-1);
  // Allocating the table space 
  int size_chitbl = dc_mixfraccount*dc_mixvarcount;
  chitable = vector <double> (size_chitbl);
  //Reading the data
  for (int jj=0;jj<dc_mixvarcount*dc_mixfraccount; jj++){
			fd2 >> dum_reader1;
			fd2 >> dum_reader2;
                	// Read the integral 
			fd2 >> chitable[jj];
			//cout << chitable[jj]<< " ";
			//cout<<endl;
  }//End of data reading 
  // Closing the file pointer
  fd2.close();
 
}





