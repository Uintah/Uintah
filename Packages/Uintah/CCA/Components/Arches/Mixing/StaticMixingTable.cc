//----- StaticMixingTable.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/StaticMixingTable.h>
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
#include <time.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for StaticMixingTable
//****************************************************************************
StaticMixingTable::StaticMixingTable():MixingModel()
{
	//srand ( time(NULL) );
}

//****************************************************************************
// Destructor
//****************************************************************************
StaticMixingTable::~StaticMixingTable()
{
}

//****************************************************************************
// Problem Setup for StaticMixingTable
//****************************************************************************
void 
StaticMixingTable::problemSetup(const ProblemSpecP& params)
{
  std::string d_inputfile;
  ProblemSpecP db = params->findBlock("StaticMixingTable");
  db->require("adiabatic",d_adiabatic);
  db->require("rxnvars",d_numRxnVars);
  db->require("mixstatvars",d_numMixStatVars);
  db->require("inputfile",d_inputfile);
  // Set up for MixingModel table
  d_numMixingVars = 1;
  // Define mixing table, which includes call reaction model constructor
  d_tableDimension = d_numMixingVars + d_numRxnVars + d_numMixStatVars + !(d_adiabatic);
  readMixingTable(d_inputfile);
  // Moving the index by adding the properties before the species 
  co2_index=co2_index+5;
  h2o_index=h2o_index+5;
  cout<<"CO2 index is " << co2_index<<endl;
  cout<<"H2O index is " << h2o_index<<endl;

}

      

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
StaticMixingTable::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  double small=1.0e-20;
  // Extracting the independent variables from the in stream 
  double mixFrac = inStream.d_mixVars[0];
  double mixFracVars = 0.0;
  if (inStream.d_mixVarVariance.size() != 0)
    mixFracVars = inStream.d_mixVarVariance[0];
  // Heat loss for adiabatic case
  double heat_loss=0.0;
  //Absolute enthalpy
  double enthalpy=0.0;
  // Adiabatic enthalp
  double adia_enthalpy=0.0;
  // Sensible enthalpy
  double sensible_enthalpy=0.0;
  if(!d_adiabatic){
	sensible_enthalpy=tableLookUp(mixFrac, mixFracVars, heat_loss, 4);
	enthalpy=inStream.d_enthalpy;
        adia_enthalpy=tableLookUp(mixFrac, 0.0, heat_loss, 3);
  	heat_loss=(adia_enthalpy-enthalpy)/(sensible_enthalpy+small);
        if(heat_loss > enthalpyLoss[d_enthalpycount-1])
		heat_loss = enthalpyLoss[d_enthalpycount-1];
	else if (heat_loss < enthalpyLoss[0])
		heat_loss = enthalpyLoss[0];
  }
  //heat_loss=-1.0+((double)(rand()%200))/100.0;
  //heat_loss=0.55;
  //mixFrac=((double)(rand()%1000))/1000.0;
  //mixFrac=0.008102625;
  //mixFracVars=4.2261461801809210526315789473683e-4;
  //heat_loss=-1.0+((double)(rand()%200))/100.0;
  //cout<< "Heat loss is :  "<< heat_loss << endl;
  //cout<< "Mixture fraction is :  "<< mixFrac << endl;
  //cout<< "Mixture fraction variance is :  "<< mixFracVars << endl;
  outStream.d_temperature=tableLookUp(mixFrac, mixFracVars, heat_loss, 0);  
  outStream.d_density=tableLookUp(mixFrac, mixFracVars, heat_loss, 1);  
  outStream.d_cp=tableLookUp(mixFrac, mixFracVars, heat_loss, 2);  
  outStream.d_enthalpy=tableLookUp(mixFrac, mixFracVars, heat_loss, 3);  
  outStream.d_co2=tableLookUp(mixFrac, mixFracVars, heat_loss, co2_index);  
  outStream.d_h2o=tableLookUp(mixFrac, mixFracVars, heat_loss, h2o_index);  
  /*  if(outStream.d_temperature < 293.0 || outStream.d_density > 1.21){
  	cout<<"Temperature of the outstream is:  "<<outStream.d_temperature<<endl;
  	cout<<"Density of the outstream is:  "<<outStream.d_density<<endl;
  	cout<<"Mixture fraction is :  "<<mixFrac<<endl;
  	cout<<"Mixture fraction variance is :  "<<mixFracVars<<endl;
  	cout<<"Heat loss is :  "<<heat_loss<<endl;
  }*/
}



double StaticMixingTable::tableLookUp(double mixfrac, double mixfracVars, double heat_loss, int var_index)
{
  double small = 1.0e-20;
  // compute index
  // nx - mixfrac, ny - mixfracVars, var_index - index of the variable
  //cout<<"Mixture fraction is :"<<mixfrac<<endl;
  //mixfrac=0.5;
  if(mixfrac > 1.0)
	mixfrac=1.0;
  else if (mixfrac < 0.0)
	mixfrac=0.0;
  if(mixfracVars < 0.0)
	mixfracVars=0.0;
  double var_limit=mixfracVars/((mixfrac*(1.0-mixfrac))+small);
  if(var_limit > 0.9)
  	mixfracVars=(2.0/3.0)*mixfracVars;
  //mixfracVars=0.0;
  //cout<< "Mixture fraction variance is :"<<mixfracVars<<endl;
  // Enthalpy loss lookup
  double fmg, fpmg,s1,s2,var_value;
  int nhl_lo=0, nhl_hi=0;
  double dhl_lo=0.0, dhl_hi=0.0;
  if(heat_loss == 0.0 && d_enthalpycount ==1){
	nhl_lo=0;
	nhl_hi=0;
        dhl_lo=1.0;
        dhl_hi=0.0;
  }
  else{
	for(int hl_index=0; hl_index < d_enthalpycount-1; hl_index++){
		dhl_lo = enthalpyLoss[hl_index]-heat_loss;
		dhl_hi = enthalpyLoss[hl_index+1]-heat_loss;
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

  // Main loop
  for(int m_index = nhl_lo; m_index <= nhl_hi; m_index++){ 
  int nx_lo, nx_hi;

  //Non-unifrom mixture fraction 
  double df1,df2; 
  for(int index=0; index < d_mixfraccount-1; index++){
	df1 = meanMix[m_index*d_mixfraccount+index][var_index]-mixfrac;
	df2 = meanMix[m_index*d_mixfraccount+index+1][var_index]-mixfrac;
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

 // cout<<"nx_lo index is  : "<<nx_lo<<endl;
  //cout<<"nx_hi index is  : "<<nx_lo<<endl;

  
  // variance is uniform table
  double max_curr_Zvar = mixfrac*(1.0-mixfrac);
  double max_Zvar = min(mixfracVars,max_curr_Zvar);
  double g, gi1, gp, gi2;
  if(meanMix[m_index*d_mixfraccount+nx_lo][var_index]<=small || abs(meanMix[m_index*d_mixfraccount+nx_lo][var_index]-1.0)<=small)
	g=0.0;
  else{
	gi1=max_Zvar*meanMix[m_index*d_mixfraccount+nx_lo][var_index]*(1.0-meanMix[m_index*d_mixfraccount+nx_lo][var_index])/(max_curr_Zvar+small);
        g=gi1*double(d_mixvarcount-1)/(meanMix[m_index*d_mixfraccount+nx_lo][var_index]*(1.0-meanMix[m_index*d_mixfraccount+nx_lo][var_index]));
  }

  if(meanMix[m_index*d_mixfraccount+nx_hi][var_index]<=small || abs(meanMix[m_index*d_mixfraccount+nx_hi][var_index]-1.0)<=small)
	gp=0.0;
  else{
	gi2=max_Zvar*meanMix[m_index*d_mixfraccount+nx_hi][var_index]*(1.0-meanMix[m_index*d_mixfraccount+nx_hi][var_index])/(max_curr_Zvar+small);
        gp=gi2*double(d_mixvarcount-1)/(meanMix[m_index*d_mixfraccount+nx_hi][var_index]*(1.0-meanMix[m_index*d_mixfraccount+nx_hi][var_index]));

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
  fmg=(g-double(k1))*table[m_index*d_mixfraccount*d_mixvarcount+nx_lo*d_mixvarcount+k1p][var_index]-(g-double(k1p))*table[m_index*d_mixfraccount*d_mixvarcount+nx_lo*d_mixvarcount+k1][var_index];
  //cout<< "FMG is :"<<fmg<<endl;
  fpmg=(gp-double(k2))*table[m_index*d_mixfraccount*d_mixvarcount+nx_hi*d_mixvarcount+k2p][var_index]-(gp-double(k2p))*table[m_index*d_mixfraccount*d_mixvarcount+nx_hi*d_mixvarcount+k2][var_index];
  //cout<< "FPMG is : "<<fpmg<<endl;

  if(m_index == nhl_lo)
  	s2=(df1*fpmg-df2*fmg)/(df1-df2);
  else
  	s1=(df1*fpmg-df2*fmg)/(df1-df2);
	
  }
  //cout<<"value of S1 is "<<s1<<endl;
  //cout<<"value of S2 is "<<s2<<endl;

  var_value = (dhl_lo*s2-dhl_hi*s1)/(dhl_lo-dhl_hi);
  return var_value; 
}


void StaticMixingTable::readMixingTable(std::string inputfile)
{
  cerr << "Preparing to read the inputfile:   " << inputfile << endl;
  ifstream fd("nonadeqlb.tbl");
  //ifstream fd(inputfile);
  if(fd.fail()){
	cout<<" Unable to open the given input file" <<endl;
	exit(1);
  }
  fd >> d_enthalpycount >> d_mixfraccount >>d_mixvarcount >> d_speciescount;
  cout << d_enthalpycount << " " << d_mixfraccount << " " << d_mixvarcount << " " << d_speciescount << endl;
  // Total number of variables in the table: Non-adaibatic table has sensibile enthalpy too
  d_varcount=d_speciescount+4+!(d_adiabatic);
  cout<<"d_var count: "<< d_varcount << endl;
  // Intitializing the mixture fraction table: 2-D vector for non-uniform tables
  mixfrac_size=d_enthalpycount*d_mixfraccount;
  meanMix = vector < vector<double> >(mixfrac_size);
  for(int ii=0; ii<mixfrac_size; ii++)
	meanMix[ii]=vector<double>(d_varcount);
  cout<<"After mixture fraction intialization"<<endl;
  // Enthalpy loss vector
  enthalpyLoss=vector<double>(d_enthalpycount);
  // Names of the species
  species_list= vector<string>(d_speciescount);
  // Reading the species names & finding the CO2 & H2O index
  for (int ii = 0; ii < species_list.size(); ii++) {
    fd >> species_list[ii];
    if(species_list[ii]== "CO2")
	    co2_index = ii;
    else if(species_list[ii]== "H2O")
	    h2o_index = ii;
    cout<<species_list[ii]<<endl;
  }
  
  // Allocating the table space 
  int size = d_enthalpycount*d_mixfraccount*d_mixvarcount;
  table = vector <vector <double> > (size);
  for (int ii = 0; ii < size; ii++)
    table[ii] = vector <double> (d_varcount);

  //Reading the data
  // Enthaply loss loop
  for (int mm=0; mm< d_enthalpycount; mm++){
	fd >> enthalpyLoss[mm];
        //cout<<enthalpyLoss[mm]<<endl;
  	// Mixture fraction loop 
  	for (int ii=0; ii< d_mixfraccount; ii++){
        	for (int ll=0; ll < d_varcount; ll++){
 			fd >> meanMix[mm*d_mixfraccount+ii][ll];
                	//cout<< meanMix[mm*d_mixfraccount+ii][ll]<< "  ";
        	}
        	//cout<<endl;
        	// Variance loop
  		for (int jj=0;jj<d_mixvarcount; jj++){
                	// Variables loop 
			for (int kk=0; kk< d_varcount; kk++){
				fd >> table[mm*(d_mixfraccount*d_mixvarcount)+ii*d_mixvarcount+jj][kk];
				//cout << table[mm*(d_mixfraccount*d_mixvarcount)+ii*d_mixvarcount+jj][kk]<< " ";
			}
			//cout<<endl;
		} // End of variance loop
  	}// End of mixture fraction loop
  }//End of enthaly loss loop
  // Closing the file pointer
  fd.close();
  //Printing the mixture fraction table
 /* cout<< "*********Mixture fraction space***************"  <<endl;
  for (int ii=0; ii< d_enthalpycount; ii++){
        cout<<"Enthalpy count: "<<ii<<endl;
	for (int jj=0; jj<d_mixfraccount; jj++){
		for( int kk=0; kk< d_varcount; kk++)
			cout<< meanMix[ii*d_mixfraccount+jj][kk]<< " ";
		cout<<endl; 
        }
  }*/
}





