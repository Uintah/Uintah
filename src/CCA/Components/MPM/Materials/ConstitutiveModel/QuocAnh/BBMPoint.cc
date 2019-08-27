#include "stdafx.h"
#include "BBMPoint.h"

using namespace std;

BBMPoint::BBMPoint(void)
{
	for (int i=0; i<6; i++)
	{
		stress [i]=0;
		strain [i]=0;
		plastic_strain[i]=0;
		MicroStress [i]=0;
		MicroStrain [i]=0;
		MicroPlasticStrain[i]=0;
	}
	strain[6]=0;
	MicroStrain[6]=0;

	for (int i=0; i<3; i++)
	{
		state[i]=0;
		MicroState[i]=0;
	}
	state [0]=1e6; //PStar, unused
	state [1]=1e8; //Yield suction, unused
	state[2]=1.5;  //Specific Volume, unused

}

BBMPoint::~BBMPoint(void)
{
}

void BBMPoint::Copy (BBMPoint* Result)
{
for (int i=0; i<6; i++)
	{
	Result->stress [i]=stress[i];
	Result->strain [i]=strain[i];
	Result->plastic_strain[i]=plastic_strain[i];
	Result->MicroStress[i]=MicroStress[i];
	Result->MicroStrain[i]=MicroStrain[i];
	Result->MicroPlasticStrain[i]=MicroPlasticStrain[i];
	}
Result->strain[6]=strain[6];
Result->MicroStrain[6]=MicroStrain[6];

for (int i=0; i<3; i++)
	{
	Result->state [i]=state[i];
	Result->MicroState [i]=MicroState[i];
	}
}

double BBMPoint::GetPStar ()
{
	return state[0];
}

void BBMPoint::UpdatePStar (double value)
{
	state[0]=value;
}

void BBMPoint::SetSuction (double value)
{
	strain[6]=value;
}

void BBMPoint::SetSpecificVolume (double value)
{
	state[2]=value;
}

void BBMPoint::SetPStar (double value) //does same as UpdatePStar, used for making code more clear
{
	state[0]=value;
}

void BBMPoint::SetYieldSuction (double value)
{
	state[1]=value;
}

double BBMPoint::GetYieldSuction ()
{
	return state[1];
}

double BBMPoint::GetSpecVol ()
{
	return state[2];
}

double BBMPoint::GetSuction ()
{
	return strain[6];
}


double BBMPoint::GetTheta ()
{
	double Theta;
	double Sin3Theta;
	double Third=GetThirdDevInvariant();
	double Second=GetSecondDevInvariant();
	if (Second!=0) Sin3Theta= sqrt(3.0)*(-1.5)*Third/pow(Second,1.5);
	else Sin3Theta=0;
	if (Sin3Theta>1) Sin3Theta=1;
	if (Sin3Theta<-1) Sin3Theta=-1;
	Theta=(asin(Sin3Theta))/3.0;

	return Theta;
}


double BBMPoint::GetThetaDeg ()
{
	double ThetaDeg;
	ThetaDeg=GetTheta();
	ThetaDeg=ThetaDeg*45.0/atan(1.0);
	if (ThetaDeg<0) ThetaDeg+=360;
	if (ThetaDeg>360) ThetaDeg-=360;

	return ThetaDeg;
}

double BBMPoint::GetThetaDeg_0 ()
{
	double ThetaDeg;
	ThetaDeg=GetTheta();
	ThetaDeg=ThetaDeg*45.0/atan(1.0);
	ThetaDeg-=30;
	if (ThetaDeg>360) ThetaDeg-=360;
	if (ThetaDeg<-360) ThetaDeg+=360;

	return ThetaDeg;
}



void BBMPoint::Update (double* PlasticStrainInc, double *Strain_Incr, double *Stress_Incr, double dPZeroStar)
{
	//double dplastic_strain [7];
	//for (int i=0; i<7; i++) dplastic_strain[i]=0  ;  //Lambda is not used
	//***** LINE ABOVE INCORRECT !!!! ************
	double MicroCheck;

	for (int i=0; i<7; i++)
	{
		MicroCheck=strain[i];
		MicroStrain[i]=MicroStrain[i]+Strain_Incr[i];
		strain[i]=strain[i]+MicroStrain[i];
		MicroStrain[i]=MicroStrain[i]-(strain[i]-MicroCheck);

	}

	for (int i=0; i<6; i++)
	{
		MicroCheck=stress[i];
		MicroStress[i]=MicroStress[i]+Stress_Incr[i];
		stress[i]=stress[i]+MicroStress[i];
		MicroStress[i]=MicroStress[i]-(stress[i]-MicroCheck);
	}

	for (int i=0; i<6; i++)
	{
		MicroCheck=plastic_strain[i];
		MicroPlasticStrain[i]=MicroPlasticStrain[i]+PlasticStrainInc[i];
		plastic_strain[i]=plastic_strain[i]+MicroPlasticStrain[i];
		MicroPlasticStrain[i]=MicroPlasticStrain[i]-(plastic_strain[i]-MicroCheck);
	}

	MicroCheck=state[0];
	MicroState[0]=MicroState[0]+dPZeroStar;
	state[0]=state[0]+MicroState[0];
	MicroState[0]=MicroState[0]-(state[0]-MicroCheck);

	MicroCheck=state[2];
	MicroState[2]=MicroState[2]+state[2]*(exp(-(Strain_Incr[0]+Strain_Incr[1]+Strain_Incr[2]))-1);
	//MicroState[2]=MicroState[2]-(Strain_Incr[0]+Strain_Incr[1]+Strain_Incr[2])*state[2];
	state[2]=state[2]+MicroState[2]; //updated specific volume
	MicroState[2]=MicroState[2]-(state[2]-MicroCheck);
}



void BBMPoint::read ()
{
// data will be read from file point.dta
// algorithm to read will be analogical as in bbmmodel.read
ifstream infile("point.dta", ios_base::in);

	// file opened
	string s;
	int slength=0, index=0, line=0;
	//double temp=0;

	do
	{

		getline (infile, s,'\n');
		line++;
		//cout << s <<" Prep: In line no:"<<line<<endl;
		//getch ();
		if (!infile.good())
		{
			cout << "Wrong Data File";
			break;
		}
	}
	while (s!="***Start of data***");
	// I ignore file until "Start of data";

//Reading data - 6 stress+7 strain+ 2 state parameters =15
double storage[15];
		for (int j=0; j<15; j++)
			{

			getline (infile, s,'\n');	//read whole line
			line++;
			//cout << s <<"In line no:"<<line<<endl;
			//getch ();
			bool notcomment=true;
			if (s=="") notcomment=false;
			if ((s[0]=='/')&&(s[1]=='/')) notcomment=false; //check whether not a comment line or empty line

			if (notcomment)
				{
					slength=s.length ();	//get length of line
					index=s.find (";");		//find where is a ; char
					if (index != 0 )
					{
						s.erase (index,slength-index);		// delete all after ;
						storage[j]=atof(s.c_str());				//converse to double
					}
					else cout<<"No ; in line:"<<line<<" May cause errors."<<endl;   //warn about lack of ; in line
				}
			else j--;
			if (!infile.good()) break;
			}

	// Moving data from storage to object variables

	for (int i=0; i<6; i++)
		{
			stress[i]=storage[i];
			strain[i]=storage[i+6];
		}
	strain[6]=storage[12];
	state[0]=storage[13];
	state[1]=storage[14];
	// finished
infile.close(); //close file
// all done

}

double BBMPoint::GetMeanStress ()
{
	return (stress[0]+stress[1]+stress[2])/3;
}

double BBMPoint::GetShearStress ()
{
	double ShearStress;
	ShearStress=(stress[0]-stress[1])*(stress[0]-stress[1])+(stress[0]-stress[2])*(stress[0]-stress[2])+(stress[1]-stress[2])*(stress[1]-stress[2]);
	ShearStress=ShearStress+6*(stress[3]*stress[3]+stress[4]*stress[4]+stress[5]*stress[5]);
	ShearStress=ShearStress/2;
	if (ShearStress<10E-14) ShearStress=0; else ShearStress=sqrt (ShearStress);  //Naylor Pande
	return ShearStress;

}


void BBMPoint::write (){};

double BBMPoint::GetFirstInvariant ()
{
	return (stress[0]+stress[1]+stress[2]);
}

double BBMPoint::GetSecondInvariant ()
{
	double I2=stress[0]*stress[1]+stress[1]*stress[2]+stress[2]*stress[0]-stress[3]*stress[3]-stress[4]*stress[4]-stress[5]*stress[5];
	return I2;
}

double BBMPoint::GetThirdInvariant ()
{
	double I3=stress[0]*stress[1]*stress[2]+2*stress[3]*stress[4]*stress[5]-stress[0]*stress[5]*stress[5]-stress[1]*stress[4]*stress[4]-stress[2]*stress[3]*stress[3];
	return I3;
}

double BBMPoint::GetFirstDevInvariant()
{
return (stress[0]+stress[1]+stress[2])/3.0;
}

double BBMPoint::GetSecondDevInvariant()
{
	double J2;
	J2=(stress[0]-stress[1])*(stress[0]-stress[1])+(stress[0]-stress[2])*(stress[0]-stress[2])+(stress[1]-stress[2])*(stress[1]-stress[2]);
	J2=J2/6.0+(stress[3]*stress[3]+stress[4]*stress[4]+stress[5]*stress[5]);
	return J2;
}

double BBMPoint::GetThirdDevInvariant()
{
	double I1=GetFirstInvariant();
	double I2=GetSecondInvariant();
	double I3=GetThirdInvariant();
	double J3=I1*I1*I1*2.0/27.0-I1*I2/3.0+I3;

	return J3;
}

void BBMPoint::GetEigen(double Eigen[3])
{

BBMMatrix Stress(3,3),EigenValues(3,3),EigenVect(3,3);

Stress.PutElement (1,1,stress[0]);
Stress.PutElement (1,2,stress[3]);
Stress.PutElement (1,3,stress[4]);

Stress.PutElement (2,1,stress[3]);
Stress.PutElement (2,2,stress[1]);
Stress.PutElement (2,3,stress[5]);

Stress.PutElement (3,1,stress[4]);
Stress.PutElement (3,2,stress[5]);
Stress.PutElement (3,3,stress[2]);

Stress.Eigen(&EigenVect,&EigenValues);
Eigen[0]=EigenValues.GetElement(1,1);
Eigen[1]=EigenValues.GetElement(2,2);
Eigen[2]=EigenValues.GetElement(3,3);

}

void BBMPoint::GetEigen(double Eigen[3], BBMMatrix* EigenVectors)
{

BBMMatrix Stress(3,3),EigenValues(3,3);

Stress.PutElement (1,1,stress[0]);
Stress.PutElement (1,2,stress[3]);
Stress.PutElement (1,3,stress[4]);

Stress.PutElement (2,1,stress[3]);
Stress.PutElement (2,2,stress[1]);
Stress.PutElement (2,3,stress[5]);

Stress.PutElement (3,1,stress[4]);
Stress.PutElement (3,2,stress[5]);
Stress.PutElement (3,3,stress[2]);

//cout<<"GetEigen: Stress matrix:"<<endl;
//Stress.Print();

Stress.Eigen(EigenVectors,&EigenValues);
//cout<<"GetEigen: Stress matrix after Eigen procedure:"<<endl;
//Stress.Print();

Eigen[0]=EigenValues.GetElement(1,1);
Eigen[1]=EigenValues.GetElement(2,2);
Eigen[2]=EigenValues.GetElement(3,3);


}

void BBMPoint::GetEigen(BBMMatrix* EigenValues, BBMMatrix* EigenVectors)
{

BBMMatrix Stress(3,3);

Stress.PutElement (1,1,stress[0]);
Stress.PutElement (1,2,stress[3]);
Stress.PutElement (1,3,stress[4]);

Stress.PutElement (2,1,stress[3]);
Stress.PutElement (2,2,stress[1]);
Stress.PutElement (2,3,stress[5]);

Stress.PutElement (3,1,stress[4]);
Stress.PutElement (3,2,stress[5]);
Stress.PutElement (3,3,stress[2]);

Stress.Eigen(EigenVectors,EigenValues);


}

void BBMPoint::SetStressEigen (double Eigen[3])
{
stress[0]=Eigen[0];
stress[1]=Eigen[1];
stress[2]=Eigen[2];
stress[3]=0.0;
stress[4]=0.0;
stress[5]=0.0;
}



bool BBMPoint::CheckIfFinite ()
{
    bool F=true;
    for (int i=0; i<6; i++) if (!isfinite(stress[i]))
    {
      F=false;
      cout<<"Stress["<<i<<"]="<<stress[i]<<endl;
      cout<<"_finite(Stress["<<i<<"])="<<isfinite(stress[i])<<endl;
    }
    for (int i=0; i<7; i++) if (!isfinite(strain[i]))
    {
      F=false;
      cout<<"Strain["<<i<<"]="<<strain[i]<<endl;
      cout<<"_finite(Strain["<<i<<"])="<<isfinite(strain[i])<<endl;
    }
    for (int i=0; i<3; i++) if (!isfinite(state[i]))
    {
      F=false;
      cout<<"State["<<i<<"]="<<state[i]<<endl;
      cout<<"_finite(State["<<i<<"])="<<isfinite(state[i])<<endl;
    }
    if (!F)
    {
    cout<<"Point internal values incorrect."<<endl;
    cout<<"Stress:"<<stress[0]<<" "<<stress[1]<<" "<<stress[2]<<" "<<stress[3]<<" "<<stress[4]<<" "<<stress[5]<<" "<<endl;
    cout<<"Strain:"<<strain[0]<<" "<<strain[1]<<" "<<strain[2]<<" "<<strain[3]<<" "<<strain[4]<<" "<<strain[5]<<" "<<strain[6]<<endl;
    cout<<"State variables:"<<state[0]<<" "<<state[1]<<" "<<state[2]<<endl;
    cout<<"Stress finite:"<<isfinite(stress[0])<<" "<<isfinite(stress[1])<<" "<<isfinite(stress[2])<<" "<<isfinite(stress[3])<<" "<<isfinite(stress[4])<<" "<<isfinite(stress[5])<<" "<<endl;
    cout<<"strain finite:"<<isfinite(strain[0])<<" "<<isfinite(strain[1])<<" "<<isfinite(strain[2])<<" "<<isfinite(strain[3])<<" "<<isfinite(strain[4])<<" "<<isfinite(strain[5])<<" "<<endl;
    cout<<"State variables finite:"<<isfinite(state[0])<<" "<<isfinite(state[1])<<" "<<isfinite(state[2])<<endl;
    }

    return F;
}
