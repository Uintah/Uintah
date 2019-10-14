#include "stdafx.h"
#include "ShengMohrCoulomb.h"

/* this Mohr-Coulomb like model
use a rounded Mohr-Coulomb surface, see
eq 13 in Sheng D, Sloan SW & Yu HS
Computations Mechanics 26:185-196 (2000)
Springer
*/

using namespace std;

extern double MAXITER, ALFACHECK, ALFACHANGE, ALFARATIO, YIELDTOL, INTEGRATION_TOL, DRIFT_CORRECTION, TOL_METHOD, USE_ERROR;
extern double STEP_MAX, STEP_MIN, ERROR_DEF, USE_ERROR_STEP, MIN_DIVISION_SIZE;
extern double BETA_FACT, ADDTOLYIELD, CHGEPSINTOL, PMIN, SUCTIONTOL, TINY;
extern int SOLUTION_ALGORITHM, ALGORITHM_TYPE, USE_NICE_SCHEME;

ShengMohrCoulomb::ShengMohrCoulomb(void)
{

	// all the parameters are initialized
	G=10000;		//shear modulus
	K=20000;		//bulk modulus
	E=9*K*G/(G+3*K);		//Young modulus
	Poisson=(3*K-2*G)/(2*G+6*K);	//Poisson ratio

	//Mohr - Coulomb parameters

	Cohesion=0;
	Phi=3.1415/6.0;
	SinPhi=sin(Phi);
	CosPhi=cos(Phi);


	//if the flow rule is not associated; not expected...
	NonAssociated=false;
	Psi=Phi;
	SinPsi=sin(Psi);
	CosPsi=cos(Psi);
	//Rounded Mohr-Coulomb parameters
	Alpha=(3.0-SinPhi)/(3.0+SinPhi);
	Alpha4=pow(Alpha,4); //Alpha^4;

}

ShengMohrCoulomb::~ShengMohrCoulomb(void)
{
}


void ShengMohrCoulomb::SetModelParameters (double ShearModulusG, double BulkModulusK, double CohesionC, double FrictionAnglePhi)
{
G=ShearModulusG;		//shear modulus
K=BulkModulusK;
E=9*K*G/(G+3*K);
Poisson=(3*K-2*G)/(2*G+6*K);
Cohesion=CohesionC;
Phi=FrictionAnglePhi*3.14159/180.0;
SinPhi=sin(Phi);
CosPhi=cos(Phi);
Alpha=(3.0-SinPhi)/(3.0+SinPhi);
Alpha4=Alpha*Alpha*Alpha*Alpha;
Psi=Phi;
SinPsi=SinPhi;
CosPsi=CosPhi;

}

double ShengMohrCoulomb::CalcStressElast (double nu0, double* s0, double* eps0,  double* deps,  double* ds)
{
double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;


ds[0]=K43G*deps[0]+K23G*(deps[1]+deps[2]);
ds[1]=K43G*deps[1]+K23G*(deps[0]+deps[2]);
ds[2]=K43G*deps[2]+K23G*(deps[0]+deps[1]);
ds[3]=2*G*deps[3];
ds[4]=2*G*deps[4];
ds[5]=2*G*deps[5];


return (0); //it is helpful not to calculate it each time...
// That's all...

}

double ShengMohrCoulomb::CalcElastic (double * Strain, BBMPoint * InitialPoint, BBMPoint * FinalPoint)
{
	double Stress[6], Zeros[6];// , InitialSuction;

for (int i=0; i<6; i++)
{
	Zeros[i]=0;
}

//InitialSuction=InitialPoint->GetSuction();

CalcStressElast (InitialPoint->GetSpecVol(), InitialPoint->stress, InitialPoint->strain,  Strain,  Stress);
//for (int i=0; i<6; i++) cout<<"Stress increment is:"<<Stress[i]<<endl;
InitialPoint->Copy(FinalPoint);
FinalPoint->Update(Zeros, Strain, Stress, 0);
return 0;
}


void ShengMohrCoulomb::CalcStressElastM (double* deps, double* ds)
{
/*
This procedure is just to check how whether the stress increment was calculated properly.
Detailed algorithm is in file with all descriptions.
*/
  //used in final calculation K+4/3G, K-2/3G...

// Calculate helpful variables:
double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;


// Calculate stress state:
ds[0]=K43G*deps[0]+K23G*(deps[1]+deps[2]);
ds[1]=K43G*deps[1]+K23G*(deps[0]+deps[2]);
ds[2]=K43G*deps[2]+K23G*(deps[0]+deps[1]);
ds[3]=2*G*deps[3];
ds[4]=2*G*deps[4];
ds[5]=2*G*deps[5];

//for (int i=0; i<6; i++) cout<<"Delta stress i="<<ds[i]<<endl;
}

void ShengMohrCoulomb::FindElStrGrad (double nu0, double* s0  , double* eps0, double* deps,  double* ds)
{
//to be easily used
double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;



	ds[0]=K43G*deps[0]+K23G*(deps[1]+deps[2]);
	ds[1]=K43G*deps[1]+K23G*(deps[0]+deps[2]);
	ds[2]=K43G*deps[2]+K23G*(deps[0]+deps[1]);
	ds[3]=2*G*deps[3];
	ds[4]=2*G*deps[4];
	ds[5]=2*G*deps[5];


	//for (int i=0; i<6; i++) cout<<"Delta stress i="<<ds[i]<<endl;
}

bool ShengMohrCoulomb::CheckYield (BBMPoint * Point)
{
	//check of the standard yield surface
	//Note: value of function is not normalised by the mean stress

	double MeanStress=Point->GetMeanStress();
	double ShearStress=Point->GetShearStress();
	double J3=Point->GetThirdDevInvariant();
	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress>TINY)
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
	    if (Factor<-1) Factor=-1;
	    if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CheckYield not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;

		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	else M=M*Alpha/pow(0.5*(1+Alpha4),0.25);

	double Value=ShearStress/M-2*Cohesion/M-MeanStress;
/*
    cout<<"Check Yield: Mean Stress="<<MeanStress;
	cout<<" Shear Stress="<<ShearStress;
	cout<<" Cohesion="<<Cohesion;
	cout<<" M="<<M;
	cout<<" Yield Function="<<Value<<endl;
*/
	if (Value>(-YIELDTOL)) return true;
	else return false;
}

bool ShengMohrCoulomb::CheckYieldNormalised (BBMPoint * Point)
{
	//check of the standard yield surface
	//Note: value of function is not normalised by the mean stress

	double MeanStress=Point->GetMeanStress();
	double ShearStress=Point->GetShearStress();
	double J3=Point->GetThirdDevInvariant();
	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress>TINY)
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
	    if (Factor<-1) Factor=-1;
	    if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CheckYield Normalised not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}

		Factor=1+Alpha4-(1-Alpha4)*Factor;

		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	else M=M*Alpha/pow(0.5*(1+Alpha4),0.25);

	double Value=ShearStress/M-2*Cohesion/M-MeanStress;

    //normalisation
	Value=Value/(fabs(MeanStress)+2*Cohesion);
/*
    cout<<"Check Yield: Mean Stress="<<MeanStress;
	cout<<" Shear Stress="<<ShearStress;
	cout<<" Cohesion="<<Cohesion;
	cout<<" M="<<M;
	cout<<" Yield Function="<<Value<<endl;
*/
	if (Value>(-YIELDTOL)) return true;
	else return false;
}



void ShengMohrCoulomb::CheckYieldNormalised (double *state, double *s, double suction, double *FValue)
{
	//check of the standard yield surface
	//Note: value of function is normalised by the mean stress+2*Cohesion

/*
	Purpose of this routine is to calculate value of yield function to determine, whether we have yielding
	or not.

	Arguments: *state - table of state parameters; in fact we need only p*.
				s- stress state
				suction - suction
	Returns: none;  value of normalized function is saved in FValue
	*/

	double MeanStress, ShearStress, Factor;

	double I1=s[0]+s[1]+s[2];
	double I2=s[0]*s[1]+s[1]*s[2]+s[2]*s[0]-s[3]*s[3]-s[4]*s[4]-s[5]*s[5];
	double I3=s[0]*s[1]*s[2]+2*s[3]*s[4]*s[5]-s[0]*s[5]*s[5]-s[1]*s[4]*s[4]-s[2]*s[3]*s[3];
	double J3=I1*I1*I1*2.0/27.0-I1*I2/3.0+I3;


	MeanStress=(s[0]+s[1]+s[2])/3;
	ShearStress=(s[0]-s[1])*(s[0]-s[1])+(s[0]-s[2])*(s[0]-s[2])+(s[1]-s[2])*(s[1]-s[2]);
	ShearStress=ShearStress+6*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
	ShearStress=ShearStress/2;
	if (ShearStress*ShearStress*ShearStress<TINY) ShearStress=0; else ShearStress=sqrt (ShearStress);  //Naylor Pande

	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress<TINY) M=M*Alpha/pow(0.5*(1+Alpha4),0.25); else
	{
		Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
		if (Factor<-1) Factor=-1;
		if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CheckYieldNormalised [parameters s etc] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;

		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	*FValue=ShearStress/M-2*Cohesion/M-MeanStress;
//	cout<<*FValue<<"  mean stress + cohesion  "<<MeanStress+2*Cohesion<<endl;
	if (!finite (*FValue))
	{
		cout<<"Yield locus value is not finite"<<endl;
		cout<<"s="<<s[0]<<' '<<s[1]<<' '<<s[2]<<' '<<s[3]<<' '<<s[4]<<' '<<s[5]<<' '<<endl;
		cout<<"I1="<<I1<<" I2="<<I2<<" I3="<<I3<<" J3="<<J3<<endl;
		cout<<"ShearStress="<<ShearStress<<" MeanStress="<<MeanStress<<" Cohesion="<<Cohesion<<endl;
		cout<<"M="<<M<<" Factor="<<Factor<<"Alpha="<<Alpha<<endl;
	}

	//normalisation
	*FValue=*FValue/(fabs(MeanStress)+2*Cohesion);
//	cout<<*FValue<<endl;
/*
    cout<<"Check Yield: Mean Stress="<<MeanStress;
	cout<<" Shear Stress="<<ShearStress;
	cout<<" Cohesion="<<Cohesion;
	cout<<" M="<<M;
	cout<<" Yield Function="<<Value<<endl;
*/

}



double ShengMohrCoulomb::ComputeYieldFunction(BBMPoint * Point)
{
//Note: value of function is not normalised by the mean stress+2*cohesion!

	double MeanStress=Point->GetMeanStress();
	double ShearStress=Point->GetShearStress();
	double J3=Point->GetThirdDevInvariant();
	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress>TINY)
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
		if (Factor<-1) Factor=-1;
		if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in ComputeYieldFunction [parameters *Point] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;
		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	else M=M*Alpha/pow(0.5*(1+Alpha4),0.25);

	double Value=ShearStress/M-2*Cohesion/M-MeanStress;

	//normalisation
	//Value=Value/(MeanStress+2*Cohesion);

	return Value;

}

double ShengMohrCoulomb::ComputeYieldFunctionNN(BBMPoint * Point)
{
//Note: value of function is normalised by the mean stress+2*cohesion!

	double MeanStress=Point->GetMeanStress();
	double ShearStress=Point->GetShearStress();
	double J3=Point->GetThirdDevInvariant();
	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress>TINY)
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
		if (Factor<-1) Factor=-1;
		if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in ComputeYieldFunctionNN [parameters *Point] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;
		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	else M=M*Alpha/pow(0.5*(1+Alpha4),0.25);

	double Value=ShearStress/M-2*Cohesion/M-MeanStress;

	return Value;

}



bool ShengMohrCoulomb::CheckIfPlastic (BBMPoint * Point)
{
//Note: value of function is normalised by the mean stress+2*cohesion!

	double MeanStress=Point->GetMeanStress();
	double ShearStress=Point->GetShearStress();
	double J3=Point->GetThirdDevInvariant();
	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress>TINY)
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
		if (Factor<-1) Factor=-1;
		if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CheckIfPlastic [parameters *Point] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;
		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	else
	M=M*Alpha/pow(0.5*(1+Alpha4),0.25);

	double Value=ShearStress/M-2*Cohesion/M-MeanStress;

	//normalisation
	Value=Value/(fabs(MeanStress)+2*Cohesion);

		if (Value>(-YIELDTOL)) return true;
	else return false;

}



void ShengMohrCoulomb::CheckYield (double *state, double *s, double suction, double *FValue)
{
	/*
	Purpose of this routine is to calculate value of yield function to determine, whether we have yielding
	or not.

	Arguments: *state - table of state parameters; in fact we need only p*.
				s- stress state
				suction - suction
	Returns: none;  value of normalized function is saved in FValue
	*/

	double MeanStress, ShearStress;

	double I1=s[0]+s[1]+s[2];
	double I2=s[0]*s[1]+s[1]*s[2]+s[2]*s[0]-s[3]*s[3]-s[4]*s[4]-s[5]*s[5];
	double I3=s[0]*s[1]*s[2]+2*s[3]*s[4]*s[5]-s[0]*s[5]*s[5]-s[1]*s[4]*s[4]-s[2]*s[3]*s[3];
	double J3=I1*I1*I1*2.0/27.0-I1*I2/3.0+I3;


	MeanStress=(s[0]+s[1]+s[2])/3;
	ShearStress=(s[0]-s[1])*(s[0]-s[1])+(s[0]-s[2])*(s[0]-s[2])+(s[1]-s[2])*(s[1]-s[2]);
	ShearStress=ShearStress+6*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
	ShearStress=ShearStress/2;
	if (ShearStress<TINY) ShearStress=0; else ShearStress=sqrt (ShearStress);  //Naylor Pande

	double M=(6*SinPhi)/(3-SinPhi);
	if (ShearStress<TINY) M=M*Alpha/pow(0.5*(1+Alpha4),0.25); else
	{
		double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
		if (Factor>1) Factor=1;
		if (Factor<-1) Factor=-1;
		if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CheckYield [parameters s etc] is not finite. Set to 1"<<endl;
		     cout<<"s="<<s[0]<<" "<<s[1]<<" "<<s[2]<<" "<<s[3]<<" "<<s[4]<<" "<<s[5]<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
		Factor=1+Alpha4-(1-Alpha4)*Factor;
		if ((Alpha/pow(0.5*Factor,0.25)<1.0)&&(Alpha/pow(0.5*Factor,0.25)>-1.0)) M=M*Alpha/pow(0.5*Factor,0.25);
	}
	*FValue=ShearStress/M-2*Cohesion/M-MeanStress;
//	cout<<*FValue<<"  mean stress + cohesion  "<<MeanStress+2*Cohesion<<endl;
	if (!finite (*FValue))
	{
		cout<<"Yield locus value is not finite"<<endl;
	}
	//normalisation
	*FValue=*FValue/(fabs(MeanStress)+2*Cohesion);
//	cout<<*FValue<<endl;
}

double ShengMohrCoulomb::ComputeNu (double *s,double *state, double suction)
{
// does nothing for SMC
return 1;
}



int ShengMohrCoulomb::CalcPlasticFaster (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
CalcPlastic (Point, EPStrain, DSIGMA, PlasticStrain, DPZeroStar, FValue, dS, dLambda);
return 0;
//so we had the increase of strenghtening parameter...
//that's all?
}



int ShengMohrCoulomb::CalcPlasticPQ (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
CalcPlastic (Point, EPStrain, DSIGMA, PlasticStrain, DPZeroStar, FValue, dS, dLambda);
return 0;
//so we had the increase of strenghtening parameter...
//that's all?
}





int ShengMohrCoulomb::CalcPlastic (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)
BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
BBMMatrix DEL (6,6); //D elastic matrix...
BBMMatrix DEPS (6,1);

for (int i=1; i<6; i++) DEPS.PutElement (i,1,EPStrain[i-1]);  //increase of epsilon copied
// we do not have g, as a == g in associated flow rule.
if (!Point.CheckIfFinite())
    {
        cout<<"Error in the CalcPlastic Procedure. Point internal values are not finite. Press any key."<<endl;
        getchar();
    }



double I1=Point.GetFirstInvariant();
double I2=Point.GetSecondInvariant();
double J2=Point.GetSecondDevInvariant ();
if (fabs(J2)<TINY) J2=TINY;
double J3=Point.GetThirdDevInvariant ();
double ShearStress=Point.GetShearStress ();
if (fabs(ShearStress)<TINY) ShearStress=TINY;
double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
if (Factor>1) Factor=1;
if (Factor<-1) Factor=-1;
if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in CalcPlastic is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
Factor=1+Alpha4-(1-Alpha4)*Factor;
double Factor025=pow(Factor,0.25);
double Factor075=pow(Factor,-0.75);
double M=(3-SinPhi)/(Alpha*SinPhi); //not M but useful in derivatives, see file
double MPsi=(3-SinPsi)/(Alpha*SinPsi);



BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);

//derivatives of the invariants
double s[6]={Point.stress[0],Point.stress[1],Point.stress[2],Point.stress[3],Point.stress[4],Point.stress[5]};

dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (2,1,1.0);dI1dSIG.PutElement (3,1,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (2,1,s[0]+s[2]);dI2dSIG.PutElement (3,1,s[0]+s[1]);
dI2dSIG.PutElement (4,1,-2*s[3]);dI2dSIG.PutElement (5,1,-2*s[4]);dI2dSIG.PutElement (6,1,-2*s[5]);

//dI2dSIG.PrintPrecise();

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (2,1,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (3,1,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (4,1,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (5,1,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (6,1,2*s[3]*s[4]-2*s[0]*s[5]);

//dI3dSIG.PrintPrecise();

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (2,1,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (3,1,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (4,1,2*s[3]);dJ2dSIG.PutElement (5,1,2*s[4]);dJ2dSIG.PutElement (6,1,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(2.0/9.0*I1*I1-I2/3.0,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)




dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
        cout<<"CalcPlastic(1) DF/dSigma not finite at A("<<i<<"). Set to 1.0. Results may be incorrect. Press any key."<<endl;
	    getchar();
		A.PutElement(i,1,0.0);
	}
}

//cout<<"dF/dJ2 part"<<endl;
//A.PrintPrecise();
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dJ3 part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
	    cout<<"CalcPlastic(2) DF/dSigma not finite at A("<<i<<"). Set to 1.0. Results may be incorrect. Press any key."<<endl;
	    getchar();
		A.PutElement(i,1,0.0);
	}
}

dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dq part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
        cout<<"CalcPlastic(3) DF/dSigma not finite at A("<<i<<"). Set to 1.0. Results may be incorrect. Press any key."<<endl;
	    getchar();
		A.PutElement(i,1,0.0);
	}
}

/*A.PrintPrecise ();
//FINISHED dF/dSigma

BBMPoint CopyPoint;
Point.Copy(&CopyPoint);
double Yield1,Yield2,dSs=0.0001;
Yield1=ComputeYieldFunction(&CopyPoint);
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;
*/


dJ2dSIG.Copy(&GG);
GG.Multiply (0.21022410391343*MPsi*Factor025/ShearStress,&GG);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*MPsi*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*MPsi*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*MPsi*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*MPsi*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);  //First correction: sign here, but most likely not enough!!!
GG.Add (&TEMP,&GG);

//FINISHED dQ/dSigma

double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;

DEL.PutElement (1,1,K43G);
DEL.PutElement (1,2,K23G);
DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
DEL.PutElement (2,1,K23G);
DEL.PutElement (2,2,K43G);
DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
DEL.PutElement (3,2,K23G);
DEL.PutElement (3,3,K43G);
DEL.PutElement (4,4,2.0*G);
DEL.PutElement (5,5,2.0*G);
DEL.PutElement (6,6,2.0*G); //rest of the matrix is filled with zeros...

//Getting lambda and Dep

A.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&DEL,&NUMERATOR); //NUMERATOR=aT*Del -->Numerator of Lambda without multiplication by dEpsilon
NUMERATOR.Multiply(&GG,&DENOMINATOR);
NUMERATOR.Multiply(&DEPS,&NUMERATOR);//NUMERATOR=aT*Del*dEps -->Numerator of Lambda without multiplication by dEpsilon

if (USE_NICE_SCHEME>0)
	{
		//NUMERATOR.PutElement(1,1,NUMERATOR.GetElement(1,1)+ComputeYieldFunction(&Point));
		NUMERATOR.PutElement(1,1,NUMERATOR.GetElement(1,1)+FValue);
		NUMERATOR.PutElement(1,1,NUMERATOR.GetElement(1,1)+(*dLambda)*DENOMINATOR.GetElement(1,1));
		*dLambda=FValue/DENOMINATOR.GetElement(1,1);
		DEL.Multiply(&GG,&TEMP);
		TEMP.Multiply(*dLambda,&TEMP);
		dS[0]=-TEMP.GetElement(1,1);
		dS[1]=-TEMP.GetElement(2,1);
		dS[2]=-TEMP.GetElement(3,1);
		dS[3]=-TEMP.GetElement(4,1);
		dS[4]=-TEMP.GetElement(5,1);
		dS[5]=-TEMP.GetElement(6,1);
	}



DEL.Multiply(&GG,&TEMP);
TEMP.Multiply(&NUMERATOR,&TEMP);
if (DENOMINATOR.GetElement(1,1)<TINY)
 cout<<"Denominator of plastic multiplier is very small. Some error may arise and results may be incorrect"<<endl;

TEMP.Multiply(1/DENOMINATOR.GetElement(1,1),&TEMP);
DEL.Multiply(&DEPS,DSIGMA);
DSIGMA->Substract(&TEMP,DSIGMA);
for (int i=1; i<7 ; i++)
{
	if (!finite (DSIGMA->GetElement(i,1)))
	{
	    cout<<"CalcPlastic: DSIGMA not finite: set to zero. Results may be erronous."<<endl;
		DSIGMA->PutElement(i,1,0.0);
	}
}
*DPZeroStar=1.0;
//Stress increment computed and in DSIGMA matrix


return 0;

//that's all?
}

void ShengMohrCoulomb::GetTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
{
/*
This procedure returns tangent matrix. If given point lies on the yield locus, the matrix is elastoplastic.
When the point lies inside the yield locus, the matrix is elastic.
Point is supposed to have all the required data about describing its state, like stress state, specific volume nu and the
hardening parameter p0*.

Input: Point, Matrix (7x6 - six 6 rows, seven 7 columns) that will be filled with coefficients corresponding to the tangent matrix
Initialise BBMMatrix (6,7);
*/

DEP->Resize(6,7);

if (CheckYield(Point))
	{
		//Calculate elasto-plastic matrix
		CalculateElastoPlasticTangentMatrix (Point, DEP);
		//DEP->Print();

		//CalculateElasticTangentMatrix (Point, DEP);
		//cout<<"Elasto-Plastic Matrix used"<<endl;
	}
else
	{
		//Calculate elastic matrix
		CalculateElasticTangentMatrix (Point, DEP);
	}

}


void ShengMohrCoulomb::GetTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP)
{
cout<<"GetTangentMatrixPQ is unsupported for Mohr-Coulomb Model"<<endl;
getchar();
//CalculateElastoPlasticTangentMatrix (Point,DEP);
}

void ShengMohrCoulomb::CalculateElasticTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP)
{
cout<<"CalculateElasticTangentMatrixPQ is unsupported for Mohr-Coulomb Model"<<endl;
getchar();
//CalculateElasticTangentMatrix (Point, DEP);
}



void ShengMohrCoulomb::CalculateElastoPlasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
{
/*************************************
Here the ElastoPlastic tangent matrix is calculated.
For more theoretical explanation what is being done below please see
'Evaluation of the tangent elastic matrix Del & elastoplastic matrix Dep'
(evaluation of the tangent matrix D.doc)

This procedure is quite similar to procedure CalcPlastic where the stress increment using
the tangent elasto-plastic matrix is calculated. The matrix itself, however, is not explicitly
calculated there.


	// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)
BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
BBMMatrix DEL (6,6); //D elastic matrix...

// we do not have g, as a == g in associated flow rule.

double I1=Point->GetFirstInvariant();
double I2=Point->GetSecondInvariant();
double J2=Point->GetSecondDevInvariant ();
double J3=Point->GetThirdDevInvariant ();
double ShearStress=Point->GetShearStress ();
double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
if (Factor>1) Factor=1;
if (Factor<-1) Factor=-1;
if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in Calculate Elasto-Plastic Tangent Matrix is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
Factor=1+Alpha4-(1-Alpha4)*Factor;
double Factor025=pow(Factor,0.25);
double Factor075=pow(Factor,-0.75);
double M=(3-SinPhi)/(6*Alpha*SinPhi);
double MPsi=(3-SinPsi)/(6*Alpha*SinPsi);



BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);

//derivatives of the invariants
double s[6]={Point->stress[0],Point->stress[1],Point->stress[2],Point->stress[3],Point->stress[4],Point->stress[5]};

dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (2,1,1.0);dI1dSIG.PutElement (3,1,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (2,1,s[0]+s[2]);dI2dSIG.PutElement (3,1,s[0]+s[1]);
dI2dSIG.PutElement (4,1,-2*s[3]);dI2dSIG.PutElement (5,1,-2*s[4]);dI2dSIG.PutElement (6,1,-2*s[5]);

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (2,1,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (3,1,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (4,1,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (5,1,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (6,1,2*s[3]*s[4]-2*s[0]*s[5]);

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (2,1,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (3,1,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (4,1,2*s[3]);dJ2dSIG.PutElement (5,1,2*s[4]);dJ2dSIG.PutElement (6,1,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(6.0/27.0*I1*I1-I2/3.0,&dJ3dSIG);
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)

dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)-
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)-
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);

//FINISHED dF/dSigma

//FINISHED dF/dSigma

dJ2dSIG.Copy(&GG);
GG.Multiply (0.21022410391343*MPsi*Factor025/ShearStress,&GG);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (1.419012700740643*(Alpha4-1)*MPsi*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)-
			   0.94600846716043*(Alpha4-1)*Cohesion*MPsi*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (2.838025401481287*(Alpha4-1)*Cohesion*MPsi*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)-
			   4.257038102221929*(Alpha4-1)*MPsi*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);  //First correction: sign here, but most likely not enough!!!
GG.Add (&TEMP,&GG);

//FINISHED dQ/dSigma

double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;

DEL.PutElement (1,1,K43G);
DEL.PutElement (1,2,K23G);
DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
DEL.PutElement (2,1,K23G);
DEL.PutElement (2,2,K43G);
DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
DEL.PutElement (3,2,K23G);
DEL.PutElement (3,3,K43G);
DEL.PutElement (4,4,2.0*G);
DEL.PutElement (5,5,2.0*G);
DEL.PutElement (6,6,2.0*G); //rest of the matrix is filled with zeros...

//Getting lambda and Dep

A.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&DEL,&NUMERATOR); //NUMERATOR=aT*Del -->Numerator of Lambda without multiplication by dEpsilon
NUMERATOR.Multiply(&GG,&DENOMINATOR);

DEL.Multiply(&GG,&TEMP);
TEMP.Multiply(&NUMERATOR,&TEMP);
TEMP.Multiply(1/DENOMINATOR.GetElement(1,1),&TEMP);
DEL.Substract(&TEMP,&DEL);
for (int i=1;i<7; i++) for (int j=1; j<7; j++) DEP->PutElement(i,j,DEL.GetElement(i,j));
//so if 6x7 is needed, it is preserved for compatibility purposes

//Stress increment computed and in DSIGMA matrix

//DEP->Print();
//getchar(); */
//FINAL MATRIX PUT TOGETHER, 6 rows, 7 columns
}

void ShengMohrCoulomb::CalculateElasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
{

double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;

DEP->PutElement (1,1,K43G);
DEP->PutElement (1,2,K23G);
DEP->PutElement (1,3,K23G); //rest of the line are zeros and rightly so
DEP->PutElement (2,1,K23G);
DEP->PutElement (2,2,K43G);
DEP->PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
DEP->PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
DEP->PutElement (3,2,K23G);
DEP->PutElement (3,3,K43G);
DEP->PutElement (4,4,2.0*G);
DEP->PutElement (5,5,2.0*G);
DEP->PutElement (6,6,2.0*G); //rest of the matrix is filled with zeros...

}


double ShengMohrCoulomb::FindGradient (double * state, double * s, double *ds, double * dF, double suction, double dsuction)
{

/*

//to be amended later on
// HACK HACK HACK
This procedure finds a gradient to the yield locus at given point.
It is used later to determine whether we have partly elastic unloading or only plastic step
The gradient may be found only when we are on "main" plastic yield locus; Otherwise results may be erroneous.

Parameters: state[], s[] (stress), ds[] (stress increment), suction,

Gradient = gradient[1], gradient[2] etc - there the result will be stored

Firs the C(s) part must be calculated, then the gradient.

returns cosinus of the angle...

*/

BBMMatrix A (6,1); //dF/dsigma


double I1=s[0]+s[1]+s[2];

double I2=s[0]*s[1]+s[1]*s[2]+s[2]*s[0]-s[3]*s[3]-s[4]*s[4]-s[5]*s[5];

double I3=s[0]*s[1]*s[2]+2*s[3]*s[4]*s[5]-s[0]*s[5]*s[5]-s[1]*s[4]*s[4]-s[2]*s[3]*s[3];

double J2=(s[0]-s[1])*(s[0]-s[1])+(s[0]-s[2])*(s[0]-s[2])+(s[1]-s[2])*(s[1]-s[2]);
J2=J2/6.0+(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);

double J3=I1*I1*I1*2.0/27.0-I1*I2/3.0+I3;



double ShearStress=sqrt(3.0*J2);
double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
if (Factor>1) Factor=1;
if (Factor<-1) Factor=-1;
if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in FindGradient [parameters *Point] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
Factor=1+Alpha4-(1-Alpha4)*Factor;
double Factor025=pow(Factor,0.25);
double Factor075=pow(Factor,-0.75);
double M=(3-SinPhi)/(6*Alpha*SinPhi);
//double MPsi=(3-SinPsi)/(6*Alpha*SinPsi);


BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1);



dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (2,1,1.0);dI1dSIG.PutElement (3,1,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (2,1,s[0]+s[2]);dI2dSIG.PutElement (3,1,s[0]+s[1]);
dI2dSIG.PutElement (4,1,-2*s[3]);dI2dSIG.PutElement (5,1,-2*s[4]);dI2dSIG.PutElement (6,1,-2*s[5]);

//dI2dSIG.PrintPrecise();

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (2,1,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (3,1,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (4,1,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (5,1,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (6,1,2*s[3]*s[4]-2*s[0]*s[5]);

//dI3dSIG.PrintPrecise();

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (2,1,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (3,1,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (4,1,2*s[3]);dJ2dSIG.PutElement (5,1,2*s[4]);dJ2dSIG.PutElement (6,1,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(2.0/9.0*I1*I1-I2/3.0,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)




dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,0.0);
	}
}

//cout<<"dF/dJ2 part"<<endl;
//A.PrintPrecise();
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dJ3 part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,0.0);
	}
}

dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dq part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,0.0);
	}
}

for (int i=0; i<6; i++) dF[i]=A.GetElementZ(i,1);

double dFlength=0, dslength=0, cosin=0;

for (int i=0; i<6; i++)
	{
		dFlength=dFlength+dF[i]*dF[i]; //calculated length
		dslength=dslength+ds[i]*ds[i];
	}

	dslength=sqrt(dslength);
	dFlength=sqrt(dFlength);

	for (int i=0; i<6; i++)
	{

	 //calculated lenght - gradient - total length of vector ==1#
		//cout<<"dF["<<i<<"]="<<dF[i]<<endl;
		cosin=cosin+dF[i]*ds[i]/(dslength*dFlength); //calculated cosinus of the thetha angle...
	}
	if (cosin<-YIELDTOL)
	{
	//cout<<"Check if no problem in the FindGradient"<<endl;
	//cout<<"cosin="<<cosin<<endl;
	//getchar();

	}


	return cosin; //cosin returned...


//derivatives of the invariants

/*

dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (1,2,1.0);dI1dSIG.PutElement (1,3,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (1,2,s[0]+s[2]);dI2dSIG.PutElement (1,3,s[0]+s[1]);
dI2dSIG.PutElement (1,4,-2*s[3]);dI2dSIG.PutElement (1,5,-2*s[4]);dI2dSIG.PutElement (1,6,-2*s[5]);

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (1,2,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (1,3,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (1,4,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (1,5,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (1,6,2*s[3]*s[4]-2*s[0]*s[5]);

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (1,2,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (1,3,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (1,4,2*s[3]);dJ2dSIG.PutElement (1,5,2*s[4]);dJ2dSIG.PutElement (1,6,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(6.0/27.0*I1*I1-I2/3.0,&dJ3dSIG);
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)

dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)-
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)-
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);

for (int i=0; i<6; i++) dF[i]=A.GetElementZ(i,1);

double dFlength=0, dslength=0, cosin=0;

for (int i=0; i<6; i++)
	{
		dFlength=dFlength+dF[i]*dF[i]; //calculated length
		dslength=dslength+ds[i]*ds[i];
	}

	dslength=sqrt(dslength);
	dFlength=sqrt(dFlength);

	for (int i=0; i<6; i++)
	{

	 //calculated lenght - gradient - total length of vector ==1#
		//cout<<"dF["<<i<<"]="<<dF[i]<<endl;
		cosin=cosin+dF[i]*ds[i]/(dslength*dFlength); //calculated cosinus of the thetha angle...
	}
	cout<<"Check if no problem in the FindGradient"<<endl;
	getchar();
	return cosin; //cosin returned... */


//in A is dF/dSigma

/*


double PZero, SuctionPressure, MeanStress, ShearStress, FValue, dFlength, dslength, cosin;

//check of the second yield surface:
	if (state[1]-suction<-SUCTIONTOL)
	{
		cout<<"yield of the suction surface occurred"<<endl;
		if (dsuction>0)
		{
			cout <<"Unable to find gradient; Whole step plastic"<<endl;
			return -2;
		//this check may be unused...
		}
	}
//check of the standard yield surface
	MeanStress=(s[0]+s[1]+s[2])/3;
	ShearStress=(s[0]-s[1])*(s[0]-s[1])+(s[0]-s[2])*(s[0]-s[2])+(s[1]-s[2])*(s[1]-s[2]);
	ShearStress=ShearStress+6*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
	ShearStress=ShearStress/2;
	//ShearStress=sqrt (ShearStress);  //Naylor Pande

	PZero=LambdaZero*((1-r)*exp(-1*Beta*suction)+r);
	PZero=(LambdaZero-KappaP)/(PZero-KappaP);
	PZero=pc*pow((state[0]/pc),PZero);
	SuctionPressure=k*suction;
	FValue=ShearStress-M*M*(MeanStress+SuctionPressure)*(PZero-MeanStress);
	FValue=FValue/((PZero+SuctionPressure)*(PZero+SuctionPressure)); //value of Yield function calculated and normalized
	if (FValue<-YIELDTOL)
	{
		cout<<"there is no yield at the beginning !!!"<<endl;
		cout<<"find gradient procedure terminated!!!"<<endl;
		return -3;
		// this check may be disabled later
		//THIS CHECK LEADS TO ERROR AS WE ARE PLAYING WITH THE TOLERANCES...
	}
	dF[0]=2*s[0]-s[1]-s[2]-M*M/3*(PZero-SuctionPressure-2/3*MeanStress);
	dF[1]=2*s[1]-s[0]-s[2]-M*M/3*(PZero-SuctionPressure-2/3*MeanStress);
	dF[2]=2*s[2]-s[0]-s[1]-M*M/3*(PZero-SuctionPressure-2/3*MeanStress);
	dF[3]=6*s[3];
	dF[4]=6*s[4];
	dF[5]=6*s[5];
	dFlength=0;
	dslength=0;
	cosin=0;

	double P0Star=state[0];
	double denominator=LambdaZero*((1-r)*exp(-1*Beta*suction)+r);
	double numerator=LambdaZero-KappaP;

	dF[6]=-1*Beta*(1-r)*exp(-1*Beta*suction);
	dF[6]=dF[6]*-1*numerator/(denominator*denominator);
	dF[6]=dF[6]*pc*log(P0Star/pc);
	dF[6]=dF[6]*pow((P0Star/pc),numerator/denominator); //it's finished with dp0/ds
	dF[6]=M*M*(MeanStress+SuctionPressure)*dF[6];
	dF[6]=-1*M*M*k*(PZero-MeanStress)-dF[6];



	for (int i=0; i<6; i++)
	{
		dFlength=dFlength+dF[i]*dF[i]; //calculated length
		dslength=dslength+ds[i]*ds[i];
	}
	dFlength=dFlength+dF[6]*dF[6];
	dslength=dslength+dsuction*dsuction;
	dslength=sqrt(dslength);
	dFlength=sqrt(dFlength);
	//cout<<"dFlength="<<dFlength<<endl;
	//cout<<"dslength="<<dslength<<endl;
	//cout<<"Mean Stress="<<MeanStress<<"  Shear Stress="<<sqrt(ShearStress)<<endl;

	for (int i=0; i<6; i++)
	{

	 //calculated lenght - gradient - total length of vector ==1#
		//cout<<"dF["<<i<<"]="<<dF[i]<<endl;
		cosin=cosin+dF[i]*ds[i]/(dslength*dFlength); //calculated cosinus of the thetha angle...
	}
	cosin=cosin-dF[6]*dsuction/(dslength*dFlength);
	return cosin; //cosin returned... */
	//return 0;
}

void ShengMohrCoulomb::GetDerivative (double MeanStress, double ShearStress, double suction,double PZero, double * state, double* deriv )
{
	cout<<"GetDerivative is not implemented for Mohr-Coulomb model."<<endl;
	getchar();
/*	double SuctionPressure=k*suction;
	deriv[0]=M*M*(2*MeanStress+SuctionPressure-PZero);  // dF/dp
	deriv[1]=2*ShearStress;	//dF/dq
	//now we have to calculate df/ds... this is the most difficult (see word file)
	//p0* is == state [0]
	//pc is a parameter
	double P0Star=state[0];
	double denominator=LambdaZero*((1-r)*exp(-1*Beta*suction)+r);
	double numerator=LambdaZero-KappaP;
	deriv[2]=-1*Beta*(1-r)*exp(-1*Beta*suction);
	//cout<<"1:"<<deriv[2]<<endl;
	deriv[2]=deriv[2]*-1*numerator/(denominator*denominator);
	//cout<<"2:"<<deriv[2]<<endl;
	deriv[2]=deriv[2]*pc*log(P0Star/pc);
	//cout<<"3:"<<deriv[2]<<endl;
	deriv[2]=deriv[2]*pow((P0Star/pc),numerator/denominator); //it's finished with dp0/ds
	//cout<<"4:"<<deriv[2]<<endl;
	deriv[2]=M*M*(MeanStress+SuctionPressure)*deriv[2];
	//cout<<"5:"<<deriv[2]<<endl;
	deriv[2]=-1*M*M*k*(PZero-MeanStress)-deriv[2];	//final result

*/
//double ShengMohrCoulomb::FindGradientPQ (double * state, double * s, double *ds, double * dF, double suction, double dsuction)

}


bool ShengMohrCoulomb::CheckGradient (BBMPoint * InitialPoint, BBMPoint * FinalPoint)
{
	double df[6]; //not initialised; will contain values of derivatives of the yield locus function
	double ds[6]; //must be initialised with tangent value of stress
	double StrainIncrement [6];

	BBMMatrix STRAIN_INC(7,1), DEL(6,7), STRESS_INC(6,1);
	for (int i=0; i<6; i++) StrainIncrement[i]=FinalPoint->strain[i]-InitialPoint->strain[i];
	// HACK normalising strain to 10E-10 value
	double max=0; for (int i=0; i<6; i++) if (fabs(StrainIncrement[i])>max) max=fabs(StrainIncrement[i]);
	for (int i=0; i<6; i++) StrainIncrement[i]=(StrainIncrement[i]/max)*10E-10;
	//The normalisation is important to catch the unloading due to shear stress 12 13 23. As the q is always positive, large
	//unloading - loading values of 12 13 or 23 component lead to larger q, the shear stress change is taken as positive
	// and there is no unloading. This fix, though not the most elegant, should work.

	for (int i=0; i<6; i++) STRAIN_INC.PutElement(i+1,1,StrainIncrement[i]);
	STRAIN_INC.PutElement (7,1,FinalPoint->GetSuction()-InitialPoint->GetSuction());
	CalculateElasticTangentMatrix (InitialPoint, &DEL);
	DEL.Multiply(&STRAIN_INC, &STRESS_INC);

	/*STRAIN_INC.Print();
	cout<<endl<<"DEL:"<<endl;
	DEL.Print();
	STRESS_INC.Print();*/

	for (int i=0; i<6; i++) ds[i]=STRESS_INC.GetElement(i+1,1); //in ds values of stress increment calculated using tangent D
	double cosinus;// , SuctionIncrement;
	//SuctionIncrement=STRAIN_INC.GetElement(7,1);
	//cout<<"Suction Increment="<<SuctionIncrement<<endl;
	cosinus=FindGradient ( InitialPoint->state ,InitialPoint->stress, ds, df, 0,0);


	if (cosinus>-YIELDTOL)
	return false;
	else
	{
	    //cout<<"Unloading occuring... cosinus="<<cosinus<<endl;
	    return true;  //(negative cosinus means unloading occurs)
	}

}


double ShengMohrCoulomb::FindGradientPQ (BBMPoint * Point, double *ds, double * dF,  double dsuction)
{
cout<<"FindGradientPQ is unsupported for Mohr-Coulomb Model"<<endl;
/*
double PZero, SuctionPressure, dMeanStress, dShearStress, FValue, dFlength, dslength, cosin;
double P0Star=Point->GetPStar();
double MeanStress=Point->GetMeanStress();
double ShearStress=Point->GetShearStress();
double Suction=Point->GetSuction();

	//check of the second yield surface:
if (Point->GetYieldSuction()-Suction<-SUCTIONTOL)
	{
		cout<<"yield of the suction surface occurred"<<endl;
		if (dsuction>0)
		{
			cout <<"Unable to find gradient; Whole step plastic"<<endl;
			return -2;
		//this check may be unused...
		}
	}
//check of the standard yield surface

	//cout<<"MeanStress="<<MeanStress<<endl;
	//cout<<"ShearStress="<<ShearStress<<endl;

	PZero=LambdaZero*((1-r)*exp(-1*Beta*Suction)+r);
    PZero=(LambdaZero-KappaP)/(PZero-KappaP);
	PZero=pc*pow((P0Star/pc),PZero);
	SuctionPressure=k*Suction;
//	cout<<"suction="<<suction<<endl;
//	cout<<"PZero="<<PZero<<endl;
	FValue=ShearStress*ShearStress-M*M*(MeanStress+SuctionPressure)*(PZero-MeanStress);
	FValue=FValue/((PZero+SuctionPressure)*(PZero+SuctionPressure)); //value of Yield function calculated and normalized
	if (FValue<-YIELDTOL)
	{
		cout<<"!!!there is no yield at the beginning !!!"<<endl;
		cout<<"FValue is="<<FValue<<endl;
		cout<<"!!!find gradient procedure terminated!!!"<<endl<<endl<<endl<<endl;
		return -3;
		// this check may be disabled later
	}

	dF[0]=M*M*(2*MeanStress+SuctionPressure-PZero);  // dF/dp
	dF[1]=2*ShearStress;	//dF/dq
	//now we have to calculate df/ds... this is the most difficult (see word file)
	//p0* is == state [0]
	//pc is a parameter

	double denominator=LambdaZero*((1-r)*exp(-1*Beta*Suction)+r);
	double numerator=LambdaZero-KappaP;
	dF[2]=-1*Beta*(1-r)*exp(-1*Beta*Suction);
	//cout<<"1:"<<dF[2]<<endl;
	dF[2]=dF[2]*-1*numerator/(denominator*denominator);
	//cout<<"2:"<<dF[2]<<endl;
	dF[2]=dF[2]*pc*log(P0Star/pc);
	//cout<<"3:"<<dF[2]<<endl;
	dF[2]=dF[2]*pow((P0Star/pc),numerator/denominator); //it's finished with dp0/ds
	//cout<<"4:"<<dF[2]<<endl;
	dF[2]=M*M*(MeanStress+SuctionPressure)*dF[2];
	//cout<<"5:"<<dF[2]<<endl;
	dF[2]=-1*M*M*k*(PZero-MeanStress)-dF[2];	//final result
	//cout<<dF[2]<<endl;
	//calculate changes in p and q...;

	dMeanStress=(ds[0]+ds[1]+ds[2])/3;
	double dds[6];
	for (int i=0; i<6; i++) dds[i]=ds[i]+Point->stress[i];


	dShearStress=(dds[0]-dds[1])*(dds[0]-dds[1])+(dds[0]-dds[2])*(dds[0]-dds[2])+(dds[1]-dds[2])*(dds[1]-dds[2]);
	dShearStress=dShearStress+6*(dds[3]*dds[3]+dds[4]*dds[4]+dds[5]*dds[5]);
	dShearStress=dShearStress/2;
	dShearStress=sqrt(dShearStress);
	//we must not calculate change of shear stress only basing on change of stress; it must be done this way... it's not linear

	dShearStress=dShearStress-ShearStress;
	//this line should be removed later
	//dF[2]=0;
	//dF[3]=0;dF[4]=0;dF[5]=0;dF[6]=0; //don't take suction into account at the moment

	//dMeanStress=ds[0];
	//dShearStress=ds[1];

	dFlength=dF[0]*dF[0]+dF[1]*dF[1]+dF[2]*dF[2]; //calculated length
	dFlength=sqrt(dFlength);
	dslength=dShearStress*dShearStress+dMeanStress*dMeanStress+dsuction*dsuction;
	dslength=sqrt(dslength);
	for (int i=0; i<3; i++) dF[i]=dF[i]/dFlength;
	cosin=(dF[0]*dMeanStress+dF[1]*dShearStress+dF[2]*dsuction)/dslength;  //it should be d vector multiplied by gradient
	cout<<"Mean Stress="<<MeanStress<<"  Shear Stress="<<ShearStress<< " Suction="<<Suction<<endl;
	cout<<"dMean Stress="<<dMeanStress<<"  dShear Stress="<<dShearStress<<" dSuction="<<dsuction<<endl;
	for (int i=0; i<3; i++) cout<<"value of F["<<i<<"] is:"<<dF[i]<<endl;
	cout<<"df length is:"<<dFlength<<endl;
	cout<<"cosinus is:"<<cosin<<endl;
	getchar (); */
//	return cosin;
	return 0;
}


void ShengMohrCoulomb::CorrectDrift (BBMPoint *Point)
{

/*
This procedure should correct the drift, as described in the word file. The method of drift correction is based on
the Potts&Zdravkovic book; It is however slightly changes so to use it in unsaturated soil model. Detailed
description of the algorithm is presented in the word file. It is noted however, that in current version the algorithm
does calculate all the derivatives in the forbidden space outside the yield locus which may be cause of concerns.

Input: Point to correct
Output: Updated point, corrected using values of stress at the end
*/

//Need dF/Ds, D, dF/dP0*, dP0*/DEpsPl, m, dF/dS
// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)

//cout<<"Correct Drift Procedure entered!"<<endl;

BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
BBMMatrix DEL (6,6); //D elastic matrix...
BBMMatrix DEPS (6,1);
BBMMatrix DSIGMA (6,1);

double DSigma[6], EPStrain[6], zeros[7];
for (int i=0; i<7; i++) zeros[i]=0;

for (int i=1; i<6; i++) DEPS.PutElement (i,1,EPStrain[i-1]);  //increase of epsilon copied
// we do not have g, as a == g in associated flow rule.

bool correct;



double I1=Point->GetFirstInvariant();
double I2=Point->GetSecondInvariant();
double J2=Point->GetSecondDevInvariant ();
if (fabs(J2)<TINY) J2=TINY;
double J3=Point->GetThirdDevInvariant ();
double ShearStress=Point->GetShearStress ();
if (fabs(ShearStress)<TINY) ShearStress=TINY;
double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
if (Factor>1) Factor=1;
if (Factor<-1) Factor=-1;
if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in Correct Drift is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
Factor=1+Alpha4-(1-Alpha4)*Factor;
double Factor025=pow(Factor,0.25);
double Factor075=pow(Factor,-0.75);
double M=(3-SinPhi)/(Alpha*SinPhi);
double MPsi=(3-SinPsi)/(Alpha*SinPsi);
double FValue;


int NumberIter=0;

do
	{
	CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue); //20 Feb 2006, preparations for the drift correction


	if ((FValue/(Point->GetMeanStress()+2*Cohesion)<-YIELDTOL)||(FValue/(Point->GetMeanStress()+2*Cohesion)>YIELDTOL)) correct=TRUE; else correct=FALSE;
		if (correct==TRUE)
		{
			NumberIter++;
			//cout<<"Drift Correction, Iteration="<<NumberIter<<" Function Value="<<FValue<<endl;
				// CORRECT FOR DRIFT
				//HERE THE DRIFT WILL BE CORRECTED BY USING THE D MATRIX FROM THE FORBIDDEN SPACE
				//ALTHOUGH BECAUSE THE DRIFT WILL BE CHECKED AGAIN, IT SHOULDN'T POSE MUCH PROBLEM.


BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);

//derivatives of the invariants
double s[6]={Point->stress[0],Point->stress[1],Point->stress[2],Point->stress[3],Point->stress[4],Point->stress[5]};

dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (2,1,1.0);dI1dSIG.PutElement (3,1,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (2,1,s[0]+s[2]);dI2dSIG.PutElement (3,1,s[0]+s[1]);
dI2dSIG.PutElement (4,1,-2*s[3]);dI2dSIG.PutElement (5,1,-2*s[4]);dI2dSIG.PutElement (6,1,-2*s[5]);

//dI2dSIG.PrintPrecise();

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (2,1,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (3,1,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (4,1,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (5,1,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (6,1,2*s[3]*s[4]-2*s[0]*s[5]);

//dI3dSIG.PrintPrecise();

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (2,1,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (3,1,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (4,1,2*s[3]);dJ2dSIG.PutElement (5,1,2*s[4]);dJ2dSIG.PutElement (6,1,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(2.0/9.0*I1*I1-I2/3.0,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//dJ3dSIG.PrintPrecise();
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)




dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,0.0);
	}
}

//cout<<"dF/dJ2 part"<<endl;
//A.PrintPrecise();
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dJ3 part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);

for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,0.0);
	}
}

dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
//cout<<"dF/dq part"<<endl;
//TEMP.PrintPrecise();
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);


for (int i=1; i<7 ; i++)
{
	if (!finite (A.GetElement(i,1)))
	{
		A.PutElement(i,1,1.0);
	}
}


double MeanStress=Point->GetMeanStress();
if (MeanStress<-Cohesion*M)
{
    //tension cut-off plane
    A.PutElement(1,1,1.0/3.0);
    A.PutElement(2,1,1.0/3.0);
    A.PutElement(3,1,1.0/3.0);
    A.PutElement(4,1,0.0);
    A.PutElement(5,1,0.0);
    A.PutElement(6,1,0.0);
}

/*A.PrintPrecise ();
//FINISHED dF/dSigma

BBMPoint CopyPoint;
Point->Copy(&CopyPoint);
double Yield1,Yield2,dSs=0.0001;
Yield1=ComputeYieldFunction(&CopyPoint);
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;

dSs=10*dSs;
CopyPoint.stress[0]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"DSigma="<<dSs<<endl;
cout<<"dF/dSigma11="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[0]-=dSs;

CopyPoint.stress[1]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma22="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[1]-=dSs;

CopyPoint.stress[2]+=dSs;
Yield2=ComputeYieldFunctionNN(&CopyPoint);
cout<<"dF/dSigma33="<<(Yield2-Yield1)/dSs<<endl;
CopyPoint.stress[2]-=dSs;
*/
dJ2dSIG.Copy(&GG);
GG.Multiply (0.21022410391343*MPsi*Factor025/ShearStress,&GG);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*MPsi*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*MPsi*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*MPsi*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*MPsi*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);  //First correction: sign here, but most likely not enough!!!
GG.Add (&TEMP,&GG);

//FINISHED dQ/dSigma

if (MeanStress<-Cohesion*M)
{
    //tension cut-off plane
    GG.PutElement(1,1,1.0/3.0);
    GG.PutElement(2,1,1.0/3.0);
    GG.PutElement(3,1,1.0/3.0);
    GG.PutElement(4,1,0.0);
    GG.PutElement(5,1,0.0);
    GG.PutElement(6,1,0.0);
}

double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;

DEL.PutElement (1,1,K43G);
DEL.PutElement (1,2,K23G);
DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
DEL.PutElement (2,1,K23G);
DEL.PutElement (2,2,K43G);
DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
DEL.PutElement (3,2,K23G);
DEL.PutElement (3,3,K43G);
DEL.PutElement (4,4,2.0*G);
DEL.PutElement (5,5,2.0*G);
DEL.PutElement (6,6,2.0*G); //rest of the matrix is filled with zeros...

//Getting lambda and Dep

A.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&DEL,&NUMERATOR); //NUMERATOR=aT*Del -->Numerator of Lambda without multiplication by dEpsilon
NUMERATOR.Multiply(&GG,&DENOMINATOR);


double Lambda=FValue/DENOMINATOR.GetElement(1,1);

A.Multiply (Lambda, &TEMP); //delta epsilon plastic= -delta epsilon elastic

				//cout<<"Delta Epsilon Plastic:"<<endl;
				//TEMP.Print();

for (int i=1; i<7; i++) EPStrain [i-1]=TEMP.GetElement (i,1); //epsilon pl change


DEL.Multiply (&GG, &TEMP);
TEMP.Multiply (-Lambda, &DSIGMA);
			//final result for stress change, with the negative sign, so the stress should be ADDED
			//be ADDED to get the right result of corrected stresses.
for (int i=0; i<6; i++) DSigma[i]=DSIGMA.GetElement (i+1,1);
Point->Update (EPStrain, zeros, DSigma,0);

		}
		if (NumberIter>10)
		{
			//cout<<"Drift Correction Procedure failed"<<endl;
			correct=FALSE;

		}
	}
while (correct==TRUE);

/*


bool correct;
double FValue, DPZeroStar, Lambda, DSigma[6], EPStrain[6], zeros[7];
for (int i=0; i<7; i++) zeros[i]=0;

double temp, PZero, MeanStress, ShearStress, Suction, PZeroStar, SpecificVolume, LambdaS, Fraction;
// we do not have g, as a == g in associated flow rule.
	do
	{
	CheckYield (PointCopy.state, PointCopy.stress, PointCopy.GetSuction(), &FValue); //20 Feb 2006, preparations for the drift correction

		if ((FValue<-YIELDTOL)||(FValue>YIELDTOL)) correct=TRUE; else correct=FALSE;
		if (correct==TRUE)
		{
				// CORRECT FOR DRIFT
				//HERE THE DRIFT WILL BE CORRECTED BY USING THE D MATRIX FROM THE FORBIDDEN SPACE
				//ALTHOUGH BECAUSE THE DRIFT WILL BE CHECKED AGAIN, IT SHOULDN'T POSE MUCH PROBLEM.
				BBMMatrix A (6,1); //dF/dsigma
				BBMMatrix P (1,1); //dF/dP0*
				BBMMatrix GG(6,1);	//dG/dsigma
				BBMMatrix PEP (1,1); //dp0* /depsvpl
				BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
				BBMMatrix DEL (6,6); //D elastic matrix...
				BBMMatrix DPZEROSTAR (1,1);
				BBMMatrix TEMP (1,1);
				BBMMatrix DSIGMA (6,1);
				MM.PutElement (1,1,1);
				MM.PutElement (2,1,1);
				MM.PutElement (3,1,1); //rest is zero as initialized


				SpecificVolume=PointCopy.GetSpecVol();  //specific volume need to be used the right one
				//cout<<"Specific Volume:"<<SpecificVolume<<endl;
				PZeroStar=PointCopy.GetPStar ();

				//convention - matrices names are made from CAPITALIZED letters


				MeanStress=PointCopy.GetMeanStress();
				ShearStress=PointCopy.GetShearStress();
				Suction=PointCopy.GetSuction();

				LambdaS=(1-r)*exp(-Beta*Suction)+r;
				LambdaS=LambdaS*LambdaZero;
				Fraction=(LambdaZero-KappaP)/(LambdaS- KappaP);
				PZero=pc*pow(PZeroStar/pc,Fraction);  //get.calculated.pzero;

				//cout<<"PZero = "<<PZero<<endl;
				//cout<<"PZeroStar = "<<PZeroStar<<endl;
				//cout<<"p = "<<MeanStress<<endl;
				//cout<<"q = "<<ShearStress<<endl;
				//cout<<"s = "<<Suction<<endl;

				temp=2*PointCopy.stress[0]-PointCopy.stress[1]-PointCopy.stress[2]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(1,1,temp);
				temp=2*PointCopy.stress[1]-PointCopy.stress[0]-PointCopy.stress[2]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(2,1,temp);
				temp=2*PointCopy.stress[2]-PointCopy.stress[0]-PointCopy.stress[1]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(3,1,temp);
				temp=6*PointCopy.stress[3];
				A.PutElement(4,1,temp);
				temp=6*PointCopy.stress[4];
				A.PutElement(5,1,temp);
				temp=6*PointCopy.stress[5];
				A.PutElement(6,1,temp);
				//cout<<"A:"<<endl; A.Print();
				//dF/dsigma - inserted into A

				if (NonAssociated)
				{
					temp=alfa*(2*Point->stress[0]-Point->stress[1]-Point->stress[2])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(1,1,temp);
					temp=alfa*(2*Point->stress[1]-Point->stress[0]-Point->stress[2])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(2,1,temp);
					temp=alfa*(2*Point->stress[2]-Point->stress[0]-Point->stress[1])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(3,1,temp);
					temp=6*alfa*Point->stress[3];
					GG.PutElement(4,1,temp);
					temp=6*alfa*Point->stress[4];
					GG.PutElement(5,1,temp);
					temp=6*alfa*Point->stress[5];
					GG.PutElement(6,1,temp);
				}
				else */ /*A.Copy (&GG);


				//d

				temp=0;
				temp=-M*M*(MeanStress+k*Suction)*Fraction*pow((PZeroStar/pc), Fraction-1);

				P.PutElement (1,1,temp);

				//cout<<"P:"<<endl; P.Print();

				temp=PZeroStar*SpecificVolume/(LambdaZero-KappaP);
				PEP.PutElement (1,1,temp); //dP0* /depsplv
				//cout<<"DPZeroStar/Depsvpl:"<<temp<<endl;
				//DEL... elastic matrix... values of K. Here we need values of K at the point...
				We need to have the K - bulk modulus of the soil calculated, and then it is
				possible to fill into the DEL Matrix...
				So, the way of doing it will be repeated, algorithm as before... , in procedure find stress elast, but, this time
				it will be made inside and the results will be put into the matrix.
				*/

/*
				// Checks whether Mean stress is large enough to hold K in right range.
				if ((MeanStress<PMIN)&&(MeanStress>(-PMIN)))
				{
					cout<<"WARNING !!! Mean stress too low. Mean stress is adjusted to PMIN value!!!"<<endl;
					MeanStress=PMIN;
				}

				K=MeanStress*SpecificVolume/KappaP;  //tangent bulk modulus K=p*specificVol/KappaP, from eq dSpecVol=-KappaP*ln(p/pzero);
				//cout<<"K="<<K<<endl;

				// ****************************** K need correcting, but with const. suction seems to be ok ******************************

				// Tiny set in stdafx to 1e-12
				// Calculate helpful variables:
				double K43G=K+4.0*G/3.0;
				double K23G=K-2.0*G/3.0;


				// Fill in the matrix:

				DEL.PutElement (1,1,K43G);
				DEL.PutElement (1,2,K23G);
				DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
				DEL.PutElement (2,1,K23G);
				DEL.PutElement (2,2,K43G);
				DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
				DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
				DEL.PutElement (3,2,K23G);
				DEL.PutElement (3,3,K43G);
				DEL.PutElement (4,4,2*G);
				DEL.PutElement (5,5,2*G);
				DEL.PutElement (6,6,2*G); //rest of the matrix is filled with zeros...

				A.Transpose (&TEMP);
				TEMP.Multiply (&DEL,&TEMP);
				TEMP.Multiply (&GG, &TEMP);

				temp=TEMP.GetElement (1,1);  //first part of the denominator
				//cout<<"First Part of Denominator done... temp="<<temp<<endl;


				P.Multiply (&PEP,&TEMP);
				MM.Transpose (&MM);
				TEMP.Multiply (&MM,&TEMP);	//MM is transposed
				TEMP.Multiply (&GG,&TEMP);

				temp=temp+TEMP.GetElement (1,1); //'end of the denominator

				//cout<<"Denominator="<<temp<<endl;

				Lambda=FValue*(PZero+Suction*k)*(PZero+Suction*k)/temp; //because we need the value, not the reduced value...

				//cout<<"Lambda="<<Lambda<<endl;

				A.Multiply (Lambda, &TEMP); //delta epsilon plastic= -delta epsilon elastic

				//cout<<"Delta Epsilon Plastic:"<<endl;
				//TEMP.Print();

				for (int i=1; i<7; i++) EPStrain [i-1]=TEMP.GetElement (i,1); //epsilon pl change
				temp=EPStrain[0]+EPStrain[1]+EPStrain[2]; //DepsilonV
				DPZeroStar=PEP.GetElement(1,1)*temp;
				//cout<<"DPZeroStar="<<*DPZeroStar<<endl;

				DEL.Multiply (&GG, &TEMP);
				TEMP.Multiply (-Lambda, &DSIGMA);
			//final result for stress change, with the negative sign, so the stress should be ADDED
			//be ADDED to get the right result of corrected stresses.

			//cout<<"Delta Sigma="<<endl;
			//DSIGMA->Print();


			//cout<<"Press any key (end of Correct Drift Procedure)"<<endl;
			//getchar();
			for (int i=0; i<6; i++) DSigma[i]=DSIGMA.GetElement (i+1,1);
			PointCopy.Update (EPStrain, zeros, DSigma, DPZeroStar);

		}
	}
	while (correct==TRUE);

PointCopy.Copy(Point);
			//this finishes the algorithm */
}



void ShengMohrCoulomb::CorrectDriftBeg (BBMPoint *Point, BBMPoint *PointOld)
{

/*
This procedure should correct the drift, as described in the word file. The method of drift correction is based on
the Potts&Zdravkovic book; It is however slightly changes so to use it in unsaturated soil model. Detailed
description of the algorithm is presented in the word file. It is noted however, that in current version the algorithm
does calculate all the derivatives in the forbidden space outside the yield locus which may be cause of concerns.

Input: Point to correct
Output: Updated point, corrected using values of stress at the beginning
*/

//Need dF/Ds, D, dF/dP0*, dP0*/DEpsPl, m, dF/dS
// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)

//cout<<"Correct Drift Procedure entered!"<<endl;


BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
BBMMatrix DEL (6,6); //D elastic matrix...
BBMMatrix DEPS (6,1);
BBMMatrix DSIGMA (6,1);

double DSigma[6], EPStrain[6], zeros[7];
for (int i=0; i<7; i++) zeros[i]=0;

for (int i=1; i<6; i++) DEPS.PutElement (i,1,EPStrain[i-1]);  //increase of epsilon copied
// we do not have g, as a == g in associated flow rule.

bool correct;


double I1=PointOld->GetFirstInvariant();
double I2=PointOld->GetSecondInvariant();
double J2=PointOld->GetSecondDevInvariant ();
if (fabs(J2)<TINY) J2=TINY;
double J3=PointOld->GetThirdDevInvariant ();
double ShearStress=PointOld->GetShearStress ();
if (fabs(ShearStress)<TINY) ShearStress=TINY;
double Factor=-27*J3/(2*ShearStress*ShearStress*ShearStress);
if (Factor>1) Factor=1;
if (Factor<-1) Factor=-1;
if (!finite(Factor))
		{
		     Factor=1.0;
		     cout<<"Factor in Correct Drift Beg [parameters *Point] is not finite. Set to 1"<<endl;
		     cout<<"Alpha4="<<Alpha4<<" J3="<<J3<<" ShearStress="<<ShearStress<<endl;
		}
Factor=1+Alpha4-(1-Alpha4)*Factor;
double Factor025=pow(Factor,0.25);
double Factor075=pow(Factor,-0.75);
double M=(3-SinPhi)/(6*Alpha*SinPhi);
double MPsi=(3-SinPsi)/(6*Alpha*SinPsi);
double FValue;

BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);

//derivatives of the invariants
double s[6]={PointOld->stress[0],PointOld->stress[1],PointOld->stress[2],PointOld->stress[3],PointOld->stress[4],PointOld->stress[5]};

dI1dSIG.PutElement (1,1,1.0); dI1dSIG.PutElement (2,1,1.0);dI1dSIG.PutElement (3,1,1.0); //{1,1,1,0,0,0}

dI2dSIG.PutElement (1,1,s[1]+s[2]);dI2dSIG.PutElement (2,1,s[0]+s[2]);dI2dSIG.PutElement (3,1,s[0]+s[1]);
dI2dSIG.PutElement (4,1,-2*s[3]);dI2dSIG.PutElement (5,1,-2*s[4]);dI2dSIG.PutElement (6,1,-2*s[5]);

//dI2dSIG.PrintPrecise();

dI3dSIG.PutElement (1,1,s[1]*s[2]-s[5]*s[5]);dI3dSIG.PutElement (2,1,s[0]*s[2]-s[4]*s[4]);dI3dSIG.PutElement (3,1,s[0]*s[1]-s[3]*s[3]);
dI3dSIG.PutElement (4,1,2*s[5]*s[4]-2*s[2]*s[3]);dI3dSIG.PutElement (5,1,2*s[3]*s[5]-2*s[1]*s[4]);dI3dSIG.PutElement (6,1,2*s[3]*s[4]-2*s[0]*s[5]);

//dI3dSIG.PrintPrecise();

dJ2dSIG.PutElement (1,1,(2*s[0]-s[1]-s[2])/3.0);dJ2dSIG.PutElement (2,1,(2*s[1]-s[0]-s[2])/3.0);dJ2dSIG.PutElement (3,1,(2*s[2]-s[0]-s[1])/3.0);
dJ2dSIG.PutElement (4,1,2*s[3]);dJ2dSIG.PutElement (5,1,2*s[4]);dJ2dSIG.PutElement (6,1,2*s[5]);

dJ2dSIG.Copy(&dqdSIG);
dqdSIG.Multiply(sqrt(3.0)*0.5/sqrt(J2),&dqdSIG);

dI1dSIG.Copy(&dJ3dSIG);
dJ3dSIG.Multiply(6.0/27.0*I1*I1-I2/3.0,&dJ3dSIG);
dI2dSIG.Copy(&TEMP);
TEMP.Multiply(-I1/3.0,&TEMP);
dJ3dSIG.Add(&TEMP,&dJ3dSIG);
dJ3dSIG.Add(&dI3dSIG,&dJ3dSIG);
//finished dJ3dSIG

//above needs to be *CORRECTED* as some errors in ij i!=j components is likely (2x too big)

dJ2dSIG.Copy(&A);
A.Multiply (0.21022410391343*M*Factor025/ShearStress,&A);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*M*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*M*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*M*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*M*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
A.Add (&TEMP,&A);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);
A.Add (&TEMP,&A);

//FINISHED dF/dSigma

dJ2dSIG.Copy(&GG);
GG.Multiply (0.21022410391343*MPsi*Factor025/ShearStress,&GG);
dJ3dSIG.Copy(&TEMP);
TEMP.Multiply (-1.419012700740643*(Alpha4-1)*MPsi*sqrt(J2)*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress)+
			   0.94600846716043*(Alpha4-1)*Cohesion*MPsi*Factor075/(ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dqdSIG.Copy(&TEMP);
TEMP.Multiply (-2.838025401481287*(Alpha4-1)*Cohesion*MPsi*J3*Factor075/(ShearStress*ShearStress*ShearStress*ShearStress)+
			   4.257038102221929*(Alpha4-1)*MPsi*sqrt(J2)*J3*Factor075/(sqrt(3.0)*ShearStress*ShearStress*ShearStress*ShearStress),&TEMP);
GG.Add (&TEMP,&GG);
dI1dSIG.Copy(&TEMP);
TEMP.Multiply(-1/3.0, &TEMP);  //First correction: sign here, but most likely not enough!!!
GG.Add (&TEMP,&GG);

//FINISHED dQ/dSigma

double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;

DEL.PutElement (1,1,K43G);
DEL.PutElement (1,2,K23G);
DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
DEL.PutElement (2,1,K23G);
DEL.PutElement (2,2,K43G);
DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
DEL.PutElement (3,2,K23G);
DEL.PutElement (3,3,K43G);
DEL.PutElement (4,4,2.0*G);
DEL.PutElement (5,5,2.0*G);
DEL.PutElement (6,6,2.0*G); //rest of the matrix is filled with zeros...

//Getting lambda and Dep

A.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&DEL,&NUMERATOR); //NUMERATOR=aT*Del -->Numerator of Lambda without multiplication by dEpsilon
NUMERATOR.Multiply(&GG,&DENOMINATOR);

int NumberIter=0;

do
	{
	NumberIter++;
	CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue); //20 Feb 2006, preparations for the drift correction

		if ((FValue/(Point->GetMeanStress()+2*Cohesion)<-YIELDTOL)||(FValue/(Point->GetMeanStress()+2*Cohesion)>YIELDTOL)) correct=TRUE; else correct=FALSE;
		if (correct==TRUE)
		{
				// CORRECT FOR DRIFT

		double Lambda=FValue/DENOMINATOR.GetElement(1,1);
		A.Multiply (Lambda, &TEMP); //delta epsilon plastic= -delta epsilon elastic

				//cout<<"Delta Epsilon Plastic:"<<endl;
				//TEMP.Print();

		for (int i=1; i<7; i++) EPStrain [i-1]=TEMP.GetElement (i,1); //epsilon pl change
		DEL.Multiply (&GG, &TEMP);
		TEMP.Multiply (-Lambda, &DSIGMA);
			//final result for stress change, with the negative sign, so the stress should be ADDED
			//be ADDED to get the right result of corrected stresses.
		for (int i=0; i<6; i++) DSigma[i]=DSIGMA.GetElement (i+1,1);
		Point->Update (EPStrain, zeros, DSigma,0);
		}
	if (NumberIter>10)
		{
			//cout<<"Drift Correction Procedure failed"<<endl;
			correct=FALSE;

		}
	}
while (correct==TRUE);


/*
BBMPoint PointCopy, PointEnd;
PointOld->Copy(&PointCopy);
Point->Copy(&PointEnd);

bool correct;
double FValue, DPZeroStar, Lambda, DSigma[6], EPStrain[6], zeros[7];
for (int i=0; i<7; i++) zeros[i]=0;

double temp, PZero, MeanStress, ShearStress, Suction, PZeroStar, SpecificVolume, LambdaS, Fraction;
// we do not have g, as a == g in associated flow rule.
	do
	{
	CheckYield (PointEnd.state, PointEnd.stress, PointEnd.GetSuction(), &FValue); //20 Feb 2006, preparations for the drift correction

	if (fabs(FValue)>YIELDTOL) correct=TRUE; else correct=FALSE;
		if (correct==TRUE)
		{
				// CORRECT FOR DRIFT
				//HERE THE DRIFT WILL BE CORRECTED BY USING THE D MATRIX FROM THE FORBIDDEN SPACE
				//ALTHOUGH BECAUSE THE DRIFT WILL BE CHECKED AGAIN, IT SHOULDN'T POSE MUCH PROBLEM.
				BBMMatrix A (6,1); //dF/dsigma
				BBMMatrix P (1,1); //dF/dP0*
				BBMMatrix GG (6,1);
				BBMMatrix PEP (1,1); //dp0* /depsvpl
				BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
				BBMMatrix DEL (6,6); //D elastic matrix...
				BBMMatrix DPZEROSTAR (1,1);
				BBMMatrix TEMP (1,1);
				BBMMatrix DSIGMA (6,1);
				MM.PutElement (1,1,1);
				MM.PutElement (2,1,1);
				MM.PutElement (3,1,1); //rest is zero as initialized



				SpecificVolume=PointCopy.GetSpecVol();  //specific volume need to be used the right one
				//cout<<"Specific Volume:"<<SpecificVolume<<endl;
				PZeroStar=PointCopy.GetPStar ();

				//convention - matrices names are made from CAPITALIZED letters


				MeanStress=PointCopy.GetMeanStress();
				ShearStress=PointCopy.GetShearStress();
				Suction=PointCopy.GetSuction();

				LambdaS=(1-r)*exp(-Beta*Suction)+r;
				LambdaS=LambdaS*LambdaZero;
				Fraction=(LambdaZero-KappaP)/(LambdaS- KappaP);
				PZero=pc*pow(PZeroStar/pc,Fraction);  //get.calculated.pzero;

				//cout<<"PZero = "<<PZero<<endl;
				//cout<<"PZeroStar = "<<PZeroStar<<endl;
				//cout<<"p = "<<MeanStress<<endl;
				//cout<<"q = "<<ShearStress<<endl;
				//cout<<"s = "<<Suction<<endl;

				temp=2*PointCopy.stress[0]-PointCopy.stress[1]-PointCopy.stress[2]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(1,1,temp);
				temp=2*PointCopy.stress[1]-PointCopy.stress[0]-PointCopy.stress[2]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(2,1,temp);
				temp=2*PointCopy.stress[2]-PointCopy.stress[0]-PointCopy.stress[1]+M*M/3*(2*MeanStress+k*Suction-PZero);
				A.PutElement(3,1,temp);
				temp=6*PointCopy.stress[3];
				A.PutElement(4,1,temp);
				temp=6*PointCopy.stress[4];
				A.PutElement(5,1,temp);
				temp=6*PointCopy.stress[5];
				A.PutElement(6,1,temp);
				//cout<<"A:"<<endl; A.Print();
				//dF/dsigma - inserted into A

				if (NonAssociated)
				{
					temp=alfa*(2*Point->stress[0]-Point->stress[1]-Point->stress[2])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(1,1,temp);
					temp=alfa*(2*Point->stress[1]-Point->stress[0]-Point->stress[2])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(2,1,temp);
					temp=alfa*(2*Point->stress[2]-Point->stress[0]-Point->stress[1])+M*M/3*(2*MeanStress+k*Suction-PZero);
					GG.PutElement(3,1,temp);
					temp=6*alfa*Point->stress[3];
					GG.PutElement(4,1,temp);
					temp=6*alfa*Point->stress[4];
					GG.PutElement(5,1,temp);
					temp=6*alfa*Point->stress[5];
					GG.PutElement(6,1,temp);
				}
				else */ /* A.Copy (&GG);

				//d

				temp=0;
				temp=-M*M*(MeanStress+k*Suction)*Fraction*pow((PZeroStar/pc), Fraction-1);

				P.PutElement (1,1,temp);

				//cout<<"P:"<<endl; P.Print();

				temp=PZeroStar*SpecificVolume/(LambdaZero-KappaP);
				PEP.PutElement (1,1,temp); //dP0* /depsplv
				//cout<<"DPZeroStar/Depsvpl:"<<temp<<endl;
				//DEL... elastic matrix... values of K. Here we need values of K at the point...
				We need to have the K - bulk modulus of the soil calculated, and then it is
				possible to fill into the DEL Matrix...
				So, the way of doing it will be repeated, algorithm as before... , in procedure find stress elast, but, this time
				it will be made inside and the results will be put into the matrix.
				*/

/*
				// Checks whether Mean stress is large enough to hold K in right range.
				if ((MeanStress<PMIN)&&(MeanStress>(-PMIN)))
				{
					cout<<"WARNING !!! Mean stress too low. Mean stress is adjusted to PMIN value!!!"<<endl;
					MeanStress=PMIN;
				}

				K=MeanStress*SpecificVolume/KappaP;  //tangent bulk modulus K=p*specificVol/KappaP, from eq dSpecVol=-KappaP*ln(p/pzero);
				//cout<<"K="<<K<<endl;

				// ****************************** K need correcting, but with const. suction seems to be ok ******************************

				// Tiny set in stdafx to 1e-12
				// Calculate helpful variables:
				double K43G=K+4.0*G/3.0;
				double K23G=K-2.0*G/3.0;


				// Fill in the matrix:

				DEL.PutElement (1,1,K43G);
				DEL.PutElement (1,2,K23G);
				DEL.PutElement (1,3,K23G); //rest of the line are zeros and rightly so
				DEL.PutElement (2,1,K23G);
				DEL.PutElement (2,2,K43G);
				DEL.PutElement (2,3,K23G); //yes, the matrix is symmetrical, but it is faster to put this 3 additional elements
				DEL.PutElement (3,1,K23G); //than just mirror all, including zeros, which are there already
				DEL.PutElement (3,2,K23G);
				DEL.PutElement (3,3,K43G);
				DEL.PutElement (4,4,2*G);
				DEL.PutElement (5,5,2*G);
				DEL.PutElement (6,6,2*G); //rest of the matrix is filled with zeros...

				A.Transpose (&TEMP);
				TEMP.Multiply (&DEL,&TEMP);
				TEMP.Multiply (&GG, &TEMP);

				temp=TEMP.GetElement (1,1);  //first part of the denominator
				//cout<<"First Part of Denominator done... temp="<<temp<<endl;


				P.Multiply (&PEP,&TEMP);
				MM.Transpose (&MM);
				TEMP.Multiply (&MM,&TEMP);	//MM is transposed
				TEMP.Multiply (&GG,&TEMP);

				temp=temp+TEMP.GetElement (1,1); //'end of the denominator

				//cout<<"Denominator="<<temp<<endl;

				Lambda=FValue*(PZero+Suction*k)*(PZero+Suction*k)/temp; //because we need the value, not the reduced value...

				//cout<<"Lambda="<<Lambda<<endl;

				A.Multiply (Lambda, &TEMP); //delta epsilon plastic= -delta epsilon elastic

				//cout<<"Delta Epsilon Plastic:"<<endl;
				//TEMP.Print();

				for (int i=1; i<7; i++) EPStrain [i-1]=TEMP.GetElement (i,1); //epsilon pl change
				temp=EPStrain[0]+EPStrain[1]+EPStrain[2]; //DepsilonV
				DPZeroStar=PEP.GetElement(1,1)*temp;
				//cout<<"DPZeroStar="<<*DPZeroStar<<endl;

				DEL.Multiply (&GG, &TEMP);
				TEMP.Multiply (-Lambda, &DSIGMA);
			//final result for stress change, with the negative sign, so the stress should be ADDED
			//be ADDED to get the right result of corrected stresses.

			//cout<<"Delta Sigma="<<endl;
			//DSIGMA->Print();


			//cout<<"Press any key (end of Correct Drift Procedure)"<<endl;
			//getchar();
			for (int i=0; i<6; i++) DSigma[i]=DSIGMA.GetElement (i+1,1);
			PointEnd.Update (EPStrain, zeros, DSigma, DPZeroStar);

		}
	}
	while (correct==TRUE);

PointEnd.Copy(Point);
//this finishes the algorithm */
}








void ShengMohrCoulomb::SetIntegrationParameters (double IntegrationTolerance,int SolutionAlgorithm, int ToleranceMethod,
							int DriftCorrection, double BetaFactor, double YieldLocTolerance, int MaxIterPegasus)
{
INTEGRATION_TOL=IntegrationTolerance; //defines general relative tolerance (0.001 default)
SOLUTION_ALGORITHM=SolutionAlgorithm; //defines solution algorithm to use (RK etc... )
TOL_METHOD=ToleranceMethod; //(0- relative EPUS, 1 - Sloan Norm EPS (default));
DRIFT_CORRECTION=DriftCorrection; //(1- no drift correction, 2 - correction at the beginning, 3 - correction at the end (default)
BETA_FACT=BetaFactor; //defines the 'safety factor' when guessing next substep size 0.9 default
YIELDTOL=YieldLocTolerance; //defines the relative Yield Locus Tolerance , 1e-10 default
MAXITER=MaxIterPegasus; //defines the maximum nuber of iteration in the Pegasus Algorithm
}

void ShengMohrCoulomb::SetDefaultIntegrationParameters ()
{

//elastic parameters, using in the Pegasus algorithm
MAXITER=200;
ALFACHECK=1;
ALFACHANGE=0.05;
ALFARATIO=10;
//Yield tolerance
YIELDTOL=1e-6;  //toleration for the Yield locus (relatively defined, used to check if the stress state is on the YL)
INTEGRATION_TOL=0.01; //integration tolerance
DRIFT_CORRECTION=3; //3-drift at the end  //1-no drift correction
TOL_METHOD=1; //1 - Sloan , 0 would be for EPUS relative error
BETA_FACT=0.9; //safety factor
if (SinPhi>0) PMIN=Cohesion*CosPhi/SinPhi; else PMIN=-1E8; //approximate only
//PMIN=0.0001;	//minimum mean stress, below this value stress is corrected to PMIN
SUCTIONTOL=1e-8; //used in checking suction yield locus. advisible not to modify...
TINY=1e-14;
//used to check for accuracies below machine accuracy, and to accept anything less as ok... beware with modification
//they may have enormous influence on performance of the code
SOLUTION_ALGORITHM=1;//1:Modified Euler, 2:RK 332, Nystrom,

//values below are not used, to be deleted
ADDTOLYIELD=0.9;
CHGEPSINTOL=0.9;
USE_ERROR=0.5;
STEP_MAX=100;
STEP_MIN=0;
ERROR_DEF=1;
USE_ERROR_STEP=100;
MIN_DIVISION_SIZE=1;
}

void ShengMohrCoulomb::Integrate (double* StrainIncrement, BBMPoint* InitialPoint)
{
/*written in mind that InitialPoint may be equal to the FinalPoint
Basically the procedure made before should be re-introduced, with some changes
due to the different approach to parameters setting. It should also incorporate the
correct value of plastic strain, instead of useless lambda. This should have been done
long time ago, but well, better late then never.
The plastic/elastic part of the strain is also linked to the
*/

BBMPoint FinalPoint;
bool PurelyElastic;
bool OnTheYieldLocus;
bool Unloading;
double PurelyElasticStrain[7],PurelyPlasticStrain[7];

//cout<<'Strains:'<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;

OnTheYieldLocus=CheckYieldNormalised (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)
CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
PurelyElastic=!(CheckIfPlastic (&FinalPoint)); //CheckIfPlastic returns true if in the plastic part of the YL. If on the YL (within Tolerance INT_TOL returns false)
//cout<<"p="<<FinalPoint.GetMeanStress()<<"  q="<<FinalPoint.GetShearStress()<<"  Yield? "<<PurelyElastic<<endl; getchar();
//PurelyElastic=TRUE;

if (PurelyElastic)
{
	//FinalPoint.Copy(InitialPoint);
	//update with what we have - all updated after the procedure
	//cout<<"Purely Elastic Step"<<endl;
}
else  // we have elasto-plastic step
{
	if (OnTheYieldLocus)
	{
		//Unloading=CheckGradient(InitialPoint, &FinalPoint); //checking if there is any unloading part; in FinalPoint purely elastic values; we care only for correct change of suction
		//cout<<endl<<endl<<"Unloading="<<Unloading<<endl;
		//TEMPORARY HACK TESTING
		Unloading=false;
		//getchar();
		if (Unloading)
		{
			cout<<endl<<endl<<"Elasto-Plastic Unloading="<<Unloading<<endl;
			FindIntersectionUnloading(StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//finding elastic part
			CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint); //calculating elastic part, updated in the same point.
			FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
			CalculatePlastic (PurelyPlasticStrain,&FinalPoint); //and subsequently plastic part
			//FinalPoint.Copy(InitialPoint);
		}
		else {
			CalculatePlastic (StrainIncrement,&FinalPoint);
			}
	}
	else
	{
	FindIntersection (StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//not on the yield locus, finding intersection and subsequently calculate elastic and plastic stress increment
	CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint);
	FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
	CalculatePlastic (PurelyPlasticStrain,&FinalPoint);
	//FinalPoint.Copy(InitialPoint);
	}

}
//for (int i=0; i<3; i++) cout<<StrainIncrement[i]<<"  "<<endl;
//for (int i=0; i<3; i++) cout<<FinalPoint.stress[i]<<"  "<<endl;

FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
FinalPoint.Copy(InitialPoint);
//getchar();
}


void ShengMohrCoulomb::Integrate (double* StrainIncrement, double SuctionIncrement, BBMPoint* InitialPoint, double * StressIncrement,
	double P0StarIncrement, double* PlasticStrainIncrement)
{
}

void ShengMohrCoulomb::IntegrateConst (double* StrainIncrement,BBMPoint* InitialPoint, int StepNo, int Method)

//Method does not work at the moment
{
BBMPoint FinalPoint;
bool PurelyElastic;
bool OnTheYieldLocus;
bool Unloading;
double PurelyElasticStrain[7],PurelyPlasticStrain[7];

OnTheYieldLocus=CheckYield (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)
CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
PurelyElastic=!(CheckIfPlastic (&FinalPoint)); //CheckIfPlastic returns true if in the plastic part of the YL. If on the YL (within Tolerance INT_TOL returns false)
//cout<<PurelyElastic<<endl;


if (PurelyElastic)
{
	//FinalPoint.Copy(InitialPoint);
	//update with what we have - all updated after the procedure
	//cout<<"Purely Elastic Step"<<endl;
}
else  // we have elasto-plastic step
{
	if (OnTheYieldLocus)
	{
		Unloading=CheckGradient(InitialPoint, &FinalPoint); //checking if there is any unloading part; in FinalPoint purely elastic values; we care only for correct change of suction
		//cout<<endl<<endl<<"Unloading="<<Unloading<<endl;
		//getchar();
		if (Unloading)
		{
			cout<<endl<<endl<<"Elasto-Plastic Unloading="<<Unloading<<endl;
			FindIntersectionUnloading(StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//finding elastic part
			CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint); //calculating elastic part, updated in the same point.
			CalculatePlasticConst (PurelyPlasticStrain,&FinalPoint, StepNo); //and subsequently plastic part
			//FinalPoint.Copy(InitialPoint);
		}
		else {
			CalculatePlasticConst (StrainIncrement,&FinalPoint, StepNo);
			}
	}
	else
	{
	FindIntersection (StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//not on the yield locus, finding intersection and subsequently calculate elastic and plastic stress increment
	CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint);
	CalculatePlasticConst (PurelyPlasticStrain,&FinalPoint, StepNo);
	//FinalPoint.Copy(InitialPoint);
	}
}
//for (int i=0; i<3; i++) cout<<StrainIncrement[i]<<"  "<<endl;
//for (int i=0; i<3; i++) cout<<FinalPoint.stress[i]<<"  "<<endl;
FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
FinalPoint.Copy(InitialPoint);
}






void ShengMohrCoulomb::FindElStrGradPQ (double nu0, double* s0  , double* eps0, double* deps,  double* ds)
{
	/*
	Assumed that into s0 is the vector p,q inserted s0[0]=p. s0[1]=q
	Output - into ds the p, q is written
	*/
/*
	double K, dEpsSV, dEpsSig, dEpsV, dEpsq;

	dEpsq=(deps[0]+deps[1])*(deps[0]+deps[1])+(deps[0]+deps[2])*(deps[0]+deps[2])+(deps[1]+deps[2])*(deps[1]+deps[2]);
	dEpsq=dEpsq+6*(deps[3]*deps[3]+deps[4]*deps[4]+deps[5]*deps[5]);
	dEpsq=sqrt(2*dEpsq);
	dEpsq=dEpsq/3;	//distortional strain computed

	K=s0[0]*nu0/KappaP;
	dEpsV=(deps[0]+deps[1]+deps[2])/3;

	dEpsSV=(KappaS/nu0)*log((eps0[6]+deps[6]+PAtmos)/(eps0[6]+PAtmos));
	dEpsSig=dEpsV-dEpsSV;	//volumetric strain due to change of stress computed
	ds[0]=K*dEpsSig;
	ds[1]=G*dEpsq;	//p & q computed  */
// !!! all stresses computed using tangent K modulus- only for gradient procedures !!!
}



double ShengMohrCoulomb::CalculatePZero (BBMPoint * Point)
{
/*	double PZero=LambdaZero*((1-r)*exp(-1*Beta*Point->GetSuction())+r);
	PZero=(LambdaZero-KappaP)/(PZero-KappaP);
	PZero=pc*pow((Point->GetPStar()/pc),PZero);
	return PZero;*/
	return 0;
}




/*
bool ShengMohrCoulomb::CheckYield (double *state, double *s, double suction)
{

	
	Purpose of this routine is to calculate value of yield function to determine, whether we have yielding
	or not.

	Arguments: *state - table of state parameters; in fact we need only p*.
				s- stress state
				suction - suction
	Returns: FALSE when NOT yielding, TRUE when yielding.
	*/
/*	double PZero, SuctionPressure, MeanStress, ShearStress, FValue;
	//check of the second yield surface:
	if (state[1]-suction<-SUCTIONTOL) return true;
	//check of the standard yield surface
	MeanStress=(s[0]+s[1]+s[2])/3;
	ShearStress=(s[0]-s[1])*(s[0]-s[1])+(s[0]-s[2])*(s[0]-s[2])+(s[1]-s[2])*(s[1]-s[2]);
	ShearStress=ShearStress+6*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
	ShearStress=ShearStress/2;
	ShearStress=sqrt (ShearStress);  //Naylor Pande

	PZero=LambdaZero*((1-r)*exp(-1*Beta*suction)+r);
	PZero=(LambdaZero-KappaP)/(PZero-KappaP);
	PZero=pc*pow((state[0]/pc),PZero);
	SuctionPressure=k*suction;
	FValue=ShearStress*ShearStress-M*M*(MeanStress+SuctionPressure)*(PZero-MeanStress);
	FValue=FValue/((PZero+SuctionPressure)*(PZero+SuctionPressure)); //value of Yield function calculated and normalized
	if (FValue<YIELDTOL) return false;
	 else return true;

}
*/


void ShengMohrCoulomb::FindYieldOriginal (double *state, double* s0, double* eps0, double* deps, double* a)
{

/*
Main purpose of this procedure is to find the intersection between yield surface and the stress vector. We need to know
at what value of epsilon (strain) we are on the surface - where plastic yielding will begin.

Parameters: state vector, initial stress s0, initial strain eps0, strain increment deps, a- to which value of alfa will be written

Changed - a

Algorithm - this is a Pegasus algorithm; we are looking for alfa such, that value of yield function F==0.
This alfa is found and written to a.

*/

	double F0, F1,Falfa, epsini[7], epsfini[7], epsalfa[7], sini[6], sfini[6], salfa[6], alfa0, alfa1, alfa;
	alfa0=0;
	alfa1=1;

	for (int i=0; i<6; i++) sini[i]=0;
	for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}

	// check of the second yield surface:
	if ((eps0[6]+deps[6])>state[1])  //so suction is larger then maximum experienced suction
	{
		alfa0=(state[1]-eps0[6])/deps[6];
		for (int i=0; i<7; i++)
		{
			epsini[i]=deps[i]*alfa0;
		}
		CalcStressElast (state[2], s0, eps0, epsini, sini);
		for (int i=0; i<6;i++) sini[i]=sini[i]+s0[i];
		CheckYield (state, sini, eps0[6]+epsini[6],&F0);
		// main F checked
		if (F0<YIELDTOL)
		{
			*a=alfa0; // so we have elastic state in other case,
			return;
		}
		else
		{
			alfa1=alfa0; //we have the upper limit of yiels
			alfa0=0;
		}
		//the second surface is the most important - and it is violated...
		//we have to find alfa depending on first surface - which is violated first
	}

//now because the Main yield surface is valid, the whole check is made...
//however we start with alfa0 different if the suction was greater than maksimum suction...



for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}

	CalcStressElast (state[2], s0, eps0, epsfini, sfini);

	for (int i=0; i<6; i++)
	{
		sfini[i]=sfini[i]+s0[i]; //otherwise we have sfini already calculated
		sini[i]=s0[i];
	}
	CheckYield (state, sini, eps0[6]+epsini[6],&F0);
	CheckYield (state, sfini , eps0[6]+epsfini[6],&F1);

	//cout <<"F0="<<F0<<endl<<"F1="<<F1<<endl;

for (int iter=0; iter<MAXITER; iter++)
{
	alfa=F0/(F0-F1);
	for (int i=0; i<7; i++)
	{
		epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
	}

	CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
	//state[2]=ComputeNu (salfa, state, eps0[6]+epsalfa[6]);
	for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];	//update stress
	CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
	//cout<<"In iteration "<<iter<<" alfa="<<alfa<<" and F="<<Falfa<<endl;


	if ((Falfa>-YIELDTOL)&&(Falfa<YIELDTOL))
		{
		*a=alfa0+alfa*(alfa1-alfa0);		//if Falfa within tolerance, we have the solution
		cout<<"Solution in FindYieldOriginal procedure was found after "<<iter<<" iterations."<<endl;
		if (iter>50) {
				getchar();
				cout<<"Large number of iterations!!! Solution is however correct... Press any key..."<<endl;
		}
		return;
		}
	if (Falfa>0)
	{
		alfa1=alfa0+alfa*(alfa1-alfa0);
		F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa
	}
	else
	{
		alfa0=alfa0+alfa*(alfa1-alfa0);
		F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa
	}
}
*a=-1;
// if we are here, we must have perforemed to many iterations...
//cout<<"Error in procedure FindYieldOriginal"<<endl;
//cout<<"After "<<MAXITER<<" iterations crossing point not found"<<endl;
//cout<<"This is likely to cause incorrect results... Results obtained should not be taken too seriously..."<<endl;
*a=-1; //set value to error...

}



void ShengMohrCoulomb::FindYieldModified (double *state, double* s0, double* eps0, double* deps, double* a)
{

/*
Main purpose of this procedure is to find the intersection between yield surface and the stress vector. We need to know
at what value of epsilon (strain) we are on the surface - where plastic yielding will begin.

Parameters: state vector, initial stress s0, initial strain eps0, strain increment deps, a- to which value of alfa will be written

Changed - a

Algorithm - this is a Pegasus algorithm; we are looking for alfa such, that value of yield function F==0.
This alfa is found and written to a.

*/

	double F0, F1,Falfa, epsini[7], epsfini[7], epsalfa[7], sini[6], sfini[6], salfa[6], alfa0, alfa1, alfa, alfaold;
	alfa0=0;
	alfa1=1;
	alfa=0.5;
	alfaold=0;
	bool problems=false;
/*
	cout<<"Parameters are set to:"<<endl;
	cout<<"Maximum number of iteration MAXITER:"<<MAXITER<<endl;
	cout<<"Value of ALFACHECK - when the additional iter. is to enter:"<<ALFACHECK<<endl;
	cout<<"Value of change of alfa in the step:"<<ALFACHANGE<<endl;
	cout<<"alfa old/alfa ratio:"<<ALFARATIO<<endl;
*/	for (int i=0; i<6; i++) sini[i]=0;
	for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}

	// check of the second yield surface:
/*
	if ((eps0[6]+deps[6])>state[1])  //so suction is larger then maximum experienced suction
	{
	cout<<"WARNING SUCTION GREATER THAN S0!!!"<<endl;

		alfa0=(state[1]-eps0[6])/deps[6];
		for (int i=0; i<7; i++)
		{
			epsini[i]=deps[i]*alfa0;
		}
		CalcStressElast (state[2], s0, eps0, epsini, sini);
		for (int i=0; i<6;i++) sini[i]=sini[i]+s0[i];
		CheckYieldNormalised (state, sini, eps0[6]+epsini[6],&F0);
		// main F checked
		if (F0<YIELDTOL) //we may finish when there is no yield at this strain at F0. so..
		{
			*a=alfa0; // so we have elastic state in other case,
			return;
		}
		else
		{
			alfa1=alfa0; // although we have the upper limit of yield, better start at zero...
			alfa0=0;
		}
		//the second surface is the most important - and it is violated...
		//we have to find alfa depending on first surface - which is violated first
	}
*/
//now because the Main yield surface is valid, the whole check is made...
//however we start with alfa0 different if the suction was greater than maksimum suction...


	for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}

	CalcStressElast (state[2], s0, eps0, epsfini, sfini);

	for (int i=0; i<6; i++)
	{
		sini[i]=s0[i];
		sfini[i]=sfini[i]+s0[i];
	}
	CheckYield (state, sini, eps0[6]+epsini[6],&F0);
	CheckYield (state, sfini , eps0[6]+epsfini[6],&F1);

	//cout <<"Find Yield Modified Procedure"<<"  F0="<<F0<<endl<<"  F1="<<F1<<endl;
	//cout <<"Value of F0 should be negative, and value of F1 should be positive."<<endl;
	//cout <<"Values should be larger than tolerance for yield:"<<YIELDTOL<<endl;

for (int iter=0; iter<MAXITER; iter++)
{
	problems=false;
	alfaold=alfa;
	alfa=F0/(F0-F1);
	//cout<<"Alfa="<<alfa0<<endl;
	for (int i=0; i<7; i++)
	{
		epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
	}
	//here we need to update specific volume


	CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
	//state[2]=ComputeNu (salfa, state, eps0[6]+epsalfa[6]);
	for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
	CheckYieldNormalised (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
	//cout<<"In iteration "<<iter<<" alfa="<<alfa<<" and F="<<Falfa<<endl;

    if ((alfa1-alfa0)<TINY) Falfa=0.0; //if the difference is below numerical the accuracy, we have the solution

	if ((Falfa>-YIELDTOL)&&(Falfa<YIELDTOL))
		{
		*a=alfa0+alfa*(alfa1-alfa0);	//if Falfa within tolerance, we have the solution
		//cout<<"Solution in FindYieldModified procedure was found after "<<iter<<" iterations."<<endl;
		if (iter>50) {
				cout<<"Large number of iterations in FindYieldModified procedure!!! Solution is however correct..."<<endl;
		}
		//cout<<"solution is:"<<*a<<endl;
		return;
		}
	if (Falfa>0)
	{
	    CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);
		if (((1-alfa)<ALFACHECK)&&((1-alfaold)/(1-alfa))<ALFARATIO) problems=true;
		//alfa1=alfa1-(1-alfa)*(alfa1-alfa0); the line below is the same but faster
		alfa1=alfa0+alfa*(alfa1-alfa0);
		F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa

		if (problems)
		{
			//cout<<"Problematic iteration entered !!!"<<endl;
		alfa=alfa1-ALFACHANGE*(alfa1-alfa0);
				for (int i=0; i<7; i++)
				{
					epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
				}

				//here to update specific volume

				CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
				//state[2]=ComputeNu (salfa, state, eps0[6]+epsalfa[6]);
				for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
				CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa

			if (Falfa>0)
				{
					alfa1=alfa0+alfa*(alfa1-alfa0);
					F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa
				}
			else
				{
				alfa0=alfa0+alfa*(alfa1-alfa0);
				F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa
				}
		}

	}
	else
	{
	    CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);
		if ((alfa<ALFACHECK)&&(alfaold/alfa)<ALFARATIO) problems=true;
		alfa0=alfa0+alfa*(alfa1-alfa0);
		F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa

		if (problems)
		{
			//cout<<"Problematic iteration entered !!!"<<endl;
			alfa=alfa0+ALFACHANGE*(alfa1-alfa0);
					for (int i=0; i<7; i++)
				{
					epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
				}
				//here to update specific volume

				CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
				//state[2]=ComputeNu (salfa, state, eps0[6]+epsalfa[6]);
				for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
				CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
				if (Falfa>0)
					{
						alfa1=alfa0+alfa*(alfa1-alfa0);
						F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa
					}
				else
					{
					alfa0=alfa0+alfa*(alfa1-alfa0);
					F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa
					}
		}

	}
}
// if we are here, we must have perforemed to many iterations...
cout<<"Error in procedure FindYieldModified"<<endl;
cout<<"After "<<MAXITER<<" iterations crossing point not found"<<endl;
cout<<"Alphamin="<<alfa0<<" Alphamax="<<alfa1<<" dAlpha="<<alfa1-alfa0;
cout<<"Yield Function value Min="<<F0<<" Max="<<F1<<endl;
cout<<"Stress:"<<s0[0]<<' '<<s0[1]<<' '<<s0[2]<<' '<<' '<<s0[3]<<' '<<s0[4]<<' '<<s0[5]<<endl;
cout<<"Strain:"<<deps[0]<<' '<<deps[1]<<' '<<deps[2]<<' '<<' '<<deps[3]<<' '<<deps[4]<<' '<<deps[5]<<endl;
cout<<"G:"<<G<<" K:"<<K<<" Cohesion:"<<Cohesion<<" Phi:"<<Phi<<endl;
cout<<"This is likely to cause incorrect results... Results obtained should not be taken too seriously..."<<endl;
getchar();
}

void ShengMohrCoulomb::FindIntersection (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElasticStrain, double * PurelyPlasticStrain)
{
double alfa;
FindYieldModified (InitialPoint->state, InitialPoint->stress, InitialPoint->strain, StrainIncrement, &alfa);
for (int i=0; i<7; i++)
	{
		PurelyElasticStrain[i]=alfa*StrainIncrement[i];
		PurelyPlasticStrain[i]=StrainIncrement[i]-PurelyElasticStrain[i];
	}
}






void ShengMohrCoulomb::MoveYieldaBit (double * state, double * s0, double *ds, double * eps0, double * deps, double * gradient, double F0)
/*

This function is to move a point a little bit from just outside the yield locus, just inside the yield locus.
This movement should be made within given tolerance. This - of course - raise some additional problems...
Especially about tolerance levels...

Algorithm is modified to what it was originally thought.

*/
{
double  ddeps[7], dds[6];
double ddF,x ;
F0=F0+ADDTOLYIELD*YIELDTOL;
for (int i=0; i<7; i++) ddeps[i]=-CHGEPSINTOL*eps0[i];
CalcStressElast (state[2], s0  ,eps0, ddeps,  dds);
for (int i=0; i<6; i++) dds[i]=dds[i]+s0[i];
CheckYield (state, dds, eps0[6]+ddeps[6],&ddF);
ddF=ddF+ADDTOLYIELD*YIELDTOL;  //so we added the same to both values...
//now x from doc file is calculated
x=(F0*CHGEPSINTOL)/(F0-ddF);
for (int i=0; i<7;i++) ddeps[i]=-x*eps0[i];
CalcStressElast (state[2], s0  ,eps0, ddeps,  dds);
for (int i=0; i<6; i++) dds[i]=dds[i]+s0[i];
CheckYield (state, dds, eps0[6]+ddeps[6],&ddF);
if (ddF>0) cout<<"Error in procedure MoveYieldaBit; Yield Function ddF is greater then zero...";
else {
	for (int i=0; i<6; i++)
	{
	deps[i]=deps[i]-ddeps[i];
	eps0[i]=eps0[i]+ddeps[i];
	s0[i]=dds[i];
	}
	deps[6]=deps[6]-ddeps[6]; //generally speaking ddeps <0 and adding means substracting...
	eps0[6]=eps0[6]+ddeps[6];
	}
return;
}

void ShengMohrCoulomb::FindIntersectionUnloading (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElastic, double * PurelyPlastic)
{
double alfa;
FindYieldYield (InitialPoint->state, InitialPoint->stress, InitialPoint->strain, StrainIncrement, &alfa);
for (int i=0; i<7; i++)
{
	PurelyElastic[i]=alfa*StrainIncrement[i];
	PurelyPlastic[i]=StrainIncrement[i]-PurelyElastic[i];
}
}


void ShengMohrCoulomb::FindYieldYield (double *state, double* s0, double* eps0, double* deps, double* a)
{

/*
Main purpose of this procedure is to find the intersection between yield surface and the stress vector. This is a
special version of the procedure used in case of elastic-plastic unloading - when we have plastic state
at the beginning of the stei and then elastic unloading and plastic again. We need to know
at what value of epsilon (strain) we are on the surface - where plastic yielding will begin.

Parameters: state vector, initial stress s0, initial strain eps0, strain increment deps, a- to which value of alfa will be written

Changed - a

Algorithm - this is a Pegasus algorithm; we are looking for alfa such, that value of yield function F==0.
This alfa is found and written to a.

*/

	double F0, F1,Falfa, epsini[7], epsfini[7], epsalfa[7], sini[6], sfini[6], salfa[6], alfa0, alfa1, alfa, alfaold;
	double oldyieldtol, delta;
	alfa0=0;
	alfa1=1;
	alfa=0.5;
	alfaold=0;
	bool problems=false;
	CheckYieldNormalised (state, s0, eps0[6],&F0); //possibly the function value will be given...
	oldyieldtol=YIELDTOL;
	if (F0<0) {
		delta=0;
		YIELDTOL=-0.9*F0;
	}
	else	{
	//cout<<"FindYieldYield : case not coded for yet"<<endl;
	delta=(F0+YIELDTOL)/2;
	YIELDTOL=0.9*(YIELDTOL-F0)/2;
	}
//cout<<"Procedure Find Yield Yield ... s="<<s0[0]<<' '<<s0[1]<<' '<<s0[2]<<' '<<s0[3]<<' '<<s0[4]<<' '<<s0[5]<<' '<<endl;
//cout<<"Procedure Find Yield Yield ... dEps="<<deps[0]<<' '<<deps[1]<<' '<<deps[2]<<' '<<deps[3]<<' '<<deps[4]<<' '<<deps[5]<<' '<<endl;
//cout<<"Initial normalised Yield Locus... F0="<<F0<<endl;

//FindYieldModified(state, s0, eps0, deps, a);
//YIELDTOL=oldyieldtol;

//	cout<<"Delta="<<delta<<endl;
//	cout<<"F0="<<F0<<endl;
	cout<<"Procedure Find Yield Yield ... Delta="<<*a<<endl;
	cout<<"Call of this procedure is generally rare and most likely improper ... Delta="<<*a<<endl;




    /*
	cout<<"Parameters are set to:"<<endl;
	cout<<"Maximum number of iteration MAXITER:"<<MAXITER<<endl;
	cout<<"Value of ALFACHECK - when the additional iter. is to enter:"<<ALFACHECK<<endl;
	cout<<"Value of change of alfa in the step:"<<ALFACHANGE<<endl;
	cout<<"alfa old/alfa ratio:"<<ALFARATIO<<endl;
    */


	for (int i=0; i<6; i++) sini[i]=0;
	for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}


	for (int i=0; i<7; i++)
	{
		epsini[i]=deps[i]*alfa0;
		epsfini[i]=deps[i]*alfa1;
	}

	CalcStressElast (state[2], s0, eps0, epsfini, sfini);

	for (int i=0; i<6; i++)
	{
		sini[i]=s0[i];
		sfini[i]=sfini[i]+s0[i];
	}
	CheckYieldNormalised (state, sini, eps0[6]+epsini[6],&F0);
	CheckYieldNormalised (state, sfini , eps0[6]+epsfini[6],&F1);
	F0=F0-delta;
	F1=F1-delta;
//	cout <<"Procedure FindYieldYield. Initial values of F0="<<F0<<endl<<"  F1="<<F1<<endl;
//	cout <<"F0 should lie on the yield locus within tolerance:"<<oldyieldtol<<endl;

for (int iter=0; iter<MAXITER; iter++)
{
	problems=false;
	alfaold=alfa;

	alfa=F0/(F0-F1);
	for (int i=0; i<7; i++)
	{
		epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
	}
	CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
	for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
	CheckYieldNormalised (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
	Falfa=Falfa-delta;
	//cout<<"In iteration "<<iter<<" alfa="<<alfa<<" and F="<<Falfa<<endl;


	if ((Falfa>-YIELDTOL)&&(Falfa<YIELDTOL))
		{
		*a=alfa0+alfa*(alfa1-alfa0);	//if Falfa within tolerance, we have the solution
//		cout<<"Solution in FindYieldYield procedure was found after "<<iter<<" iterations."<<endl;
		if (iter>50) {
				getchar();
				cout<<"Large number of iterations!!! Solution is however correct... Press any key..."<<endl;
		}
//		cout<<"Value of the yield function is equal to:"<<Falfa<<endl;
//		cout<<"Modified value of tolerance is:"<<YIELDTOL<<endl;
//		cout<<"Value of delta is:"<<delta<<endl;
		//cout<<"Alfa is equal to: "<<*a<<endl;
		YIELDTOL=oldyieldtol;
//		cout<<"Yield tolerance is set back to:"<<YIELDTOL<<endl;
		return;
		}
	if (Falfa>0)
	{
		if (((1-alfa)<ALFACHECK)&&((1-alfaold)/(1-alfa))<ALFARATIO) problems=true;
		//alfa1=alfa1-(1-alfa)*(alfa1-alfa0); the line below is the same but faster
		alfa1=alfa0+alfa*(alfa1-alfa0);
		F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa

		if (problems)
		{
			//cout<<"Problematic iteration entered !!!"<<endl;
		alfa=alfa1-ALFACHANGE*(alfa1-alfa0);
				for (int i=0; i<7; i++)
				{
					epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
				}
				CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
				for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
				CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
				Falfa=Falfa-delta;
			if (Falfa>0)
				{
					alfa1=alfa0+alfa*(alfa1-alfa0);
					F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa
				}
			else
				{
				alfa0=alfa0+alfa*(alfa1-alfa0);
				F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa
				}
		}

	}
	else
	{
		if ((alfa<ALFACHECK)&&(alfaold/alfa)<ALFARATIO) problems=true;
		alfa0=alfa0+alfa*(alfa1-alfa0);
		F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa

		if (problems)
		{
			//cout<<"Problematic iteration entered !!!"<<endl;
			alfa=alfa0+ALFACHANGE*(alfa1-alfa0);
					for (int i=0; i<7; i++)
				{
					epsalfa[i]=alfa0*deps[i]+alfa*(alfa1-alfa0)*deps[i];
				}
				CalcStressElast (state[2] ,s0, eps0, epsalfa, salfa);	//calculated stress increment for current alfa
				for (int i=0; i<6; i++) salfa[i]=salfa[i]+s0[i];  //update stress
				CheckYield (state, salfa, eps0[6]+epsalfa[6],&Falfa);		// calculated yield function for current alfa
				Falfa=Falfa-delta;
				if (Falfa>0)
					{
						alfa1=alfa0+alfa*(alfa1-alfa0);
						F1=Falfa; //if Falfa >0 - we are yielding - max alfa set to current alfa
					}
				else
					{
					alfa0=alfa0+alfa*(alfa1-alfa0);
					F0=Falfa;	// if Falfa <0 - we are elastic - minimum alfa is set to current alfa
					}
		}

	}
}
YIELDTOL=oldyieldtol;
// if we are here, we must have perforemed to many iterations...
cout<<"Error in procedure FindYieldYield"<<endl;
cout<<"After "<<MAXITER<<" iterations crossing point not found"<<endl;
cout<<"This is likely to cause incorrect results... Results obtained should not be taken too seriously..."<<endl;



}


void ShengMohrCoulomb::read ()
{
	//reads data from file "ShengMohrCoulomb.dta"
	ifstream infile("ShengMohrCoulomb.dta", ios_base::in);

	// file opened
	string s;
	int slength=0, index=0, line=0;
	//double temp=0;


	do
	{

		getline (infile, s,'\n');
		line++;
		//cout << s <<" Prep: In line no:"<<line<<endl;
		//getchar ();
		if (!infile.good())
		{
			cout << "Wrong Data File";
			break;
		}
	}
	while (s!="***Start of data***");
	// I ignore file until "Start of data";

//Reading data - 5 material parameters
double storage[12];
		for (int j=0; j<4; j++)
			{

			getline (infile, s,'\n');	//read whole line
			line++;
			//cout << s <<"In line no:"<<line<<endl;
			//getchar ();
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

	SetModelParameters (storage[0],storage[1],storage[2],storage[3]); //G, K, Cohesion, Friction Angle


/*	KappaP=storage[1];
	KappaS=storage[2];
	PAtmos=storage[3];
	pc=storage[4];
	k=storage[5];
	r=storage[6];
	Beta=storage[7];
	LambdaZero=storage[8];
	NZero=storage[9];
	M=storage[10];
	if (storage [11]==0) NonAssociated=false; else NonAssociated=true;
if (NonAssociated)
	{
		alfa=(M*(M-9)*(M-3))/(9*(6-M)*(1-KappaP/LambdaZero));
		cout<<"Non associated flow rule used. Value of alfa:"<<alfa<<endl;
	}
else {
	alfa=1;
	}

	// finished  */
infile.close(); //close file
// all done
}


void ShengMohrCoulomb::write()
{
/*
	cout<<"Model Parameters:"<<endl;
	cout<<"Shear Modulus G="<<G<<endl;
	cout<<"Kappa p="<<KappaP<<endl;
	cout<<"Kappa s="<<KappaS<<endl;
	cout<<"Pc="<<pc<<endl;
	cout<<"k="<<k<<endl;
	cout<<"r="<<r<<endl;
	cout<<"Beta="<<Beta<<endl;
	cout<<"Lambda (0)="<<LambdaZero<<endl;
	cout<<"N (0)="<<NZero<<endl;
	cout<<"M="<<M<<" deg"<<endl;

	cout<<"Computed parameters:"<<endl;
	cout<<"K="<<K<<endl;
	cout<<"Ks="<<Ks<<endl;
	cout<<"Kp="<<Kp<<endl;*/
}


void ShengMohrCoulomb::PaintLocus (double *state, double suction, int Max)
{
//FILE *stream;
//stream = fopen( "yieldshape.dta", "w" );

 /* Reassign "stderr" to "freopen.out": */
//locus starts from -s0 and finishes at P0.
//double minP, maxP, PZero, difference;
/*
//cout<<Max<<endl; // in first line we put how many points we have...
fprintf( stream, "%d\n", Max );
minP=-k*suction;
PZero=LambdaZero*((1-r)*exp(-1*Beta*suction)+r);
PZero=(LambdaZero-KappaP)/(PZero-KappaP);
PZero=pc*pow((state[0]/pc),PZero);
maxP=PZero;
//cout<<minP<<endl;	//minimum
//cout<<maxP<<endl;	//maximum, used to set the view...
fprintf( stream, "%f\n",minP );
fprintf( stream, "%f\n",maxP );
difference=maxP-minP;	//this is the max difference...
//cout<<difference<<endl;
double p,q;

for (int i=0; i<Max; i++)
{
	p=i*difference/(Max-1)+minP;
	q=M*sqrt((p-minP)*(maxP-p));

	fprintf( stream, "%f\n", p);
	fprintf( stream, "%f\n", q);
 //point written to the file
}
*/
//fclose( stream ); //closing stream...
}


double ShengMohrCoulomb::CalculatePlasticConst (double * PurelyPlasticStrain, BBMPoint* Point, int StepNo)
{
double time,StressIncrAbs[7], RelativeError;
int NumberIter=StepNo;
int MethodSteps=6;

double MethodOrder=5;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 4th order solution, not an error estimate

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */
double C[6]={0.0 , 0.2 , 0.3 , 0.6 , 1 , 0.875};
double BRes[6]={37.0/378.0 , 0 , 250.0/621.0 , 125.0/594.0 , 0 , 512.0/1771.0};
double B[6]={2825.0/27648.0 , 0 , 18575.0/48384.0 , 13525.0/55296.0 , 277.0/14336.0 , 0.25};

A[0][0]=0;

A[1][0]=0.2;

A[2][0]=0.075;
A[2][1]=0.225;

A[3][0]=0.3;
A[3][1]= -0.9;
A[3][2]=1.2;

A[4][0]= -11.0/54.0;
A[4][1]= 2.5;
A[4][2]= -70.0/27.0;
A[4][3]= 35.0/27.0;

A[5][0]=1631.0/55296.0;
A[5][1]= 175.0/512.0;
A[5][2]=575.0/13824.0;
A[5][3]=44275.0/110592.0;
A[5][4]=253.0/4096.0;

time=RungeKuttaEqualStep (A, B,BRes, C, Point, PurelyPlasticStrain, StressIncrAbs, &RelativeError,   NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}


double ShengMohrCoulomb::CalculatePlastic (double * PurelyPlasticStrain, BBMPoint* Point)
{
double time=0,StressIncrAbs[7];
int NumberIter;



switch (SOLUTION_ALGORITHM)
{
case 1:
	{
	//calculate using Modified Euler scheme
	time=PlasticRKME221(Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 2:
	{
		//calculate using 3rd order RK scheme (Nystrom)
		time=PlasticRK332(Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 3:
	{
		//calculate using 3rd order RK scheme (Bogacki - Shampine)
		time=PlasticRKBog432(Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 4:
	{
		//calculate using 4th order RK scheme
		time=PlasticRK543(Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 5:
	{
		//calculate using 5th order RK scheme (England)
		time=PlasticRKEng654 (Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 6:
	{
		//calculate using 5th order RK scheme (Cash - Karp)
		time=PlasticRKCK654 (Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 7:
	{
		//calculate using 5th order RK scheme (Dormand - Prince)
		time=PlasticRKDP754 (Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 8:
	{
		//calculate using 5th order RK scheme (Bogacki - Shampine)
		time=PlasticRKErr8544 (Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
case 9:
	{
		//calculate using extrapolation method (Bulirsch - Stoer)
		time=PlasticExtrapol (Point,PurelyPlasticStrain,StressIncrAbs,&NumberIter);
	}
	break;
default:
	{
		cout<<"Unknown Solution Algorithm. Value of SOLUTION_ALGORITHM variable is set to:"<<SOLUTION_ALGORITHM<<endl;
		cout<<"Acceptable values are ints from 1 to 9. Procedure Calculate Plastic exits without any calculations done. Please press any key to continue"<<endl; getchar ();
	}
	break;
}



return time;
}





double ShengMohrCoulomb::PlasticEuler (BBMPoint* Point, double* EPStrain, double* AbsStress, int NumberIterations)
{

BBMMatrix DSIGMA (6,1);
double DPZeroStar;
double DSigma[6];
double CurrentStrain [7], PlasticStrain[6];
double PZeroStarTot=0;
double dSigmaDrift[6], dLambdaDrift;
BBMPoint OldPoint;
Point->Copy (&OldPoint);

vector <double> StressInc;
vector <double> StrainInc;
vector <double> PZeroLambda;
vector <double>::iterator Iter;

//cout<<"Euler Procedure Entered"<<endl;
//cout<<"Number of iterations in Euler Algorithm:"<<NumberIterations<<endl;

clock_t StartTime, EndTime;
StartTime=clock () ;

for (int i=0;i<7;i++) AbsStress[i]=0;
for (int i=0;i<7;i++) CurrentStrain[i]=EPStrain[i]/NumberIterations;
for (int loop=0; loop<NumberIterations; loop++)
{
	double FValue=0;
	if (USE_NICE_SCHEME>0) FValue=ComputeYieldFunction (Point);
	CalcPlastic (*Point, CurrentStrain, &DSIGMA, PlasticStrain, &DPZeroStar,FValue,dSigmaDrift,&dLambdaDrift);
	for (int i=0; i<6; i++) DSigma[i]=DSIGMA.GetElement(i+1,1);
	Point->Update (PlasticStrain, CurrentStrain, DSigma, DPZeroStar);

	PZeroLambda.push_back (DPZeroStar);
	PZeroLambda.push_back (PlasticStrain[0]);
	PZeroStarTot=PZeroStarTot+DPZeroStar;

	for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(DSigma[i]);
	AbsStress[6]=AbsStress[6]+fabs(DPZeroStar);

	/*for (int i=0; i<6; i++)
		{
		StressInc.push_back (DSigma[i]);  //is ok, as updated total stress?
		StrainInc.push_back (CurrentStrain [i]);
		}
	StrainInc.push_back (CurrentStrain [6]);*/
}

EndTime=clock () ;


/*
cout<<"Calculation took:"<<double(EndTime-StartTime)/CLOCKS_PER_SEC<<" s."<<endl;
cout<<"The INTEGRATION_TOL parameter is equal to:"<<INTEGRATION_TOL<<endl;
cout<<"Total number of steps done:"<<NumberIterations<<endl;

cout<<"Total PZeroStar change: "<<PZeroStarTot<<endl;
CheckYield (Point->state, Point->stress, Point->strain[6],&FValue);
cout<<"Yield Function value is:"<<FValue<<endl;

cout<<"Over the whole step change of stress is:"<<endl;
for (int i=0; i<6; i++)
{
	cout<<"s["<<i<<"]="<<Point->stress[i]-OldPoint.stress[i]<<endl;
}

cout<<"Initial specific volume="<<OldPoint.GetSpecVol()<<endl;
cout<<"Final specific volume="<<Point->GetSpecVol()<<endl;
cout<<"Change of specific volume is equal to="<<(OldPoint.GetSpecVol()-Point->GetSpecVol())<<endl;
cout<<"Change of mean stress p is equal to:"<<Point->GetMeanStress()-OldPoint.GetMeanStress()<<endl;
cout<<"Change of shear stress q is equal to:"<<Point->GetShearStress()-OldPoint.GetShearStress()<<endl;



FILE * StressFile;
FILE * StrainFile;
FILE * PZeroFile;
StressFile = fopen( "StressIncEuler.dta", "w" );

for (Iter=StressInc.begin() ; Iter!=StressInc.end(); )
{
for (int i=0; i<6; i++)
	{
	fprintf( StressFile, "%.20f , ",*Iter);
	Iter++;
	}
fprintf( StressFile, "\n");
}
fclose (StressFile);

StrainFile = fopen( "StrainIncEuler.dta", "w" );

for (Iter=StrainInc.begin() ; Iter!=StrainInc.end(); )
{
for (int i=0; i<7; i++)
	{
	fprintf( StrainFile, "%.20f , ",*Iter);
	Iter++;
	}
fprintf( StrainFile, "\n");
}
fclose (StrainFile);


PZeroFile = fopen( "PZeroIncEuler.dta", "w" );
for (Iter=PZeroLambda.begin() ; Iter!=PZeroLambda.end(); )
{
for (int i=0; i<2; i++)
	{
	fprintf( PZeroFile, "%.25f , ",*Iter);
	Iter++;
	}
fprintf( PZeroFile, "\n");
}
fclose (PZeroFile);



cout<<"Press any key..."<<endl;
getchar();
*/

return (EndTime-StartTime);
}


double ShengMohrCoulomb::RungeKutta (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter,
							 double MethodOrder, int MethodSteps, bool ErrorEstimate)
/*
This procedure calculate any Runge - Kutta pair, given the coefficient of
stress in A for each x used to calculate values in C, where B gives the coefficient to calculate
error estimate (or lower order solution) and BRes to calculate result. The procedure will integrate whole step of strain
re-using the initial derivative for rejected steps.
MethodOrder contain order of the method (required for substep prediction)
MethodSteps contain the number of stages to get the result
ErrorEstimate if true - the B table contains error estimate. If false, it contains 4th order solution
*/
{
BBMMatrix DSIGMA (6,1);
BBMPoint MidPoint[8], OldPoint, TrialPoint;

double CriticalStepSize=1E-4; //HACK HACK HACK - prevents algorithm to have more than 1e4 steps; cost: accuracy reduction in some unusual cases, but the whole thing will keep on going

double DSigma[8][6], DSigmaTemp[7], Result[7];
double DPZeroStar[8],DPZeroStarTemp, PlasticStrainTemp[6], PlasticStrain[8][6];
double Error[7],ReUseRes[7],RError, NewStepSize,TotalSize, StepLength, Temp, MethodPower;
double dSigmaDrift[6], dSigmaDriftConst[6], dSigmaDriftOldConst[6], dLambdaDrift;

double Frequency=100000/MethodOrder; //how often display info about steps
bool ReUseStep=false;

StepLength=1;
TotalSize=0; NewStepSize=1;
MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;

for (int i=0; i<MethodOrder; i++) {
	for (int j=0; j<6; j++) DSigma[i][j]=0;
	DPZeroStar[i]=0;
	dSigmaDrift[i]=0;
	}

bool Finished=false, StepAccepted=false;
double SubstepStrain [7], CurrentStrain[7];
double MicroStep=0;
double StepAccuracyCheck;

for (int i=0; i<7; i++)
{
	SubstepStrain[i]=EPStrain[i];
	AbsStress[i]=0;
}
int StepNo=0;
vector <double> StressInc;
vector <double> StrainInc;
vector <double> PZeroLambda;
vector <double>::iterator Iter;

clock_t StartTime, EndTime;
StartTime=clock();

NewStepSize=0;
for (int i=0; i<6; i++) NewStepSize=NewStepSize+fabs(SubstepStrain[i]);
NewStepSize=0.01/(NewStepSize*MethodOrder);
//cout<<"NewStepSize="<<NewStepSize<<endl;
if (NewStepSize>1) NewStepSize=1;


do
{
StepAccepted=false;
StepNo++;

/*if (StepNo>1)
{
	if (NewStepSize>10) NewStepSize=10;
	if (NewStepSize<0.1) NewStepSize=0.1;
}*/   //limiting step increase/decrease does not improve results/ enables faster convergence...

StepLength=StepLength*NewStepSize;  // size of a step
if (StepLength<CriticalStepSize) StepLength=CriticalStepSize; //HACK here, it is a fairly large value
if ((StepLength+TotalSize)>1) StepLength=1-TotalSize; //check whether the step not exceed the whole increment


for (int i=0; i<7; i++)
{
	SubstepStrain[i]=StepLength*EPStrain[i]; // strain increment in current step
	CurrentStrain[i]=0;
}
RError=0;

//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<CurrentStrain[0]<<endl;

for (int i=0; i<MethodSteps; i++) Point->Copy (&MidPoint[i]); //point is unchanged in  procedure

//Below the main R-K loop to calculate the value of intermediate stresses; values stored in DSigmaTemp[][]
//ReUseStep=false;
if (ReUseStep)
{
	for (int i=0; i<6; i++)
	{
		DSigma[0][i]=ReUseRes[i];
		PlasticStrain[0][i]=PlasticStrainTemp[i];
	}
	DPZeroStar[0]=ReUseRes[6];
	//add line about plastic strain...
}
else
{
	switch (ALGORITHM_TYPE)
	{
	case 0:
		{
	CalcPlastic (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0], ComputeYieldFunctionNN(Point),dSigmaDrift,&dLambdaDrift);

/*	BBMMatrix DEPTEST (6,7), SIGTEST (6,1), EPSTEST (7,1);
	for (int i=1; i<8; i++) EPSTEST.PutElement (i,1,SubstepStrain [i-1]);
	CalculateElastoPlasticTangentMatrix (&MidPoint[0],&DEPTEST);
	DEPTEST.Multiply (&EPSTEST,&SIGTEST);
	SIGTEST.Print ();
	DSIGMA.Print();
	getchar (); */
		}
	break;

	case 1:
    CalcPlasticFaster (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0],ComputeYieldFunctionNN(Point),dSigmaDrift,&dLambdaDrift);
	break;

	case 2:
	CalcPlasticPQ (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0],ComputeYieldFunctionNN(Point),dSigmaDrift,&dLambdaDrift);
	/*DSIGMA.Print();
	cout<<"DPZeroStar="<<DPZeroStar[0]<<endl;
	CalcPlastic (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0]);
	DSIGMA.Print();
	cout<<"DPZeroStar="<<DPZeroStar[0]<<endl; */

	break;

	default:
	CalcPlasticFaster (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0],ComputeYieldFunctionNN(Point),dSigmaDrift,&dLambdaDrift);
	break;
	}

	for (int i=0; i<6; i++)
	{
		DSigma[0][i]=DSIGMA.GetElement(i+1,1);
		PlasticStrain[0][i]=PlasticStrainTemp[i];
	}
}

if (USE_NICE_SCHEME==3) for (int i=0; i<6; i++) dSigmaDriftConst[i]=dSigmaDrift[i];
	else for (int i=0; i<6; i++) dSigmaDriftConst[i]=0.0;
if (USE_NICE_SCHEME==4) for (int i=0; i<6; i++)
{
	dSigmaDriftConst[i]=dSigmaDrift[i]+dSigmaDriftOldConst[i];
	dSigmaDriftOldConst[i]=dSigmaDrift[i];//+0.5*dSigmaDriftOldConst[i];
}


for (int rkloop=1; rkloop<MethodSteps; rkloop++)
{
	for (int i=0; i<6; i++)
	{DSigmaTemp[i]=0; PlasticStrainTemp[i]=0;}
	DPZeroStarTemp=0;
	for (int i=0; i<7; i++) CurrentStrain[i]=C[rkloop]*SubstepStrain[i];  //set the beginning point of the procedure
	for (int i=0; i<rkloop; i++)
		{
		for (int j=0; j<6; j++)
		{
			DSigmaTemp[j]=DSigmaTemp[j]+A[rkloop][i]*DSigma[i][j];
			PlasticStrainTemp[j]=PlasticStrainTemp[j]+A[rkloop][i]*PlasticStrain[i][j];
		}
		DPZeroStarTemp=DPZeroStarTemp+A[rkloop][i]*DPZeroStar[i];
		}
	MidPoint[rkloop].Update (PlasticStrainTemp, CurrentStrain, DSigmaTemp, DPZeroStarTemp);
	//double dummy;
	//BBMPoint TempPoint;
	//BBMMatrix TEMPMATRIX (6,7), TEMPEPSILON(7,1);
	//SubstepStrain[6]=0;
	//for (int i=1; i<8; i++) TEMPEPSILON.PutElement(i,1,SubstepStrain[i-1]);
	//MidPoint[rkloop].Copy(&TempPoint);
	double YieldFunctionValue;

	switch (USE_NICE_SCHEME)
	{
	case 0:
		break;

	case 1:
		{
		//nice scheme with yield function correction
		YieldFunctionValue=ComputeYieldFunctionNN(Point);
		dLambdaDrift=0;
		}
		break;

	case 2:
		{
			//nice scheme with carrying lambda
		YieldFunctionValue=0;
		}
		break;

	case 3:
		{
			//nice scheme with carrying stresses only
			YieldFunctionValue=0;
			dLambdaDrift=0;
		}
		break;

	case 4:
		{
			// predictive nice scheme
			cout<<"WARNING: Predictive NICE scheme is not accurate and may lead to large errors. Please use for debug purposes only"<<endl;
			YieldFunctionValue=0;
			dLambdaDrift=0;
		}
		break;


	default:
		{
		YieldFunctionValue=ComputeYieldFunctionNN(Point);
		dLambdaDrift=0;
		}
	}

	switch (ALGORITHM_TYPE)
	{
	case 0:
	CalcPlastic (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],YieldFunctionValue,dSigmaDrift,&dLambdaDrift);
	break;

	case 1:
    CalcPlasticFaster (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],YieldFunctionValue,dSigmaDrift,&dLambdaDrift);
	break;

	case 2:
	CalcPlasticPQ (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],YieldFunctionValue,dSigmaDrift,&dLambdaDrift);
	break;

	default:
	CalcPlastic (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],YieldFunctionValue,dSigmaDrift,&dLambdaDrift);
	break;
	}

	//CalcPlastic (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop]);
	//CalculateElastoPlasticTangentMatrix(&TempPoint,&TEMPMATRIX);
	//TEMPMATRIX.Multiply(&TEMPEPSILON,&TEMPMATRIX);
/*	for (int i=1; i<7; i++)
	{
		dummy=fabs(DSIGMA.GetElement(i,1)-TEMPMATRIX.GetElement(i,1));
		if (fabs(DSIGMA.GetElement(i,1))>TINY) dummy=dummy/fabs(DSIGMA.GetElement(i,1));
		if (dummy>0.01)
		{
			cout<<"Problems with the alternative matrix."<<i<<" Change of stress is:"<<DSIGMA.GetElement(i,1)<<endl;
			cout<<"Calculated change of stress with the DEP matrix is:"<<TEMPMATRIX.GetElement(i,1)<<endl;
			getchar();
		}
		else cout<<"*";
	}


*/
	for (int i=0; i<6; i++)
	{
		DSigma[rkloop][i]=DSIGMA.GetElement(i+1,1)+dSigmaDriftConst[i];
		PlasticStrain[rkloop][i]=PlasticStrainTemp[i];
	}
}

//needed: result, error

for (int i=0; i<6; i++)
{
	Result[i]=0;
	PlasticStrainTemp[i]=0;
	for (int j=0; j<MethodSteps; j++)
	{
		Result[i]=Result[i]+BRes[j]*DSigma[j][i];
		PlasticStrainTemp[i]=PlasticStrainTemp[i]+BRes[j]*PlasticStrain[j][i];
	}
}
Result[6]=0;
for (int j=0; j<MethodSteps; j++) Result[6]=Result[6]+BRes[j]*DPZeroStar[j];

for (int i=0; i<7; i++) Error[i]=0;

for (int i=0; i<MethodSteps; i++)
		{
		for (int j=0; j<6; j++) Error[j]=Error[j]+B[i]*DSigma[i][j];
		Error[6]=Error[6]+B[i]*DPZeroStar[i];
		}
if (!ErrorEstimate) for (int i=0; i<7; i++) Error[i]=Error[i]-Result[i]; //error estimate calculated in case we have lower order solution instead of error estimate

//Check the error norm

switch (int (TOL_METHOD))
	{
	case 0:
		{
		//RError=CheckNorm (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		RError=CheckNorm (Result, Result[6], Point, Error);	//returns RError
		}
		break;
	case 1:
		{
			//SLOAN NORM
		//RError=CheckNormSloan (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		RError=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}

for (int i=0; i<7; i++) if (!finite(Result[i]))
	{
	cout<<"Results not a number. Correcting issue, but results may be incorrect..."<<endl;
	Result [i]=0;
	RError=TINY;
	//if (RError<MethodPower) RError=MethodPower;
	}


//if ((Point->GetMeanStress()+(Result[0]+Result[1]+Result[2]))/3<0) if (RError<MethodPower) RError=MethodPower;
//if ((Point->GetPStar()+Result[6])<0) if (RError<MethodPower) RError=MethodPower;

if (RError<INTEGRATION_TOL)
{
	StepAccepted=true ;
}
else
{
	StepAccepted=false;
	if (StepLength<=CriticalStepSize) {StepAccepted=true;
                        //StepLength=1E-10;
                        //HACK here - you need to adjust all the check in the procedure to the same value
                            }
}

if (RError<TINY) RError=TINY;
if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));

//cout<<BETA_FACT;
//cout<<"What is going on????"<<endl;

if (!StepAccepted)
{
	//here we should take care about correct re - usage of the first evaluation of derivative
	ReUseStep=true;
	for (int i=0; i<6; i++)
	{
		ReUseRes[i]=DSigma[0][i]*NewStepSize;
		PlasticStrainTemp[i]=PlasticStrain[0][i]*NewStepSize;
	}
	ReUseRes[6]=DPZeroStar[0]*NewStepSize;
}
else
{
	//here we need to update all the point data.
	Point->Copy(&OldPoint);
	Point->Update (PlasticStrainTemp, SubstepStrain, Result, Result[6]);
	Point->Copy(&TrialPoint);
	//this value is not used in any calculations, it is just to show at the end of the step the p0* increase

	if (DRIFT_CORRECTION==3) CorrectDrift (Point);
	if (DRIFT_CORRECTION==2) CorrectDriftBeg(Point, &OldPoint); //value of OldPoint copied before updating the point

	//re - evaluate the error in the point:

	for (int i =0; i<6; i++)
	{
		Error[i]=Error[i]+Point->stress[i]-TrialPoint.stress[i];

	}
	Error[6]=Error[6]+Point->GetPStar()-TrialPoint.GetPStar();
	//error vector updated, norm should be re-evaluated:

	switch (int (TOL_METHOD))
	{
	case 0:
		{
		//Temp=CheckNorm (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		Temp=CheckNorm (Result, Result[6], Point, Error);	//returns RError
		//if (Temp>RError) RError=Temp;
		}
		break;
	case 1:
		{
			//SLOAN NORM
		//Temp=CheckNormSloan (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		Temp=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
		//if (Temp>RError) RError=Temp;
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}
if (!finite(RError)) if (RError<MethodPower) RError=MethodPower;

if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));

for (int i=0; i<6; i++) dSigmaDriftOldConst[i]=dSigmaDriftOldConst[i]*NewStepSize;
//cout<<dSigmaDriftOldConst[0]<<endl; getchar();

if (RError<INTEGRATION_TOL) StepAccepted=true ;
else
{
	StepAccepted=false;
	if (StepLength<=CriticalStepSize)
            {StepAccepted=true;
            //StepLength=1E-6;
            }
	//HACK HERE: this is relatively large value
}

if (!StepAccepted)
{
	ReUseStep=true;
	for (int i=0; i<6; i++)
	{
		ReUseRes[i]=DSigma[0][i]*NewStepSize;
		PlasticStrainTemp[i]=PlasticStrain[0][i]*NewStepSize;
	}
	ReUseRes[6]=DPZeroStar[0]*NewStepSize;
	OldPoint.Copy(Point);
	Temp=double(StepNo)/Frequency;
	if (modf(Temp,&Temp)==0)
	{
		cout<<"Step number:"<<StepNo<<endl;
		cout<<"Total size done is: "<<TotalSize<<" of whole step. Current StepLength="<<StepLength<<endl;
		cout<<"Stress state"<<Point->stress[0]<<' '<<Point->stress[1]<<' '<<Point->stress[2]<<' '
		<<Point->stress[3]<<' '<<Point->stress[4]<<' '<<Point->stress[5]<<' '<<endl;
	 }
}
else
{
	//this may be done only after successful re-evaluation of the substep
	for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(Point->stress[i]-OldPoint.stress[i]);
	AbsStress[6]=AbsStress[6]+fabs(Point->GetPStar()-OldPoint.GetPStar());
	ReUseStep=false;
	StepAccuracyCheck=TotalSize;
	MicroStep=MicroStep+StepLength;
	TotalSize=TotalSize+MicroStep; //total part of step done updated
	MicroStep=MicroStep-(TotalSize-StepAccuracyCheck);
	if (TotalSize>=1) Finished=true;
	Temp=double(StepNo)/Frequency;
	if (modf(Temp,&Temp)==0)
	{
		cout<<"Step number:"<<StepNo<<endl;
		cout<<"Total size done is: "<<TotalSize<<" of whole step. Current StepLength="<<StepLength<<endl;
		cout<<"Stress state"<<Point->stress[0]<<' '<<Point->stress[1]<<' '<<Point->stress[2]<<' '
		<<Point->stress[3]<<' '<<Point->stress[4]<<' '<<Point->stress[5]<<' '<<endl;
	 }
/*
	//Debug
	StrainInc.push_back (StepLength);
	for (int i=0; i<6; i++) StressInc.push_back (Point->stress[i]);
	StressInc.push_back (Point->GetPStar());
	//End Debug */
}

}

}
while (!Finished);
EndTime=clock () ;
*NumberIter=StepNo;
/*
//debug
FILE * ResultsFile;
ResultsFile = fopen( "ResultsRK.dta", "w" );

for (Iter=StressInc.begin();Iter!=StressInc.end();)
{
	for (int i=0; i<7; i++)
			{
			fprintf( ResultsFile, "%.15g , ",*Iter);
			Iter++;
			}
	fprintf( ResultsFile, "\n");
}
fclose (ResultsFile);

ResultsFile = fopen( "StrainRK.dta", "w" );
for (Iter=StrainInc.begin();Iter!=StrainInc.end();)
{
	fprintf( ResultsFile, "%.15g \n",*Iter);
	Iter++;
}
fclose (ResultsFile);
//end debug */

return(EndTime-StartTime);
}




double ShengMohrCoulomb::RungeKuttaEqualStep (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
									  double* RelError, int NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate)
/*
This procedure calculate any Runge - Kutta pair, given the coefficient of
stress in A for each x used to calculate values in C, where B gives the coefficient to calculate
error estimate (or lower order solution) and BRes to calculate result. The procedure will integrate whole step of strain
re-using the initial derivative for rejected steps.
MethodOrder contain order of the method (required for substep prediction)
MethodSteps contain the number of stages to get the result
ErrorEstimate if true - the B table contains error estimate. If false, it contains 4th order solution
*/

{
BBMMatrix DSIGMA (6,1);
BBMPoint MidPoint[8], OldPoint, TrialPoint;

double DSigma[8][6], DSigmaTemp[7], Result[7];
double DPZeroStar[8],DPZeroStarTemp, PlasticStrain[8][6], PlasticStrainTemp[6];
double Error[7],RError, TotRError, TotalSize, StepLength, Temp, MethodPower;
double Frequency=15000/MethodOrder; //how often display info about steps


for (int i=0; i<MethodOrder; i++) {
	for (int j=0; j<6; j++) DSigma[i][j]=0;
	DPZeroStar[i]=0;	}

//bool Finished=false, StepAccepted=false;
double SubstepStrain [7], CurrentStrain[7];
double MicroStep=0;
double StepAccuracyCheck;
double dStressDrift[6], dLambdaDrift;

int StepNo=0;
vector <double> StressInc;
vector <double> StrainInc;
vector <double> PZeroLambda;
vector <double>::iterator Iter;

StepLength=1.0/NumberIter;

for (int i=0; i<7; i++)
{
	CurrentStrain[i]=0;
	SubstepStrain[i]=StepLength*EPStrain[i]; // strain increment in all steps (equally sized)
	AbsStress[i]=0;
}
TotalSize=0; //NewStepSize=1;
MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;
//StepAccepted=true;
RError=0;
TotRError=0;

for (int loop=0; loop<NumberIter; loop++)
{

StepNo++;

/*if (StepNo>1)
{
	if (NewStepSize>10) NewStepSize=10;
	if (NewStepSize<0.1) NewStepSize=0.1;
}*/   //limiting step increase/decrease does not improve results/ enables faster convergence...



//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<CurrentStrain[0]<<endl;

for (int i=0; i<MethodSteps; i++) Point->Copy (&MidPoint[i]); //point is unchanged in CalcPlastic procedure

//Below the main R-K loop to calculate the value of intermediate stresses; values stored in DSigmaTemp[][]
//ReUseStep=false;


for (int rkloop=0; rkloop<MethodSteps; rkloop++)
{
	for (int i=0; i<6; i++) {DSigmaTemp[i]=0; PlasticStrainTemp[i]=0;}
	DPZeroStarTemp=0;
	for (int i=0; i<7; i++) CurrentStrain[i]=C[rkloop]*SubstepStrain[i];  //set the beginning point of the procedure
	for (int i=0; i<rkloop; i++)
		{
		for (int j=0; j<6; j++)
		{
			DSigmaTemp[j]=DSigmaTemp[j]+A[rkloop][i]*DSigma[i][j];
			PlasticStrainTemp[j]=PlasticStrainTemp[j]+A[rkloop][i]*PlasticStrain[i][j];
		}
		DPZeroStarTemp=DPZeroStarTemp+A[rkloop][i]*DPZeroStar[i];
		}
	MidPoint[rkloop].Update (PlasticStrainTemp, CurrentStrain, DSigmaTemp, DPZeroStarTemp);
	CalcPlastic (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);
	for (int i=0; i<6; i++)
	{
		DSigma[rkloop][i]=DSIGMA.GetElement(i+1,1);
		PlasticStrain[rkloop][i]=PlasticStrainTemp[i];
	}
	}

//needed: result, error

for (int i=0; i<6; i++)
{
	Result[i]=0;
	PlasticStrainTemp[i]=0;
	for (int j=0; j<MethodSteps; j++)
	{
		Result[i]=Result[i]+BRes[j]*DSigma[j][i];
		PlasticStrainTemp[i]=PlasticStrainTemp[i]+BRes[j]*PlasticStrain[j][i];
	}
}
Result[6]=0;
for (int j=0; j<MethodSteps; j++) Result[6]=Result[6]+BRes[j]*DPZeroStar[j];

for (int i=0; i<7; i++) Error[i]=0;

for (int i=0; i<MethodSteps; i++)
		{
		for (int j=0; j<6; j++) Error[j]=Error[j]+B[i]*DSigma[i][j];
		Error[6]=Error[6]+B[i]*DPZeroStar[i];
		}
if (!ErrorEstimate) for (int i=0; i<7; i++) Error[i]=Error[i]-Result[i]; //error estimate calculated in case we have lower order solution instead of error estimate


//Check the error norm

switch (int (TOL_METHOD))
	{
	case 0:
		{
		RError=CheckNorm (Result, Result[6], Point, Error);	//returns RError
		}
		break;
	case 1:
		{
		//SLOAN NORM
		RError=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
		//cout<<"Runge - Kutta constant size procedure. RError="<<RError<<endl;
		//cout<<"Error[0]="<<Error[0]<<"  Result[0]="<<Result[0]<<endl;
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}

//cout<<"Procedure R-K constant. RError="<<RError<<endl;

for (int i=0; i<7; i++) if (!finite(Result[i]))
	{
	Result [i]=0;
	if (RError<MethodPower) RError=MethodPower;
	}


if ((Point->GetMeanStress()+(Result[0]+Result[1]+Result[2]))/3<0)
{
	//cout<<"Mean Stress less then 0!!! Result:"<<((Result[0]+Result[1]+Result[2])/3)<<"  Mean stress:"<<Point->GetMeanStress()<<endl;

	if (RError<MethodPower) RError=MethodPower;
}
if ((Point->GetPStar()+Result[6])<0)
{
	//cout<<"P Zero Star less then 0!!!"<<endl;
	if (RError<MethodPower) RError=MethodPower;
}
//here we need to update all the point data.
Point->Copy(&OldPoint);
Point->Update (PlasticStrainTemp, SubstepStrain, Result, Result[6]);
Point->Copy(&TrialPoint);
//this value is not used in any calculations, it is just to show at the end of the step the p0* increase

//cout<<"Procedure R-K constant. RError="<<RError<<endl;

if (DRIFT_CORRECTION==3) CorrectDrift (Point);
if (DRIFT_CORRECTION==2) CorrectDriftBeg(Point, &OldPoint); //value of OldPoint copied before updating the point

//re - evaluate the error in the point:

for (int i =0; i<6; i++)
{
	Error[i]=Error[i]+Point->stress[i]-TrialPoint.stress[i];

}
Error[6]=Error[6]+Point->GetPStar()-TrialPoint.GetPStar();
//error vector updated, norm should be re-evaluated:

switch (int (TOL_METHOD))
{
case 0:
	{
	Temp=CheckNorm (Result, Result[6], Point, Error);	//returns RError
	if (Temp>RError) RError=Temp;
	}
	break;
case 1:
	{
		//SLOAN NORM
	Temp=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
	if (Temp>RError) RError=Temp;
	}
	break;
default:
	{
		cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
		getchar ();
	}
}
if (!finite(RError)) if (RError<MethodPower) RError=MethodPower;

//cout<<"Procedure R-K constant. RError="<<RError<<endl;
TotRError=TotRError+RError;
//cout<<"Procedure R-K constant. Total RError="<<TotRError<<endl;

//this is done anyway
for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(Point->stress[i]-OldPoint.stress[i]);
AbsStress[6]=AbsStress[6]+fabs(Point->GetPStar()-OldPoint.GetPStar());
StepAccuracyCheck=TotalSize;
MicroStep=MicroStep+StepLength;
TotalSize=TotalSize+MicroStep; //total part of step done updated
MicroStep=MicroStep-(TotalSize-StepAccuracyCheck);
//if (TotalSize>=1) Finished=true;
Temp=double(StepNo)/Frequency;
/*if (modf(Temp,&Temp)==0)
	{
		cout<<"Step number:"<<StepNo<<endl;
		cout<<"Total size done is: "<<TotalSize<<" of whole step. Current StepLength="<<StepLength<<endl;
	 }*/
/*
	//Debug
	StrainInc.push_back (StepLength);
	for (int i=0; i<6; i++) StressInc.push_back (Point->stress[i]);
	StressInc.push_back (Point->GetPStar());
	//End Debug */
}

*RelError=TotRError/NumberIter;

/*
//debug
FILE * ResultsFile;
ResultsFile = fopen( "ResultsRK.dta", "w" );

for (Iter=StressInc.begin();Iter!=StressInc.end();)
{
	for (int i=0; i<7; i++)
			{
			fprintf( ResultsFile, "%.15g , ",*Iter);
			Iter++;
			}
	fprintf( ResultsFile, "\n");
}
fclose (ResultsFile);

ResultsFile = fopen( "StrainRK.dta", "w" );
for (Iter=StrainInc.begin();Iter!=StrainInc.end();)
{
	fprintf( ResultsFile, "%.15g \n",*Iter);
	Iter++;
}
fclose (ResultsFile);
//end debug */

return(0);
}

double ShengMohrCoulomb::RungeKuttaExtrapol (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
									  int* NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate)
{
/*
this procedure tend to use Runge Kutta scheme. In case the step is not accepted, and the substep size is just a bit smaller
than current one , instead of cancelling the step, it goes into extrapolation procedure, just to save the substep done.
*/


BBMMatrix DSIGMA (6,1);
BBMPoint TrialPoint, OldPoint;
double WORTH_EXTRAPOL=3;

//double DSigma[8][6];
//double DPZeroStar[8];
double RError, NewStepSize, TotalSize, StepLength, Temp;// MethodPower;
double Frequency=15000/MethodOrder; //how often display info about steps
//bool ReUseStep=false;

StepLength=1;
TotalSize=0; NewStepSize=1;
//MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;

//for (int i=0; i<MethodOrder; i++) {
	//for (int j=0; j<6; j++) DSigma[i][j]=0;
	//DPZeroStar[i]=0;	}

bool Finished=false, StepAccepted=false;
double SubstepStrain[7];// CurrentStrain[7];
double MicroStep=0;
double StepAccuracyCheck;

for (int i=0; i<7; i++)
{
	SubstepStrain[i]=EPStrain[i];
	AbsStress[i]=0;
}
int StepNo=0;
vector <double> StressInc;
vector <double> StrainInc;
vector <double> PZeroLambda;
vector <double>::iterator Iter;

clock_t StartTime, EndTime;
StartTime=clock();

//assuming that maximum tolerable step is 0.5% for order 2 Initial step size is being calculated:
NewStepSize=0;
for (int i=0; i<6; i++) NewStepSize=NewStepSize+fabs(SubstepStrain[i]);
NewStepSize=0.01/(NewStepSize*MethodOrder);
//cout<<"NewStepSize="<<NewStepSize<<endl;
if (NewStepSize>1) NewStepSize=1;
//getchar(); */

do
{
StepAccepted=false;

/*if (StepNo>1)
{
	if (NewStepSize>10) NewStepSize=10;
	if (NewStepSize<0.1) NewStepSize=0.1;
}*/   //limiting step increase/decrease does not improve results/ enables faster convergence...



StepLength=StepLength*NewStepSize;  // size of a step
if ((StepLength+TotalSize)>1) StepLength=1-TotalSize; //check whether the step not exceed the whole increment

for (int i=0; i<7; i++)
{
	SubstepStrain[i]=StepLength*EPStrain[i]; // strain increment in current step
	//CurrentStrain[i]=0;
}
RError=0;


Point->Copy(&TrialPoint);
Point->Copy(&OldPoint);
RungeKuttaEqualStep (A, B,BRes, C,  &TrialPoint, SubstepStrain, AbsStress,
									   &RError, 2, MethodOrder, MethodSteps, ErrorEstimate);

StepNo=StepNo+2;
//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<SubstepStrain[0]<<endl;
//cout<<"RError="<<RError<<endl;
//getchar();


if (RError<INTEGRATION_TOL)
{
	StepAccepted=true ;
}
else
{
	StepAccepted=false;
	if (StepLength<1e-20) StepAccepted=true;
}

if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));


if (!StepAccepted)
{
	//check the extrapolation and go into it, if it looks sensible. otherwise reject the step...
	if (NewStepSize<WORTH_EXTRAPOL)
	{
		//double Result[7];
		int TempNumber=0;
		//for (int i=0; i<6; i++) Result[i]=TrialPoint.stress[i]-OldPoint.stress[i];
		//Result[6]=TrialPoint.GetPStar()-OldPoint.GetPStar();
		TrialPoint.Copy(&OldPoint);
		Point->Copy(&TrialPoint);
		RKExtrapolation(A, B,BRes, C, &TrialPoint, SubstepStrain, AbsStress
								  , &OldPoint, &RError, &TempNumber, MethodOrder, MethodSteps, ErrorEstimate); // Extrapolate and finally accept the step
		StepNo=StepNo+TempNumber;
		StepAccepted=true;
		Point->Copy(&OldPoint);
	}
	else ;
	/*	//here we should take care about correct re - usage of the first evaluation of derivative
	ReUseStep=true;
	for (int i=0; i<6; i++) ReUseRes[i]=DSigma[0][i]*NewStepSize;
	ReUseRes[6]=DPZeroStar[0]*NewStepSize; */ //No step re-using so far
	 //reject the step.
}

if (StepAccepted)
{
	//here we need to update all the point data. and nothing else...
	for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(OldPoint.stress[i]-TrialPoint.stress[i]);
	AbsStress[6]=AbsStress[6]+fabs(Point->GetPStar()-TrialPoint.GetPStar());
	TrialPoint.Copy(Point);
		//Point->Update(0,SubstepStrain,DSigmaTemp,OldPoint.GetPStar()-TrialPoint.GetPStar());
	//drift is already corrected
	//ReUseStep=false;
	StepAccuracyCheck=TotalSize;
	MicroStep=MicroStep+StepLength;
	TotalSize=TotalSize+MicroStep; //total part of step done updated
	MicroStep=MicroStep-(TotalSize-StepAccuracyCheck);
	if (TotalSize>=1) Finished=true;
	Temp=double(StepNo)/Frequency;
	if (modf(Temp,&Temp)==0)
	{
		cout<<"Step number:"<<StepNo<<endl;
		cout<<"Total size done is: "<<TotalSize<<" of whole step. Current StepLength="<<StepLength<<endl;
	 }
/*
	//Debug
	StrainInc.push_back (StepLength);
	for (int i=0; i<6; i++) StressInc.push_back (Point->stress[i]);
	StressInc.push_back (Point->GetPStar());
	//End Debug */
}

}
while (!Finished);
EndTime=clock () ;
*NumberIter=StepNo;
/*
//debug
FILE * ResultsFile;
ResultsFile = fopen( "ResultsRK.dta", "w" );

for (Iter=StressInc.begin();Iter!=StressInc.end();)
{
	for (int i=0; i<7; i++)
			{
			fprintf( ResultsFile, "%.15g , ",*Iter);
			Iter++;
			}
	fprintf( ResultsFile, "\n");
}
fclose (ResultsFile);

ResultsFile = fopen( "StrainRK.dta", "w" );
for (Iter=StrainInc.begin();Iter!=StrainInc.end();)
{
	fprintf( ResultsFile, "%.15g \n",*Iter);
	Iter++;
}
fclose (ResultsFile);
//end debug */
return(EndTime-StartTime);
}


double ShengMohrCoulomb::PlasticRKErr8544 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/* the procedure uses the embedded Runge - Kutta integration scheme with Adaptive Stepsize Control
the constants are as proposed by Bogacki and Shampine (1996), An efficient R-K (4,5) pair, Computers Math Applic, Vol 32 No 6 pp 15-28
with FSAL feauture the method allows for getting the error estimate and calculating value in one go
It is arguably better than any other 5(4) RK pair; It has double 4th order error estimate
*/

double A[8][8]; //matrix for aij components for RK method, as in the Fortran source www.netlib.org/ode/rksuite

A[0][0]=0;
A[1][0]= 1.0/6.0 ;
A[2][0]= 2.0/27.0;
A[2][1]= 4.0/27.0;
A[3][0]= 183.0/1372.0;
A[3][1]= -162.0/343.0;
A[3][2]= 1053.0/1372.0;
A[4][0]= 68.0/297.0;
A[4][1]= -4.0/11.0;
A[4][2]= 42.0/143.0;
A[4][3]= 1960.0/3861.0;
A[5][0]= 597.0/22528.0;
A[5][1]= 81.0/352.0;
A[5][2]= 63099.0/585728.0;
A[5][3]= 58653.0/366080.0;
A[5][4]= 4617.0/20480.0;
A[6][0]= 174197.0/959244.0;
A[6][1]= -30942.0/79937.0;
A[6][2]= 8152137.0/19744439.0;
A[6][3]= 666106.0/1039181.0;
A[6][4]= -29421.0/29068.0;
A[6][5]= 482048.0/414219.0;
A[7][0]= 587.0/8064.0;
A[7][1]= 0.0;
A[7][2]= 4440339.0/15491840.0;
A[7][3]= 24353.0/124800.0;
A[7][4]= 387.0/44800.0;
A[7][5]= 2152.0/5985.0;
A[7][6]= 7267.0/94080.0;

double B[8];//  The coefficients B[*] refer to the formula of order 4.

B[0] = 2479.0/34992.0;
B[1] = 0.0;
B[2] = 123.0/416.0;
B[3] = 612941.0/3411720.0;
B[4] = 43.0/1440.0;
B[5] = 2272.0/6561.0;
B[6] = 79937.0/1113912.0;
B[7] = 3293.0/556956.0;

/*  The coefficients E(*) refer to an estimate of the local error based on
C  the first formula of order 4.  It is the difference of the fifth order
C  result, here located in A(8,*), and the fourth order result.  By
C  construction both ErrorCoef[1] and ErrorCoef[6] are zero. */

double ErrorCoef [7];  //first error estimate, does not require knowing the result

ErrorCoef[0] = -3.0/1280.0;
ErrorCoef[1] = 0.0;
ErrorCoef[2] = 6561.0/632320.0;
ErrorCoef[3] = -343.0/20800.0;
ErrorCoef[4] = 243.0/12800.0;
ErrorCoef[5] = -1.0/95.0;
ErrorCoef[6] = 0.0;

double C[8]; //ci matrix, parameters for x

C[0] = 0.0;
C[1] = 1.0/6.0;
C[2] = 2.0/9.0;
C[3] = 3.0/7.0;
C[4] = 2.0/3.0;
C[5] = 3.0/4.0;
C[6] = 1.0;
C[7] = 1.0;

double BRes[8];
for (int i=0; i<7; i++) BRes[i]=A[7][i];
BRes[7]=0;  //as this scheme is the FSAL scheme

//All the RK matrices are put from the Fortran code, the indices are reduced by 1 and C notation is used
//BBMMatrix DSIGMA (6,1);
//BBMPoint MidPoint[8], OldPoint, TrialPoint;

int MethodSteps=8;
double MethodOrder=5.0;
bool ErrorEstimate=false;

//double time;
//time=RungeKutta (A,B,BRes,C,Point,EPStrain,AbsStress,NumberIter,MethodOrder,MethodSteps,false);
//return time;


BBMMatrix DSIGMA (6,1);
BBMPoint MidPoint[8], OldPoint, TrialPoint;

double DSigma[8][6], DSigmaTemp[7], Result[7];
double DPZeroStar[8],DPZeroStarTemp, PlasticStrainTemp[6], PlasticStrain[8][6];
double Error[7],ErrorOther[7],ReUseRes[7],RError,RErrorOther, NewStepSize,TotalSize, StepLength, Temp, MethodPower;
double Frequency=15000/MethodOrder; //how often display info about steps
double dStressDrift[6], dLambdaDrift;

bool ReUseStep=false;

StepLength=1;
TotalSize=0; NewStepSize=1;
MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;

for (int i=0; i<MethodOrder; i++) {
	for (int j=0; j<6; j++) DSigma[i][j]=0;
	DPZeroStar[i]=0;	}

bool Finished=false, StepAccepted=false;
double SubstepStrain [7], CurrentStrain[7];
double MicroStep=0;
double StepAccuracyCheck;

for (int i=0; i<7; i++)
{
	SubstepStrain[i]=EPStrain[i];
	AbsStress[i]=0;
}
int StepNo=0;
vector <double> StressInc;
vector <double> StrainInc;
vector <double> PZeroLambda;
vector <double>::iterator Iter;

clock_t StartTime, EndTime;
StartTime=clock();

NewStepSize=0;
for (int i=0; i<6; i++) NewStepSize=NewStepSize+fabs(SubstepStrain[i]);
NewStepSize=0.01/(NewStepSize*MethodOrder);
//cout<<"NewStepSize="<<NewStepSize<<endl;
if (NewStepSize>1) NewStepSize=1;


do
{
StepAccepted=false;
StepNo++;

/*if (StepNo>1)
{
	if (NewStepSize>10) NewStepSize=10;
	if (NewStepSize<0.1) NewStepSize=0.1;
}*/   //limiting step increase/decrease does not improve results/ enables faster convergence...

StepLength=StepLength*NewStepSize;  // size of a step
if ((StepLength+TotalSize)>1) StepLength=1-TotalSize; //check whether the step not exceed the whole increment


for (int i=0; i<7; i++)
{
	SubstepStrain[i]=StepLength*EPStrain[i]; // strain increment in current step
	CurrentStrain[i]=0;
}
RError=0;

//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<CurrentStrain[0]<<endl;

for (int i=0; i<MethodSteps; i++) Point->Copy (&MidPoint[i]); //point is unchanged in  procedure

//Below the main R-K loop to calculate the value of intermediate stresses; values stored in DSigmaTemp[][]
//ReUseStep=false;
if (ReUseStep)
{
	for (int i=0; i<6; i++)
	{
		DSigma[0][i]=ReUseRes[i];
		PlasticStrain[0][i]=PlasticStrainTemp[i];
	}
	DPZeroStar[0]=ReUseRes[6];
	//add line about plastic strain...
}
else
{
	CalcPlastic (MidPoint[0], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[0],ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);
	for (int i=0; i<6; i++)
	{
		DSigma[0][i]=DSIGMA.GetElement(i+1,1);
		PlasticStrain[0][i]=PlasticStrainTemp[i];
	}
}

for (int rkloop=1; rkloop<MethodSteps; rkloop++)
{
	for (int i=0; i<6; i++)
	{DSigmaTemp[i]=0; PlasticStrainTemp[i]=0;}
	DPZeroStarTemp=0;
	for (int i=0; i<7; i++) CurrentStrain[i]=C[rkloop]*SubstepStrain[i];  //set the beginning point of the procedure
	for (int i=0; i<rkloop; i++)
		{
		for (int j=0; j<6; j++)
		{
			DSigmaTemp[j]=DSigmaTemp[j]+A[rkloop][i]*DSigma[i][j];
			PlasticStrainTemp[j]=PlasticStrainTemp[j]+A[rkloop][i]*PlasticStrain[i][j];
		}
		DPZeroStarTemp=DPZeroStarTemp+A[rkloop][i]*DPZeroStar[i];
		}
	MidPoint[rkloop].Update (PlasticStrainTemp, CurrentStrain, DSigmaTemp, DPZeroStarTemp);
	//double dummy;
	//BBMPoint TempPoint;
	//BBMMatrix TEMPMATRIX (6,7), TEMPEPSILON(7,1);
	//SubstepStrain[6]=0;
	//for (int i=1; i<8; i++) TEMPEPSILON.PutElement(i,1,SubstepStrain[i-1]);
	//MidPoint[rkloop].Copy(&TempPoint);
	CalcPlastic (MidPoint[rkloop], SubstepStrain, &DSIGMA, PlasticStrainTemp, &DPZeroStar[rkloop],ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);
	//CalculateElastoPlasticTangentMatrix(&TempPoint,&TEMPMATRIX);
	//TEMPMATRIX.Multiply(&TEMPEPSILON,&TEMPMATRIX);
/*	for (int i=1; i<7; i++)
	{
		dummy=fabs(DSIGMA.GetElement(i,1)-TEMPMATRIX.GetElement(i,1));
		if (fabs(DSIGMA.GetElement(i,1))>TINY) dummy=dummy/fabs(DSIGMA.GetElement(i,1));
		if (dummy>0.01)
		{
			cout<<"Problems with the alternative matrix."<<i<<" Change of stress is:"<<DSIGMA.GetElement(i,1)<<endl;
			cout<<"Calculated change of stress with the DEP matrix is:"<<TEMPMATRIX.GetElement(i,1)<<endl;
			getchar();
		}
		else cout<<"*";
	}


*/
	for (int i=0; i<6; i++)
	{
		DSigma[rkloop][i]=DSIGMA.GetElement(i+1,1);
		PlasticStrain[rkloop][i]=PlasticStrainTemp[i];
	}
}

//needed: result, error

for (int i=0; i<6; i++)
{
	Result[i]=0;
	PlasticStrainTemp[i]=0;
	for (int j=0; j<MethodSteps; j++)
	{
		Result[i]=Result[i]+BRes[j]*DSigma[j][i];
		PlasticStrainTemp[i]=PlasticStrainTemp[i]+BRes[j]*PlasticStrain[j][i];
	}
}
Result[6]=0;
for (int j=0; j<MethodSteps; j++) Result[6]=Result[6]+BRes[j]*DPZeroStar[j];

for (int i=0; i<7; i++) Error[i]=0;

for (int i=0; i<MethodSteps; i++)
		{
		for (int j=0; j<6; j++) Error[j]=Error[j]+B[i]*DSigma[i][j];
		Error[6]=Error[6]+B[i]*DPZeroStar[i];
		}
if (!ErrorEstimate) for (int i=0; i<7; i++) Error[i]=Error[i]-Result[i]; //error estimate calculated in case we have lower order solution instead of error estimate

for (int i=0; i<7; i++) ErrorOther[i]=0;
for (int i=0; i<(MethodSteps-1); i++)
{
	for (int j=0; j<6; j++) ErrorOther[j]=ErrorOther[j]+ErrorCoef[i]*DSigma[i][j];
	ErrorOther[6]=ErrorOther[6]+ErrorCoef[i]*DPZeroStar[i];
}



//Check the error norm

switch (int (TOL_METHOD))
	{
	case 0:
		{
		//RError=CheckNorm (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		RError=CheckNorm (Result, Result[6], Point, Error);	//returns RError
		RErrorOther=CheckNorm (Result, Result[6], Point, ErrorOther);
		if (RError<RErrorOther) RError=RErrorOther;
		}
		break;
	case 1:
		{
			//SLOAN NORM
		//RError=CheckNormSloan (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		RError=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
		RErrorOther=CheckNormSloan (Result, Result[6], Point, ErrorOther);	//returns RError
		if (RError<RErrorOther) RError=RErrorOther;
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}

for (int i=0; i<7; i++) if (!finite(Result[i]))
	{
	Result [i]=0;
	if (RError<MethodPower) RError=MethodPower;
	}


if ((Point->GetMeanStress()+(Result[0]+Result[1]+Result[2]))/3<0) if (RError<MethodPower) RError=MethodPower;
if ((Point->GetPStar()+Result[6])<0) if (RError<MethodPower) RError=MethodPower;

if (RError<INTEGRATION_TOL)
{
	StepAccepted=true ;
}
else
{
	StepAccepted=false;
	if (StepLength<1e-20) StepAccepted=true;
}

if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));


if (!StepAccepted)
{
	//here we should take care about correct re - usage of the first evaluation of derivative
	ReUseStep=true;
	for (int i=0; i<6; i++)
	{

		ReUseRes[i]=DSigma[0][i]*NewStepSize;
		PlasticStrainTemp[i]=PlasticStrain[0][i]*NewStepSize;
	}
	ReUseRes[6]=DPZeroStar[0]*NewStepSize;
}
else
{
	//here we need to update all the point data.
	Point->Copy(&OldPoint);
	Point->Update (PlasticStrainTemp, SubstepStrain, Result, Result[6]);
	Point->Copy(&TrialPoint);
	//this value is not used in any calculations, it is just to show at the end of the step the p0* increase

	if (DRIFT_CORRECTION==3) CorrectDrift (Point);
	if (DRIFT_CORRECTION==2) CorrectDriftBeg(Point, &OldPoint); //value of OldPoint copied before updating the point

	//re - evaluate the error in the point:

	for (int i =0; i<6; i++)
	{
		Error[i]=Error[i]+Point->stress[i]-TrialPoint.stress[i];
		ErrorOther[i]=ErrorOther[i]+Point->stress[i]-TrialPoint.stress[i];
	}
	Error[6]=Error[6]+Point->GetPStar()-TrialPoint.GetPStar();
	ErrorOther[6]=ErrorOther[6]+Point->GetPStar()-TrialPoint.GetPStar();
	//error vector updated, norm should be re-evaluated:

	switch (int (TOL_METHOD))
	{
	case 0:
		{
		//Temp=CheckNorm (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		Temp=CheckNorm (Result, Result[6], Point, Error);	//returns RError
		RErrorOther=CheckNorm (Result, Result[6], Point, ErrorOther);	//returns RError
		if (Temp>RError) RError=Temp;
		if (RErrorOther>RError)RError=RErrorOther;
		}
		break;
	case 1:
		{
			//SLOAN NORM
		//Temp=CheckNormSloan (DSigmaTemp, DPZeroStarTemp, Point, Error);	//returns RError
		Temp=CheckNormSloan (Result, Result[6], Point, Error);	//returns RError
		RErrorOther=CheckNormSloan (Result, Result[6], Point, ErrorOther);	//returns RError
		if (Temp>RError) RError=Temp;
		if (RErrorOther>RError)RError=RErrorOther;
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}
if (!finite(RError)) if (RError<MethodPower) RError=MethodPower;

if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));



if (RError<INTEGRATION_TOL) StepAccepted=true ;
else
{
	StepAccepted=false;
	if (StepLength<1e-20) StepAccepted=true;
}

if (!StepAccepted)
{
	ReUseStep=true;
	for (int i=0; i<6; i++)
	{
		ReUseRes[i]=DSigma[0][i]*NewStepSize;
		PlasticStrainTemp[i]=PlasticStrain[0][i]*NewStepSize;
	}
	ReUseRes[6]=DPZeroStar[0]*NewStepSize;
	OldPoint.Copy(Point);
}
else
{
	//this may be done only after successful re-evaluation of the substep
	for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(Point->stress[i]-OldPoint.stress[i]);
	AbsStress[6]=AbsStress[6]+fabs(Point->GetPStar()-OldPoint.GetPStar());
	ReUseStep=false;
	StepAccuracyCheck=TotalSize;
	MicroStep=MicroStep+StepLength;
	TotalSize=TotalSize+MicroStep; //total part of step done updated
	MicroStep=MicroStep-(TotalSize-StepAccuracyCheck);
	if (TotalSize>=1) Finished=true;
	Temp=double(StepNo)/Frequency;
	if (modf(Temp,&Temp)==0)
	{
		cout<<"Step number:"<<StepNo<<endl;
		cout<<"Total size done is: "<<TotalSize<<" of whole step. Current StepLength="<<StepLength<<endl;
	 }
/*
	//Debug
	StrainInc.push_back (StepLength);
	for (int i=0; i<6; i++) StressInc.push_back (Point->stress[i]);
	StressInc.push_back (Point->GetPStar());
	//End Debug */
}

}

}
while (!Finished);
EndTime=clock () ;
*NumberIter=StepNo;
/*
//debug
FILE * ResultsFile;
ResultsFile = fopen( "ResultsRK.dta", "w" );

for (Iter=StressInc.begin();Iter!=StressInc.end();)
{
	for (int i=0; i<7; i++)
			{
			fprintf( ResultsFile, "%.15g , ",*Iter);
			Iter++;
			}
	fprintf( ResultsFile, "\n");
}
fclose (ResultsFile);

ResultsFile = fopen( "StrainRK.dta", "w" );
for (Iter=StrainInc.begin();Iter!=StrainInc.end();)
{
	fprintf( ResultsFile, "%.15g \n",*Iter);
	Iter++;
}
fclose (ResultsFile);
//end debug */

return(EndTime-StartTime);


}


double ShengMohrCoulomb::PlasticRKNoExTry (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/* It is a non - extrapolation procedure !!! */

/*
This procedure calculate stress increment using Runge - Kutta pair as given by England. The procedure consists of 6 stages and should be less
efficient than the RKErr8544 that uses Bogacki - Shimpine pair.
*/

{
int MethodSteps=6;

double MethodOrder=5;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 2nd order solution
/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */
double C[6]={0.0 , 1.0/3.0 , 0.4 , 1 , 2.0/3.0 , 0.8};
double BRes[6]={23/192.0 , 0 , 125/192.0 , 0.0 , -81.0/192.0 , 125.0/192.0};
double B[6]={ -0.5 , 1.5 , 0 , 0 , 0 , 0};

A[0][0]=0;

A[1][0]=1.0/3.0;

A[2][0]=4.0/25.0;
A[2][1]=6.0/25.0;

A[3][0]=0.25;
A[3][1]= -3.0;
A[3][2]=3.75;

A[4][0]= 6.0/81.0;
A[4][1]= 90/81.0;
A[4][2]= -50/81.0;
A[4][3]= 8.0/81.0;

A[5][0]=6.0/75.0;
A[5][1]=36.0/75.0;
A[5][2]=10.0/75.0;
A[5][3]=8.0/75.0;
A[5][4]=0.0;

time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}



double ShengMohrCoulomb::PlasticRKDP754 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using Runge - Kutta pair as given by DORMAND - PRINCE. The
used pair is known also as DOPRI5 or RK5(4)7FM. The procedure consists of 7 stages and should be less
efficient than the RKErr8544 that uses Bogacki - Shimpine pair.
*/

{
int MethodSteps=7;
double MethodOrder=5;
double time;
double A[8][8];
bool ErrorEstimate=false; //we give a 4th order solution, not an error estimate

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */
double C[7]={0.0 , 0.2 , 0.3 , 0.8 , 8.0/9.0 , 1 , 1};
double BRes[7]={35.0/384.0 , 0 , 500.0/1113.0 , 125.0/192.0 , -2187.0/6784.0 , 11.0/84.0 ,0};
double B[7]={5179.0/57600.0 , 0 , 7571.0/16695.0 , 393.0/640.0 , -92097.0/339200.0 , 187.0/2100.0 , 0.025};

A[0][0]=0;

A[1][0]=0.2;

A[2][0]=0.075;
A[2][1]=0.225;

A[3][0]=44.0/45.0;
A[3][1]= -56.0/15.0;
A[3][2]=32.0/9.0;

A[4][0]=19372.0/6561.0;
A[4][1]= -25360.0/2187.0;
A[4][2]=64448.0/6561.0;
A[4][3]= -212.0/729.0;

A[5][0]=9017.0/3168.0;
A[5][1]= -355.0/33.0;
A[5][2]=46732.0/5247.0;
A[5][3]=49.0/176.0;
A[5][4]= -5103.0/18656.0;

A[6][0]=35.0/384.0;
A[6][1]=0;
A[6][2]=500.0/1113.0;
A[6][3]=125.0/192.0;
A[6][4]= -2187.0/6784.0;
A[6][5]=11.0/84.0;

//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}

double ShengMohrCoulomb::PlasticRKCK654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using Runge - Kutta pair as given by Cash - Karp. The coeficients
are given in Numerical Recipes (Cambrige Univ Press) or Cash Karp (1990) ACM Transactions on Mathematical
software, vol 16, pp 201-222. The procedure consists of 6 stages and should be less
efficient than the RKErr8544 that uses Bogacki - Shimpine pair.
*/

{
int MethodSteps=6;

double MethodOrder=5;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 4th order solution, not an error estimate

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */
double C[6]={0.0 , 0.2 , 0.3 , 0.6 , 1 , 0.875};
double BRes[6]={37.0/378.0 , 0 , 250.0/621.0 , 125.0/594.0 , 0 , 512.0/1771.0};
double B[6]={2825.0/27648.0 , 0 , 18575.0/48384.0 , 13525.0/55296.0 , 277.0/14336.0 , 0.25};

A[0][0]=0;

A[1][0]=0.2;

A[2][0]=0.075;
A[2][1]=0.225;

A[3][0]=0.3;
A[3][1]= -0.9;
A[3][2]=1.2;

A[4][0]= -11.0/54.0;
A[4][1]= 2.5;
A[4][2]= -70.0/27.0;
A[4][3]= 35.0/27.0;

A[5][0]=1631.0/55296.0;
A[5][1]= 175.0/512.0;
A[5][2]=575.0/13824.0;
A[5][3]=44275.0/110592.0;
A[5][4]=253.0/4096.0;

time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}


double ShengMohrCoulomb::PlasticRKEng654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using Runge - Kutta pair as given by Sloan (1987). The coeficients
are given in Sloan (1987), Ordinary and Partial Differential Equations routines in... by Lee & Schiesser, Chapman & Hall 2003
or originally by England (1969) Error Estimates for Runge - Kutta type solutions to systems of ordinary differential equations,
Computer Journal 12 - 166-170. The procedure consists of 6 stages and should be the least
efficient from all the R-K pairs presented
*/

{
int MethodSteps=6;

double MethodOrder=5;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=true; //we give a 4th order error estimate

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */
double C[6]={0.0 , 0.5 , 0.5 , 1 , 2.0/3.0 , 0.2};
double BRes[6]={14.0/336.0 , 0 , 0 , 35.0/336.0 , 162.0/336.0 , 125.0/336.0};
double B[6]={ -42.0/336.0 , 0 , -224.0/336.0 , -21.0/336.0 , 162.0/336.0 , 125.0/336.0};

A[0][0]=0;

A[1][0]=0.5;

A[2][0]=0.25;
A[2][1]=0.25;

A[3][0]=0;
A[3][1]= -1.0;
A[3][2]=2.0;

A[4][0]= 7.0/27.0;
A[4][1]= 10.0/27.0;
A[4][2]= 0;
A[4][3]= 1.0/27.0;

A[5][0]=28.0/625.0;
A[5][1]=  -125.0/625.0;
A[5][2]=546.0/625.0;
A[5][3]=54.0/625.0;
A[5][4]= -378.0/625.0;

time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}

double ShengMohrCoulomb::PlasticRK543 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using 4-3 Runge - Kutta pair. The coeficients
are given in  Ordinary and Partial Differential Equations routines in... by Lee & Schiesser, Chapman & Hall 2003
The procedure consists of 5 stages and is 4th order accurate.
*/

{
int MethodSteps=5;

double MethodOrder=4;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=true; //we give a 3th order error estimate

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */

/*
double C[5]={0.0 , 1.0 , 1.0 , 1.5 , 3};
double LowOrdSolution[5]={0.5 , 0 , 0 , 2 , 0.5}
double BRes[5]={0.3 , 0 , 0.9 , 1.2 , 0.6};
double B[5]={ -0.2 , 0 , 0.9 , -0.8 , 0.1};

A[0][0]=0;

A[1][0]=1.0;

A[2][0]=0.5;
A[2][1]=0.5;

A[3][0]=0.375;
A[3][1]=0;
A[3][2]=1.125;

A[4][0]=1.5;
A[4][1]=0;
A[4][2]= -4.5;
A[4][3]= 6;

double TempStrain[7];
for (int i=0; i<7; i++) TempStrain[i]=EPStrain[i]/3; */

double C[5]={0.0 , 1.0/3.0 , 1.0/3.0 , 0.5 , 1};
//double LowOrdSolution[5]={1.0/6.0 , 0 , 0 , 2.0/3.0 , 1.0/6.0}
double BRes[5]={0.1 , 0 , 0.3 , 0.4 , 0.2};
double B[5]={ -1.0/15.0 , 0 , 0.3 , -4.0/15.0 , 1.0/30.0};

A[0][0]=0;

A[1][0]=1.0/3.0;

A[2][0]=0.5/3.0;
A[2][1]=0.5/3.0;

A[3][0]=0.125;
A[3][1]=0;
A[3][2]=0.375;

A[4][0]=0.5;
A[4][1]=0;
A[4][2]= -1.5;
A[4][3]=2.0;


time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}

double ShengMohrCoulomb::PlasticRK332 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using 3-2 Runge - Kutta pair. The coeficients
are given in  Ordinary and Partial Differential Equations routines in... by Lee & Schiesser, Chapman & Hall 2003
The procedure consists of 3 stages and is 3th order accurate.
*/

{
int MethodSteps=3;

double MethodOrder=3;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 2nd order solution

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */

double C[3]={0.0 , 2.0/3.0 , 2.0/3.0};
double BRes[3]={0.25 , 0.375 , 0.375};
double B[3]={ 0.25 , 0.75 , 0};

A[0][0]=0;

A[1][0]=2.0/3.0;

A[2][0]=0;
A[2][1]=2.0/3.0;


time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}

double ShengMohrCoulomb::PlasticRKBog432 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using 3-2 Runge - Kutta pair. The coeficients
are given in "A 3(2) Pair of Runge-Kutta Formulas" by P. Bogacki and L.F. Shampine, Appl. Math. Lett., 2, pp. 321-325, 1989.
The procedure consists of 3 stages and is 3th order accurate.
*/

{
int MethodSteps=4;

double MethodOrder=3;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 2nd order solution

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */

double C[4]={0.0 , 0.5 , 0.75 , 1.0};
double BRes[4]={2.0/9.0 , 1.0/3.0 , 4.0/9.0, 0};
double B[4]={ 7.0/24.0 , 0.25 , 1/3.0 ,0.125};

A[0][0]=0;

A[1][0]=0.5;

A[2][0]=0;
A[2][1]=0.75;

A[3][0]=2.0/9.0;
A[3][1]=1.0/3.0;
A[3][2]=4.0/9.0;


time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
return time;
}

double ShengMohrCoulomb::PlasticRKME221 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
/*
This procedure calculate stress increment using Modified Euler method
*/

{
int MethodSteps=2;
double MethodOrder=2;
double time;
double A[8][8]; //matrix must be this size to be used in the RungeKutta method
bool ErrorEstimate=false; //we give a 2nd order solution

/*A - matrix with coefficients, B - error estimate, BRes - result coefficients, C - x coefficients. */


double C[2]={0.0 , 1.0};
double BRes[2]={0.5 , 0.5};
double B[2]={ 1.0 , 0};
A[0][0]=0;
A[1][0]=1.0;


//Matrix for midpoint method
/*
double C[2]={0.0 , 0.5};
double BRes[2]={0 , 1.0};
double B[2]={ 1.0 , 0};
A[0][0]=0;
A[1][0]=0.5; */



*NumberIter=0;

//time=RungeKuttaEqualStep (A, B,BRes, C, Point, EPStrain, AbsStress, &RError, *NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//cout<<"RKME221. Error="<<RError<<endl;
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//cout<<"Runge Kutta Extrapol Finished. Number of iterations:"<<*NumberIter<<endl;
time=RungeKutta (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);
//time=RungeKuttaExtrapol (A, B,BRes, C, Point, EPStrain, AbsStress, NumberIter, MethodOrder, MethodSteps, ErrorEstimate);

return time;
}

 double ShengMohrCoulomb::PlasticMidpointGallipoli (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/*this procedure is to calculate the stress increase using the midpoint method with given number of iterations NumberIter.
is made mainly for use in the extrapolation procedure. It has no adaptive substepping or error control. It just integrates
the strain to get the stress in given number of substeps using the midpoint method
*/
//clock_t StartTime, EndTime;
//StartTime=clock();

BBMPoint NewPoint, MidPoint;
BBMMatrix SIGMA(6,1);
//vector <double> Result;
vector <double>::iterator Iter;

double DSigma[6], CurrentStrain[7], HalfCurrentStrain[7], PlasticStrain[6];
double h, DPZeroStar=0;
double dStressDrift[6], dLambdaDrift;

h=*NumberIter;
h=1/h;
for (int i=0;i<7;i++) AbsStress[i]=0;  //7 components...
for (int i=0; i<7; i++)
{
	CurrentStrain[i]=EPStrain[i]*h; // strain increment in current step
	HalfCurrentStrain[i]=0.5*CurrentStrain[i];
}
//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<CurrentStrain[0]<<endl;
/*
for (int i=0; i<6; i++) Result.push_back(Point->stress[i]);
Result.push_back(Point->GetPStar());
Result.push_back(Point->GetMeanStress());
Result.push_back(Point->GetShearStress());
*/

Point->Copy (&NewPoint); //point is unchanged in CalcPlastic procedure
Point->Copy (&MidPoint); //point is unchanged in CalcPlastic procedure
CalcPlastic (NewPoint,HalfCurrentStrain, &SIGMA, PlasticStrain, &DPZeroStar,ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);  //calculate the plastic stresses...
for (int i=0; i<6; i++) DSigma[i]=SIGMA.GetElement(i+1,1);
NewPoint.Update (PlasticStrain, HalfCurrentStrain, DSigma, DPZeroStar);
CalcPlastic (NewPoint,HalfCurrentStrain, &SIGMA, PlasticStrain, &DPZeroStar,ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);  //calculate the plastic stresses...

for (int loop=0; loop<2*(*NumberIter); loop++)
{

	MidPoint.Update (PlasticStrain, HalfCurrentStrain, DSigma, DPZeroStar);
	MidPoint.Copy (&NewPoint);
	NewPoint.Update(PlasticStrain, HalfCurrentStrain, DSigma, DPZeroStar);
	CalcPlastic (NewPoint,HalfCurrentStrain, &SIGMA, PlasticStrain, &DPZeroStar,ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);  //calculate the plastic stresses...
	for (int i=0; i<6; i++) DSigma[i]=SIGMA.GetElement(i+1,1);

	/*for (int i=0; i<6; i++) Result.push_back(Point->stress[i]);
	Result.push_back(Point->GetPStar());
	Result.push_back(Point->GetMeanStress());
	Result.push_back(Point->GetShearStress());*/
}
MidPoint.Copy(Point);
//EndTime=clock();
//return (EndTime-StartTime);
/*FILE *File;
File=fopen ("ConstStep.dta", "a+");

fprintf(File, "\n");
fprintf(File, "Data for %d substeps. \n", *NumberIter);
for (Iter=Result.begin() ; Iter!=Result.end(); )
		{
			for (int i=0; i<9; i++)
			{
			fprintf( File, "%.20f , ",*Iter);
			Iter++;
			}
		fprintf(File, "\n");
		}
fprintf(File, "\n");
fclose (File); */
return 0;
}



double ShengMohrCoulomb::PlasticMidpoint (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/*this procedure is to calculate the stress increase using the midpoint method with given number of iterations NumberIter.
is made mainly for use in the extrapolation procedure. It has no adaptive substepping or error control. It just integrates
the strain to get the stress in given number of substeps using the midpoint method
*/
//clock_t StartTime, EndTime;
//StartTime=clock();

BBMPoint NewPoint, MidPoint;
BBMMatrix SIGMA(6,1);
//vector <double> Result;
vector <double>::iterator Iter;

double DSigma[6], CurrentStrain[7], HalfCurrentStrain[7], PlasticStrain[6];
double h, DPZeroStar=0;
double dStressDrift[6], dLambdaDrift;

h=*NumberIter;
h=1/h;
for (int i=0;i<7;i++) AbsStress[i]=0;  //7 components...
for (int i=0; i<7; i++)
{
	CurrentStrain[i]=EPStrain[i]*h; // strain increment in current step
	HalfCurrentStrain[i]=0.5*CurrentStrain[i];
}
//cout<<"Step Length="<<StepLength<<endl;
//cout<<"Current strain [0]="<<CurrentStrain[0]<<endl;
/*
for (int i=0; i<6; i++) Result.push_back(Point->stress[i]);
Result.push_back(Point->GetPStar());
Result.push_back(Point->GetMeanStress());
Result.push_back(Point->GetShearStress());
*/
for (int loop=0; loop<*NumberIter; loop++)
{
	Point->Copy (&NewPoint); //point is unchanged in CalcPlastic procedure
	Point->Copy (&MidPoint);
	CalcPlastic (NewPoint,HalfCurrentStrain, &SIGMA, PlasticStrain, &DPZeroStar,ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);  //calculate the plastic stresses...
	for (int i=0; i<6; i++) DSigma[i]=SIGMA.GetElement(i+1,1);
	MidPoint.Update(PlasticStrain, HalfCurrentStrain, DSigma, DPZeroStar);
	CalcPlastic (MidPoint, HalfCurrentStrain, &SIGMA, PlasticStrain, &DPZeroStar,ComputeYieldFunctionNN(Point),dStressDrift, &dLambdaDrift);  //calculate the plastic stresses...
	for (int i=0; i<6; i++)
	{
		DSigma[i]=2*SIGMA.GetElement(i+1,1);
		PlasticStrain[i]=2*PlasticStrain[i];
	}
	DPZeroStar=2*DPZeroStar;

	for (int i=0; i<6; i++) AbsStress[i]=AbsStress[i]+fabs(DSigma[i]);
	AbsStress[6]=AbsStress[6]+fabs(DPZeroStar);
	Point->Update (PlasticStrain, CurrentStrain, DSigma, DPZeroStar);
	/*for (int i=0; i<6; i++) Result.push_back(Point->stress[i]);
	Result.push_back(Point->GetPStar());
	Result.push_back(Point->GetMeanStress());
	Result.push_back(Point->GetShearStress());*/
}
//EndTime=clock();
//return (EndTime-StartTime);
/*FILE *File;
File=fopen ("ConstStep.dta", "a+");

fprintf(File, "\n");
fprintf(File, "Data for %d substeps. \n", *NumberIter);
for (Iter=Result.begin() ; Iter!=Result.end(); )
		{
			for (int i=0; i<9; i++)
			{
			fprintf( File, "%.20f , ",*Iter);
			Iter++;
			}
		fprintf(File, "\n");
		}
fprintf(File, "\n");
fclose (File); */
return 0;
}


double ShengMohrCoulomb::PlasticExtrapol  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
//Here the DRIFT_CORRECTION parameter is used. 0=no correction, 1 - correction at the beginning (A),
//2 - correction at the end (B), 3 - Zero Drift Algorithm.
/*
switch (int(DRIFT_CORRECTION))
{
case 1 :
	{
		cout<<"Procedure proceed with standard algorithm and no drift correction."<<endl;
		break;
	}
case 2 :
	{
		cout<<"Procedure proceed with standard algorithm and drift correction at the beginning (point A)."<<endl;
		break;
	}

default :
	{
		cout<<"Unknown DRIFT_CORRECTION parameter. Parameter read is:"<<DRIFT_CORRECTION<<endl;
		break;
	}
}
*/

//no drift correction as may worsen the results. at least possibly.

clock_t StartTime, EndTime;
StartTime=clock();


int STEPMAX=15;
//int DivisionsInt[15]={2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
//int DivisionsInt[15]={2,4,6,8,10,12,14,16,18,20,22,24,26,28,30};
//int DivisionsInt[15]={2,4,6,8,12,16,24,32,48,64,96,128,192,256,384};
//int DivisionsInt[15]={2,4,6,8,12,16,24,32,48,64,96,128,192,256,384};
//int DivisionsInt[15]={12,16,24,32,48,64,96,128,160,192,256,320,384,448,512}; //fastests so far
int DivisionsInt[15]={32,48,64,96,128,160,192,256,320,384,448,512,608,736,992}; //fastests so far
//int DivisionsInt[15]={2,4,6,8,10,14,20,28,38,52,72,100,138,190,262,362,500,690,952}; //N+N-4
//int DivisionsInt[15]={2,4,6,8,10,12,16,22,30,40,52,68,90,120,170,222,290,380,500,670,892}; //N+N-5
//int DivisionsInt[15]={28,32,40,52,64,78,94,120,154,200,240,290,330,380,440};
//int DivisionsInt[15]={32,40,52,64,78,94,120,154,200,240,290,330,380,440,520};
//int DivisionsInt[15]={32,36,40,46,52,60,70,82,96,112,130,150,176,220,380};
//int DivisionsInt[15]={32,36,40,52,68,92,114,154,200,240,290,330,380,440,520};
//int DivisionsInt[15]={32,36,40,44,50,56,62,68,76,84,92,102,112,124,136}; //n=n-1*1.1, doesn't converge too often
//int DivisionsInt[15]={32,38,46,56,66,80,96,114,138,166,198,238,286,344,412}; //n=n-1*1.2
double Zeros[6]={0,0,0,0,0,0};

double Divisions[15]; for (int i=0; i<15;i++) Divisions[i]=DivisionsInt[i];

double ApproximationTable[16][20];  //six stresses, p0*, six absolute stresses, absolute p0*, 6 plastic strains
double ApproximationTableOld[16][20];
double hSquareTable [16][15];
double DError[15], RError;
double InitialStress[7], InitialPlasticStrain[6], PlasticStrain[6];
double DSigma[6];
/*
double TestTable[5]={0.375,0.37109375,0.36945588,0.36879683,0.36829712};
double TestApproxTable[5]; double TestApproxTableOld[5];
for (int i=0; i<5; i++)
{
	TestApproxTable[i]=0;
	TestApproxTableOld[i]=0;
}
*/

BBMPoint NewPoint, CopyPoint, OldPoint;
Point->Copy(&CopyPoint);
Point->Copy(&OldPoint);

for (int i=0; i<16; i++)
{
	for (int j=0; j<20; j++)
	{
	ApproximationTable[i][j]=0;
	ApproximationTableOld[i][j]=0;
	}
	for (int j=0; j<15; j++) hSquareTable[i][j]=0;

}

for (int i=1; i<16; i++)
{
	for (int j=0; j<i;j++)
	{
	hSquareTable[i][j]=(Divisions[i]/Divisions[j])*(Divisions[i]/Divisions[j]);
	//cout<<"Divisions["<<i<<"]="<<Divisions[i]<<"  Divisions["<<j<<"]="<<Divisions[j]<<endl;
	//cout<<"hSquareTable["<<i<<"]["<<j<<"]="<<hSquareTable[i][j]<<endl;
	}

}
for (int i=0; i<6; i++)
{
	InitialStress[i]=Point->stress[i];
	InitialPlasticStrain[i]=Point->plastic_strain[i];
}
InitialStress[6]=Point->GetPStar();

int loop=0;
for (; loop<STEPMAX; loop++)
{
//calculating stress increment using MidPoint rule

CopyPoint.Copy(&NewPoint);
PlasticMidpoint (&NewPoint, EPStrain, AbsStress, &DivisionsInt[loop]); //calculation of stress increment using the midpoint procedure
//TestApproxTable[0]=TestTable[loop];

//saving data using MidPoint rule
for (int i=0; i<6; i++)
{
	ApproximationTable[0][i]=NewPoint.stress[i]-InitialStress[i];
	ApproximationTable[0][i+7]=AbsStress[i];
	ApproximationTable[0][i+14]=NewPoint.plastic_strain[i]-InitialPlasticStrain[i];
}
ApproximationTable[0][6]=NewPoint.GetPStar()-InitialStress[6];
ApproximationTable[0][13]=AbsStress[6];

//all data from the Midpoint Rule saved in the ResultsTable.


for (int i=0; i<loop; i++)
{

	for (int j=0; j<20;j++)
	{
		ApproximationTable[i+1][j]=ApproximationTable[i][j]+
		(ApproximationTable[i][j]-ApproximationTableOld[i][j])/(hSquareTable[loop][loop-i-1]-1);
	}
}

//for (i=0; i<loop+1; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl<<endl;


//approximations are calculated
//two possibilities of error control. In literature rather the second one; this is the more stringent, but first one should be enough
//(1) use ApproximationTable[loop][i] - ApproximationTable[loop-1][i]
//(2) use ApproximationTable[loop][i] - ApproximationTableOld [loop-1][i]
//still using relative error definition...
//OldPoint (used for calculating the norm in q) is set accordingly

//FIRST ONE OK, SEE Deuflhard Bornemann 2002 Springer p.206



RError=0;
if (loop>0)
{
	for (int i=0; i<6; i++) DSigma[i]=ApproximationTable[loop][i];
	for (int i=0; i<7; i++)
	{
		//DError[i]=ApproximationTable[loop][i] - ApproximationTableOld[loop-1][i];
		DError[i]=ApproximationTable[loop][i] - ApproximationTable[loop-1][i];
	}

	switch (int (TOL_METHOD))
		{
		case 0:
			{
			RError=CheckNorm (DSigma,ApproximationTable[loop][6],&CopyPoint,DError);	//returns RError
			}
			break;
		case 1:
			{
				//SLOAN NORM
				//cout<<"SLOAN NORM"<<endl;
			RError=CheckNormSloan (DSigma,ApproximationTable[loop][6],&CopyPoint,DError);	//returns RError
			}
			break;
		default:
			{
				cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
				getchar ();
			}
		}	//end switch
}
else RError=2*INTEGRATION_TOL;

//cout<<"Relative error after iteration "<<loop<<" is equal to:"<<RError<<endl;
for (int i=0; i<loop+1; i++) for (int j=0; j<20; j++) ApproximationTableOld[i][j]=ApproximationTable[i][j];
//for (int i=0; i<loop+1; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl;
if (RError<INTEGRATION_TOL)
{
	//error less then requested...check for drift and add the error
	CopyPoint.Copy(&OldPoint);
	OldPoint.Update(Zeros,EPStrain,DSigma,ApproximationTable[loop][6]);
	OldPoint.Copy(&NewPoint);
	if (DRIFT_CORRECTION==3) CorrectDrift (&NewPoint);
	if (DRIFT_CORRECTION==2) CorrectDriftBeg(&NewPoint, &CopyPoint); //value of OldPoint copied before updating the point
	for (int i=0; i<6; i++) DError[i]=DError[i]+(NewPoint.stress[i]-OldPoint.stress[i]);
	DError[6]=DError[6]+NewPoint.GetPStar()-OldPoint.GetPStar();
	switch (int (TOL_METHOD))
		{
		case 0:
			{
			RError=CheckNorm (DSigma,ApproximationTable[loop][6],&CopyPoint,DError);	//returns RError
			}
			break;
		case 1:
			{
				//SLOAN NORM
				//cout<<"SLOAN NORM"<<endl;
			RError=CheckNormSloan (DSigma,ApproximationTable[loop][6],&CopyPoint,DError);	//returns RError
			}
			break;
		default:
			{
				cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
				getchar ();
			}
		}	//end switch
		if (RError<INTEGRATION_TOL) {
			loop=loop+100; //no more interations after - finished
		}
		else
		{
		//one more loop, do not need to change anything, as point anyway copied back later on.
		}
}
else {//NewPoint.Copy(&OldPoint);
		}

}

if (loop>100)
{
loop=loop-101;
//done - time to update everything and sent time and no of iterations...

for (int i=0; i<6; i++)
{
	AbsStress[i]=ApproximationTable[loop][i+7];
	PlasticStrain[i]=ApproximationTable[loop][i+14];
	//cout<<"DSigma="<<DSigma[i]<<"   AbsStress="<<AbsStress[i]<<endl;
}
AbsStress[6]=ApproximationTable[loop][13];
//cout<<AbsStress[6]<<endl;
Point->Update (PlasticStrain, EPStrain, DSigma, ApproximationTable[loop][6]);
EndTime=clock ();
*NumberIter=0;
for (int i=0; i<loop+1; i++) *NumberIter=*NumberIter+Divisions[i];
//cout<<"Procedure has coverged after"<<*NumberIter<<" iterations."<<endl;
//getchar();
}
else
{
loop--;
double DSigma[6];
for (int i=0; i<6; i++)
{
	DSigma[i]=ApproximationTable[loop][i];
	AbsStress[i]=ApproximationTable[loop][i+7];
	PlasticStrain[i]=ApproximationTable[loop][i+14];
	//cout<<DSigma[i]<<endl;
}
AbsStress[6]=ApproximationTable[loop][13];
Point->Update (PlasticStrain, EPStrain, DSigma, ApproximationTable[loop][6]);
EndTime=clock ();
*NumberIter=0;
for (int i=0; i<loop+1; i++) *NumberIter=*NumberIter+Divisions[i];
//cout<<"Procedure has NOT CONVERGED after"<<*NumberIter<<" iterations."<<endl;
//getchar();
}
//cout<<"Calculation took:"<<double(EndTime-StartTime)/CLOCKS_PER_SEC<<" s."<<endl;

/*
switch (int(DRIFT_CORRECTION))
{case 1 :{cout<<"Values calculated with standard algorithm and no drift correction."<<endl;break;}
case 2 :{cout<<"Values calculated with standard algorithm and drift correction at the beginning (point A)."<<endl;break;}
case 3 :{cout<<"Values calculated with standard algorithm and drift correction at the end (point B)."<<endl;break;}
case 4 :{cout<<"Values calculated with zero drift algorithm."<<endl;break;}
default :{cout<<"Unknown DRIFT_CORRECTION parameter. Parameter read is:"<<DRIFT_CORRECTION<<endl;break;}
}

cout<<"The INTEGRATION_TOL parameter is equal to:"<<INTEGRATION_TOL<<endl;
cout<<"Total number of steps done:"<<StepNo<<endl;
cout<<"Total parameter lambda change: "<<LambdaTot<<endl;
cout<<"Total PZeroStar change: "<<PZeroStarTot<<endl;
CheckYield (Point->state, Point->stress, Point->strain[6],&FValue);
cout<<"Yield Function value is:"<<FValue<<endl;
cout<<"Number of drift correction performed:"<<CorrectDriftCount<<endl;
cout<<"Over the whole step change of stress is:"<<endl;
for (int i=0; i<6; i++)
{
	cout<<"s["<<i<<"]="<<Point->stress[i]-OldPoint.stress[i]<<endl;
}

//double DEpsV=0, dNu, SpecificVolume;

//SpecificVolume=OldPoint.GetSpecVol();
//LambdaS=LambdaZero*(r+(1-r)*exp(-Beta*OldPoint.GetSuction()));



cout<<"Initial specific volume="<<OldPoint.GetSpecVol()<<endl;
cout<<"Final specific volume="<<Point->GetSpecVol()<<endl;
cout<<"Change of specific volume is equal to="<<(OldPoint.GetSpecVol()-Point->GetSpecVol())<<endl;
cout<<"Change of mean stress p is equal to:"<<Point->GetMeanStress()-OldPoint.GetMeanStress()<<endl;
cout<<"Change of shear stress q is equal to:"<<Point->GetShearStress()-OldPoint.GetShearStress()<<endl;



FILE * StressFile;
FILE * StrainFile;
FILE * PZeroFile;
StressFile = fopen( "StressInc.dta", "w" );

for (Iter=StressInc.begin() ; Iter!=StressInc.end(); )
{
for (int i=0; i<6; i++)
	{
	fprintf( StressFile, "%.20f , ",*Iter);
	Iter++;
	}
fprintf( StressFile, "\n");
}
fclose (StressFile);

StrainFile = fopen( "StrainInc.dta", "w" );

for (Iter=StrainInc.begin() ; Iter!=StrainInc.end(); )
{
for (int i=0; i<7; i++)
	{
	fprintf( StrainFile, "%.20f , ",*Iter);
	Iter++;
	}
fprintf( StrainFile, "\n");
}
fclose (StrainFile);


PZeroFile = fopen( "PZeroInc.dta", "w" );
for (Iter=PZeroLambda.begin() ; Iter!=PZeroLambda.end(); )
{
for (int i=0; i<2; i++)
	{
	fprintf( PZeroFile, "%.25f , ",*Iter);
	Iter++;
	}
fprintf( PZeroFile, "\n");
}
fclose (PZeroFile);
*/


//cout<<"Press any key..."<<endl;
//getchar();
//*NumberIter=StepNo;
/*
FILE *File;
File=fopen ("ConstStep.dta", "a+");

fprintf(File, "\n");
fprintf(File, "Extrapolation Method Results.\n", *NumberIter);
for (int i=0; i<7;i++ )
		{
			fprintf( File, "%.20f , ",ApproximationTable[loop][i]);
		}
fprintf( File, "%.20f , ",(ApproximationTable[loop][0]+ApproximationTable[loop][1]+ApproximationTable[loop][2])/3);
fprintf( File, "%.20f , ",ApproximationTable[loop][0]-ApproximationTable[loop][1]);

fprintf(File, "\n");
fclose (File); */
return (EndTime-StartTime);
}



double ShengMohrCoulomb::RKExtrapolation  (double A[][8], double* B,double *BRes, double *C, BBMPoint* Point, double* EPStrain, double* AbsStress
								  , BBMPoint* OldPoint, double* RelError, int* NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate)
{
//Here the DRIFT_CORRECTION parameter is used. 0=no correction, 1 - correction at the beginning (A),
//2 - correction at the end (B), 3 - Zero Drift Algorithm.
/*
switch (int(DRIFT_CORRECTION))
{
case 1 :
	{
		cout<<"Procedure proceed with standard algorithm and no drift correction."<<endl;
		break;
	}
case 2 :
	{
		cout<<"Procedure proceed with standard algorithm and drift correction at the beginning (point A)."<<endl;
		break;
	}

default :
	{
		cout<<"Unknown DRIFT_CORRECTION parameter. Parameter read is:"<<DRIFT_CORRECTION<<endl;
		break;
	}
}
*/

//no drift correction as may worsen the results. at least possibly.

clock_t StartTime, EndTime;
StartTime=clock();


int STEPMAX=15;
//int DivisionsInt[15]={2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
//int DivisionsInt[15]={2,4,6,8,10,12,14,16,18,20,22,24,26,28,30};
int DivisionsInt[15]={2,4,6,8,12,16,24,32,48,64,96,128,192,256,384};
//int DivisionsInt[15]={32,48,64,96,128,160,192,256,320,384,448,512,608,736,992}; //fastests so far
//int DivisionsInt[15]={2,4,6,8,10,14,20,28,38,52,72,100,138,190,262,362,500,690,952}; //N+N-4
//int DivisionsInt[15]={2,4,6,8,10,12,16,22,30,40,52,68,90,120,170,222,290,380,500,670,892}; //N+N-5
//int DivisionsInt[15]={28,32,40,52,64,78,94,120,154,200,240,290,330,380,440};
//int DivisionsInt[15]={32,40,52,64,78,94,120,154,200,240,290,330,380,440,520};
//int DivisionsInt[15]={32,36,40,46,52,60,70,82,96,112,130,150,176,220,380};
//int DivisionsInt[15]={32,36,40,52,68,92,114,154,200,240,290,330,380,440,520};
//int DivisionsInt[15]={32,36,40,44,50,56,62,68,76,84,92,102,112,124,136}; //n=n-1*1.1, doesn't converge too often
//int DivisionsInt[15]={32,38,46,56,66,80,96,114,138,166,198,238,286,344,412}; //n=n-1*1.2


double Divisions[15]; for (int i=0; i<15;i++) Divisions[i]=DivisionsInt[i];

double ApproximationTable[16][20]; //6 stresses, p0*, 6 absolute stresses, absolute p0*, 6 plastic strains
double ApproximationTableOld[16][20];
double hSquareTable [16][15];
double DError[15], RError;
double InitialStress[7], InitialPlasticStrain[6], PlasticStrain[6];
double DSigma[6];
bool StepAccepted=false;
/*
double TestTable[5]={0.375,0.37109375,0.36945588,0.36879683,0.36829712};
double TestApproxTable[5]; double TestApproxTableOld[5];
for (int i=0; i<5; i++)
{
	TestApproxTable[i]=0;
	TestApproxTableOld[i]=0;
}
*/

BBMPoint NewPoint, CopyPoint;
Point->Copy(&CopyPoint);
//Point->Copy(&OldPoint);

for (int i=0; i<16; i++)
{
	for (int j=0; j<20;j++)
	{
	ApproximationTable[i][j]=0;
	ApproximationTableOld[i][j]=0;
	}
	for (int j=0; j<15; j++) hSquareTable[i][j]=0;

}

for (int i=1; i<16; i++)
{
	for (int j=0; j<i;j++)
	{
	hSquareTable[i][j]=(Divisions[i]/Divisions[j])*(Divisions[i]/Divisions[j]);
	//cout<<"Divisions["<<i<<"]="<<Divisions[i]<<"  Divisions["<<j<<"]="<<Divisions[j]<<endl;
	//cout<<"hSquareTable["<<i<<"]["<<j<<"]="<<hSquareTable[i][j]<<endl;
	}

}
for (int i=0; i<6; i++)
{
	InitialStress[i]=Point->stress[i];
	InitialPlasticStrain[i]=Point->plastic_strain[i];
}
InitialStress[6]=Point->GetPStar();


for (int i=0; i<6; i++)
{
	ApproximationTable[0][i]=OldPoint->stress[i]-InitialStress[i];
	ApproximationTable[0][i+7]=AbsStress[i];
	ApproximationTable[0][i+14]=OldPoint->plastic_strain[i]-InitialPlasticStrain[i];
}
ApproximationTable[0][6]=OldPoint->GetPStar()-InitialStress[6];
ApproximationTable[0][13]=AbsStress[6];

int loop=1;


//for (int i=0; i<loop+1; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl;

for (int i=0; i<loop; i++)
{

	for (int j=0; j<20;j++)
	{
		ApproximationTable[i+1][j]=ApproximationTable[i][j]+
		(ApproximationTable[i][j]-ApproximationTableOld[i][j])/(hSquareTable[loop][loop-i-1]-1);
	}
}
for (int i=0; i<loop+1; i++) for (int j=0; j<20; j++) ApproximationTableOld[i][j]=ApproximationTable[i][j];

//cout<<"Initial"<<endl;
//for (int i=0; i<loop; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl<<endl;
//loop=0;

for (; loop<STEPMAX; loop++)
{
//calculating stress increment using MidPoint rule

CopyPoint.Copy(&NewPoint);
RungeKuttaEqualStep (A, B, BRes, C,  &NewPoint, EPStrain, AbsStress,
					 &RError, DivisionsInt[loop],MethodOrder,  MethodSteps,  ErrorEstimate); //calculation of stress increment using the RK procedure
//TestApproxTable[0]=TestTable[loop];

//saving data using MidPoint rule
for (int i=0; i<6; i++)
{
	ApproximationTable[0][i]=NewPoint.stress[i]-InitialStress[i];
	ApproximationTable[0][i+7]=AbsStress[i];
	ApproximationTable[0][i+14]=NewPoint.plastic_strain[i]-InitialPlasticStrain[i];
}
ApproximationTable[0][6]=NewPoint.GetPStar()-InitialStress[6];
ApproximationTable[0][13]=AbsStress[6];

//all data from the Midpoint Rule saved in the ResultsTable.

for (int i=0; i<loop; i++)
{

	for (int j=0; j<20;j++)
	{
		ApproximationTable[i+1][j]=ApproximationTable[i][j]+
		(ApproximationTable[i][j]-ApproximationTableOld[i][j])/(hSquareTable[loop][loop-i-1]-1);
	}
}

//for (i=0; i<loop+1; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl;
//getchar();

//approximations are calculated
//two possibilities of error control. In literature rather the second one; this is the more stringent, but first one should be enough
//(1) use ApproximationTable[loop][i] - ApproximationTable[loop-1][i]
//(2) use ApproximationTable[loop][i] - ApproximationTableOld [loop-1][i]
//still using relative error definition...
//OldPoint (used for calculating the norm in q) is set accordingly

StepAccepted=false;
if(RError<INTEGRATION_TOL) StepAccepted=true;

for (int i=0; i<6; i++) DSigma[i]=ApproximationTable[loop][i];
for (int i=0; i<7; i++)
{
	DError[i]=ApproximationTable[loop][i] - ApproximationTableOld[loop-1][i];
}

switch (int (TOL_METHOD))
	{
	case 0:
		{
		RError=CheckNorm (DSigma,ApproximationTable[loop][6],OldPoint,DError);	//returns RError
		}
		break;
	case 1:
		{
			//SLOAN NORM
			//cout<<"SLOAN NORM"<<endl;
		RError=CheckNormSloan (DSigma,ApproximationTable[loop][6],OldPoint,DError);	//returns RError
		}
		break;
	default:
		{
			cout<<"ERROR !!!! Improper TOL_METHOD in increment.dta"<<endl;
			getchar ();
		}
	}	//end switch


//cout<<"Relative error after iteration "<<loop<<" is equal to:"<<RError<<endl;
for (int i=0; i<loop+1; i++) for (int j=0; j<20; j++) ApproximationTableOld[i][j]=ApproximationTable[i][j];
//for (int i=0; i<loop+1; i++) cout<<ApproximationTable[i][0]<<"  ";
//cout<<endl;
if (RError<INTEGRATION_TOL) StepAccepted=true;
if (StepAccepted) loop=loop+100; //no more interations after - finished
else NewPoint.Copy(OldPoint);
}

if (loop>100)
{
loop=loop-101;
//done - time to update everything and sent time and no of iterations...

for (int i=0; i<6; i++)
{
	AbsStress[i]=ApproximationTable[loop][i+7];
	PlasticStrain[i]=ApproximationTable[loop][i+14];
	//cout<<"DSigma="<<DSigma[i]<<"   AbsStress="<<AbsStress[i]<<endl;
}
AbsStress[6]=ApproximationTable[loop][13];
//cout<<AbsStress[6]<<endl;
Point->Update (PlasticStrain, EPStrain, DSigma, ApproximationTable[loop][6]);
EndTime=clock ();
*NumberIter=0;
for (int i=0; i<loop+1; i++) *NumberIter=*NumberIter+Divisions[i];
//cout<<"Procedure has coverged after"<<*NumberIter<<" iterations."<<endl;
//getchar();
return (EndTime-StartTime);
}
else
{
loop--;
double DSigma[6];
for (int i=0; i<6; i++)
{	DSigma[i]=ApproximationTable[loop][i];
	AbsStress[i]=ApproximationTable[loop][i+7];
	PlasticStrain[i]=ApproximationTable[loop][i+14];
	//cout<<DSigma[i]<<endl;
}
AbsStress[6]=ApproximationTable[loop][13];
Point->Update (PlasticStrain, EPStrain, DSigma, ApproximationTable[loop][6]);
EndTime=clock ();
*NumberIter=0;
for (int i=0; i<loop+1; i++) *NumberIter=*NumberIter+Divisions[i];
cout<<"Procedure has NOT CONVERGED after"<<*NumberIter<<" iterations."<<endl;
//getchar();
}
return (EndTime-StartTime);
}


double ShengMohrCoulomb::CheckNorm (double* DSigma, double DPZeroStar, BBMPoint* InitialPoint, double* DError)	//returns RError
{
/*
Procedure returns value of the relative error RError
The value is calculated as maximum of all the values of RErrors
RErrors used:
	- each stress component
	- error in p0*
	- error in p
	- error in q

INPUT: DSigma [6] - values of stress increment (min 6, but may be more; values over six ignored)
		DPZeroStar - value of the p0* increment
		InitialPoint - point used to calculate the values of the step
		DError [7] - error estimates for all values of DSigma  DError[i]  - error estimate for DSigma[i];
*/

double DRError[9];
double RError=0;

//standard norm:
for (int i=0; i<6; i++)
	{
		if (fabs(DSigma[i])>TINY) DRError[i]=DError[i]/DSigma[i];
		else DRError[i]=DError[i]/TINY;
	}
if (fabs(DPZeroStar)>TINY) DRError[6]=DError[6]/DPZeroStar;
	else DRError[6]=DError[6]/TINY;

//norm in mean stress p:

double P, ErrorP;

P=fabs (DSigma[0]+DSigma[1]+DSigma[2])/3;
ErrorP=fabs (DError[0]+DError[1]+DError[2])/3;
if (P>TINY) DRError[7]=ErrorP/P;
	else DRError[7]=ErrorP/TINY;

//norm of q...


double InitialShear, FinalShearMin, FinalShearMax, FinalShear;
double ShearError, DShear;
double SigmaTemp[6], InitialSigma[6];

InitialShear=InitialPoint->GetShearStress();

for (int i=0; i<6; i++)
{
	InitialSigma[i]=InitialPoint->stress[i];
	SigmaTemp[i]=InitialSigma[i]+DSigma[i];
}

//computing shear stress like this is much more effective than calling the point to give the shear stress
//as the point would have to be updated and copied. It is not time effective, as there are more
//things to update than just the stresses; Even if zeros are given, they are still updated what is time consuming
//This is less elegant but more efficient

FinalShear=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShear=FinalShear+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShear=sqrt (0.5*FinalShear);
DShear=fabs(FinalShear-InitialShear);

for (int i=0; i<6; i++) SigmaTemp[i]=SigmaTemp[i]+DError[i];
FinalShearMax=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShearMax=FinalShearMax+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShearMax=sqrt (0.5*FinalShearMax);

for (int i=0; i<6; i++) SigmaTemp[i]=SigmaTemp[i]-2*DError[i];
FinalShearMin=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShearMin=FinalShearMin+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShearMin=sqrt (0.5*FinalShearMin);

ShearError=fabs (FinalShearMax-FinalShear);
if (fabs(FinalShearMin-FinalShear)>ShearError) ShearError=fabs (FinalShearMin-FinalShear);
if (DShear>TINY) DRError[8]=ShearError/DShear;
	else DRError[8]=ShearError/TINY;


//final check: max norm
for (int i=0; i<9; i++)
{
	DRError[i]=fabs(DRError[i]);
	if (DRError[i]>RError)RError=DRError[i];
}
return RError;
}


double ShengMohrCoulomb::CheckNormSloan (double* DSigma, double DPZeroStar, BBMPoint* InitialPoint, double* DError)	//returns RError
{
/*
Procedure returns value of the relative error RError
The value is calculated as maximum of all the values of RErrors
RErrors used:
	- each stress component
	- error in p0*
	- error in p
	- error in q

INPUT: DSigma [6] - values of stress increment (min 6, but may be more; values over six ignored)
		DPZeroStar - value of the p0* increment
		InitialPoint - point used to calculate the values of the step
		DError [7] - error estimates for all values of DSigma  DError[i]  - error estimate for DSigma[i];
*/

double DRError[9];
double RError=0;
double InitialSigma[6], SigmaTemp[6], DPZeroStarEnd;

for (int i=0; i<6; i++)
{
	InitialSigma[i]=InitialPoint->stress[i];
	SigmaTemp[i]=InitialSigma[i]+DSigma[i];
}
DPZeroStarEnd=InitialPoint->GetPStar()+DPZeroStar;

//standard norm:
for (int i=0; i<6; i++)
	{
		if (fabs(SigmaTemp[i])>TINY) DRError[i]=DError[i]/SigmaTemp[i];
		else DRError[i]=DError[i]/TINY;
	}
if (fabs(DPZeroStarEnd)>TINY) DRError[6]=DError[6]/DPZeroStarEnd;
	else DRError[6]=DError[6]/TINY;

//norm in mean stress p:

double P, ErrorP;

P=fabs (InitialSigma[0]+InitialSigma[1]+InitialSigma[2]+DSigma[0]+DSigma[1]+DSigma[2])/3;
ErrorP=fabs (DError[0]+DError[1]+DError[2])/3;
if (P>TINY) DRError[7]=ErrorP/P;
	else DRError[7]=ErrorP/TINY;

//norm of q...


double FinalShearMin, FinalShearMax, FinalShear;//InitialShear;
double ShearError;


//InitialShear=InitialPoint->GetShearStress();

//computing shear stress like this is much more effective than calling the point to give the shear stress
//as the point would have to be updated and copied. It is not time effective, as there are more
//things to update than just the stresses; Even if zeros are given, they are still updated what is time consuming
//This is less elegant but more efficient

FinalShear=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShear=FinalShear+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShear=sqrt (0.5*FinalShear);


for (int i=0; i<6; i++) SigmaTemp[i]=SigmaTemp[i]+DError[i];
FinalShearMax=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShearMax=FinalShearMax+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShearMax=sqrt (0.5*FinalShearMax);

for (int i=0; i<6; i++) SigmaTemp[i]=SigmaTemp[i]-2*DError[i];
FinalShearMin=(SigmaTemp[0]-SigmaTemp[1])*(SigmaTemp[0]-SigmaTemp[1])
	+(SigmaTemp[0]-SigmaTemp[2])*(SigmaTemp[0]-SigmaTemp[2])+(SigmaTemp[1]-SigmaTemp[2])*(SigmaTemp[1]-SigmaTemp[2]);
FinalShearMin=FinalShearMin+6*(SigmaTemp[3]*SigmaTemp[3]+SigmaTemp[4]*SigmaTemp[4]+SigmaTemp[5]*SigmaTemp[5]);
FinalShearMin=sqrt (0.5*FinalShearMin);

ShearError=fabs (FinalShearMax-FinalShear);
if (fabs(FinalShearMin-FinalShear)>ShearError) ShearError=fabs (FinalShearMin-FinalShear);
if (FinalShear>TINY) DRError[8]=ShearError/FinalShear;
	else DRError[8]=ShearError/TINY;

//final check: max norm
for (int i=0; i<9; i++)
{
	DRError[i]=fabs(DRError[i]);
	if (DRError[i]>RError) RError=DRError[i];
}
return RError;
}






void ShengMohrCoulomb::ComputeG1 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G1)
{
/*
See Word document for details

Retention Model:
1 - State surface
2 - Van Genuchten
3 - Gallipoli Wheeler & Karstunen

Retention Parameters:
1: [a, b]
2: irrelevant (matrix ==0)
3: [fi, psi, n, m]

G1- double[6];
*/
switch (RetentionModel)
{
	case 1:  //state surface model.
		{
		double a,b,s;
		a=RetentionParameters[0];
		b=RetentionParameters[1];
		s=InitialPoint->GetSuction();
		G1[0]=(1-exp(b*s))*a/3; G1[1]=G1[0]; G1[2]=G1[0];
		G1[3]=0; G1[4]=0; G1[5]=0;
		}
		break;
	case 2:   //Van Genuchten Model
		{
		for (int i=0; i<6; i++) G1[i]=0;
		}
		break;
	case 3:  //Gallipoli, Wheeler, Karstunen
		{
		cout<<"Calculation of G1 Matrix for Gallipoli Wheeler Karstunen model not implemented yet..."<<endl;
		for (int i=0; i<6; i++) G1[i]=0;
		}
		break;
	default:
		{
			cout<<"Procedure Compute G1. Unknown Retention Model... No matrix G1 calculated. Please press any key to continue..."<<endl;
			getchar();
		}

}

}
void ShengMohrCoulomb::ComputeG2 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G2)
{
/*
See Word document for details

Retention Model:
1 - State surface
2 - Van Genuchten
3 - Gallipoli Wheeler & Karstunen

Retention Parameters:
1: [a, b]
2: [Ew, Fw, Ssn, Sir]
3: [fi, psi, n, m]

G2- double;
*/

switch (RetentionModel)
{
	case 1:  //state surface model.
		{
		double a,b,s;
		a=RetentionParameters[0];
		b=RetentionParameters[1];
		s=InitialPoint->GetSuction();
		*G2=b*exp(b*s)*(0.5-a*InitialPoint->GetMeanStress());
		}
		break;
	case 2:   //Van Genuchten Model
		{
		double Ew,Fw,Ssn,Sir,s,Numerator,Denominator;
		Ew=RetentionParameters[0];
		Fw=RetentionParameters[1];
		Ssn=RetentionParameters[2];
		Sir=RetentionParameters[3];
		s=InitialPoint->GetSuction();
		Numerator=(Ssn-Sir)*(1-Fw)*Ew*pow(Ew*s,Fw-1);
		Denominator=1+pow(Ew*s,Fw);
		Denominator=Fw*pow(Denominator,1-1/Fw);
		*G2=Numerator/Denominator;
		}
		break;
	case 3:  //Gallipoli, Wheeler, Karstunen
		{
		double Fi, Psi, n, m, nu, s, Numerator, Denominator;
		Fi=RetentionParameters[0];
		Psi=RetentionParameters[1];
		n=RetentionParameters[2];
		m=RetentionParameters[3];
		s=InitialPoint->GetSuction();
		nu=InitialPoint->GetSpecVol();
		Numerator=Fi*(1-nu)*Psi*s;
		Numerator=Fi*(1-nu)*Psi*pow(Numerator,n-1);
		Denominator=Numerator*s+1;
		Denominator=pow(Denominator,m+1);
		*G2=Numerator/Denominator;
		}
		break;
	default:
		{
			cout<<"Procedure Compute G1. Unknown Retention Model... No matrix G1 calculated. Please press any key to continue..."<<endl;
			getchar();
		}

}


}



double ShengMohrCoulomb::GetLambda (double * deriv, double stresspq[3], double strainpq[3])
{
	double numerator[3], denominator[3];
	//get K from calcstresselast, hmmmm, later
	double K=1;
	numerator[0]=K*strainpq[0];
	numerator[1]=3*G*strainpq[1];
    numerator[2]=strainpq[2];
	double DeltaPZero=0;

	for (int i=0; i<3; i++) numerator[i]=numerator[i]*deriv[i]; //now it needed to be added to get the first part of the numerator
	for (int i=0; i<3; i++) numerator[i]=numerator[i]+deriv[i]*DeltaPZero; //here it does nothing at the moment, because DeltaPZero==0

	numerator[0]=numerator[0]+numerator[1]+numerator[2]; //numerator finished; it's a number stored in [0]

	denominator[0]=K*deriv[0];
	denominator[1]=3*G*deriv[1];
	denominator[2]=deriv[2];

	for (int i=0; i<3; i++) denominator[i]=denominator[i]*deriv[i];
	denominator[0]=denominator[0]+denominator[1]+denominator[2]; //denominator finished; it's a number stored in [0]

	return numerator[0]/denominator[0]; //numerator/denominator=lambda
}


double ShengMohrCoulomb::Getk ()
{
	return 0;// k;
}
double ShengMohrCoulomb::GetLambdaZero ()
{
	return 0;// LambdaZero;
}
double ShengMohrCoulomb::Getr ()
{
return 0;//r;
}
double ShengMohrCoulomb::GetBeta ()
{
	return 0;// Beta;
}
double ShengMohrCoulomb::GetKappaP ()
{
	return 0;// KappaP;
}

double ShengMohrCoulomb::Getpc ()
{
	return 0;//pc;
}




