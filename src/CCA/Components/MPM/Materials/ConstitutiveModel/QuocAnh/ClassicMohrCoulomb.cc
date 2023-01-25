#include "stdafx.h"
#include "ClassicMohrCoulomb.h"

/* this ia a classic Mohr-Coulomb model

***** integration works based on the principal stress space
***** note that the strain must be rotated into the same coordinate system as stress

*/

using namespace std;

extern double MAXITER, ALFACHECK, ALFACHANGE, ALFARATIO, YIELDTOL, INTEGRATION_TOL, DRIFT_CORRECTION, TOL_METHOD, USE_ERROR;
extern double STEP_MAX, STEP_MIN, ERROR_DEF, USE_ERROR_STEP, MIN_DIVISION_SIZE;
extern double BETA_FACT, ADDTOLYIELD, CHGEPSINTOL, PMIN, SUCTIONTOL, TINY;
extern int SOLUTION_ALGORITHM, ALGORITHM_TYPE, USE_NICE_SCHEME;
extern double CRITICAL_STEP_SIZE;

ClassicMohrCoulomb::ClassicMohrCoulomb(void)
{

	// all the parameters are initialized
	G=10000;		//shear modulus
	K=20000;		//bulk modulus
	E=9*K*G/(G+3*K);		//Young modulus
	Poisson=(3*K-2*G)/(2*G+6*K);	//Poisson ratio

	Pi=3.1415926535897932384626433832795;
	//Mohr - Coulomb parameters

	Cohesion=0;
	Phi=Pi/6.0;
	SinPhi=sin(Phi);
	CosPhi=cos(Phi);


	//if the flow rule is not associated; not expected...
	NonAssociated=false;
	Psi=Phi;
	SinPsi=sin(Psi);
	CosPsi=cos(Psi);


}

ClassicMohrCoulomb::~ClassicMohrCoulomb(void)
{
}


void ClassicMohrCoulomb::SetModelParameters (double ShearModulusG, double BulkModulusK, double CohesionC, double FrictionAnglePhi, double DilationAnglePsi)
{
G=ShearModulusG;		//shear modulus
K=BulkModulusK;
E=9*K*G/(G+3*K);
Poisson=(3*K-2*G)/(2*G+6*K);
Cohesion=CohesionC;
Phi=FrictionAnglePhi*Pi/180.0;
SinPhi=sin(Phi);
CosPhi=cos(Phi);
Psi=DilationAnglePsi*Pi/180.0;;
SinPsi=sin(Psi);
CosPsi=cos(Psi);

}
/*
ClassicMohrCoulomb::ClassicMohrCoulomb(void)
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


}

ClassicMohrCoulomb::~ClassicMohrCoulomb(void)
{
}


void ClassicMohrCoulomb::SetModelParameters (double ShearModulusG, double BulkModulusK, double CohesionC, double FrictionAnglePhi, double DilationAnglePsi)
{
G=ShearModulusG;		//shear modulus
K=BulkModulusK;
E=9*K*G/(G+3*K);
Poisson=(3*K-2*G)/(2*G+6*K);
Cohesion=CohesionC;
Phi=FrictionAnglePhi*3.14159/180.0;
SinPhi=sin(Phi);
CosPhi=cos(Phi);
Psi=DilationAnglePsi*3.14159/180.0;;
SinPsi=sin(Psi);
CosPsi=cos(Psi);

}
*/

double ClassicMohrCoulomb::CalcStressElast (double nu0, double* s0, double* eps0,  double* deps,  double* ds)
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

double ClassicMohrCoulomb::CalcElastic (double * Strain, BBMPoint * InitialPoint, BBMPoint * FinalPoint)
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


void ClassicMohrCoulomb::CalcStressElastM (double* deps, double* ds)
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

void ClassicMohrCoulomb::FindElStrGrad (double nu0, double* s0  , double* eps0, double* deps,  double* ds)
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

bool ClassicMohrCoulomb::CheckYield (BBMPoint * Point)
{
	//check of the standard yield surface
	//Note: value of function is not normalised by the sum of eigenvalues


//Check Yield to be written...

double Eigen[3];
double Value;

Point->GetEigen (Eigen);
Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
//Value=Value/(Eigen[0]+Eigen[2])

if (Value>YIELDTOL) return true;
else return false;

/*
    cout<<"Check Yield: Mean Stress="<<MeanStress;
	cout<<" Shear Stress="<<ShearStress;
	cout<<" Cohesion="<<Cohesion;
	cout<<" M="<<M;
	cout<<" Yield Function="<<Value<<endl;
*/

}

bool ClassicMohrCoulomb::CheckYieldNormalised (BBMPoint * Point)
{
	//check of the standard yield surface
	//Note: value of function is normalised by the sum of eigenvalues


double Eigen[3];
double Value;

Point->GetEigen (Eigen);
Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
Value=Value/(fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion);

if (Value>(YIELDTOL)) return true;
else return false;

}



void ClassicMohrCoulomb::CheckYieldNormalised (double *state, double *s, double suction, double *FValue)
{
	//check of the standard yield surface
	//Note: value of function is normalised by the eigenvalues

/*
	Purpose of this routine is to calculate value of yield function to determine, whether we have yielding
	or not.

	Arguments: *state - table of state parameters; in fact we need only p*.
				s- stress state
				suction - suction
	Returns: none;  value of normalized function is saved in FValue
	*/


BBMMatrix Stress(3,3),EigenValues(3,3),EigenVect(3,3);
double Eigen[3];

Stress.PutElement (1,1,s[0]);
Stress.PutElement (1,2,s[3]);
Stress.PutElement (1,3,s[4]);

Stress.PutElement (2,1,s[3]);
Stress.PutElement (2,2,s[1]);
Stress.PutElement (2,3,s[5]);

Stress.PutElement (3,1,s[4]);
Stress.PutElement (3,2,s[5]);
Stress.PutElement (3,3,s[2]);

Stress.Eigen(&EigenVect,&EigenValues);
Eigen[0]=EigenValues.GetElement(1,1);
Eigen[1]=EigenValues.GetElement(2,2);
Eigen[2]=EigenValues.GetElement(3,3);

*FValue=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
*FValue=*FValue/(fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion);

}



double ClassicMohrCoulomb::ComputeYieldFunction(BBMPoint * Point)
{
//Note: value of function is not normalised by the eigenvalues

double Eigen[3];
double Value;

Point->GetEigen (Eigen);
//cout<<"EigenValues:"<<Eigen[0]<<' '<<Eigen[1]<<' '<<Eigen[2]<<endl;
//cout<<"SinPhi:"<<SinPhi<<" Cohesion:"<<Cohesion<<" CosPhi:"<<CosPhi<<endl;

Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
//Value=Value/(Eigen[0]+Eigen[2])

return Value;

}

double ClassicMohrCoulomb::ComputeYieldFunctionNN(BBMPoint * Point)
{
//Note: value of function is not normalised by the eigenvalues

double Eigen[3];
double Value;

Point->GetEigen (Eigen);
Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
//Value=Value/(Eigen[0]+Eigen[2])

return Value;

}

double ClassicMohrCoulomb::ComputeYieldFunctionEigen(double Eigen[3])
{
//Note: value of function is not normalised by the eigenvalues


double Value;

Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
//Value=Value/(fabs(Eigen[0])+fabs(Eigen[2]));

return Value;

}


double ClassicMohrCoulomb::ComputeYieldFunctionEigenNorm(double Eigen[3])
{
//Note: value of function is normalised by the eigenvalues


double Value;

Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
Value=Value/(fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion);

return Value;

}



bool ClassicMohrCoulomb::CheckIfPlastic (BBMPoint * Point)
{
//Note: value of function is not normalised by the mean stress+2*cohesion!

double Eigen[3];
double Value;

Point->GetEigen (Eigen);
Value=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
Value=Value/(fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion);

if (Value>(YIELDTOL)) return true;
	else return false;

}





void ClassicMohrCoulomb::CheckYield (double *state, double *s, double suction, double *FValue)
{
	/*
	Purpose of this routine is to calculate value of yield function to determine, whether we have yielding
	or not.

	Arguments: *state - table of state parameters; in fact we need only p*.
				s- stress state
				suction - suction
	Returns: none;  value of not normalized function is saved in FValue
	*/

BBMMatrix Stress(3,3),EigenValues(3,3),EigenVect(3,3);
double Eigen[3];

Stress.PutElement (1,1,s[0]);
Stress.PutElement (1,2,s[3]);
Stress.PutElement (1,3,s[4]);

Stress.PutElement (2,1,s[3]);
Stress.PutElement (2,2,s[1]);
Stress.PutElement (2,3,s[5]);

Stress.PutElement (3,1,s[4]);
Stress.PutElement (3,2,s[5]);
Stress.PutElement (3,3,s[2]);

Stress.Eigen(&EigenVect,&EigenValues);
Eigen[0]=EigenValues.GetElement(1,1);
Eigen[1]=EigenValues.GetElement(2,2);
Eigen[2]=EigenValues.GetElement(3,3);

*FValue=(Eigen[0]-Eigen[2])-(Eigen[0]+Eigen[2])*SinPhi-2*Cohesion*CosPhi;

//Normalisation
//*FValue=*FValue/(Eigen[0]+Eigen[2])
	//normalisation
	//*FValue=*FValue/(MeanStress+2*Cohesion);
//	cout<<*FValue<<endl;
}

double ClassicMohrCoulomb::ComputeNu (double *s,double *state, double suction)
{
// does nothing for MC
return 1;
}



int ClassicMohrCoulomb::CalcPlasticFaster (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
CalcPlastic (Point, EPStrain, DSIGMA, PlasticStrain, DPZeroStar, FValue, dS, dLambda);
return 0;
//so we had the increase of strenghtening parameter...
//that's all?
}



int ClassicMohrCoulomb::CalcPlasticPQ (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
CalcPlastic (Point, EPStrain, DSIGMA, PlasticStrain, DPZeroStar, FValue, dS, dLambda);
return 0;
//so we had the increase of strenghtening parameter...
//that's all?
}





int ClassicMohrCoulomb::CalcPlastic (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue, double *dS, double* dLambda)
{
// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)
BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1), NUMERATOR(1,1), TEMP(1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix MM (6,1); // will be vector(1,1,1,0,0,0)T
BBMMatrix DEL (6,6); //D elastic matrix...
BBMMatrix DEPS (6,1);

for (int i=1; i<6; i++) DEPS.PutElement (i,1,EPStrain[i-1]);  //increase of epsilon copied
//cout<<"DEPS:"<<endl;
//DEPS.Print();
//cout<<"Mean Stress:"<<Point.GetMeanStress()<<" PMIN="<<PMIN<<endl;


//int Alert=0;

	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	//A.PutElement(2,1,0.0);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	//GG.PutElement(2,1,0.0);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma

/*
if (fabs(Point.stress[0]-Point.stress[1])<TINY) Alert=1;
if (fabs(Point.stress[0]-Point.stress[2])<TINY) Alert=2;
if (fabs(Point.stress[1]-Point.stress[2])<TINY) Alert=3;
if (Point.GetMeanStress()<=PMIN) Alert=4;



switch (Alert)
{

case 1:
	{
		//sigma 1= sigma 2 so we define a new gradient in the corner
		//gradient is defined as a

	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(2,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(2,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma

	}
	break;

case 2:
	{
			//sigma 1= sigma 3 so we are at the tip of the cone (as sigma 2 is in between)
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma

	}
	break;
case 3:
	{
	//sigma 2= sigma 3 so we define a new gradient in the corner
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(2,1,-1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(2,1,-1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
	}
	break;
//
case 4:
	{
	// we are at the top of the cone
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma
	}
	break;

default:
	{
	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	//A.PutElement(2,1,0.0);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	//GG.PutElement(2,1,0.0);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
	}
}
*/

/*
if (Point.GetMeanStress()>PMIN)
{
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(2,1,0.0);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(2,1,0.0);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
}
else
{
// we are at the top of the cone
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma
}
*/

//cout<<"A:"<<endl;
//A.Print();
//cout<<"G:"<<endl;
//GG.Print();


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
//cout<<"Numerator before multiplying by Epsilon:"<<endl;
//cout<<"Numerator:";NUMERATOR.Print();cout<<endl;
NUMERATOR.Multiply(&GG,&DENOMINATOR); //Denominator finished
//cout<<"Denominator:"<<DENOMINATOR.GetElement(1,1)<<endl;
//NUMERATOR.Multiply(&DEPS,&NUMERATOR);//NUMERATOR=aT*Del*dEps -->Numerator of Lambda without multiplication by dEpsilon
//cout<<"Numerator:"<<NUMERATOR.GetElement(1,1)<<endl;
//cout<<"Lambda:"<<NUMERATOR.GetElement(1,1)/DENOMINATOR.GetElement(1,1)<<endl;

/*
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
*/
DEL.Multiply(&GG,&TEMP);
TEMP.Multiply(&NUMERATOR,&TEMP);
if (DENOMINATOR.GetElement(1,1)<TINY)  cout<<"Denominator of plastic multiplier is very small. Some error may arise and results may be incorrect"<<endl;

TEMP.Multiply(1.0/DENOMINATOR.GetElement(1,1),&TEMP);
DEL.Substract(&TEMP,&TEMP);
//cout<<"Dep=";TEMP.Print();cout<<endl; elasto-plastic matrix in TEMP
DEL.Multiply(&DEPS,DSIGMA);
//cout<<"DSigma elastic:"<<DSIGMA->GetElement(1,1)<<" "<<DSIGMA->GetElement(2,1)<<" "<<DSIGMA->GetElement(3,1)<<" "<<endl;
//DSIGMA->Substract(&TEMP,DSIGMA);
//DEPS.Print();
TEMP.Multiply(&DEPS,DSIGMA);
//cout<<"DSigma:"<<DSIGMA->GetElement(1,1)<<" "<<DSIGMA->GetElement(2,1)<<" "<<DSIGMA->GetElement(3,1)<<" "<<endl;
//getchar();

for (int i=1; i<7 ; i++)
{
	if (!isfinite (DSIGMA->GetElement(i,1)))
	{
	    cout<<"Stress increment in the CalcPlastic Procedure not a number. Putting zero instead. May cause further issues in the code and the results obtained may be incorrect."<<endl;
		DSIGMA->PutElement(i,1,0.0);
	}
}
*DPZeroStar=1.0;
//Stress increment computed and in DSIGMA matrix


return 0;

//that's all?
}

void ClassicMohrCoulomb::GetTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
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


void ClassicMohrCoulomb::GetTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP)
{
cout<<"GetTangentMatrixPQ is unsupported for Mohr-Coulomb Model"<<endl;
getchar();
//CalculateElastoPlasticTangentMatrix (Point,DEP);
}

void ClassicMohrCoulomb::CalculateElasticTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP)
{
cout<<"CalculateElasticTangentMatrixPQ is unsupported for Mohr-Coulomb Model"<<endl;
getchar();
//CalculateElasticTangentMatrix (Point, DEP);
}



void ClassicMohrCoulomb::CalculateElastoPlasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
{

// at the beginning we need to calculate the derivatives a, b, c, d, g, p...)
BBMMatrix A (6,1); //dF/dsigma
BBMMatrix DENOMINATOR (1,1), NUMERATOR(1,1), TEMP(1,1);

BBMMatrix GG (6,1); //dG/dsigma, in case of associated flow (NonAssociated==false) same as A
BBMMatrix DEL (6,6); //D elastic matrix...
//cout<<"Mean Stress:"<<Point.GetMeanStress()<<" PMIN="<<PMIN<<endl;


if (Point->GetMeanStress()>PMIN)
{
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
}
else
{
// we are at the top of the cone
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma
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

A.Transpose(&TEMP);
TEMP.Multiply(&DEL,&DENOMINATOR); //DENOMINATOR=aT*Del -->Numerator of Lambda without multiplication by dEpsilon
DENOMINATOR.Multiply(&GG,&DENOMINATOR); //Denominator finished
if(DENOMINATOR.GetElement(1,1)<TINY)
{
	cout<<"Denominator in Lambda in the Dep matrix is very small... May lead to serious stability issues and wrong results"<<endl;
	DENOMINATOR.PutElement(1,1,TINY);
}


DEL.Multiply(&GG,&NUMERATOR);
NUMERATOR.Multiply(&TEMP,&NUMERATOR);
NUMERATOR.Multiply(&DEL,&NUMERATOR); //Numerator: DEL*G*AT*DEL
NUMERATOR.Multiply(1/DENOMINATOR.GetElement(1,1),&TEMP);
DEL.Substract(&TEMP,&TEMP);
TEMP.Copy(DEP);

}

void ClassicMohrCoulomb::CalculateElasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP)
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


double ClassicMohrCoulomb::FindGradient (double * state, double * s, double *ds, double * dF, double suction, double dsuction)
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

A.PutElement(1,1,1.0-SinPhi);
A.PutElement(3,1,-1.0-SinPhi);


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
		cosin=cosin+dF[i]*ds[i]/(dslength*dFlength); //calculated cosinus of the Theta angle...
	}
	return cosin; //cosin returned... */

}

void ClassicMohrCoulomb::GetDerivative (double MeanStress, double ShearStress, double suction,double PZero, double * state, double* deriv )
{
	cout<<"GetDerivative is not implemented for Mohr-Coulomb model."<<endl;
	getchar();

}


bool ClassicMohrCoulomb::CheckGradient (BBMPoint * InitialPoint, BBMPoint * FinalPoint)
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
	//cout<<"Cosinus is:"<<cosinus<<endl;

	if (cosinus>-YIELDTOL) return false; else return true;  //(negative cosinus means unloading occurs)
}


double ClassicMohrCoulomb::FindGradientPQ (BBMPoint * Point, double *ds, double * dF,  double dsuction)
{
cout<<"FindGradientPQ is unsupported for Mohr-Coulomb Model"<<endl;
	return 0;
}

void ClassicMohrCoulomb::CorrectDriftMC (BBMPoint *Point)
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

//for all stresses different
A.PutElement(1,1,1.0-SinPhi);
A.PutElement(3,1,-1.0-SinPhi);
//A.PrintPrecise ();
//FINISHED dF/dSigma
GG.PutElement(1,1,1.0-SinPsi);
GG.PutElement(3,1,-1.0-SinPsi);
//Finished dG/dSigma

int NumberIter=0;
double CorrFactor, Eigen[3];

Point->GetEigen(Eigen);
CorrFactor=fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion;
double FValue;

BBMMatrix EigenVectors(3,3);
double EigenValues [3];


CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue);
if ((FValue/CorrFactor)<-YIELDTOL||(FValue/CorrFactor)>YIELDTOL) correct=TRUE; else correct=FALSE;

//use check yield here!!!

if (correct==TRUE)
		{
		NumberIter++;
		BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);
		//1: Find Stress EigenValues and EigenVectors
		//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//EigenVectors.Print();
		Point->GetEigen(EigenValues,&EigenVectors);
		//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//cout<<"EigenVectores:"<<endl;
		//EigenVectors.Print();
		EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
		//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		Point->SetStressEigen(EigenValues);
		//2:Rotate the strain
		EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
		EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
		//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;

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
		//4:Rotate back the computed stress
		EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress after calculations:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		} //end if Corrrect==TRUE
//End of first correction

CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue);
if ((FValue/CorrFactor)<-YIELDTOL||(FValue/CorrFactor)>YIELDTOL) correct=TRUE; else correct=FALSE;

if (correct==TRUE)
		{
		NumberIter++;
		BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);
		//1: Find Stress EigenValues and EigenVectors
		//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//EigenVectors.Print();
		Point->GetEigen(EigenValues,&EigenVectors);
		//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//cout<<"EigenVectores:"<<endl;
		//EigenVectors.Print();
		EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
		//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		Point->SetStressEigen(EigenValues);
		//2:Rotate the strain
		EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
		EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
		//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;

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
		//4:Rotate back the computed stress
		EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress after calculations:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		} //end if Corrrect==TRUE

//End of second correction

CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue);
if ((FValue/CorrFactor)<-YIELDTOL||(FValue/CorrFactor)>YIELDTOL) correct=TRUE; else correct=FALSE;

if (correct==TRUE)
		{
		NumberIter++;
		BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);
		//1: Find Stress EigenValues and EigenVectors
		//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//EigenVectors.Print();
		Point->GetEigen(EigenValues,&EigenVectors);
		//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//cout<<"EigenVectores:"<<endl;
		//EigenVectors.Print();
		EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
		//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		Point->SetStressEigen(EigenValues);
		//2:Rotate the strain
		EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
		EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
		//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;

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
		//4:Rotate back the computed stress
		EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress after calculations:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		} //end if Corrrect==TRUE
//end of third and final correction

//cout<<"Drift Correction: Drift correction done in "<<NumberIter<<" iterations."<<endl;
	//this finishes the algorithm */
}




void ClassicMohrCoulomb::CorrectDrift (BBMPoint *Point)
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

//int Alert;



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

	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma



int NumberIter=0;
double CorrFactor, Eigen[3];

Point->GetEigen(Eigen);
CorrFactor=fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion;
double FValue;

BBMMatrix EigenVectors(3,3);
double EigenValues [3];

do
	{
	CheckYield (Point->state, Point->stress, Point->GetSuction(), &FValue); //20 Feb 2006, preparations for the drift correction

	if ((FValue/CorrFactor)<-YIELDTOL||(FValue/CorrFactor)>YIELDTOL) correct=TRUE; else correct=FALSE;

	//use check yield here!!!

if (correct==TRUE)
		{
			NumberIter++;
			//cout<<"Drift Correction, Iteration="<<NumberIter<<" Function Value="<<FValue<<endl;
				// CORRECT FOR DRIFT
				//HERE THE DRIFT WILL BE CORRECTED BY USING THE D MATRIX FROM THE FORBIDDEN SPACE
				//ALTHOUGH BECAUSE THE DRIFT WILL BE CHECKED AGAIN, IT SHOULDN'T POSE MUCH PROBLEM.


BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);



    //1: Find Stress EigenValues and EigenVectors
	//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
	//EigenVectors.Print();
    Point->GetEigen(EigenValues,&EigenVectors);
	//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
	//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
	//cout<<"EigenVectores:"<<endl;
	//EigenVectors.Print();
	EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
	//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
    Point->SetStressEigen(EigenValues);
    //2:Rotate the strain
	EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
	EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
    //cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
	//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;



//Getting lambda and Dep
//Alert=0;

	/*
if (fabs(Point->stress[0]-Point->stress[1])<TINY) Alert=1;
if (fabs(Point->stress[0]-Point->stress[2])<TINY) Alert=2;
if (fabs(Point->stress[1]-Point->stress[2])<TINY) Alert=3;
if (Point->GetMeanStress()<=PMIN) Alert=4;



switch (Alert)
{

case 1:
	{
		//sigma 1= sigma 2 so we define a new gradient in the corner
		//gradient is defined as a

	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(2,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(2,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma

	}
	break;

case 2:
	{
			//sigma 1= sigma 3 so we are at the tip of the cone (as sigma 2 is in between)
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma

	}
	break;
case 3:
	{
	//sigma 2= sigma 3 so we define a new gradient in the corner
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(2,1,-1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(2,1,-1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
	}
	break;

case 4:
	{
	// we are at the top of the cone
	A.PutElement(1,1,-1.0/3.0);
	A.PutElement(2,1,-1.0/3.0);
	A.PutElement(3,1,-1.0/3.0);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,-1.0/3.0);
	GG.PutElement(2,1,-1.0/3.0);
	GG.PutElement(3,1,-1.0/3.0);
	//Finished dG/dSigma
	}
	break;

default:
	{
	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma
	}
}
*/



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



    //4:Rotate back the computed stress
    EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
	EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
	EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);
//cout<<"Stress after calculations:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
} //end if Corrrect==TRUE

if (NumberIter>50)
		{
			cout<<"Drift Correction Procedure reached more than 50 iterations..."<<endl;
			correct=FALSE;

		}
	}   //end do-while loop
while (correct==TRUE);


//cout<<"Drift Correction: Drift correction done in "<<NumberIter<<" iterations."<<endl;
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


void ClassicMohrCoulomb::CorrectDriftOnce (BBMPoint *Point)
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

//int Alert;



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

	//for all stresses different
	A.PutElement(1,1,1.0-SinPhi);
	A.PutElement(3,1,-1.0-SinPhi);

	//A.PrintPrecise ();
	//FINISHED dF/dSigma

	GG.PutElement(1,1,1.0-SinPsi);
	GG.PutElement(3,1,-1.0-SinPsi);
	//Finished dG/dSigma



int NumberIter=0;
double CorrFactor, Eigen[3];

Point->GetEigen(Eigen);
CorrFactor=fabs(Eigen[0])+fabs(Eigen[2])+2*Cohesion;
double FValue;

BBMMatrix EigenVectors(3,3);
double EigenValues [3];


FValue=ComputeYieldFunctionNN(Point); //20 Feb 2006, preparations for the drift correction
/*
	BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);
		//1: Find Stress EigenValues and EigenVectors
		//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//EigenVectors.Print();
		Point->GetEigen(EigenValues,&EigenVectors);
		//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
		//cout<<"EigenVectores:"<<endl;
		//EigenVectors.Print();
		//EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
		//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
		Point->SetStressEigen(EigenValues);
		//2:Rotate the strain
		EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
		EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
		//cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
		//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;

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
		//4:Rotate back the computed stress
		EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);

*/









	if ((FValue/CorrFactor)<-YIELDTOL||(FValue/CorrFactor)>YIELDTOL) correct=TRUE; else correct=FALSE;

	//use check yield here!!!



if (correct==TRUE)
		{
			NumberIter++;
			//cout<<"Drift Correction, Iteration="<<NumberIter<<" Function Value="<<FValue<<endl;
				// CORRECT FOR DRIFT
				//HERE THE DRIFT WILL BE CORRECTED BY USING THE D MATRIX FROM THE FORBIDDEN SPACE
				//ALTHOUGH BECAUSE THE DRIFT WILL BE CHECKED AGAIN, IT SHOULDN'T POSE MUCH PROBLEM.


	BBMMatrix dJ2dSIG (6,1), dJ3dSIG (6,1), dqdSIG (6,1), dI1dSIG (6,1), dI2dSIG (6,1), dI3dSIG (6,1), TEMP(6,1), NUMERATOR(6,1);

    //1: Find Stress EigenValues and EigenVectors
	//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
	//EigenVectors.Print();
    Point->GetEigen(EigenValues,&EigenVectors);
	//cout<<"Stress b4 rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
	//cout<<"EigenValues:"<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
	//cout<<"EigenVectores:"<<endl;
	//EigenVectors.Print();
	//EigenVectors.RotateToEigen(Point->stress,&EigenVectors);
	//cout<<"Stress after rotation:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
    Point->SetStressEigen(EigenValues);
    //2:Rotate the strain
	EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
	EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);
    //cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
	//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;



//Getting lambda and Dep
//Alert=0;
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



    //4:Rotate back the computed stress
    EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
	EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
	EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);
//cout<<"Stress after calculations:"<<Point->stress[0]<<" "<<Point->stress[1]<<" "<<Point->stress[2]<<" "<<Point->stress[3]<<" "<<Point->stress[4]<<" "<<Point->stress[5]<<endl;
} //end if Corrrect==TRUE

//*/



}




void ClassicMohrCoulomb::CorrectDriftBeg (BBMPoint *Point, BBMPoint *PointOld)
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

//For Classical MC it is the same as Correct Drift. Thus just call the procedure.

CorrectDrift (Point);

}








void ClassicMohrCoulomb::SetIntegrationParameters (double IntegrationTolerance,int SolutionAlgorithm, int ToleranceMethod,
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

void ClassicMohrCoulomb::SetDefaultIntegrationParameters ()
{

//elastic parameters, using in the Pegasus algorithm
MAXITER=200;
ALFACHECK=1;
ALFACHANGE=0.05;
ALFARATIO=10;
//Yield tolerance
YIELDTOL=1e-10;  //toleration for the Yield locus (relatively defined, used to check if the stress state is on the YL)
INTEGRATION_TOL=0.01; //integration tolerance
DRIFT_CORRECTION=3; //3-drift at the end  //1-no drift correction
TOL_METHOD=1; //1 - Sloan , 0 would be for EPUS relative error
BETA_FACT=0.9; //safety factor

if (SinPhi>0) PMIN=-Cohesion*CosPhi/SinPhi; else PMIN=-1E8; //approximate only, large value for Tresca

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
CRITICAL_STEP_SIZE=1e-6;
}

void ClassicMohrCoulomb::IntegrateMCIClassic (double* StrainIncrement, BBMPoint* InitialPoint, int* Region)
{
// Modified implicit, should be accurate and equal to the explicit
// take into account that different surfaces may be active during integration

/*written in mind that InitialPoint may be equal to the FinalPoint
Basically the procedure made before should be re-introduced, with some changes
due to the different approach to parameters setting. It should also incorporate the
correct value of plastic strain, instead of useless lambda. This should have been done
long time ago, but well, better late then never.
The plastic/elastic part of the strain is also linked to the
*/


bool PurelyElastic;
//bool OnTheYieldLocus;
//bool Unloading;
//double PurelyElasticStrain[7],PurelyPlasticStrain[7], IntersectionStrain[7], NewIntersectionStrain[7];

BBMMatrix EigenVectors(3,3);
//double EigenValues [3];

//cout<<"Strains:"<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;



//InitialPoint->GetEigen(EigenValues,&EigenVectors);
//InitialPoint->SetStressEigen(EigenValues);
//EigenVectors.RotateToEigen (StrainIncrement,&EigenVectors);
CalcElastic (StrainIncrement, InitialPoint, InitialPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
//cout<<"Stress after elastic step:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;


PurelyElastic=!(CheckIfPlastic (InitialPoint)); //CheckIfPlastic returns true if in the plastic part of the YL. If on the YL (within Tolerance INT_TOL returns false)
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
	ReturnImplicit (InitialPoint, Region);
}
//for (int i=0; i<3; i++) cout<<StrainIncrement[i]<<"  "<<endl;
//for (int i=0; i<3; i++) cout<<FinalPoint.stress[i]<<"  "<<endl;

//FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));

//EigenVectors.RotateToOrigin(InitialPoint->stress,&EigenVectors);
//EigenVectors.RotateToOrigin(InitialPoint->strain,&EigenVectors);
//EigenVectors.RotateToOrigin(InitialPoint->plastic_strain,&EigenVectors);
//getchar();
//*/
}

//int ClassicMohrCoulomb::IntegrateMCIA (double* StrainIncrement, BBMPoint* InitialPoint)
//{
// Modified implicit, should be accurate and equal to the explicit
// take into account that different surfaces may be active during integration

//not efficient and should not be used

/*written in mind that InitialPoint may be equal to the FinalPoint
Basically the procedure made before should be re-introduced, with some changes
due to the different approach to parameters setting. It should also incorporate the
correct value of plastic strain, instead of useless lambda. This should have been done
long time ago, but well, better late then never.
The plastic/elastic part of the strain is also linked to the


BBMPoint FinalPoint, TrialPoint;
bool PurelyElastic;
bool OnTheYieldLocus;
bool Unloading;
int Region;
double PurelyElasticStrain[7],PurelyPlasticStrain[7], IntersectionStrain[7], NewIntersectionStrain[7];

BBMMatrix EigenVectors(3,3);
double EigenValues [3];

//cout<<"Strains:"<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;

OnTheYieldLocus=CheckYieldNormalised (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)


CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
//cout<<"Stress after elastic step:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;


PurelyElastic=!(CheckIfPlastic (&FinalPoint)); //CheckIfPlastic returns true if in the plastic part of the YL. If on the YL (within Tolerance INT_TOL returns false)
//cout<<"p="<<FinalPoint.GetMeanStress()<<"  q="<<FinalPoint.GetShearStress()<<"  Yield? "<<PurelyElastic<<endl; getchar();
//PurelyElastic=TRUE;

if (PurelyElastic)
{
	//FinalPoint.Copy(InitialPoint);
	//update with what we have - all updated after the procedure
	cout<<"Purely Elastic Step"<<endl;
	Region=0;
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
			FindIntersectionUnloading(StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//finding elastic part/ need to update this part of the computations
			CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint); //calculating elastic part, updated in the same point.
			//FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));

			//CalculatePlasticMC (PurelyPlasticStrain,&TrialPoint);

			if (FindIntersectionPlastic (PurelyPlasticStrain,&FinalPoint,IntersectionStrain))
			{
				//intersection== true

				if (CheckStayInCorner (&FinalPoint,IntersectionStrain))
				{
					//yes: we are done: final stress is achieved
				}
				else
				{
					CalculatePlasticMC (IntersectionStrain,&FinalPoint);
					FindIntersectionPlastic (IntersectionStrain,&FinalPoint,NewIntersectionStrain);
					//only twice we can do that- there should be no intersection again
				}

			}
			else
			{
				Region=CalculatePlasticMC_NEW (IntersectionStrain,&FinalPoint);
				//all well, we don't need to do anything more
			}
					
					Major rehaul of this procedure is needed...
					Note: intersection is already found, so there is no trouble about that...
					Consists of:
						- note that we are already on the yield locus
						- computing the plastic part up to the corner (checking whether corner is crossed, if not, up to the corner)
						- checking whether in the corner we have any other single active plane - if so - compute further up to the other corner
						- otherwise, we just stay in the corner.
						For the crossing - maybe use the indicators or something similar, see the Bicanic paper etc. Also Clausen + Crisfield

						Actually - the loop should be put here... not in the CalcualatePlastic MC procedure. That will be more clear
						Question remaining is whether to use the indicators used in the papers or derive own ones...
						Additional question (leading to a paper) is whether with the updated method a consistent tangent matrix may be derived
						...

					

			//this should be not concluded here, but several steps (due to crossing the active surface lines) should be invoked
			//and subsequently plastic part
			//FinalPoint.Copy(InitialPoint);
		}
		else {
			Region=CalculatePlasticMC_NEW (StrainIncrement,&FinalPoint);
			}
	}
	else
	{
	FindIntersection (StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//not on the yield locus, finding intersection and subsequently calculate elastic and plastic stress increment
	CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint);
	//FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
	//cout<<"Integrate MC:End of elastic calculations. Stress:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;
	Region=CalculatePlasticMC_NEW (PurelyPlasticStrain,&FinalPoint);
	//FinalPoint.Copy(InitialPoint);
	}

}
//for (int i=0; i<3; i++) cout<<StrainIncrement[i]<<"  "<<endl;
//for (int i=0; i<3; i++) cout<<FinalPoint.stress[i]<<"  "<<endl;

FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
FinalPoint.Copy(InitialPoint);
//getchar();
//*/
//return Region;
//}



bool ClassicMohrCoulomb::FindIntersectionPlastic (double *PurelyPlasticStrain,BBMPoint * FinalPoint,double *IntersectionStrain)
{


// DO NOT USE --- DOES NOT WORK!!!!


/*
this procedure finds the intersection with the corner, if the stress state reaches the corner
idea is simple: get the Dep, compute as if no intersection
the line gets at the corner of existing yield surface, i.e. if the corner is reached for the existing yield surface,
which is when sigma1=sigma2 or sigma3=sigma2 or sigma1=sigma2=sigma3 (tip of the cone)

This can be computed relatively easily given p; and having the stress state on the yield surface.
Generally, plastic part of the increment is the one corresponding to the strain normal to the potential surface
thus, the elastic is the rest of the strain...
should be analytical to find out the purely plastic part.

The algorithm should be like that:
Take Dep, compute final state.
Use the active surface to return.
If the return is in the active surface - finish.
Otherwise, compute how far from the active surface it is (i.e. % outside compared to % inside)
Using this % compute again, the active part. This part of increment should return to the corner [or not?]

Once in the corner: check whether inside of the other surface is active. Simplest: compute DEP using the other surface and see whether
it will return to it. If it wont (and the previous yield surface would become active), corner is the place to be
If it won't, and the new surface becomes active, another corner is the place to be
If it does return to new point on the surface, it is the new - and correct - point for integration.

sad

*/

//1: Find out how much strain you need for step 1 of algorith (i.e. determining the strain required to reach the corner).

//we need to know current stress and nearest corner stresses:
//note that the nearest corner is the one where sigma1=sigma2 or sigma2=sigma3
//as such difference between sigma1 and sigma2 and sigma 3 is indicative

double HowFarStress;
double HowFarStrain, Alpha;// Factor[2];
BBMPoint TestPoint;
BBMMatrix EigenVectors(3,3),DEPS (6,1);
double EigenValues[3], EigenValuesInitial[3];// Delta[3];

FinalPoint->Copy(&TestPoint);
FinalPoint->GetEigen(EigenValuesInitial);

//stresses are 0>1>2
HowFarStress=FinalPoint->stress[0]-FinalPoint->stress[1];
if ((FinalPoint->stress[1]-FinalPoint->stress[2])<HowFarStress) HowFarStress=FinalPoint->stress[1]-FinalPoint->stress[2];

//now we know how far we are.
//we need to check whether we are already in the corner
if (HowFarStress<TINY*FinalPoint->stress[1])
{
	//we are in the corner
	//strain required to get to the corner == 0
}
else
{
	//we need to assess strain increment needed for such change...
	//simplifying a lot, and assuming elasticity, it is connected by E
	//additional safety comes from the point that we are elasto-plastic

	E=9*K*G/(3*K+G);
	HowFarStrain=0.1*fabs(HowFarStress/E);
	Alpha=1.0;
	for (int i=0;i<6;i++)
	{
		if (fabs(PurelyPlasticStrain[i])>HowFarStrain*Alpha) Alpha=fabs(HowFarStrain/PurelyPlasticStrain[i]);
	}
	if (Alpha>=1.0)
		{
			CalculatePlasticMC (PurelyPlasticStrain,FinalPoint);
			//we compute whole strain safely without changing active surface
		}
	else
	{

		/*we compute just part of the strain and see how it goes, than interpolate until we get to the corner...
		get the increment, turn round to get
		1: Get Dep, multiply by alpha*strain, get stresses increment, rotate them, see how the principal stresses
		change, get the part of strain which would lead to something like corner crossing, maybe rinse & repeat
		2: Once we get the first intersection, return to the corner, see whether further return should be
		to the corner or not. [there or not is whole point of this algorithm - other stop and return all to the corner]
		*/
		for (int i=0; i<6; i++) IntersectionStrain[i]=Alpha*PurelyPlasticStrain[i];
		CalculatePlasticMC (IntersectionStrain,&TestPoint);

		//rotate the stress to get the principal stresses
		TestPoint.GetEigen(EigenValues);
		//for (int i=0; i<3; i++) Delta[i]=EigenValuesInitial[i]-EigenValues[i];
		//now we have the increment in EigenValues corresponding to the increment in strains.
		//Assuming linear relationship, we can extrapolate to get the right amount of stress

		//we need to decide whether we will get sigma1=sigma2 or sigma2=sigma3 first.
		//Factor[0]=1/Alpha*(EigenValuesInitial[0]-EigenValuesInitial[1])/(Delta[1]-Delta[0]);
		//Factor[1]=1/Alpha*(EigenValuesInitial[1]-EigenValuesInitial[2])/(Delta[2]-Delta[1]);
		//now, we can compute things with given Factor. If Factor more than 1, we should be able to get the whole thing done.
		//note that if both factors are less than one, it seems that we will get to the tip of the cone...
		//also, not sure, but those Factors may be approximate only - maybe using Pegasus to get them right would be best...



	for (int i=1; i<7; i++) DEPS.PutElement(i,1,PurelyPlasticStrain[i-1]);
//	cout<<"DEPS before rotation:"<<endl;
//	DEPS.Print();
	TestPoint.GetEigen(EigenValues,&EigenVectors);
//	cout<<"Initial Eigenvalues:"<<EigenValues[0]<<' '<<EigenValues[1]<<' '<<EigenValues[2]<<' '<<endl;
//	cout<<"Initial Eigenvectors:"<<endl;
//	EigenVectors.Print();

    TestPoint.SetStressEigen(EigenValues);
    //2:Rotate the strain
    EigenVectors.RotateToEigen(&DEPS,&EigenVectors);

	}
}






return false;
}


bool ClassicMohrCoulomb::CheckStayInCorner (BBMPoint *FinalPoint, double *IntersectionStrain)
{
	return false;
}


/*
void ClassicMohrCoulomb::IntegrateMC (double* StrainIncrement, BBMPoint* InitialPoint)
{
written in mind that InitialPoint may be equal to the FinalPoint
Basically the procedure made before should be re-introduced, with some changes
due to the different approach to parameters setting. It should also incorporate the
correct value of plastic strain, instead of useless lambda. This should have been done
long time ago, but well, better late then never.
The plastic/elastic part of the strain is also linked to the
*/

/*
BBMPoint FinalPoint;
bool PurelyElastic;
bool OnTheYieldLocus;
bool Unloading;
double PurelyElasticStrain[7],PurelyPlasticStrain[7];

BBMMatrix EigenVectors(3,3);
double EigenValues [3];

//cout<<"Strains:"<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;

OnTheYieldLocus=CheckYieldNormalised (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)


CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
//cout<<"Stress after elastic step:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;


PurelyElastic=!(CheckIfPlastic (&FinalPoint)); //CheckIfPlastic returns true if in the plastic part of the YL. If on the YL (within Tolerance INT_TOL returns false)
//cout<<"p="<<FinalPoint.GetMeanStress()<<"  q="<<FinalPoint.GetShearStress()<<"  Yield? "<<PurelyElastic<<endl; getchar();
//PurelyElastic=TRUE;

if (PurelyElastic)
{
	//FinalPoint.Copy(InitialPoint);
	//update with what we have - all updated after the procedure
	cout<<"Purely Elastic Step"<<endl;
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
			FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
			CalculatePlasticMC (PurelyPlasticStrain,&FinalPoint); //and subsequently plastic part
			//FinalPoint.Copy(InitialPoint);
		}
		else {
			CalculatePlasticMC (StrainIncrement,&FinalPoint);
			}
	}
	else
	{
	FindIntersection (StrainIncrement,InitialPoint,PurelyElasticStrain,PurelyPlasticStrain);	//not on the yield locus, finding intersection and subsequently calculate elastic and plastic stress increment
	CalcElastic (PurelyElasticStrain, InitialPoint,&FinalPoint);
	FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
	//cout<<"Integrate MC:End of elastic calculations. Stress:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;
	CalculatePlasticMC (PurelyPlasticStrain,&FinalPoint);
	//FinalPoint.Copy(InitialPoint);
	}

}
//for (int i=0; i<3; i++) cout<<StrainIncrement[i]<<"  "<<endl;
//for (int i=0; i<3; i++) cout<<FinalPoint.stress[i]<<"  "<<endl;

FinalPoint.SetSpecificVolume(ComputeNu(FinalPoint.stress,FinalPoint.state, FinalPoint.GetSuction()));
FinalPoint.Copy(InitialPoint);
//getchar();
//*/
//}


/*
void ClassicMohrCoulomb::IntegrateMCI (double* StrainIncrement, BBMPoint* InitialPoint)
{
Note: does not work correctly
written in mind that InitialPoint may be equal to the FinalPoint
Basically the procedure made before should be re-introduced, with some changes
due to the different approach to parameters setting. It should also incorporate the
correct value of plastic strain, instead of useless lambda. This should have been done
long time ago, but well, better late then never.
The plastic/elastic part of the strain is also linked to the


BBMPoint FinalPoint;
bool PurelyElastic;
bool OnTheYieldLocus;
bool Unloading;
double PurelyElasticStrain[7],PurelyPlasticStrain[7];

BBMMatrix EigenVectors(3,3);
double EigenValues [3];

//cout<<"Strains:"<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;

OnTheYieldLocus=CheckYieldNormalised (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)


CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress
//cout<<"Stress after elastic step:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;
CorrectDriftMC(&FinalPoint);
FinalPoint.Copy(InitialPoint);

}
*/


void ClassicMohrCoulomb::Integrate (double* StrainIncrement, BBMPoint* InitialPoint)
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

BBMMatrix EigenVectors(3,3);
//double EigenValues [3];

//cout<<'Strains:'<<StrainIncrement[0]<<' '<<StrainIncrement[1]<<' '<<StrainIncrement[2]<<' '<<StrainIncrement[3]<<' '<<StrainIncrement[4]<<' '<<StrainIncrement[5]<<endl;

OnTheYieldLocus=CheckYieldNormalised (InitialPoint);  //checking if on the YL (Returns true if on the YL or outside, in the plastic part)


CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress

//cout<<"Stress"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;

// this rotation is not necessary, done just to test the algorithm
/*
//cout<<"Strain"<<StrainIncrement[0]<<" "<<StrainIncrement[1]<<" "<<StrainIncrement[2]<<" "<<StrainIncrement[3]<<" "<<StrainIncrement[4]<<" "<<StrainIncrement[5]<<endl;
    //1: Find Stress EigenValues and EigenVectors
    InitialPoint->GetEigen(EigenValues,&EigenVectors);
    InitialPoint->SetStressEigen(EigenValues);
    //2:Rotate the strain
	EigenVectors.RotateToEigen(InitialPoint->strain,&EigenVectors);
	EigenVectors.RotateToEigen(InitialPoint->plastic_strain,&EigenVectors);
    EigenVectors.RotateToEigen(StrainIncrement,&EigenVectors);
    //3:Recompute the strain increment

cout<<"Rotated Strain"<<StrainIncrement[0]<<" "<<StrainIncrement[1]<<" "<<StrainIncrement[2]<<" "<<StrainIncrement[3]<<" "<<StrainIncrement[4]<<" "<<StrainIncrement[5]<<endl;


CalcElastic (StrainIncrement, InitialPoint, &FinalPoint);    //Initial Point copied to the final Point, later Final point updated with the elastic stress


	EigenVectors.RotateToOrigin(InitialPoint->stress,&EigenVectors);
	EigenVectors.RotateToOrigin(InitialPoint->strain,&EigenVectors);
	EigenVectors.RotateToOrigin(InitialPoint->plastic_strain,&EigenVectors);
    EigenVectors.RotateToOrigin(FinalPoint.stress,&EigenVectors);
	EigenVectors.RotateToOrigin(FinalPoint.strain,&EigenVectors);
	EigenVectors.RotateToOrigin(FinalPoint.plastic_strain,&EigenVectors);
	EigenVectors.RotateToOrigin(StrainIncrement,&EigenVectors);

//cout<<"Strain increment after rotation"<<StrainIncrement[0]<<" "<<StrainIncrement[1]<<" "<<StrainIncrement[2]<<" "<<StrainIncrement[3]<<" "<<StrainIncrement[4]<<" "<<StrainIncrement[5]<<endl;
//cout<<"Stress after rotation"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;
//cout<<"Press any key!"<<endl;
//getchar();
*/

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
		Unloading=CheckGradient(InitialPoint, &FinalPoint); //checking if there is any unloading part; in FinalPoint purely elastic values; we care only for correct change of suction
		//cout<<endl<<endl<<"Unloading="<<Unloading<<endl;
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
	//cout<<"End of elastic calculations. Stress:"<<FinalPoint.stress[0]<<" "<<FinalPoint.stress[1]<<" "<<FinalPoint.stress[2]<<" "<<FinalPoint.stress[3]<<" "<<FinalPoint.stress[4]<<" "<<FinalPoint.stress[5]<<endl;
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

double ClassicMohrCoulomb::CalculatePlasticMC (double * PurelyPlasticStrain, BBMPoint* Point)
{
	//Note that the elasto-plastic matrix is constant in MC
	//As such we get the DEP matrix first , check if we get to the corners and if not we are done. If we do, we need to come out with a special algorithm
	BBMMatrix DEP(6,6),DSIGMA(6,1),DEPS(6,1);
	BBMMatrix EigenVectors(3,3);
	double EigenValues [3];


	for (int i=1; i<7; i++) DEPS.PutElement(i,1,PurelyPlasticStrain[i-1]);
//	cout<<"DEPS before rotation:"<<endl;
//	DEPS.Print();
	Point->GetEigen(EigenValues,&EigenVectors);
//	cout<<"Initial Eigenvalues:"<<EigenValues[0]<<' '<<EigenValues[1]<<' '<<EigenValues[2]<<' '<<endl;
//	cout<<"Initial Eigenvectors:"<<endl;
//	EigenVectors.Print();

    Point->SetStressEigen(EigenValues);
    //2:Rotate the strain
    EigenVectors.RotateToEigen(&DEPS,&EigenVectors);
//	cout<<"DEPS after rotation:"<<endl;
//	DEPS.Print();

	CalculateElastoPlasticTangentMatrix(Point,&DEP); //matrix must be rotated to eigenvalues first

	DEP.Multiply(&DEPS,&DSIGMA);
//	cout<<"Stress increment in Eigenvalues coordinates:"<<endl;
//	DSIGMA.Print();

	for (int i=1; i<7; i++) Point->stress[i-1]=Point->stress[i-1]+DSIGMA.GetElement(i,1);
	CorrectDrift(Point);

	EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
/*
	Point->GetEigen(EigenValues,&EigenVectors);
	cout<<"Final Eigenvalues:"<<EigenValues[0]<<' '<<EigenValues[1]<<' '<<EigenValues[2]<<' '<<endl;
	cout<<"Final Eigenvectors:"<<endl;
	EigenVectors.Print();
	getchar();
*/
	return 0;
}




void ClassicMohrCoulomb::ReturnImplicit (BBMPoint* Point, int* Region)
{

BBMMatrix EigenVectors(3,3);
double EigenValues [3];
for (int i=0; i<6; i++) Point->stress[i]=-Point->stress[i];

//cout<<"Point stress initial:"<<Point->stress[0]<<"  "<<Point->stress[1]<<"  "<<Point->stress[2]<<"  "<<Point->stress[3]<<"  "<<Point->stress[4]<<"  "<<Point->stress[5]<<"  "<<endl;
Point->GetEigen(EigenValues,&EigenVectors);
Point->SetStressEigen(EigenValues);
EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
EigenVectors.RotateToEigen(Point->plastic_strain,&EigenVectors);

double k=(1+SinPhi)/(1-SinPhi);
double m=(1+SinPsi)/(1-SinPsi);

BBMMatrix DEL (3,3), NUMERATOR (1,1), DENOMINATOR (1,1);

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

BBMMatrix APEX (3,1),SIGMA (3,1), SIGMAAPEX (3,1);

for (int i=0; i<3; i++)
	{
	APEX.PutElementZ(i,0,2*Cohesion*sqrt(k)/(k-1));
	SIGMA.PutElementZ(i,0,Point->stress[i]);
	SIGMA.Substract(&APEX,&SIGMAAPEX);
	}

BBMMatrix RP1 (3,1), A1 (3,1), B1(3,1);

A1.PutElement(1,1,k);A1.PutElement(3,1,-1.0);
B1.PutElement(1,1,m);B1.PutElement(3,1,-1.0);

DEL.Multiply(&B1,&RP1);
A1.Transpose(&DENOMINATOR);
DENOMINATOR.Multiply(&DEL,&DENOMINATOR);
DENOMINATOR.Multiply(&B1,&DENOMINATOR);
RP1.Multiply(1.0/DENOMINATOR.GetElement(1,1),&RP1);
//RP1.Print();
//first update direction done

BBMMatrix NI_II (3,1), R1 (3,1);
//vector product
R1.PutElement(1,1,1.0);R1.PutElement(2,1,1.0);R1.PutElement(3,1,k);

RP1.VectorProduct(&R1,&NI_II);
//NI_II.Print();
NI_II.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&SIGMAAPEX,&NUMERATOR);

double pI_II=NUMERATOR.GetElement(1,1);


BBMMatrix NI_III (3,1), R2 (3,1);
//vector product
R2.PutElement(1,1,1.0);R2.PutElement(2,1,k);R2.PutElement(3,1,k);

RP1.VectorProduct(&R2,&NI_III);
NI_III.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&SIGMAAPEX,&NUMERATOR);

double pI_III=NUMERATOR.GetElement(1,1);


BBMMatrix RP2 (3,1), A2 (3,1), B2(3,1);

A2.PutElement(2,1,k);A2.PutElement(3,1,-1.0);
B2.PutElement(2,1,m);B2.PutElement(3,1,-1.0);

DEL.Multiply(&B2,&RP2);
A2.Transpose(&DENOMINATOR);
DENOMINATOR.Multiply(&DEL,&DENOMINATOR);
DENOMINATOR.Multiply(&B2,&DENOMINATOR);
RP2.Multiply(1.0/DENOMINATOR.GetElement(1,1),&RP2);
//RP2.Print();
BBMMatrix N2 (3,1);

RP1.VectorProduct(&RP2,&N2);

R1.Transpose(&DENOMINATOR);
DENOMINATOR.Multiply(&N2, &DENOMINATOR);
N2.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&SIGMAAPEX,&NUMERATOR);
double t1=NUMERATOR.GetElement(1,1)/DENOMINATOR.GetElement(1,1);

//first line finished

BBMMatrix RP3 (3,1), A3 (3,1), B3(3,1);

A3.PutElement(1,1,k);A3.PutElement(2,1,-1.0);
B3.PutElement(1,1,m);B3.PutElement(2,1,-1.0);

DEL.Multiply(&B3,&RP3);
A3.Transpose(&DENOMINATOR);
DENOMINATOR.Multiply(&DEL,&DENOMINATOR);
DENOMINATOR.Multiply(&B3,&DENOMINATOR);
RP3.Multiply(1.0/DENOMINATOR.GetElement(1,1),&RP3);
//RP3.Print();
BBMMatrix N3 (3,1);

RP1.VectorProduct(&RP3,&N3);
//RP3.Print();

R2.Transpose(&DENOMINATOR);
//R2.Print();
DENOMINATOR.Multiply(&N3, &DENOMINATOR);
N3.Transpose(&NUMERATOR);

//SIGMAAPEX.Print();
NUMERATOR.Multiply(&SIGMAAPEX,&NUMERATOR);
double t2=NUMERATOR.GetElement(1,1)/DENOMINATOR.GetElement(1,1);
//cout<<"t2="<<t2<<" Denominator="<<DENOMINATOR.GetElement(1,1)<<" N3="<<endl;
//N3.Print();

//Alternative calculations for SinPhi=0 [i.e. apex--> inf] is needed



if ((t1>0.0)&& (t2>0.0)) {
	*Region=4;
	for (int i=0; i<3; i++) Point->stress[i]=APEX.GetElement(1,1);
	}
else if (pI_II<0){
	*Region=2;
	R1.Multiply(t1,&SIGMA);
	SIGMA.Add(&APEX,&SIGMA);
	for (int i=0; i<3; i++) Point->stress[i]=SIGMA.GetElementZ(i,0);
	}
else if (pI_III<=0) {
	*Region=1;
	double f=k*Point->stress[0]-Point->stress[2]-2*Cohesion*sqrt(k);
	RP1.Multiply(f,&NUMERATOR);
	SIGMA.Substract(&NUMERATOR,&SIGMA);
	for (int i=0; i<3; i++) Point->stress[i]=SIGMA.GetElementZ(i,0);
	}
else
	{
		*Region=3;
		R2.Multiply(t2,&SIGMA);
		//SIGMA.Print();
		//APEX.Print();
		SIGMA.Add(&APEX,&SIGMA);
		//SIGMA.Print();
		for (int i=0; i<3; i++) Point->stress[i]=SIGMA.GetElementZ(i,0);
	}

EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
//cout<<"Region:"<<*Region<<endl;
//cout<<"Point stress:"<<Point->stress[0]<<"  "<<Point->stress[1]<<"  "<<Point->stress[2]<<"  "<<Point->stress[3]<<"  "<<Point->stress[4]<<"  "<<Point->stress[5]<<"  "<<endl;
for (int i=0; i<6; i++) Point->stress[i]=-Point->stress[i];
EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);


}




int ClassicMohrCoulomb::CalculatePlasticMC_NEW (double * PurelyPlasticStrain, BBMPoint* Point)
{
/*The principle of this routine is to use a double [sometimes triple] step for implicit correction of the stresses to the yield surface
Idea is to:


1: check whether the strain increment is large enough to extend over single surface of the yield locus [Y/N]

2a: if N: just do the normal return
2b: if Y:
	-compute the part of the strain increment to extend over single surface
	-do a usual single step return on that
	-do a single step return on the remaining part of the increment
		-- note an additional caveat: the return may be to the edge or to the surface, depending on the strain increment

Step 1: details

Actually it is not trivial to find whether the increment would lead to return to more than one surface

However, need to check first whether the *increases* in principal stresses are linear with the strain increment - they should be.

As such, we can get dEpsilon increment causing dSigma stress change which is dS in principal stress state.
Now, it is easy to check how much of that is needed to actually reach the moment when the single return is fine;
For dual return we can get use of the formulas from de Brost Pankaj Bicanic or similar
The final case is to return to first yield surface, compute the remaining increment again and return to the other yield surface.


One assumption to test is that the principal stress increment is proportional to the strain increment; if so, the algorithm can be made rather simple; otherwise it is not.


=====================================================================================================================================================================================
It can be shown that the assumption above is incorrect. As such, a new algorithm is proposed:

1: Elstic strain increment is coaxial to the stress increment, assumed that initial stress is equal to zero (this is because the strain increment is coaxial in piecewise manner,
i.e. the dSigma is coaxial to dEpsilon. Therefore, having an initial stress state Sigma_Zero and strain increment dEpsilon, the stress increment dSigma is coaxial with the
strain increment dEpsilon, but *is not* coaxial with the total stress increment dSigma+Sigma_Zero.

As such, there are things to check:

1: it seems that the increment in principal stresses from increase from Sigma_Zero to Sigma_Zero+Delta_Sigma caused by Delta_Epsilon is, in some cases, not proportional;
this means that the increments of Sigma1, Sigma2, Sigma3 are not linearly dependent on increments of Strain, when checked from the Sigma_Zero stress state.
The principal stress increments are proportional to the total strain change, i.e. Epsilon_Zero_Elastic + Delta_Epsilon

*Note: Check that first; the checks made suggested otherwise when Epsilon_ij i!=j was non-zero*
*Note 2: maybe bug in the algorithm?* it *SHOULD* be a line

Note 3: When there is elastic and plastic part, the final stresses are coaxial with the Epsilon_Zero_Elastic+Delta_Epsilon_Elastic
Those can be very different to Epsilon_Zero_Elastic+Delta_Epsilon as the components of the strain are not changing linearly (in some cases,yes, most often not).


Preliminary algorithm:

1: find intersection with the yield surface: corresponding to Epsilon_Zero; remaining strain is Delta_Epsilon
2: find amount of Delta_Epsilon such that the yield surface flips; i.e what is the amount for Sigma1=Sigma2 and Sigma2=Sigma3. Possible special case Sigma1=Sigma3
3: check if any of those happen within the Delta_Epsilon

No: we have a single return, or return to the edge, or return to the tip; algorithms described work properly
Yes: use smallest amount of Epsilon required and do the return to the edge. It should be the value which returns stress to the edge with a single surface return.
This will be less than the value computed in 2. For the remaining amount of strain, a single return should be enough. If it is not, a return to the edge of whole strain is required.

Only in case of *yes* the algorithm acts differently to those described. The described algorithms would generally in case 'Yes' use the single return for the whole strain to the surface.
The return of whole strain to the surface leads to incorrect result (part of the strain should be returned using the other surface/edge).

Deatils:

a) need to derive analytical expression for 1 - i.e. find how much strain is needed
b) need to derive/copy analytical expression for the return to the vertex
c) make the whole algorith work  + testing.
d) find cases when it matters; see how significant it is.

===========================================================================================================================================================
See On Mohr Coulomb Stress integration. doc for details; the algorithm described above has been changed.

In short the algorithm is:

1: Return to the plane. If a single return successful, check whether the Theta is not showing we changed planes; if not, finished; otherwise, find out how much belongs to first plane, how much (if any) to the other one
2: If return to plane unsuccessful, fird amount of strain which is the maximum for valid return to plane; this should return to the edge. Subsequently follow with the return to the edge
3: If return to the edge unsuccessful, find largest strain returning to the edge. Once found, the stress state should be on the tip of the yield locus and subsequent return should lead to no further change in stress state.

============================================================================================================================================================

//At the beginning, the stress state of the point is on the yield locus and we have strain to integrate.




*/
int Region=1;
	//Note that the elasto-plastic matrix is constant in MC
	//As such we get the DEP matrix first , check if we get to the corners and if not we are done. If we do, we need to come out with a special algorithm
	BBMMatrix DEP(6,6),DSIGMA(6,1),DEPS(6,1);
	BBMPoint PointCopy, PointEigen;
	BBMMatrix EigenVectors(3,3);
	double EigenValues[3], Strain[6];// Percentage;
	Point->Copy(&PointCopy);

	for (int i=0; i<6; i++) Strain[i]=PurelyPlasticStrain[i];
	//for (int i=1; i<7; i++) DEPS.PutElement(i,1,PurelyPlasticStrain[i-1]);
//	cout<<"DEPS before rotation:"<<endl;
//	DEPS.Print();
	PointCopy.GetEigen(EigenValues,&EigenVectors);
//	cout<<"Initial Eigenvalues:"<<EigenValues[0]<<' '<<EigenValues[1]<<' '<<EigenValues[2]<<' '<<endl;
//	cout<<"Initial Eigenvectors:"<<endl;
//	EigenVectors.Print();

    //PointCopy.SetStressEigen(EigenValues);
	PointCopy.Copy(&PointEigen);
    //2:Rotate the strain
    //EigenVectors.RotateToEigen(&DEPS,&EigenVectors);
	//EigenVectors.RotateToEigen(Strain,&EigenVectors);
	CalcElastic (Strain,&PointCopy,&PointCopy);
	//double YieldE=ComputeYieldFunction(&PointCopy);
	CorrectDriftOnce (&PointCopy);	//	CorrectDriftOnce (&PointCopy);
	//EigenVectors.RotateToOrigin(PointCopy.stress,&EigenVectors);
	//double Yield=ComputeYieldFunction(&PointCopy);
//	cout<<"DEPS after rotation:"<<endl;
//	DEPS.Print();

	//CalculateElastoPlasticTangentMatrix(Point,&DEP); //stress must be rotated to eigenvalues first

	//DEP.Multiply(&DEPS,&DSIGMA);
	//for (int i=0; i<6; i++) PointCopy.stress[i]+=DSIGMA.GetElementZ(i,0);
/* the DSIGMA is the correct one if we are on the yield locus. If we are not on the yield locus we need to find the amount
of strain that would correspond to getting to the edge of the yield locus and integrate the rest of the strain separately.
The remaining strain may need to be corrected to the line (1), line and tip (2) or other yield surface (3)

So the step now is to check whether we are on the YL, if not, find the appropriate amount of strain that would lead to stress
state being on the yield locus.
*/
	if (CheckYield(&PointCopy))
	{
			//Percentage=FindPercentageBis(Strain,&PointEigen);
			Point->Copy(&PointEigen);
			//double PartStrain[6]; for (int i=0; i<6; i++) PartStrain[i]=Percentage*Strain[i];
			CalcElastic (Strain,&PointEigen,&PointEigen);
			CorrectDriftOnce (&PointEigen);
			PointEigen.GetEigen(EigenValues,&EigenVectors);
			if (EigenValues[0]-EigenValues[1]>EigenValues[1]-EigenValues[2]) Region=3;
				else Region=2;
			CorrectDriftEdge (&PointCopy,Region);
			if (CheckYield(&PointCopy))
				{
				CorrectDriftTip (&PointCopy);
				Region=4;
				}

	}



/*
		if (CheckYield(&PointCopy))
		{
			//EigenVectors.RotateToEigen(Strain,&EigenVectors);
			Point->Copy(&PointCopy);
			Percentage=FindPercentageBis(Strain,&PointCopy);
			//Percentage=0;
			// From the edge we can try elastic return for the remaining part of strain, if it returns to the surface, we are done
			for (int i=0; i<6 ; i++)RemainingStrain [i]=Percentage*Strain[i];
			//otherwise, we should return to the edge
			PointEigen.Copy (&PointCopy);
			CalcElastic (RemainingStrain,&PointCopy,&PointCopy);
			CorrectDriftOnce (&PointCopy);//we are on the edge, now try standard return
			for (int i=0; i<6 ; i++)RemainingStrain[i] =(1.0-Percentage)*Strain[i];		//now, up to here, same as with the explicit - found the ridge correctly. However, later, the correction is kind of wrong direction...
			//Now we are at the edge
			EigenVectors.RotateToOrigin(PointCopy.stress,&EigenVectors);
			EigenVectors.RotateToOrigin(RemainingStrain,&EigenVectors);

			//need to story the EigenVectors again, so I can rotate back!!! Best rotate back first, and get new set of EigenVectors/Values.
			//same with strains...

			PointCopy.GetEigen(EigenValues,&EigenVectors);
		    PointCopy.SetStressEigen(EigenValues);
			EigenVectors.RotateToEigen(RemainingStrain,&EigenVectors);

//
			PointCopy.Copy(&PointEigen);
			// those may be not necessary
			//Yield=ComputeYieldFunction(&PointCopy);
			CalcElastic (RemainingStrain,&PointCopy,&PointCopy);
			//CorrectDriftOnce (&PointCopy);	//CorrectDriftOnce (&PointCopy); //CorrectDriftOnce (&PointCopy);  //the values are not that bad here...
			//Yield=ComputeYieldFunction(&PointCopy);



			if (CheckYield(&PointCopy))
			{
			//if the surface return is not successful, we need to return to the edge
			cout<<"Return to an edge"<<endl;
			//PointEigen.GetEigen(EigenValues,&EigenVectors);
			if (EigenValues[0]-EigenValues[1]>EigenValues[1]-EigenValues[2]) Region=3;
				else Region=2;

			PointEigen.Copy (&PointCopy);
			CalcElastic (RemainingStrain,&PointCopy,&PointCopy);
			//CorrectDriftOnce (&PointCopy);	//CorrectDriftOnce (&PointCopy); CorrectDriftOnce (&PointCopy);	CorrectDriftOnce (&PointCopy); CorrectDriftOnce (&PointCopy);//now rotation is switched somehow as compared with original before if... find why.
			CorrectDriftEdge (&PointCopy,Region); //this correction totally wrong...
			Yield=ComputeYieldFunction(&PointCopy);

			//if correct drift to the edge is not successful, we know the correction should be to the tip of the yield locus
			//thus
			if (CheckYield(&PointCopy))
				{
				PointEigen.Copy (&PointCopy);
				CorrectDriftTip (&PointCopy);
				Region=4;
				}
			}
			else
			{
				//The result is correct at the edge
			}
		}
		else
		{
			//The results is correct - a single correction to the other yield surface suffice
		}

*/

//EigenVectors.RotateToOrigin(PointCopy.stress,&EigenVectors);

//note:no strain plastic/elastic is computed
PointCopy.Copy (Point);
//end



/*

//now lets do some tests:

	double TempStress [6], EigenStress[3], TempEigen [3];
	BBMMatrix TempEigenVectors (3,3);
	BBMPoint TempPoint;
	double TotalEigen [10][3];

	Point->Copy(&TempPoint);
	for (int j=0; j<10; j++)
	{
		for (int i=0; i<6; i++)
		{
			TempStress[i]=DSIGMA.GetElementZ(i,0);
			TempStress[i]=TempStress[i]*double(j)/10000.0;
			TempPoint.stress[i]=TempPoint.stress[i]+TempStress[i];
		}
	EigenVectors.RotateToOrigin(TempPoint.stress,&EigenVectors);
	TempPoint.GetEigen (TempEigen,&TempEigenVectors);
	for (int i=0; i<3; i++) TotalEigen[j][i]=TempEigen[i]-EigenValues[i];
	for (int i=0; i<3; i++) cout<<"j="<<j<<"  "<<TempEigen[i]<<"  "<<endl;
	if (j>0) cout<<"Ratio="<<TotalEigen[j][0]/TotalEigen[j-1][0]<<"  "<<TotalEigen[j][1]/TotalEigen[j-1][1]<<" "<<TotalEigen[j][2]/TotalEigen[j-1][2]<<endl;
	}

//end tests



//	cout<<"Stress increment in Eigenvalues coordinates:"<<endl;
//	DSIGMA.Print();

				for (int i=1; i<7; i++) Point->stress[i-1]=Point->stress[i-1]+DSIGMA.GetElement(i,1);
	CorrectDrift(Point);

	EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
	*/
/*
	Point->GetEigen(EigenValues,&EigenVectors);
	cout<<"Final Eigenvalues:"<<EigenValues[0]<<' '<<EigenValues[1]<<' '<<EigenValues[2]<<' '<<endl;
	cout<<"Final Eigenvectors:"<<endl;
	EigenVectors.Print();
	getchar();
*/
	return Region;
}

void ClassicMohrCoulomb::CorrectDriftEdge (BBMPoint* InitialPoint, int Region)
{
//The procedure checks if we are on edge or on the tip of the yield; If we are on the tip, nothing is done; otherwise a correction to the edge the stress is on currently is performed
BBMMatrix R(3,1),RU(3,1);
double r[3],ru[3];




if (SinPhi!=1)
{
	if (Region==3)
	{
		//Sigma 2== Sigma 3
		r[0]=(2*Cohesion*CosPhi)/(1-SinPhi);
		r[1]=0;
		r[2]=0;
	}
	else
	{
		//Sigma 1== Sigma 2
		r[0]=(2*Cohesion*CosPhi)/(1-SinPhi);
		r[1]=r[0];
		r[2]=0;
	}
}



if (SinPsi!=1)
{
	if (Region==3)
	{
		//Sigma 2== Sigma 3
		ru[0]=(2*Cohesion*CosPsi)/(1-SinPsi);
		ru[1]=0;
		ru[2]=0;
	}
	else
	{
		//Sigma 1== Sigma 2
		ru[0]=(2*Cohesion*CosPsi)/(1-SinPsi);
		ru[1]=ru[0];
		ru[2]=0;

	}
}


if (SinPhi!=0)	for (int i=0; i<3; i++)
{
	r[i]+=Cohesion*CosPhi/SinPhi;

}

if (SinPsi!=0) for (int i=0; i<3; i++)

{
	ru[i]+=Cohesion*CosPsi/SinPsi;
}

//*/

/*
if (SinPhi!=1)
{
	if (Region==3)
	{
		//Sigma 2== Sigma 3
		r[0]=(1+SinPhi)/(1-SinPhi);
		r[1]=1;
		r[2]=1;
	}
	else
	{
		//Sigma 1== Sigma 2
		r[0]=(1+SinPhi)/(1-SinPhi);
		r[1]=r[0];
		r[2]=1;
	}
}



if (SinPsi!=1)
{
	if (Region==3)
	{
		//Sigma 2== Sigma 3
		ru[0]=(1+SinPsi)/(1-SinPsi);
		ru[1]=1;
		ru[2]=1;
	}
	else
	{
		//Sigma 1== Sigma 2
		ru[0]=(1+SinPsi)/(1-SinPsi);
		ru[1]=ru[0];
		ru[2]=1;

	}
}


//*/






if (SinPhi==0)
{

		r[0]=1.0;
		r[1]=1.0;
		r[2]=1.0;
}



if (SinPsi==0)
{

		ru[0]=1.0;
		ru[1]=1.0;
		ru[2]=1.0;


}


for (int i=0; i<3; i++) {
	R.PutElementZ(i,0,r[i]);
	RU.PutElementZ(i,0,ru[i]);
	}
//remaining elements initialised with zero anyway

BBMMatrix EigenVectors(3,3);
double EigenValues[3];
InitialPoint->GetEigen(EigenValues,&EigenVectors);
InitialPoint->SetStressEigen(EigenValues);


BBMMatrix DEL(3,3), DELINV(3,3);
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

DEL.Inverse(&DELINV);
//BBMMatrix Test (1,1);
//DELINV.Print();


BBMMatrix SCONST (3,1);

if (SinPhi!=0) for (int i=0; i<3; i++) SCONST.PutElementZ(i,0,-Cohesion*CosPhi/SinPhi);
//for (int i=0; i<3; i++) SCONST.PutElementZ(i,0,2*Cohesion*sqrt((1+SinPhi)/(1-SinPhi))/((1+SinPhi)/(1-SinPhi)-1));
else
{
if (Region==3)
	{
		//Sigma 2== Sigma 3
	SCONST.PutElement(1,1,4*Cohesion/3.0);
	SCONST.PutElement(2,1,-2*Cohesion/3.0);
	SCONST.PutElement(3,1,-2*Cohesion/3.0);
	}
	else
	{
		//Sigma 1== Sigma 2
	SCONST.PutElement(1,1,2*Cohesion/3.0);
	SCONST.PutElement(2,1,2*Cohesion/3.0);
	SCONST.PutElement(3,1,-4*Cohesion/3.0);
	}
}


double t;
BBMMatrix SIGMA (3,1);
for (int i=0; i<3; i++) SIGMA.PutElementZ(i,0,InitialPoint->stress[i]);
BBMMatrix NUMERATOR (1,1), DENOMINATOR (1,1);

RU.Transpose(&NUMERATOR);
NUMERATOR.Multiply(&DELINV,&NUMERATOR);
NUMERATOR.Copy(&DENOMINATOR);
//SIGMA.Print();
SIGMA.Substract(&SCONST,&SIGMA);

//SIGMA.Print();
NUMERATOR.Multiply(&SIGMA,&NUMERATOR);
DENOMINATOR.Multiply(&R,&DENOMINATOR);

t=NUMERATOR.GetElementZ(0,0)/DENOMINATOR.GetElementZ(0,0);


R.Multiply(t,&SIGMA);
SIGMA.Add(&SCONST,&SIGMA); //corrected stress in SIGMA matrix
//the strain will be updated in the end, getting sigma and computing the inverse matrix

for (int i=0; i<3; i++) InitialPoint->stress[i]=SIGMA.GetElementZ(i,0);
EigenVectors.RotateToOrigin(InitialPoint->stress,&EigenVectors);

}


void ClassicMohrCoulomb::CorrectDriftEdgeJC (BBMPoint* InitialPoint, int Region)
{
//The procedure checks if we are on edge or on the tip of the yield; If we are on the tip, nothing is done; otherwise a correction to the edge the stress is on currently is performed
BBMMatrix R(3,1),RU(3,1);
double r[3],ru[3];


double k=(1+SinPhi)/(1-SinPhi);
double m=(1+SinPsi)/(1-SinPsi);

if (SinPhi!=1)
{
	if (Region==2)
	{
		//Sigma 2== Sigma 3
		r[0]=1;
		r[1]=1;
		r[2]=k;
	}
	else
	{
		//Sigma 1== Sigma 2
		r[0]=1;
		r[1]=k;
		r[2]=k;
	}
}



if (SinPsi!=1)
{
	if (Region==2)
	{
		//Sigma 2== Sigma 3
		ru[0]=1;
		ru[1]=1;
		ru[2]=m;
	}
	else
	{
		//Sigma 1== Sigma 2
		ru[0]=1;
		ru[1]=m;
		ru[2]=m;

	}
}


/*
if (SinPhi==0)
{

		r[0]=1.0;
		r[1]=1.0;
		r[2]=1.0;
}



if (SinPsi==0)
{

		ru[0]=1.0;
		ru[1]=1.0;
		ru[2]=1.0;


}
*/

for (int i=0; i<3; i++) {
	R.PutElementZ(i,0,r[i]);
	RU.PutElementZ(i,0,ru[i]);
	}
//remaining elements initialised with zero anyway

BBMMatrix EigenVectors(3,3);
double EigenValues[3];
InitialPoint->GetEigen(EigenValues,&EigenVectors);
InitialPoint->SetStressEigen(EigenValues);


BBMMatrix DEL(3,3), DELINV(3,3);
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

DEL.Copy(&DELINV);
DEL.Inverse3x3(&DELINV);
DELINV.Print();
//BBMMatrix Test (1,1);

BBMMatrix SCONST (3,1);

if (SinPhi!=0) for (int i=0; i<3; i++) SCONST.PutElementZ(i,0,2*Cohesion*sqrt(k)/(k-1));
/*
else
{
if (Region==3)
	{
		//Sigma 2== Sigma 3
	SCONST.PutElement(1,1,4*Cohesion/3.0);
	SCONST.PutElement(2,1,-2*Cohesion/3.0);
	SCONST.PutElement(3,1,-2*Cohesion/3.0);
	}
	else
	{
		//Sigma 1== Sigma 2
	SCONST.PutElement(1,1,2*Cohesion/3.0);
	SCONST.PutElement(2,1,2*Cohesion/3.0);
	SCONST.PutElement(3,1,-4*Cohesion/3.0);
	}
}
*/

double t;
BBMMatrix SIGMA (3,1);
	SIGMA.PutElementZ(0,0,-InitialPoint->stress[2]);
	SIGMA.PutElementZ(1,0,-InitialPoint->stress[1]);
	SIGMA.PutElementZ(2,0,-InitialPoint->stress[0]);
BBMMatrix NUMERATOR (1,1), DENOMINATOR (1,1);

//SIGMA.Print();
//SCONST.Print();
//BBMMatrix TEMP(3,3); SIGMA.Substract(&SCONST,&TEMP); TEMP.Print();

RU.Transpose(&NUMERATOR);
//NUMERATOR.Print();
NUMERATOR.Multiply(&DELINV,&NUMERATOR);
NUMERATOR.Print(); //ru*D-1
NUMERATOR.Copy(&DENOMINATOR);
//SIGMA.Print();
SIGMA.Substract(&SCONST,&SIGMA);
SIGMA.Print();
NUMERATOR.Multiply(&SIGMA,&NUMERATOR);
NUMERATOR.Print();

/*
SIGMA.Transpose(&SIGMA);
SIGMA.Copy(&NUMERATOR);
NUMERATOR.Multiply(&DELINV,&NUMERATOR);
NUMERATOR.Multiply(&RU,&NUMERATOR);
*/

DENOMINATOR.Multiply(&R,&DENOMINATOR);
DENOMINATOR.Print();
		t=NUMERATOR.GetElementZ(0,0)/DENOMINATOR.GetElementZ(0,0);
	cout<<"t="<<t<<endl;

R.Multiply(t,&SIGMA);
SIGMA.Add(&SCONST,&SIGMA); //corrected stress in SIGMA matrix
//the strain will be updated in the end, getting sigma and computing the inverse matrix
SIGMA.Print();

//for (int i=0; i<3; i++) InitialPoint->stress[i]=-SIGMA.GetElementZ(i,0);

	InitialPoint->stress[0]=-SIGMA.GetElementZ(2,0);
	InitialPoint->stress[1]=-SIGMA.GetElementZ(1,0);
	InitialPoint->stress[2]=-SIGMA.GetElementZ(0,0);
EigenVectors.RotateToOrigin(InitialPoint->stress,&EigenVectors);

}


void ClassicMohrCoulomb::CorrectDriftTip (BBMPoint* InitialPoint)
{
	for (int i=0; i<3; i++) InitialPoint->stress[i]=-Cohesion*CosPhi/SinPhi;
	for (int i=3; i<6; i++) InitialPoint->stress[i]=0.0;
}


double ClassicMohrCoulomb::FindPercentage (double * Strain, BBMPoint* InitialPoint)
{
	//Procedure finds the percentage of strain required to get principal stresses equal (S1=S2 or S2=S3).

	double Percentage=1.0,Delta;
	double PercLow=0,PercHigh=1,Theta,ThetaOld=0,Theta1,Theta2,TargetDeg;
	double TempStrain[6], Eigens[3];
	BBMPoint FinalPoint;
	InitialPoint->Copy(&FinalPoint);
	CalcElastic(Strain,InitialPoint,&FinalPoint);
	CorrectDriftOnce(&FinalPoint);
	FinalPoint.GetEigen(Eigens);
	//double YieldF=ComputeYieldFunction (&FinalPoint);

	//1: we need to find out whether we are going for S1=S2 or S2=S3 case. To do that, a Theta check is made

	Theta1=InitialPoint->GetThetaDeg_0();
	Theta2=FinalPoint.GetThetaDeg_0();

	//Now, when we go for S1=S2

	//we are looking for Theta closest to Theta 2, with round 60 deg, i.e. 0,60,120,180,240,300,360
	if (Theta2>Theta1) TargetDeg=60*(int(Theta2)/60);
	else TargetDeg=60*(int(Theta2)/60)+60;

	//This is one of the things to check whether the procedure works as intended up to this point

	//Now the strain corresponding to Theta=target is to be found with the Pegasus algorithm
	Theta1-=TargetDeg;
	Theta2-=TargetDeg;
	ThetaOld=Theta2;


	int Exit=0;
	do
	{
	//Delta=Theta2-Theta1;
	Percentage=PercLow-Theta1*(PercHigh-PercLow)/(Theta2-Theta1);
	for (int i=0; i<6; i++)  TempStrain[i]=Percentage*Strain[i];
	InitialPoint->Copy(&FinalPoint);
	CalcElastic(TempStrain,InitialPoint,&FinalPoint);
	CorrectDriftOnce(&FinalPoint);
	Theta=FinalPoint.GetThetaDeg_0()-TargetDeg;

	if (Theta>0)
	{
		if (ThetaOld*Theta>0)
		{
		Theta1=Theta1*Theta2/(Theta+Theta2);
		}
	Theta2=Theta;
	PercHigh=Percentage;
	}
	else
	{
		if (ThetaOld*Theta>0)
		{
		Theta2=Theta2*Theta1/(Theta+Theta1);
		}
	Theta1=Theta;
	PercLow=Percentage;
	}
	Exit++;
	if (Exit>100)
		{
		cout<<"FindPercentageProcedure, exiting after reaching 100 iterations without full convergence. Current Delta is "<<Delta<<" whereas for convergence Delta is assumed to be 1E-8"<<endl;
		}
	}
	while (!((fabs(Theta)<1E-8)||(Exit>10)));

	return Percentage;
}



double ClassicMohrCoulomb::FindPercentageBis (double * Strain, BBMPoint* InitialPoint)
{
	//Finding Percentage by bisection

	double Percentage=1.0;
	double PercLow=0,PercHigh=1;
	double TempStrain[6], Eigens[3];
	BBMPoint FinalPoint;
	InitialPoint->Copy(&FinalPoint);
	CalcElastic(Strain,InitialPoint,&FinalPoint);
	CorrectDriftOnce(&FinalPoint);
	FinalPoint.GetEigen(Eigens);
	//double YieldF=ComputeYieldFunction (&FinalPoint);


	int Exit=0;
	do
	{
	//Delta=Theta2-Theta1;
	Percentage=PercLow+(PercHigh-PercLow)/2;
	for (int i=0; i<6; i++)  TempStrain[i]=Percentage*Strain[i];
	InitialPoint->Copy(&FinalPoint);
	CalcElastic(TempStrain,InitialPoint,&FinalPoint);
	CorrectDriftOnce(&FinalPoint);
	if (CheckYield(&FinalPoint))PercHigh=Percentage;
	else PercLow=Percentage;
	}
	while (!(((PercHigh-PercLow)<1E-8)||(Exit>100)));

	return Percentage;
}


void ClassicMohrCoulomb::Integrate (double* StrainIncrement, double SuctionIncrement, BBMPoint* InitialPoint, double * StressIncrement,
	double P0StarIncrement, double* PlasticStrainIncrement)
{
    cout<<"Classic Mohr Coulomb:Integrate not implemented for use without Point structure."<<endl;
}

void ClassicMohrCoulomb::IntegrateConst (double* StrainIncrement,BBMPoint* InitialPoint, int StepNo, int Method)

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






void ClassicMohrCoulomb::FindElStrGradPQ (double nu0, double* s0  , double* eps0, double* deps,  double* ds)
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



double ClassicMohrCoulomb::CalculatePZero (BBMPoint * Point)
{
/*	double PZero=LambdaZero*((1-r)*exp(-1*Beta*Point->GetSuction())+r);
	PZero=(LambdaZero-KappaP)/(PZero-KappaP);
	PZero=pc*pow((Point->GetPStar()/pc),PZero);
	return PZero;*/
	return 0;
}




/*
bool ClassicMohrCoulomb::CheckYield (double *state, double *s, double suction)
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


void ClassicMohrCoulomb::FindYieldOriginal (double *state, double* s0, double* eps0, double* deps, double* a)
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



void ClassicMohrCoulomb::FindYieldModified (double *state, double* s0, double* eps0, double* deps, double* a)
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

void ClassicMohrCoulomb::FindIntersection (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElasticStrain, double * PurelyPlasticStrain)
{
double alfa;
FindYieldModified (InitialPoint->state, InitialPoint->stress, InitialPoint->strain, StrainIncrement, &alfa);
for (int i=0; i<7; i++)
	{
		PurelyElasticStrain[i]=alfa*StrainIncrement[i];
		PurelyPlasticStrain[i]=StrainIncrement[i]-PurelyElasticStrain[i];
	}
}






void ClassicMohrCoulomb::MoveYieldaBit (double * state, double * s0, double *ds, double * eps0, double * deps, double * gradient, double F0)
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

void ClassicMohrCoulomb::FindIntersectionUnloading (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElastic, double * PurelyPlastic)
{
double alfa;
FindYieldYield (InitialPoint->state, InitialPoint->stress, InitialPoint->strain, StrainIncrement, &alfa);
for (int i=0; i<7; i++)
{
	PurelyElastic[i]=alfa*StrainIncrement[i];
	PurelyPlastic[i]=StrainIncrement[i]-PurelyElastic[i];
}
}


void ClassicMohrCoulomb::FindYieldYield (double *state, double* s0, double* eps0, double* deps, double* a)
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
	delta=(F0+YIELDTOL)/2;
	YIELDTOL=0.9*(YIELDTOL-F0)/2;
	}

	cout<<"Delta="<<delta<<endl;
	cout<<"F0="<<F0<<endl;

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

	// check of the second yield surface:
/*
	if ((eps0[6]+deps[6])>state[1])  //so suction is larger then maximum experienced suction
	{
		alfa0=(state[1]-eps0[6])/deps[6];
		for (int i=0; i<7; i++)
		{
			epsini[i]=deps[i]*alfa0;
		}
		CalcStressElast (state[2], s0, eps0, epsini, sini);
		for (int i=0; i<6;i++) sini[i]=sini[i]+s0[i];
		CheckYieldNormalised (state, sini, eps0[6]+epsini[6],&F0);
		F0=F0-delta;
	// main F checked
		if (F0<YIELDTOL) //we may finish when there is no yield at this strain at F0. so..
		{
			*a=alfa0; // so we have elastic state in other case,
			YIELDTOL=oldyieldtol;
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
	CheckYieldNormalised (state, sini, eps0[6]+epsini[6],&F0);
	CheckYieldNormalised (state, sfini , eps0[6]+epsfini[6],&F1);
	F0=F0-delta;
	F1=F1-delta;
	cout <<"Procedure FindYieldYield. Initial values of F0="<<F0<<endl<<"  F1="<<F1<<endl;
	cout <<"F0 should lie on the yield locus within tolerance:"<<oldyieldtol<<endl;

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
		cout<<"Solution in FindYieldYield procedure was found after "<<iter<<" iterations."<<endl;
		if (iter>50) {
				getchar();
				cout<<"Large number of iterations!!! Solution is however correct... Press any key..."<<endl;
		}
		cout<<"Value of the yield function is equal to:"<<Falfa<<endl;
		cout<<"Modified value of tolerance is:"<<YIELDTOL<<endl;
		cout<<"Value of delta is:"<<delta<<endl;
		//cout<<"Alfa is equal to: "<<*a<<endl;
		YIELDTOL=oldyieldtol;
		cout<<"Yield tolerance is set back to:"<<YIELDTOL<<endl;
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


void ClassicMohrCoulomb::read ()
{
	//reads data from file "ClassicMohrCoulomb.dta"
	ifstream infile("ClassicMohrCoulomb.dta", ios_base::in);

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

	SetModelParameters (storage[0],storage[1],storage[2],storage[3],storage[3]); //G, K, Cohesion, Friction Angle


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


void ClassicMohrCoulomb::write()
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


void ClassicMohrCoulomb::PaintLocus (double *state, double suction, int Max)
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


double ClassicMohrCoulomb::CalculatePlasticConst (double * PurelyPlasticStrain, BBMPoint* Point, int StepNo)
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


double ClassicMohrCoulomb::CalculatePlastic (double * PurelyPlasticStrain, BBMPoint* Point)
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

//cout<<"Runge Kutta result computed in "<<NumberIter<<" iterations."<<endl;

return time;
}

double ClassicMohrCoulomb::PlasticEuler (BBMPoint* Point, double* EPStrain, double* AbsStress, int NumberIterations)
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


double ClassicMohrCoulomb::RungeKutta (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter,
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
BBMMatrix DSIGMA (6,1),EigenVectors(3,3);
double EigenValues [3];
BBMPoint MidPoint[8], OldPoint, TrialPoint;

//double CriticalStepSize=1E-3; //HACK HACK HACK

double DSigma[8][6], DSigmaTemp[7], Result[7];
double DPZeroStar[8],DPZeroStarTemp, PlasticStrainTemp[6], PlasticStrain[8][6];
double Error[7],ReUseRes[7],RError, NewStepSize,TotalSize, StepLength, Temp, MethodPower;
double dSigmaDrift[6], dSigmaDriftConst[6], dSigmaDriftOldConst[6], dLambdaDrift;

double Frequency=1000000;//15000/MethodOrder; //how often display info about steps
bool ReUseStep=false;

StepLength=1;
TotalSize=0; NewStepSize=1;
MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;


Point->Copy(&MidPoint[0]);
    MidPoint[0].GetEigen(EigenValues,&EigenVectors);
    MidPoint[0].SetStressEigen(EigenValues);
    EigenVectors.RotateToOrigin(MidPoint[0].stress,&EigenVectors);
	if ((MidPoint[0].stress[0]-Point->stress[0])>0.1||(MidPoint[0].stress[1]-Point->stress[1])>0.1||(MidPoint[0].stress[2]-Point->stress[2])>0.1) {
		cout<<"Problem with initial test rotation procedure in the Runge-Kutta procedure. Stress tensor rotated back to origin is not equal to the initial stress. Please investigate... "<<endl;
		getchar();
	}


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

NewStepSize=1.0;
//for (int i=0; i<6; i++) NewStepSize=NewStepSize+fabs(SubstepStrain[i]);
//NewStepSize=0.01/(NewStepSize*MethodOrder);
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
if (StepLength<CRITICAL_STEP_SIZE) StepLength=CRITICAL_STEP_SIZE; //HACK here, it is a fairly large value
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
    //1: Find Stress EigenValues and EigenVectors
    MidPoint[0].GetEigen(EigenValues,&EigenVectors);
    MidPoint[0].SetStressEigen(EigenValues);
    //2:Rotate the strain
	//	cout<<"Stress Origin: Midpoint[0]"<<MidPoint[0].stress[0]<<" "<<MidPoint[0].stress[1]<<" "<<MidPoint[0].stress[2]<<" "<<MidPoint[0].stress[3]<<" "<<MidPoint[0].stress[4]<<" "<<MidPoint[0].stress[5]<<endl;
	//EigenVectors.RotateToEigen(MidPoint[0].stress,&EigenVectors);
	//	cout<<"Stress Eigen: Midpoint[0]"<<MidPoint[0].stress[0]<<" "<<MidPoint[0].stress[1]<<" "<<MidPoint[0].stress[2]<<" "<<MidPoint[0].stress[3]<<" "<<MidPoint[0].stress[4]<<" "<<MidPoint[0].stress[5]<<endl;
	//EigenVectors.RotateToOrigin(MidPoint[0].stress,&EigenVectors);
	//	cout<<"Stress Origin: Midpoint[0]"<<MidPoint[0].stress[0]<<" "<<MidPoint[0].stress[1]<<" "<<MidPoint[0].stress[2]<<" "<<MidPoint[0].stress[3]<<" "<<MidPoint[0].stress[4]<<" "<<MidPoint[0].stress[5]<<endl;


	//EigenVectors.RotateToEigen(MidPoint[0].stress,&EigenVectors);
	EigenVectors.RotateToEigen(MidPoint[0].strain,&EigenVectors);
	EigenVectors.RotateToEigen(MidPoint[0].plastic_strain,&EigenVectors);
    EigenVectors.RotateToEigen(SubstepStrain,&EigenVectors);
    //3:Recompute the strain increment


	//cout<<"Stress: Midpoint[0]"<<MidPoint[0].stress[0]<<" "<<MidPoint[0].stress[1]<<" "<<MidPoint[0].stress[2]<<" "<<MidPoint[0].stress[3]<<" "<<MidPoint[0].stress[4]<<" "<<MidPoint[0].stress[5]<<endl;
	//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;

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
	//4:Rotate back the computed stress
    EigenVectors.RotateToOrigin(MidPoint[0].stress,&EigenVectors);
	//cout<<"Stress Midpoint["<<0<<"] "<<MidPoint[0].stress[0]<<" "<<MidPoint[0].stress[1]<<" "<<MidPoint[0].stress[2]<<" "<<MidPoint[0].stress[3]<<" "<<MidPoint[0].stress[4]<<" "<<MidPoint[0].stress[5]<<endl;
	EigenVectors.RotateToOrigin(MidPoint[0].strain,&EigenVectors);
	EigenVectors.RotateToOrigin(MidPoint[0].plastic_strain,&EigenVectors);
	EigenVectors.RotateToOrigin(SubstepStrain,&EigenVectors);
	EigenVectors.RotateToOrigin(PlasticStrainTemp,&EigenVectors);
	EigenVectors.RotateToOrigin(&DSIGMA,&EigenVectors);
	//DSIGMA.Print();
	//getchar();


	for (int i=0; i<6; i++)
	{
		DSigma[0][i]=DSIGMA.GetElement(i+1,1);
		PlasticStrain[0][i]=PlasticStrainTemp[i];
	}
}

//NICE SCHEME does NOT work for the Mohr Coulomb

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

    //1: Find Stress EigenValues and EigenVectors

    //2:Rotate the strain

    //3:Recompute the strain increment

    //1: Find Stress EigenValues and EigenVectors
	//cout<<"Before EigenValues["<<rkloop-1<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
    MidPoint[rkloop].GetEigen(EigenValues,&EigenVectors);
	//cout<<"New EigenValues["<<rkloop<<"] "<<EigenValues[0]<<" "<<EigenValues[1]<<" "<<EigenValues[2]<<endl;
    MidPoint[rkloop].SetStressEigen(EigenValues);
    //2:Rotate the strain
	EigenVectors.RotateToEigen(MidPoint[rkloop].strain,&EigenVectors);
	EigenVectors.RotateToEigen(MidPoint[rkloop].plastic_strain,&EigenVectors);
    EigenVectors.RotateToEigen(SubstepStrain,&EigenVectors);
	//cout<<"Stress Midpoint["<<rkloop<<"] "<<MidPoint[rkloop].stress[0]<<" "<<MidPoint[rkloop].stress[1]<<" "<<MidPoint[rkloop].stress[2]<<" "<<MidPoint[rkloop].stress[3]<<" "<<MidPoint[rkloop].stress[4]<<" "<<MidPoint[rkloop].stress[5]<<endl;
	//cout<<"SubstepStrain after rot"<<SubstepStrain[0]<<" "<<SubstepStrain[1]<<" "<<SubstepStrain[2]<<" "<<SubstepStrain[3]<<" "<<SubstepStrain[4]<<" "<<SubstepStrain[5]<<endl;


    //3:Recompute the strain increment

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


    //4:Rotate back the computed stress
    EigenVectors.RotateToOrigin(MidPoint[rkloop].stress,&EigenVectors);
	EigenVectors.RotateToOrigin(MidPoint[rkloop].strain,&EigenVectors);
	EigenVectors.RotateToOrigin(MidPoint[rkloop].plastic_strain,&EigenVectors);
    EigenVectors.RotateToOrigin(SubstepStrain,&EigenVectors);
	EigenVectors.RotateToOrigin(PlasticStrainTemp,&EigenVectors);
	EigenVectors.RotateToOrigin(&DSIGMA,&EigenVectors);


	//DSIGMA.Print();
	//getchar();


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
		DSigma[rkloop][i]=DSIGMA.GetElement(i+1,1)+dSigmaDriftConst[i];  //dSigmaDriftConst==0 if NICE is not equal to 3 or 4
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

for (int i=0; i<7; i++) if (!isfinite(Result[i]))
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
	if (StepLength<=CRITICAL_STEP_SIZE) {StepAccepted=true;
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




	if (DRIFT_CORRECTION==3)
	{
	    // DRift correction: need to rotate to eigenvalues again
		Point->GetEigen(EigenValues,&EigenVectors);
		Point->SetStressEigen(EigenValues);
    //2:Rotate the strain
		EigenVectors.RotateToEigen(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);


	    CorrectDrift (Point);

		EigenVectors.RotateToOrigin(Point->stress,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->strain,&EigenVectors);
		EigenVectors.RotateToOrigin(Point->plastic_strain,&EigenVectors);



		//and rotate back
	}
	if (DRIFT_CORRECTION==2)
	{
        // DRift correction: need to rotate to eigenvalues again
		cout<<"CorrectDriftBeginning is not well implemented for the Classical Mohr Coulomb. No drift correction made!"<<endl;
	    //CorrectDriftBeg(Point, &OldPoint); //value of OldPoint copied before updating the point
	    //and rotate back
	}

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
if (!isfinite(RError)) if (RError<MethodPower) RError=MethodPower;

if (TOL_METHOD==0) NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/(MethodOrder-1.0)));
else NewStepSize=BETA_FACT*pow(INTEGRATION_TOL/RError,(1/MethodOrder));

for (int i=0; i<6; i++) dSigmaDriftOldConst[i]=dSigmaDriftOldConst[i]*NewStepSize;
//cout<<dSigmaDriftOldConst[0]<<endl; getchar();

if (RError<INTEGRATION_TOL) StepAccepted=true ;
else
{
	StepAccepted=false;
	if (StepLength<=CRITICAL_STEP_SIZE)
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
		cout<<"Strain is:"<<SubstepStrain[0]<<" , "<<SubstepStrain[1]<<" , "<<SubstepStrain[2]<<" , "<<endl;
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
		cout<<"Strain is:"<<SubstepStrain[0]<<" , "<<SubstepStrain[1]<<" , "<<SubstepStrain[2]<<" , "<<endl;
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




double ClassicMohrCoulomb::RungeKuttaEqualStep (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
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
    cout<<"RungeKutta Equal step does not work for the Classic Mohr Coulomb model. Would need an update regarding rotation of the stress and strain tensors"<<endl;

BBMMatrix DSIGMA (6,1);
BBMPoint MidPoint[8], OldPoint, TrialPoint;

double DSigma[8][6], DSigmaTemp[7], Result[7];
double DPZeroStar[8],DPZeroStarTemp, PlasticStrain[8][6], PlasticStrainTemp[6];
double Error[7],RError, TotRError,TotalSize, StepLength, Temp, MethodPower;
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

for (int i=0; i<7; i++) if (!isfinite(Result[i]))
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
if (!isfinite(RError)) if (RError<MethodPower) RError=MethodPower;

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

double ClassicMohrCoulomb::RungeKuttaExtrapol (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
									  int* NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate)
{
/*
this procedure tend to use Runge Kutta scheme. In case the step is not accepted, and the substep size is just a bit smaller
than current one , instead of cancelling the step, it goes into extrapolation procedure, just to save the substep done.
*/

cout<<"RungeKuttaExtrapol does not work for the Classic Mohr Coulomb model."<<endl;


BBMMatrix DSIGMA (6,1);
BBMPoint TrialPoint, OldPoint;
double WORTH_EXTRAPOL=3;

//double DSigma[8][6];
//double DPZeroStar[8];
double RError, NewStepSize,TotalSize, StepLength, Temp;
double Frequency=15000/MethodOrder; //how often display info about steps
//bool ReUseStep=false;

StepLength=1;
TotalSize=0; NewStepSize=1;
//MethodPower=pow(2.0,MethodOrder)*INTEGRATION_TOL;

//for (int i=0; i<MethodOrder; i++) {
	//for (int j=0; j<6; j++) DSigma[i][j]=0;
	//DPZeroStar[i]=0;	}

bool Finished=false, StepAccepted=false;
double SubstepStrain [7];
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


double ClassicMohrCoulomb::PlasticRKErr8544 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/* the procedure uses the embedded Runge - Kutta integration scheme with Adaptive Stepsize Control
the constants are as proposed by Bogacki and Shampine (1996), An efficient R-K (4,5) pair, Computers Math Applic, Vol 32 No 6 pp 15-28
with FSAL feauture the method allows for getting the error estimate and calculating value in one go
It is arguably better than any other 5(4) RK pair; It has double 4th order error estimate
*/

cout<<"PlasticRKErr8544 procedure is not updated for the CLassical Mohr Coulomb Model and the results obtained are incorrect"<<endl;

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

for (int i=0; i<7; i++) if (!isfinite(Result[i]))
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
if (!isfinite(RError)) if (RError<MethodPower) RError=MethodPower;

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


double ClassicMohrCoulomb::PlasticRKNoExTry (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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



double ClassicMohrCoulomb::PlasticRKDP754 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

double ClassicMohrCoulomb::PlasticRKCK654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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


double ClassicMohrCoulomb::PlasticRKEng654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

double ClassicMohrCoulomb::PlasticRK543 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

double ClassicMohrCoulomb::PlasticRK332 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

double ClassicMohrCoulomb::PlasticRKBog432 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

double ClassicMohrCoulomb::PlasticRKME221 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
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

 double ClassicMohrCoulomb::PlasticMidpointGallipoli (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/*this procedure is to calculate the stress increase using the midpoint method with given number of iterations NumberIter.
is made mainly for use in the extrapolation procedure. It has no adaptive substepping or error control. It just integrates
the strain to get the stress in given number of substeps using the midpoint method
*/
//clock_t StartTime, EndTime;
//StartTime=clock();


cout<<"PlasticMidpointGallipoli procedure is wrong"<<endl;

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



double ClassicMohrCoulomb::PlasticMidpoint (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{
/*this procedure is to calculate the stress increase using the midpoint method with given number of iterations NumberIter.
is made mainly for use in the extrapolation procedure. It has no adaptive substepping or error control. It just integrates
the strain to get the stress in given number of substeps using the midpoint method
*/
//clock_t StartTime, EndTime;
//StartTime=clock();

cout<<"Plastic Midpoint procedure does not work as intended"<<endl;

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


double ClassicMohrCoulomb::PlasticExtrapol  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter)
{

    cout<<"PlasticExtrapol procedure is not updated for use for the Classic Mohr Coulomb model"<<endl;
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



double ClassicMohrCoulomb::RKExtrapolation  (double A[][8], double* B,double *BRes, double *C, BBMPoint* Point, double* EPStrain, double* AbsStress
								  , BBMPoint* OldPoint, double* RelError, int* NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate)
{

cout<<"RKExtrapolation procedure does not work as intended"<<endl;
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


double ClassicMohrCoulomb::CheckNorm (double* DSigma, double DPZeroStar, BBMPoint* InitialPoint, double* DError)	//returns RError
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


double ClassicMohrCoulomb::CheckNormSloan (double* DSigma, double DPZeroStar, BBMPoint* InitialPoint, double* DError)	//returns RError
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


double FinalShearMin, FinalShearMax, FinalShear;
double ShearError;

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






void ClassicMohrCoulomb::ComputeG1 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G1)
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
void ClassicMohrCoulomb::ComputeG2 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G2)
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

double ClassicMohrCoulomb::GetLambda (double * deriv, double stresspq[3], double strainpq[3])
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

double ClassicMohrCoulomb::Getk ()
{
	return 0;// k;
}
double ClassicMohrCoulomb::GetLambdaZero ()
{
	return 0;// LambdaZero;
}
double ClassicMohrCoulomb::Getr ()
{
return 0;//r;
}
double ClassicMohrCoulomb::GetBeta ()
{
	return 0;// Beta;
}
double ClassicMohrCoulomb::GetKappaP ()
{
	return 0;// KappaP;
}

double ClassicMohrCoulomb::Getpc ()
{
	return 0;//pc;
}