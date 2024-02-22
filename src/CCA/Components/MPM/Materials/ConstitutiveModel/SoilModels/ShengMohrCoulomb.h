#pragma once
#include <CCA/Components/MPM/Materials/ConstitutiveModel/SoilModels/BBMMatrix.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/SoilModels/BBMPoint.h>


/* this Mohr-Coulomb like model
 use a rounded Mohr-Coulomb surface, see
eq 13 in Sheng D, Sloan SW & Yu HS
Computations Mechanics 26:185-196 (2000)
Springer
*/

class ShengMohrCoulomb
{

private:
        // model parameters
        //elastic parameters
        double G;               //shear modulus
        double K;               //bulk modulus
        double E;               //Young modulus
        double Poisson; //Poisson ratio

        //Mohr - Coulomb parameters

        double Cohesion;
        double Phi;
        double SinPhi;
        double CosPhi;

        //if the flow rule is not associated; not expected...
        bool NonAssociated;
        double Psi;
        double SinPsi;
        double CosPsi;


        //Rounded Mohr-Coulomb parameters
        double Alpha;   //For the M parameter
        double Alpha4; //Alpha^4;

        int CalcPlastic (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue,double *dS, double* dLambda);
        int CalcPlasticPQ (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue,double *dS, double* dLambda);
        int CalcPlasticFaster (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* PlasticStrain, double* DPZeroStar, double FValue,double *dS, double* dLambda);

public:

        ShengMohrCoulomb(void);
        ~ShengMohrCoulomb(void);
        void SetDefaultIntegrationParameters ();
        void SetModelParameters (double ShearModulusG, double BulkModulusK, double CohesionC, double FrictionAnglePhi);
        void SetIntegrationParameters (double IntegrationTolerance,int SolutionAlgorithm, int ToleranceMethod,
                                                        int DriftCorrection, double BetaFactor, double YieldLocTolerance, int MaxIterPegasus);
        void Integrate (double* StrainIncrement,BBMPoint* InitialPoint);
        void Integrate (double* StrainIncrement, double SuctionIncrement, BBMPoint* InitialPoint, double * StressIncrement,
                double P0StarIncrement, double* PlasticStrainIncrement);
        void IntegrateConst (double* StrainIncrement,BBMPoint* InitialPoint, int StepNo, int Method);
        double CalcStressElast (double nu0, double* s0  , double* eps0, double* deps,  double* ds);
        double CalcElastic (double * Strain, BBMPoint * InitialPoint, BBMPoint * FinalPoint);
        void CalcStressElastM (double* deps,  double* ds);
        void FindElStrGradPQ (double nu0, double* s0  , double* eps0, double* deps,  double* ds);
        bool CheckGradient (BBMPoint * InitialPoint, BBMPoint * FinalPoint);
        void FindElStrGrad (double nu0, double* s0  , double* eps0, double* deps,  double* ds);
        double CalculatePZero (BBMPoint * Point);
        bool CheckYield (BBMPoint * Point);
        bool CheckYieldNormalised (BBMPoint * Point);
        void CheckYield (double *state, double* s, double suction, double *FValue);
        void CheckYieldNormalised (double *state, double* s, double suction, double *FValue);
        bool CheckIfPlastic (BBMPoint * Point);
        double ComputeYieldFunction (BBMPoint *Point);
        double ComputeYieldFunctionNN(BBMPoint * Point);
        //bool CheckYield (double *state, double* s, double suction);
        void FindYieldOriginal (double *state, double*s0, double* eps0, double* deps, double *a);
        // double FindYieldOriginal (double *state, double*s0, double* eps0, double* deps);
        void FindYieldModified (double *state, double*s0, double* eps0, double* deps, double *a);
        double ComputeNu (double* s, double* state, double suction);
        double FindGradient (double * state, double * s, double *ds, double * dF, double suction, double dsuction);
        double FindGradientPQ (BBMPoint * Point, double *ds, double * dF,  double dsuction);
        void MoveYieldaBit (double * state, double * s, double *ds, double * eps0, double * deps, double * gradient, double F0);
        void FindYieldYield (double *state, double*s0, double* eps0, double* deps, double *a);
        void FindIntersectionUnloading (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElastic, double * PurelyPlastic);
        void FindIntersection (double * StrainIncrement, BBMPoint * InitialPoint, double * PurelyElasticStrain, double * PurelyPlasticStrain);
        void PaintLocus (double *state, double suction, int Max);
        void ComputeG1 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G1);
        void ComputeG2 (BBMPoint * InitialPoint,int RetentionModel, double * RetentionParameters, double * G2);


        double Getk ();
        double GetLambdaZero ();
        double Getr ();
        double GetBeta ();
        double GetKappaP ();
        double Getpc ();
        void read();
        void write();

// *********************************** Plastic Procedures below ******************************************************

        void GetTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP);
        void CalculateElastoPlasticTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP);
        void CalculateElasticTangentMatrixPQ (BBMPoint * Point, BBMMatrix* DEP);
        void GetTangentMatrix (BBMPoint * Point, BBMMatrix* DEP);
        void CalculateElastoPlasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP);
        void CalculateElasticTangentMatrix (BBMPoint * Point, BBMMatrix* DEP);
        void GetDerivative (double MeanStress, double ShearStress, double suction,double PZero,double * state, double* deriv);
        double GetLambda (double * deriv, double stresspq[3], double strainpq[3]);
        double PlasticEuler (BBMPoint* Point, double* EPStrain, double* AbsStress, int NumberIterations);       //returns elapsed time of computations
        double RungeKutta (double  A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter,
                                                         double MethodOrder, int MethodSteps, bool ErrorEstimate);
        double RungeKuttaEqualStep (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
                                                                          double* RelError, int NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate);
        double RungeKuttaExtrapol (double A[][8], double* B,double *BRes, double *C,  BBMPoint* Point, double* EPStrain, double* AbsStress,
                                                                          int*NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate);
//Runge Kutta schemes
        double CalculatePlastic (double * PurelyPlasticStrain, BBMPoint* Point);
        double CalculatePlasticConst (double * PurelyPlasticStrain, BBMPoint* Point, int StepNo);
        double PlasticRKErr8544 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //Bogacki - Shimpine
        double PlasticRKDP754 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);  //Dormand Prince
        double PlasticRKCK654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //Cash - Karp
        double PlasticRKEng654 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //England as given by Sloan
        double PlasticRK543 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //4th order with 3rd ord estimate
        double PlasticRK332 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //3rd order R-K scheme
        double PlasticRKBog432 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //Bogacki - Shimpine 3rd order Runge Kutta scheme
        double PlasticRKME221 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //Modified Euler
        double PlasticRKNoExTry (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter); //using not in an extrapolation way
//Extrapolation Schemes
        double PlasticExtrapol  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        double RKExtrapolation  (double A[][8], double* B,double *BRes, double *C, BBMPoint* Point, double* EPStrain, double* AbsStress
                                                                  , BBMPoint* OldPoint, double* RelError, int* NumberIter, double MethodOrder, int MethodSteps, bool ErrorEstimate);
        double PlasticMidpoint (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        double PlasticMidpointGallipoli (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        double CheckNorm (double* DSigma, double DPZeroStar,BBMPoint* InitialPoint, double* DError);    //returns RError
        double CheckNormSloan (double* DSigma, double DPZeroStar,BBMPoint* InitialPoint, double* DError);       //returns RError
        void CorrectDrift (BBMPoint* Point);
        void CorrectDriftBeg (BBMPoint* EndPoint, BBMPoint *PointOld);

//Used Procedures before, not updated anymore, though, mostly, working.
        //void DriftCorrect (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* Lambda, double* DPZeroStar, double* FValue);
        //void CorrectDriftBeg (BBMPoint Point, double* EPStrain, BBMMatrix* DSIGMA, double* Lambda, double* DPZeroStar, double* FValue);
        //double Plastic (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);               //returns elapsed time of computations
        //double PlasticNewSlow (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIterations);  //returns elapsed time of computations
        //double PlasticRKErr6 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIterations);   //returns elapsed time of computations
        //double PlasticRKErr75 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);

        //double PlasticRK5Err4_2 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRK4Err3  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRK4Err3v2  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRK3Err2  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRK3Err2v2  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRK3Err2v3  (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticRKSloan (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIterations);  //returns elapsed time of computations
        //double PlasticMidpointC (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticMidpointCN (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticMidpointC4 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);
        //double PlasticMidpointC6 (BBMPoint* Point, double* EPStrain, double* AbsStress, int* NumberIter);

};
