/*
 *  DigitalFilterInlet.cc
 *  \author Alex Abboud
 *  \date August 2012
 *  \brief generates a turbulent inlet that is derived from a digital filter method
 *  an input file to this functions is needed which prescribes the basic geometry,
 *  mean velocities, integral length scales, Reynold's stresses
 *  refer to StandAlone/inputs/ARCHES/DigitalFilterInlet/ChannelInletGenerator.txt
 *  or User Guide for specific layout
 */


#include <fstream>
#include <Core/IO/UintahZlibUtil.h>
#include <zlib.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/BoundaryConditions/RectangleBCData.h>
#include <Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Core/Grid/BoundaryConditions/AnnulusBCData.h>
#include <Core/Grid/BoundaryConditions/EllipseBCData.h>


#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif


using namespace std;
using namespace SCIRun;
using namespace Uintah;


/***********************************************************************
 Filter Coefficient Functions
 ************************************************************************/
double bTilde ( int k, double n ) {
  double b;
  b = exp( - PI / 2.0 * k * k / n / n);
  return(b);
}

//function for summation portion for each b_i
double BSum ( int k, double n, int N ) {
  double SumTerm = 0.0;
  for (int j = -N; j<N; j++) {
    SumTerm	= SumTerm + bTilde(j, n);
  }
  SumTerm = sqrt(SumTerm);
  double BK = bTilde(k, n)/SumTerm;
  return(BK);
}


/***********************************************************************
 Wall Distance functions
 ************************************************************************/
//__________________________________________
// Circle
vector<vector <double> > setCircleD ( int jSize, int kSize, string faceSide, double charDim, 
                                     vector<double> origin, vector<double> Dx, vector<int> minCell,
                                     vector<double> gridLoPts )
{
  vector<vector <double> > distProf  (jSize, vector<double> (kSize) );
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      
      if (faceSide=="x-") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                        ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      } else if (faceSide=="x+") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                        ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) ); 

      } else if (faceSide=="y-") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                        ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      } else if (faceSide=="y+") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                        ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );;
        
      } else if (faceSide=="z-") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                        ((minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1]) ); 
      } else if (faceSide=="z+") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                        ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      }
      distProf[j][k] = distProf[j][k]/charDim*2.0; //normalize distance
    }
  }
  
  return distProf;
}

//__________________________________________
// Annulus
vector< vector< double> > setAnnulusD ( int jSize, int kSize, string faceSide, double minor_r, double major_r,
                                       vector<double> origin, vector<double> Dx, vector<int> minCell, vector<double> gridLoPts)
{
  vector<vector <double> > distProf  (jSize, vector<double> (kSize) );
  double middle_radius = (minor_r + major_r) * 0.5; //get center fo annulus channel
  double charDim = middle_radius - minor_r;
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      //get distance from origin at each pt
      if (faceSide=="x-") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                              ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      } else if (faceSide=="x+") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                              ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) ); 
        
      } else if (faceSide=="y-") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                              ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      } else if (faceSide=="y+") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                              ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );;
        
      } else if (faceSide=="z-") {
        distProf[j][k] = sqrt( ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) * ((minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0]) +
                              ((minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1]) ); 
      } else if (faceSide=="z+") {
        distProf[j][k] = sqrt( ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) * ((minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1]) +
                              ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) * ((minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2]) );
      }
      
      distProf[j][k] = (distProf[j][k] - middle_radius)/charDim; //normalize distance
    }
  }
  
  return distProf;
  
}

//__________________________________________
// Ellipse
vector<vector <double> > setEllipseD ( int jSize, int kSize, string faceSide, double minor_r, double major_radius,
                                      double angle, vector<double> origin, vector<double> Dx, vector<int> minCell,
                                      vector<double> gridLoPts, vector<IntVector> ptList, vector<int> resolution )
{
  vector<vector <double> > distProf  (jSize, vector<double> (kSize) );

  double x1, x2;
  vector<double> rotation (4);
  double r1, r2;
  angle = angle*PI/180.0;
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      
      if (faceSide=="x-") {
        x1 =  (minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1] ;//ydist
        x2 =  (minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2] ;//zdist
        //r11 r12 r21 r22
        rotation[0] = cos(angle);
        rotation[1] = sin(angle);
        rotation[2] = -sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
        
      } else if (faceSide=="x+") {
        x1 =  (minCell[1]+j+0.5)*Dx[1] +gridLoPts[1] - origin[1] ;//ydist
        x2 =  (minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2] ;//zdist
        rotation[0] = cos(angle);
        rotation[1] = -sin(angle);
        rotation[2] = sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
        
      } else if (faceSide=="y-") {
        x1 =  (minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2] ;//zdist
        x2 =  (minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0] ;//xdist
        rotation[0] = cos(angle);
        rotation[1] = sin(angle);
        rotation[2] = -sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
        
      } else if (faceSide=="y+") {
        x1 =  (minCell[2]+k+0.5)*Dx[2] +gridLoPts[2] - origin[2] ;//zdist
        x2 =  (minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0] ;//xdist
        
        rotation[0] = cos(angle);
        rotation[1] = -sin(angle);
        rotation[2] = sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
        
      } else if (faceSide=="z-") {
        x1 =  (minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0] ;//xdist
        x2 =  (minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1] ;//ydist
        
        rotation[0] = cos(angle);
        rotation[1] = sin(angle);
        rotation[2] = -sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
      } else if (faceSide=="z+") {
        x1 =  (minCell[0]+j+0.5)*Dx[0] +gridLoPts[0] - origin[0] ;//xdist
        x2 =  (minCell[1]+k+0.5)*Dx[1] +gridLoPts[1] - origin[1] ;//ydist
        
        rotation[0] = cos(angle);
        rotation[1] = -sin(angle);
        rotation[2] = sin(angle);
        rotation[3] = cos(angle);
        r1 = x1*rotation[0] + x2*rotation[2];
        r2 = x1*rotation[1] + x2*rotation[3];
      }
      distProf[j][k] = sqrt(  r1/major_radius*r1/major_radius + r2/minor_r*r2/minor_r ) ; 
      
    }
  }
  
  return distProf;
}

//__________________________________________
// Box
vector <vector <double > > setPlaneD( int jSize, int kSize, string faceSide, double charDim,
                                      vector<double> origin, vector<double> Dx, vector<int> minCell,
                                      vector<double> gridLoPts, string varyDim)
{
  vector<vector <double> >  distProf (jSize, vector<double> (kSize) );
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      if (varyDim == "x") {
        distProf[j][k] = abs( (minCell[0] + j+ 0.5)*Dx[0] + gridLoPts[0] - origin[0] );
      } else if (varyDim == "y") {
        if (faceSide == "x-" || faceSide == "x+" ) {
          distProf[j][k] = abs( (minCell[1] + j+ 0.5)*Dx[1] + gridLoPts[1] - origin[1] );
        } else {
          distProf[j][k] = abs( (minCell[1] + k+ 0.5)*Dx[1] + gridLoPts[1] - origin[1] );
        }
      } else if (varyDim == "z") {
        distProf[j][k] = abs( (minCell[2] + k+ 0.5)*Dx[2] + gridLoPts[2] - origin[2] );
      } else {
        cout << "Enter Valid Variation Direction (x/y/z) " << endl;
        exit(1);
      }
      
      distProf[j][k] = distProf[j][k]/charDim*2.0; //use center as char distance, ratehr than whole channel width
    }
  }

  return distProf;
}


/***********************************************************************
 Velocity profile functions
 ************************************************************************/

//__________________________________________
// Constant V
vector<vector <vector <double > > > setConstV( double magVelocity, int jSize, int kSize, string faceSide)
{
  vector< vector<vector <double> > > vProf (jSize, vector<vector<double> > (kSize, vector<double> (3, 0.0) ) );
  
  for (int j = 0; j<jSize; j++) {
    for (int k = 0; k<kSize; k++ ) {
      if (faceSide=="x-") {
        vProf[j][k][0] = magVelocity;
      } else if (faceSide =="x+" ) {
        vProf[j][k][0] = -magVelocity;
      } else if (faceSide=="y-") {
        vProf[j][k][1] = magVelocity;
      } else if (faceSide=="y+") {
        vProf[j][k][1] = -magVelocity;
      } else if (faceSide=="z-") {
        vProf[j][k][2] = magVelocity;
      } else if (faceSide=="z+") {
        vProf[j][k][2] = -magVelocity;
      }
    }
  }
  
  return vProf;
}

//__________________________________________
//Laminar V
vector<vector <vector <double > > > setLaminarV( double magVelocity, int jSize, int kSize, string faceSide, vector<vector<double> > distProf)
{
  vector< vector<vector <double> > > vProf (jSize, vector<vector<double> > (kSize, vector<double> (3) ) );
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      if (faceSide == "x-" ) {
        vProf[j][k][0] = magVelocity* (1 - distProf[j][k]*distProf[j][k]);
      } else if (faceSide == "x+") {
        vProf[j][k][0] = -( magVelocity* (1 - distProf[j][k]*distProf[j][k]) );
      } else if (faceSide == "y-") {
        vProf[j][k][1] =  magVelocity* (1 - distProf[j][k]*distProf[j][k]);
      } else if (faceSide == "y+") {
        vProf[j][k][1] = -(  magVelocity* (1 - distProf[j][k]*distProf[j][k]) );
      } else if (faceSide == "z-") {
        vProf[j][k][2] =  magVelocity* (1 - distProf[j][k]*distProf[j][k] );
      } else if (faceSide == "z+") {
        vProf[j][k][2] = -(  magVelocity* (1 - distProf[j][k] * distProf[j][k] ) );
      }
      
    }
  }
  
  return vProf;
}

//__________________________________________
//Tanh V
vector<vector <vector <double > > > setTanhV( double magVelocity, int jSize, int kSize, string faceSide, vector<vector<double> > distProf)
{
  vector< vector<vector <double> > > vProf (jSize, vector<vector<double> > (kSize, vector<double> (3) ) );
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      if (faceSide == "x-" ) {
        vProf[j][k][0] = magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) );
      } else if (faceSide == "x+") {
        vProf[j][k][0] = -(magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) ) );
      } else if (faceSide == "y-") {
        vProf[j][k][1] = magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) );
      } else if (faceSide == "y+") {
        vProf[j][k][1] = -( magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) ) );
      } else if (faceSide == "z-") {
        vProf[j][k][2] = magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) );
      } else if (faceSide == "z+") {
        vProf[j][k][2] = -( magVelocity* 0.5 + magVelocity* 0.5 * tanh(10.0 * (1.0-distProf[j][k]) ) );
      }
      
    }
  }
  
  return vProf;
}

//__________________________________________
//power law V
vector<vector <vector <double > > > setPowerV( double magVelocity, int jSize, int kSize, string faceSide, vector<vector<double> > distProf)
{
  vector< vector<vector <double> > > vProf (jSize, vector<vector<double> > (kSize, vector<double> (3) ) );
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      if (faceSide == "x-" ) {
        vProf[j][k][0] = magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 );
      } else if (faceSide == "x+") {
        vProf[j][k][0] = -( magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 ) );
      } else if (faceSide == "y-") {
        vProf[j][k][1] =  magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 );
      } else if (faceSide == "y+") {
        vProf[j][k][1] = -(  magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 ) );
      } else if (faceSide == "z-") {
        vProf[j][k][2] =  magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 );
      } else if (faceSide == "z+") {
        vProf[j][k][2] = -(  magVelocity* pow( (1 - distProf[j][k]), 1.0/7.0 ) );
      }
      
    }
  }
  
  return vProf;
}

/************************************************************************
 Reynold's Stress Profile functions
 *************************************************************************/ 

//__________________________________________
vector< vector <vector <double> > > setConstAij( vector<vector<double> > Aij, int jSize, int kSize)
{
  vector< vector< vector<double> > > AijProf (jSize, vector<vector<double> > (kSize, vector<double> (6)));
  
  for (int j = 0; j<jSize; j++) {
    for (int k =0; k<kSize; k++) {
      AijProf[j][k][0] = Aij[0][0];
      AijProf[j][k][1] = Aij[1][0];
      AijProf[j][k][2] = Aij[1][1];
      AijProf[j][k][3] = Aij[2][0];
      AijProf[j][k][4] = Aij[2][1];
      AijProf[j][k][5] = Aij[2][2];
    }
  }
  return AijProf;
}

//_____________________________
// Jet stress profile - as open mixing region with quasi parabolic centered around shear layer
vector< vector <vector <double> > > setJetAij( double magVelocity, int jSize, int kSize, const string faceSide, vector<vector<double> > distProf)
{
  vector< vector< vector<double> > > AijProf (jSize, vector<vector<double> > (kSize, vector<double> (6)));
  vector<double> Rij (6);

  //make this piecewise function
  double USq = magVelocity * magVelocity;
  
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      
      if (faceSide=="x-" || faceSide=="x+") {
        Rij[0] = (- 0.03 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.035) * USq;  //uu
        Rij[1] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //uv
        Rij[2] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //vv
        Rij[3] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //uw
        Rij[4] = (- 0.001 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.001) * USq; //vw
        Rij[5] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //ww
        
      } else if (faceSide=="y-" || faceSide=="y+") {
        Rij[0] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //uu
        Rij[1] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //uv
        Rij[2] = (- 0.03 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.035) * USq;  //vv
        Rij[3] = (- 0.001 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.001) * USq; //uw
        Rij[4] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //vw
        Rij[5] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //ww
        
      } else if (faceSide=="z-" || faceSide=="z+") {
        Rij[0] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //uu
        Rij[1] = (- 0.001 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.001) * USq; //uv
        Rij[2] = (- 0.015 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.016) * USq; //vv
        Rij[3] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //uw
        Rij[4] = (- 0.01 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.011) * USq;  //vw
        Rij[5] = (- 0.03 * (1.0 - distProf[j][k] ) * (1.0 - distProf[j][k] ) + 0.035) * USq;  //ww
      }
      
      //calc Lund transform matrix at this point
      AijProf[j][k][0] = sqrt(Rij[0]);
      AijProf[j][k][1] = Rij[1]/AijProf[j][k][0];
      AijProf[j][k][2] = sqrt(Rij[2]-AijProf[j][k][1]*AijProf[j][k][1]);
      AijProf[j][k][3] = Rij[3]/AijProf[j][k][0];
      AijProf[j][k][4] = (Rij[4] - AijProf[j][k][1] * AijProf[j][k][3]) / AijProf[j][k][2];
      AijProf[j][k][5] = sqrt(Rij[5] - AijProf[j][k][3] *AijProf[j][k][3] - AijProf[j][k][4] * AijProf[j][k][4] );
      
    }
  }  
  return AijProf;
}

//_____________________________
// set channel stress correlations, piecewise functions to try to match 
vector< vector <vector <double> > > setChannelAij( double magVelocity, int jSize, int kSize, const string faceSide, vector<vector<double> > distProf)
{
  vector< vector< vector<double> > > AijProf (jSize, vector<vector<double> > (kSize, vector<double> (6)));
  vector<double> Rij (6);
  
  double USq = magVelocity * magVelocity;
  double wallDist;
  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      wallDist = 1.0 - distProf[j][k];
      if (faceSide=="x-" || faceSide=="x+") {
     
        if ( wallDist <= 0.1) {
          Rij[0] = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;   //uu  
          Rij[1] = 0.02 * wallDist * USq; //uv
          Rij[2] = 0.6*wallDist*USq;      //vv
          Rij[3] = 0.02 * wallDist * USq; //uw
          Rij[4] = 0.001*USq;             //vw
          Rij[5] = 0.6*wallDist*USq;      //ww
        } else if ( wallDist <= 0.3 ) {
          Rij[0] = 0.014 / sqrt(wallDist);               //uu
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq;  //uv
          Rij[2] = (-0.0444*wallDist + 0.0644) *USq;     //vv
          Rij[3] = (-0.00225*wallDist + .00225 ) * USq;  //uw
          Rij[4] = 0.001*USq;                            //vw
          Rij[5] = (-0.0444*wallDist + 0.0644) *USq;     //ww
        } else { //bulk
          Rij[0] = 0.02*USq;                            //uu
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq; //uv
          Rij[2] = 0.02*USq;                            //vv
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq; //uw
          Rij[4] = 0.001*USq;                           //vw
          Rij[5] = 0.02*USq;                            //ww
        }
        
      } else if (faceSide=="y-" || faceSide=="y+") {

        if ( wallDist <= 0.1) {
          Rij[0] = 0.6*wallDist*USq;      //uu  
          Rij[1] = 0.02 * wallDist * USq; //uv
          Rij[2] = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;      //vv
          Rij[3] = 0.001*USq;             //uw
          Rij[4] = 0.02 * wallDist * USq; //vw
          Rij[5] = 0.6*wallDist*USq;      //ww
        } else if ( wallDist <= 0.3 ) {
          Rij[0] = (-0.0444*wallDist + 0.0644) *USq;     //uu
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq;  //uv
          Rij[2] = 0.014 / sqrt(wallDist);               //vv
          Rij[3] = 0.001*USq;                            //uw
          Rij[4] = (-0.00225*wallDist + .00225 ) * USq;  //vw
          Rij[5] = (-0.0444*wallDist + 0.0644) *USq;     //ww
        } else { //bulk
          Rij[0] = 0.02*USq;                            //uu
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq; //uv
          Rij[2] = 0.02*USq;                            //vv
          Rij[1] = 0.001*USq;                           //uw
          Rij[4] = (-0.00225*wallDist + .00225 ) * USq; //vw
          Rij[5] = 0.02*USq;                            //ww
        }
        
      } else if (faceSide=="z-" || faceSide=="z+") {

        if ( wallDist <= 0.1) {
          Rij[0] = 0.6*wallDist*USq;      //uu  
          Rij[1] = 0.001*USq;             //uv
          Rij[2] = 0.6*wallDist*USq;      //vv
          Rij[3] = 0.02 * wallDist * USq; //uw
          Rij[4] = 0.02 * wallDist * USq; //vw
          Rij[5] = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;      //ww
        } else if ( wallDist <= 0.3 ) {
          Rij[0] = (-0.0444*wallDist + 0.0644) *USq;     //uu
          Rij[1] = 0.001*USq;                            //uv
          Rij[2] = (-0.0444*wallDist + 0.0644) *USq;     //vv
          Rij[3] = (-0.00225*wallDist + .00225 ) * USq;  //uw
          Rij[4] = (-0.00225*wallDist + .00225 ) * USq;  //vw
          Rij[5] = 0.014 / sqrt(wallDist);               //ww
        } else { //bulk
          Rij[0] = 0.02*USq;                            //uu
          Rij[1] = 0.001*USq;                           //uv
          Rij[2] = 0.02*USq;                            //vv
          Rij[1] = (-0.00225*wallDist + .00225 ) * USq; //uw
          Rij[4] = (-0.00225*wallDist + .00225 ) * USq; //vw
          Rij[5] = 0.02*USq;                            //ww
        }
      }
      
      //calc Lund xform matrix at this point
      AijProf[j][k][0] = sqrt(Rij[0]);
      AijProf[j][k][1] = Rij[1]/AijProf[j][k][0];
      AijProf[j][k][2] = sqrt(Rij[2]-AijProf[j][k][1]*AijProf[j][k][1]);
      AijProf[j][k][3] = Rij[3]/AijProf[j][k][0];
      AijProf[j][k][4] = (Rij[4] - AijProf[j][k][1] * AijProf[j][k][3]) / AijProf[j][k][2];
      AijProf[j][k][5] = sqrt(Rij[5] - AijProf[j][k][3] *AijProf[j][k][3] - AijProf[j][k][4] * AijProf[j][k][4] );
      
    }
  }  
  return AijProf;
}

/************************************************************************
 Length Scale Profile Fucntions
 *************************************************************************/ 

//__________________________________________
vector< vector <vector <double> > > setConstL( vector<double> L_a, int jSize, int kSize)
{
  vector< vector <vector <double> > > LProf (jSize, vector<vector<double> > (kSize, vector<double> (3) ) );
  
  for (int j = 0; j<jSize; j++) {
    for (int k = 0; k<kSize; k++ ) {
      LProf[j][k][0] = L_a[0];
      LProf[j][k][1] = L_a[1];
      LProf[j][k][2] = L_a[2];
    }
  }
  return LProf;
}


//__________________________________________
vector< vector <vector <double> > > setChannelL( vector<double> L_a, int jSize, int kSize, string faceSide, vector<vector<double> > distProf)
{
  vector< vector <vector <double> > > LProf (jSize, vector<vector<double> > (kSize, vector<double> (3) ) );

  for (int j = 0; j< jSize; j++) {
    for (int k = 0; k<kSize; k++) {

      if (distProf[j][k] > 0.9) {
        LProf[j][k][0] = L_a[0]/2.0;
        LProf[j][k][1] = L_a[1]/2.0;
        LProf[j][k][2] = L_a[2]/2.0;
      } else {
        LProf[j][k][0] = L_a[0];
        LProf[j][k][1] = L_a[1];
        LProf[j][k][2] = L_a[2];
      }

    }
  }
  
  return LProf;
}

/************************************************************************
 Fucntions for actually writing the input file for ARCHES
 *************************************************************************/ 

//function for homogenous turbulence
void spatiallyConstant( const string outputfile,
                       const string faceSide,
                       vector < vector <vector<double> > > randomFieldx,
                       vector < vector <vector<double> > > randomFieldy,
                       vector < vector <vector<double> > > randomFieldz,
                       vector < vector <vector<double> > > filterCoefficients,
                       vector < vector <double > > Aij,
                       vector <int> filterSize,
                       vector <double> aveU,
                       int NT, int jSize, int kSize,
                       bool angleVelocity, 
                       vector<vector<double> > rotationMatrix,
                       vector<int> minCell)
{
  vector <vector<double> > uBasex(jSize, vector<double>(kSize));
  vector <vector<double> > uBasey(jSize, vector<double>(kSize));
  vector <vector<double> > uBasez(jSize, vector<double>(kSize));
  
  vector <vector <vector<double> > > uFluctx( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));
  vector <vector <vector<double> > > uFlucty( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));
  vector <vector <vector<double> > > uFluctz( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));
  
  cout << "calculating fluctuations" << endl;
  
  //fill in full fluctuations
  for (int t = 0; t<NT+1; t++) {
    //each timestep
    cout << "Writing TimeStep: "<< t << endl;
    for (int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++) {
        uBasex[j][k] = 0.0;
        uBasey[j][k] = 0.0;
        uBasez[j][k] = 0.0;
        
        uFluctx[t][j][k] = 0.0;    
        uFlucty[t][j][k] = 0.0;
        uFluctz[t][j][k] = 0.0;
        
        for (int iprime = 0; iprime<filterSize[0]; iprime++) {
          for (int jprime = 0; jprime<filterSize[1]; jprime++) {
            for (int kprime = 0; kprime<filterSize[2]; kprime++) {
              
              if (faceSide == "x-" || faceSide=="x+") {
                uBasex[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldx[iprime+t][j+jprime][k+kprime];
                uBasey[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldy[iprime+t][j+jprime][k+kprime];
                uBasez[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldz[iprime+t][j+jprime][k+kprime];
              } else if (faceSide=="y-" || faceSide=="y+") {
                uBasex[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldx[iprime+j][jprime+t][k+kprime];
                uBasey[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldy[iprime+j][jprime+t][k+kprime];
                uBasez[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldz[iprime+j][jprime+t][k+kprime];
              } else if (faceSide=="z-" || faceSide=="z+") {
                uBasex[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldx[iprime+j][k+jprime][kprime+t];
                uBasey[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldy[iprime+j][k+jprime][kprime+t];
                uBasez[j][k] += filterCoefficients[iprime][jprime][kprime] * randomFieldz[iprime+j][k+jprime][kprime+t];
              }
              
            }
          }
        } //end iprime
        
        if (!angleVelocity) {
          uFluctx[t][j][k] = aveU[0] + Aij[0][0] * uBasex[j][k];
          uFlucty[t][j][k] = aveU[1] + Aij[1][0] * uBasex[j][k] + Aij[1][1]*uBasey[j][k];
          uFluctz[t][j][k] = aveU[2] + Aij[2][0] * uBasex[j][k] + Aij[2][1]*uBasey[j][k] + Aij[2][2]*uBasez[j][k];
        } else {
          double tempX, tempY, tempZ;
          tempX = Aij[0][0] * uBasex[j][k] ;
          tempY = Aij[1][0] * uBasex[j][k] + Aij[1][1]*uBasey[j][k];
          tempZ = Aij[2][0] * uBasex[j][k] + Aij[2][1]*uBasey[j][k] + Aij[2][2]*uBasez[j][k];
          
          uFluctx[t][j][k] = aveU[0] + rotationMatrix[0][0] * tempX + rotationMatrix[1][0]*tempY + rotationMatrix[2][0]*tempZ;
          uFlucty[t][j][k] = aveU[1] + rotationMatrix[0][1] * tempX + rotationMatrix[1][1]*tempY + rotationMatrix[2][1]*tempZ;
          uFluctz[t][j][k] = aveU[2] + rotationMatrix[0][2] * tempX + rotationMatrix[1][2]*tempY + rotationMatrix[2][2]*tempZ;
        }
        
        //prevent negative flow on velocity normal to the inlet
        if ( faceSide=="x-" && uFluctx[t][j][k] < 0.0) {
          uFluctx[t][j][k] = 0.0;
        } else if (faceSide=="x+" && uFluctx[t][j][k] > 0.0 ) {
          uFluctx[t][j][k] = 0.0;
        } else if ( faceSide=="y-"  && uFlucty[t][j][k] < 0.0) {
          uFlucty[t][j][k] = 0.0;
        } else if ( faceSide=="y+" && uFlucty[t][j][k] > 0.0 ) {
          uFlucty[t][j][k] = 0.0;
        } else if ( faceSide=="z-"  && uFluctz[t][j][k] < 0.0) {
          uFluctz[t][j][k] = 0.0;
        } else if ( faceSide=="z+" && uFluctz[t][j][k] > 0.0 ) {
          uFluctz[t][j][k] = 0.0; 
        }

      }
    } //end j
  }
  
  //open file to write out
  ofstream myfile; 
  myfile.open( outputfile.c_str(), ios::out );
  myfile << "#Table Dimensions" << endl;
  myfile << NT+1 << " " << jSize << " " << kSize << endl; 
  myfile << "#Minimum Cell Index" << endl;
  myfile << minCell[0] << " " << minCell[1] << " " << minCell[2] << endl;
  myfile.precision(15);
  for (int t=0; t<NT+1; t++) {
    for (int j=0; j<jSize; j++) {
      for (int k=0; k<kSize; k++) {
        myfile << t << " " << j << " " << k << "  "; 
        myfile << uFluctx[t][j][k] << "  " << uFlucty[t][j][k] << "  " << uFluctz[t][j][k] << "\n";
      }
    }
  }
  myfile.close();
  cout << "File written and saved as: " << outputfile << endl;
}

//__________________________________________
//function for spatial variatinos in the flow
void spatiallyVariable (const string outputfile,
                        const string faceSide,
                        vector < vector <vector<double> > > randomFieldx,
                        vector < vector <vector<double> > > randomFieldy,
                        vector < vector <vector<double> > > randomFieldz,
                        vector<vector <vector <double> > > velocityProfile,
                        vector<vector <vector <double> > > AijProf,
                        vector<vector <vector <double> > > lengthScaleProfile,
                        int NT, int jSize, int kSize,
                        bool angleVelocity,
                        vector<vector <double> > rotationMatrix,
                        vector<double> Dx,
                        double AccuracyCoef,
                        vector<int> minCell) 
{
  cout << "Calculating Fluctuations" << endl;
  //fidn the filter support for each length scale at each point
  vector< vector<vector <double> > > n_a (jSize, vector<vector<double> > (kSize, vector<double> (3,0) ) );
  vector< vector<vector <int> > > N_a (jSize, vector<vector<int> > (kSize, vector<int> (3,0) ) );
  vector< vector<vector <int> > > filterSize (jSize, vector<vector<int> > (kSize, vector<int> (3,0) ) );
  vector<int> maxFilterSize (3,0);
  
  //set up lengths of filter coefficients
  for ( int j = 0; j<jSize; j++) {
    for (int k = 0; k<kSize; k++ ) {
      for (int i =0; i<3; i++) {
        n_a[j][k][i] = lengthScaleProfile[j][k][i]/Dx[i];
        N_a[j][k][i] = (int) ceil(n_a[j][k][i] * AccuracyCoef);
        filterSize[j][k][i] = 2*N_a[j][k][i] + 1;
        
        if (filterSize[j][k][i] > maxFilterSize[i]) {
          maxFilterSize[i] = filterSize[j][k][i];
        }
      }
    }
  }
  //NOTE: need the length scales from the input file to be the maximum over whole domain
  vector< vector<vector<vector<vector <double> > > > > filterCoefficients ( jSize, vector<vector<vector<vector<double> > > > 
                                                                           (kSize, vector<vector<vector<double> > >
                                                                            (maxFilterSize[0], vector<vector<double> >
                                                                             (maxFilterSize[1], vector<double>
                                                                              (maxFilterSize[2]) ) ) ) ) ;


  for (int j = 0; j<jSize; j++) {
    for (int k = 0; k<kSize; k++) {
      //calc filter coeff at each pt
      for (int iprime = 0; iprime < filterSize[j][k][0]; iprime++) {
        for (int jprime = 0; jprime < filterSize[j][k][1]; jprime++) {
          for (int kprime = 0; kprime < filterSize[j][k][2]; kprime++) {
            filterCoefficients[j][k][iprime][jprime][kprime] = BSum(iprime, n_a[j][k][0], N_a[j][k][0]) *
                                                               BSum(jprime, n_a[j][k][1], N_a[j][k][1]) *
                                                               BSum(kprime, n_a[j][k][2], N_a[j][k][2]);
          }
        }
      } //end iprime
    }
    cout << "Filter Coefficeint done for j = " << j << endl;
  }
                                                                          
  vector <vector<double> > uBasex(jSize, vector<double>(kSize));
  vector <vector<double> > uBasey(jSize, vector<double>(kSize));
  vector <vector<double> > uBasez(jSize, vector<double>(kSize));
  
  vector <vector <vector<double> > > uFluctx( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));
  vector <vector <vector<double> > > uFlucty( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));
  vector <vector <vector<double> > > uFluctz( NT+1, vector<vector<double> >(jSize, vector<double>(kSize)));                                                                                                                             
  
  cout << "flucts" << endl;
  for (int t = 0; t<NT+1; t++) {
    //each timestep
    cout << "Writing TimeStep: "<< t << endl;
    for (int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++) {
        uBasex[j][k] = 0.0;
        uBasey[j][k] = 0.0;
        uBasez[j][k] = 0.0;
        
        uFluctx[t][j][k] = 0.0;    
        uFlucty[t][j][k] = 0.0;
        uFluctz[t][j][k] = 0.0;
        
        for (int iprime = 0; iprime<filterSize[j][k][0]; iprime++) {
          for (int jprime = 0; jprime<filterSize[j][k][1]; jprime++) {
            for (int kprime = 0; kprime<filterSize[j][k][2]; kprime++) {
              
              if (faceSide == "x-" || faceSide=="x+") {
                uBasex[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldx[iprime+t][j+jprime][k+kprime];
                uBasey[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldy[iprime+t][j+jprime][k+kprime];
                uBasez[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldz[iprime+t][j+jprime][k+kprime];
              } else if (faceSide=="y-" || faceSide=="y+") {
                uBasex[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldx[iprime+j][jprime+t][k+kprime];
                uBasey[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldy[iprime+j][jprime+t][k+kprime];
                uBasez[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldz[iprime+j][jprime+t][k+kprime];
              } else if (faceSide=="z-" || faceSide=="z+") {
                uBasex[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldx[iprime+j][k+jprime][kprime+t];
                uBasey[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldy[iprime+j][k+jprime][kprime+t];
                uBasez[j][k] += filterCoefficients[j][k][iprime][jprime][kprime] * randomFieldz[iprime+j][k+jprime][kprime+t];
              }
              
            }
          }
        } //end iprime
        
        if (!angleVelocity) {
          uFluctx[t][j][k] = velocityProfile[j][k][0] + AijProf[j][k][0] * uBasex[j][k];
          uFlucty[t][j][k] = velocityProfile[j][k][1] + AijProf[j][k][1] * uBasex[j][k] + AijProf[j][k][2]*uBasey[j][k];
          uFluctz[t][j][k] = velocityProfile[j][k][2] + AijProf[j][k][3] * uBasex[j][k] + AijProf[j][k][4]*uBasey[j][k] + AijProf[j][k][5]*uBasez[j][k];
        } else {
          double tempX, tempY, tempZ;
          tempX = velocityProfile[j][k][0] + AijProf[j][k][0] * uBasex[j][k];
          tempY = velocityProfile[j][k][1] + AijProf[j][k][1] * uBasex[j][k] + AijProf[j][k][2]*uBasey[j][k];
          tempZ = velocityProfile[j][k][2] + AijProf[j][k][3] * uBasex[j][k] + AijProf[j][k][4]*uBasey[j][k] + AijProf[j][k][5]*uBasez[j][k];
          
          uFluctx[t][j][k] = rotationMatrix[0][0] * tempX + rotationMatrix[1][0]*tempY + rotationMatrix[2][0]*tempZ;
          uFlucty[t][j][k] = rotationMatrix[0][1] * tempX + rotationMatrix[1][1]*tempY + rotationMatrix[2][1]*tempZ;
          uFluctz[t][j][k] = rotationMatrix[0][2] * tempX + rotationMatrix[1][2]*tempY + rotationMatrix[2][2]*tempZ;
        }
        
        //prevent negative flow on velocity normal to the inlet
        if ( faceSide=="x-" && uFluctx[t][j][k] < 0.0) {
          uFluctx[t][j][k] = 0.0;
        } else if (faceSide=="x+" && uFluctx[t][j][k] > 0.0 ) {
          uFluctx[t][j][k] = 0.0;
        } else if ( faceSide=="y-"  && uFlucty[t][j][k] < 0.0) {
          uFlucty[t][j][k] = 0.0;
        } else if ( faceSide=="y+" && uFlucty[t][j][k] > 0.0 ) {
          uFlucty[t][j][k] = 0.0;
        } else if ( faceSide=="z-"  && uFluctz[t][j][k] < 0.0) {
          uFluctz[t][j][k] = 0.0;
        } else if ( faceSide=="z+" && uFluctz[t][j][k] > 0.0 ) {
          uFluctz[t][j][k] = 0.0; 
        }
        
      }
    } //end j
  }
  
  //open file to write out
  ofstream myfile;
  myfile.open( outputfile.c_str(), ios::out );
  myfile << "#Table Dimensions" << endl;
  myfile << NT+1 << " " << jSize << " " << kSize << endl; 
  myfile << "#Minimum Cell Index" << endl;
  myfile << minCell[0] << " " << minCell[1] << " " << minCell[2] << endl;
  myfile.precision(15);
  for (int t=0; t<NT+1; t++) {
    for (int j=0; j<jSize; j++) {
      for (int k=0; k<kSize; k++) {
        myfile << t << " " << j << " " << k << "  "; 
        myfile << uFluctx[t][j][k] << "  " << uFlucty[t][j][k] << "  " << uFluctz[t][j][k] << "\n";
      }
    }
  }
  myfile.close();
  cout << "File written and saved as: " << outputfile << endl;
}

/************************************************************************
 MAIN Function
 *************************************************************************/ 
int main( int argc, const char* argv[] ) 
{
  string inputfile, outputfile;
  if (argc == 3) { 
    inputfile = argv[1];
    outputfile = argv[2];
  } else if (argc == 2) {
    inputfile = argv[1];
    outputfile = "DFGInlet.txt";
  } else {
    cout << "Usage is    ./DigitalFilterGenerator inputfile outputfile" << endl;
    cout << "or          ./DigitalFilterGenerator inputfile" << endl;
    cout << "with outputfile default set as DFGInlet.txt" << endl;
    exit(1);
  }
  
  gzFile gzFp = gzopen( inputfile.c_str(), "r" );
  
  if( gzFp == NULL ) {
    cout << "Error with gz in opening file: " << inputfile << endl; 
    exit(1);
  }
  
  BCGeomBase* bcGeom;
  
  //parse grid variables
  vector<int> resolution (3);
  vector<double> gridLoPts (3);
  vector<double> gridHiPts (3);
  vector<double> Dx (3);
  
  resolution[0] = getInt(gzFp); resolution[1] = getInt(gzFp); resolution[2] = getInt(gzFp);
  gridLoPts[0] = getDouble(gzFp); gridLoPts[1] = getDouble(gzFp); gridLoPts[2] = getDouble(gzFp);
  gridHiPts[0] = getDouble(gzFp); gridHiPts[1] = getDouble(gzFp); gridHiPts[2] = getDouble(gzFp);
  Dx[0] = (gridHiPts[0] - gridLoPts[0])/(double) resolution[0];
  Dx[1] = (gridHiPts[1] - gridLoPts[1])/(double) resolution[1];
  Dx[2] = (gridHiPts[2] - gridLoPts[2])/(double) resolution[2];
  
  cout << "Grid Spacing" << endl;
  cout << Dx[0] << " " << Dx[1] << " " << Dx[2] << endl;
  cout << "Period should be approx dx/dt flow direction" << endl;
  
  //Parse in values;
  string faceSide;
  faceSide = getString( gzFp );
  string faceShape;
  faceShape = getString( gzFp );
  
  
  vector<double> lower (3); vector<double> upper (3);
  vector<double> origin (3);
  double radius;
  double major_radius, minor_radius;
  double angle;
  double charDim = 0.0; //charateristic dimension needed for spatial correlations
  
  if (faceShape=="box" || faceSide=="rectangle") {
    lower[0] = getDouble( gzFp );
    lower[1] = getDouble( gzFp );
    lower[2] = getDouble( gzFp );
    
    upper[0] = getDouble( gzFp );
    upper[1] = getDouble( gzFp );
    upper[2] = getDouble( gzFp );
    
    origin[0] = (lower[0] + upper[0]) * 0.5;
    origin[1] = (lower[1] + upper[1]) * 0.5;
    origin[2] = (lower[2] + upper[2]) * 0.5;
    
    vector<double> diff (3);
    diff[0] = upper[0] - lower[0]; diff[1] = upper[1] - lower[1]; diff[2] = upper[2] - lower[2];
    //take smallest non-zero as char dimension
    charDim = diff[0];
    if (diff[0] == 0.0 ) 
      charDim = diff[1];    
    if (diff[1] < charDim && diff[1] != 0.0)
      charDim = diff[1];
    if (diff[2] < charDim && diff[2] != 0.0)
      charDim = diff[2];

    Point l(lower[0],lower[1],lower[2]),u(upper[0],upper[1],upper[2]);
    bcGeom = new RectangleBCData(l,u);
  } else if (faceShape=="circle") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    radius = getDouble(gzFp);
    charDim = 2*radius;
    Point p(origin[0], origin[1], origin[2]);
    bcGeom = new CircleBCData(p, radius);
  } else if (faceShape=="ellipse") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    major_radius = getDouble(gzFp);
    minor_radius = getDouble(gzFp);
    charDim = 2*minor_radius;
    angle = getDouble(gzFp);
    Point p(origin[0], origin[1], origin[2]);    
    bcGeom = new EllipseBCData(p, minor_radius, major_radius, faceSide, angle);
    cout << "made ellipse" <<endl;
  } else if (faceShape=="annulus") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    major_radius = getDouble(gzFp);
    minor_radius = getDouble(gzFp);  
    charDim = major_radius - minor_radius;
    Point p(origin[0], origin[1], origin[2]);
    bcGeom = new AnnulusBCData(p,minor_radius, major_radius);
  } else {
    cout << "Specify a valid BC shape" << endl;
    exit(1);
  }

  
  // create a list of int vector points for the face
  // to find the minimum cell value later
  IntVector C;
  Point pointTest;
  vector< IntVector > ptList;
  
  if ( faceSide == "x-") {
    int i = 0;
    for (int j = 0; j<resolution[1]; j++) {
      for (int k = 0; k<resolution[2]; k++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1] + gridLoPts[1], (k + 0.5) * Dx[2] + gridLoPts[2]);
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  } else if ( faceSide == "x+" ) {
    int i = resolution[0] - 1;
    for (int j = 0; j<resolution[1]; j++) {
      for (int k = 0; k<resolution[2]; k++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1] + gridLoPts[1], (k + 0.5) * Dx[2]+ gridLoPts[2] );
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  } else if ( faceSide == "y-" ) {
    int j = 0;
    for (int i = 0; i<resolution[0]; i++) {
      for (int k = 0; k<resolution[2]; k++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1] + gridLoPts[1], (k + 0.5) * Dx[2]+ gridLoPts[2] );
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  } else if ( faceSide == "y+" ) {
    int j = resolution[1] - 1;
    for (int i = 0; i<resolution[0]; i++) {
      for (int k = 0; k<resolution[2]; k++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1] + gridLoPts[1], (k + 0.5) * Dx[2] + gridLoPts[2]);
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  } else if ( faceSide == "z-" ) {
    int k = 0;
    for (int i = 0; i<resolution[0]; i++) {
      for (int j = 0; j<resolution[1]; j++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1] + gridLoPts[1], (k + 0.5) * Dx[2] + gridLoPts[2]);
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  } else if (faceSide=="z+" ) {
    int k = resolution[2] - 1;
    for (int i = 0; i<resolution[0]; i++) {
      for (int j = 0; j<resolution[1]; j++) {
        C = IntVector( i, j, k);  
        pointTest = Point( (i + 0.5) * Dx[0] + gridLoPts[0], (j+ 0.5) * Dx[1]+ gridLoPts[1], (k + 0.5) * Dx[2] + gridLoPts[2]);
        if (bcGeom->inside( pointTest ) ) {
          ptList.push_back(C);
        }
      }
    }
  }

  if (ptList.empty() ) {
    cout << "No points found for the geom object on this face, check input file" << endl;
    exit(1);
  }  
  
  vector<int> minCell (3,0);
  vector<int> maxCell (3,0);
  //find bounding box
  minCell[0] = ptList[0].x(); minCell[1] = ptList[0].y(); minCell[2] = ptList[0].z();
  for (int i = 0; i< ptList.size(); i++) {
    if (minCell[0] > ptList[i].x() )
      minCell[0] = ptList[i].x();
    if (minCell[1] > ptList[i].y() )
      minCell[1] = ptList[i].y();
    if (minCell[2] > ptList[i].z() )
      minCell[2] = ptList[i].z();
    
    if (maxCell[0] < ptList[i].x() )
      maxCell[0] = ptList[i].x();
    if (maxCell[1] < ptList[i].y() )
      maxCell[1] = ptList[i].y();
    if (maxCell[2] < ptList[i].z() )
      maxCell[2] = ptList[i].z();
  }
  
  int NT; //number of timesteps before repeat
  NT = getInt( gzFp );
  int M_x, M_y, M_z; //resolution size across inlet
  
  M_x = maxCell[0] - minCell[0] + 1; //add one since need cell inclusion from the min/max list
  M_y = maxCell[1] - minCell[1] + 1;
  M_z = maxCell[2] - minCell[2] + 1;
  
  cout << "Inlet Dimen: " << M_x << " " << M_y << " " << M_z << endl;
  
  vector<double> L_a (3);
  L_a[0] = getDouble(gzFp);
  L_a[1] = getDouble(gzFp);
  L_a[2] = getDouble(gzFp);
  
  double AccuracyCoef;
  AccuracyCoef = getDouble(gzFp);
  if (AccuracyCoef < 2.0)
    AccuracyCoef = 2.0;  //2 is min rqmt for algorithm
  
  vector<double> aveU (3);
  aveU[0] = getDouble(gzFp);
  aveU[1] = getDouble(gzFp);
  aveU[2] = getDouble(gzFp);
  
  vector<double> Rij (6); //mean reynold's stress tensor
  //read r11 r31 r22 r31 r32 r33
  for (int i=0; i<6; i++) {
    double v = getDouble(gzFp);
    Rij[i] = v;
  }
  
  vector< vector<double> > Aij (3, vector<double>(3) );
  Aij[0][0] = sqrt(Rij[0]);
  Aij[1][0] = Rij[1]/Aij[0][0];
  Aij[1][1] = sqrt(Rij[2]-Aij[1][0]*Aij[1][0]);
  Aij[2][0] = Rij[3]/Aij[0][0];
  Aij[2][1] = (Rij[4] - Aij[1][0] * Aij[2][0]) / Aij[1][1];
  Aij[2][2] = sqrt(Rij[5] - Aij[2][0]*Aij[2][0] - Aij[2][1] * Aij[2][1] );
  
  if (Rij[0] < 0) {
    cout << "<u u> must be > zero" << endl;
    exit(1);
  } else if ( (Rij[2]-Aij[1][0]*Aij[1][0]) < 0 ) {
    cout << "<u v> too large must be " << endl;
    exit(1);
  } else if ( (Rij[5] - Aij[2][0]*Aij[2][0] - Aij[2][1] * Aij[2][1]) < 0 ) {
    cout << "<u w> too large must be..." << endl;
    exit(1);
  }
  
  
  vector<double> n_a (3); //multiple of length scale
  vector<int> N_a (3); //Required support filter
  
  vector<int> filterSize (3);
  //set up lengths of filter coefficients
  for (int i =0; i<3; i++) {
    n_a[i] = L_a[i]/Dx[i];
    N_a[i] = (int) ceil(n_a[i] * AccuracyCoef);
    filterSize[i] = 2*N_a[i] + 1;
  }
  
  //TEST input params
  cout <<	"face: " << faceSide << endl;
  cout << "shape: " << faceShape << endl;
  cout << "Timesteps till repeat: " << NT << endl;
  
  cout << "n_a: " << n_a[0] << " " << n_a[1] << " " << n_a[2] << endl;
  cout << "N_a: " << N_a[0] << " " << N_a[1] << " " << N_a[2] << endl;
  
  vector<int> randomSize (3);
  //set up size of random fields note: NT +2 instead of +1 is temporary
  //added in an extra point to test if periodic table was set correctly
  if ( faceSide == "x-" || faceSide == "x+") { 
    randomSize[0] = 2*N_a[0]+ NT + 1 + 1; 
    randomSize[1] = 2*N_a[1] + M_y;
    randomSize[2] = 2*N_a[2] + M_z;
  } else if ( faceSide == "y-" || faceSide == "y+" ) {
    randomSize[0] = 2*N_a[0] + M_x;
    randomSize[1] = 2*N_a[1] + 1 + NT +1 ;
    randomSize[2] = 2*N_a[2] + M_z;
  } else if (faceSide == "z-" || faceSide == "z+" ) {
    randomSize[0] = 2*N_a[0] + M_x;
    randomSize[1] = 2*N_a[1] + M_y;
    randomSize[2] = 2*N_a[2] + 1 + NT +1 ;
  }
  
  cout << "Random Field Size: " << randomSize[0] << " " << randomSize[1] << " " << randomSize[2] << endl;
  
  vector < vector <vector<double> > > randomFieldx( randomSize[0], vector< vector<double> > (randomSize[1], vector<double> (randomSize[2])));
  vector < vector <vector<double> > > randomFieldy( randomSize[0], vector< vector<double> > (randomSize[1], vector<double> (randomSize[2])));
  vector < vector <vector<double> > > randomFieldz( randomSize[0], vector< vector<double> > (randomSize[1], vector<double> (randomSize[2])));
  ;
  srand( time(NULL) );
  
  if ( faceSide == "x-" || faceSide == "x+" ) {
    for (int i=0; i<NT; i++) {
      for (int j=0; j<randomSize[1]; j++) {
        for (int k=0; k<randomSize[2]; k++) {
          randomFieldx[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldy[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldz[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
        }
      }
    }
    for (int i = NT; i<randomSize[0]; i++) {
      for (int j=0; j<randomSize[1]; j++) {
        for (int k=0; k<randomSize[2]; k++) {
          randomFieldx[i][j][k] = randomFieldx[i-NT][j][k];
          randomFieldy[i][j][k] = randomFieldy[i-NT][j][k];
          randomFieldz[i][j][k] = randomFieldz[i-NT][j][k];
        }
      }
    }
  
  } else if ( faceSide == "y-" || faceSide == "y+" ) {
    for (int i=0; i<randomSize[0]; i++) {
      for (int j=0; j<NT; j++) {
        for (int k=0; k<randomSize[2]; k++) {
          randomFieldx[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldy[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldz[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
        }
      }
    }
    for (int i = 0; i<randomSize[0]; i++) {
      for (int j=NT; j<randomSize[1]; j++) {
        for (int k=0; k<randomSize[2]; k++) {
          randomFieldx[i][j][k] = randomFieldx[i][j-NT][k];
          randomFieldy[i][j][k] = randomFieldy[i][j-NT][k];
          randomFieldz[i][j][k] = randomFieldz[i][j-NT][k];
        }
      }
    }
    
  } else if ( faceSide == "z-" || faceSide == "z+" ) {
    for (int i=0; i<randomSize[0]; i++) {
      for (int j=0; j<randomSize[1]; j++) {
        for (int k=0; k<NT; k++) {
          randomFieldx[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldy[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
          randomFieldz[i][j][k] = ((double)rand()/RAND_MAX - 0.5)*sqrt(3.0)/0.5;
        }
      }
    }
    for (int i = 0; i<randomSize[0]; i++) {
      for (int j=0; j<randomSize[1]; j++) {
        for (int k=NT; k<randomSize[2]; k++) {
          randomFieldx[i][j][k] = randomFieldx[i][j][k-NT];
          randomFieldy[i][j][k] = randomFieldy[i][j][k-NT];
          randomFieldz[i][j][k] = randomFieldz[i][j][k-NT];
        }
      }
    }
  }
  
  double magVelocity;
  magVelocity = sqrt(aveU[0]*aveU[0] + aveU[1]*aveU[1] + aveU[2]*aveU[2]);
  cout << "Velocity Magnitude: " << magVelocity << endl;
  
  //check if the velocity vector is at an angle from normal plane
  bool angleVelocity = false;
  if (aveU[0] != 0.0 && aveU[1] != 0.0) {
    angleVelocity = true;
  } else if (aveU[0] != 0.0 && aveU[2] != 0.0 ){
    angleVelocity = true;
  } else if (aveU[1] != 0.0 && aveU[2] != 0.0) {
    angleVelocity = true;
  }

  vector< vector <double> > rotationMatrix (3, vector<double> (3) );  
  vector<double> normVec (3);
  vector<double> rotVec (3);
  double rotAngle = 0.0;
  double magRotVec = 0.0;
  if (angleVelocity) {
    if (faceSide=="x-") {
      normVec[0] = 1.0;
      rotVec[0] = 0.0; rotVec[1] = -aveU[2]; rotVec[2]= aveU[1];
      rotAngle = acos(normVec[0] * aveU[0] / magVelocity);
    } else if (faceSide=="x+") {
      normVec[0] = -1.0;
      rotVec[0] = 0.0; rotVec[1] = aveU[2]; rotVec[2]= -aveU[1];
      rotAngle = acos(normVec[0] * aveU[0] / magVelocity);
    } else if (faceSide=="y-") {
      normVec[1] = 1.0;
      rotVec[0] = aveU[2]; rotVec[1] = 0.0; rotVec[2]= -aveU[0];
      rotAngle = acos(normVec[1] * aveU[1] / magVelocity);
    } else if (faceSide=="y+") {
      normVec[1] = -1.0;
      rotVec[0] = -aveU[2]; rotVec[1] = 0.0; rotVec[2]= aveU[0];
      rotAngle = acos(normVec[1] * aveU[1] / magVelocity);
    } else if (faceSide=="z-") {
      normVec[2] = 1.0;
      rotVec[0] = -aveU[1]; rotVec[1] = aveU[0]; rotVec[2]= 0.0;
      rotAngle = acos(normVec[2] * aveU[2] / magVelocity);
    } else if (faceSide=="z+") {
      normVec[2] = -1.0;
      rotVec[0] = aveU[1]; rotVec[1] = -aveU[0]; rotVec[2]= 0.0;
      rotAngle = acos(normVec[2] * aveU[2] / magVelocity);
    }
    
    //normalize rotation axis
    magRotVec = sqrt(rotVec[0]*rotVec[0]+rotVec[1]*rotVec[1]+rotVec[2]*rotVec[2]);
    rotVec[0] = - rotVec[0]/magRotVec;
    rotVec[1] = - rotVec[1]/magRotVec;
    rotVec[2] = - rotVec[2]/magRotVec;
    
    //evaluate rotation matrix
    rotationMatrix[0][0] = cos(rotAngle)+rotVec[0]*rotVec[0]*(1-cos(rotAngle));
    rotationMatrix[0][1] = rotVec[0]*rotVec[1]*(1-cos(rotAngle))-rotVec[2]*sin(rotAngle);
    rotationMatrix[0][2] = rotVec[0]*rotVec[2]*(1-cos(rotAngle))+rotVec[1]*sin(rotAngle);
    rotationMatrix[1][0] = rotVec[0]*rotVec[1]*(1-cos(rotAngle))+rotVec[2]*sin(rotAngle);
    rotationMatrix[1][1] = cos(rotAngle)+rotVec[1]*rotVec[1]*(1-cos(rotAngle));
    rotationMatrix[1][2] = rotVec[1]*rotVec[2]*(1-cos(rotAngle))-rotVec[0]*sin(rotAngle);
    rotationMatrix[2][0] = rotVec[0]*rotVec[2]*(1-cos(rotAngle))-rotVec[1]*sin(rotAngle);
    rotationMatrix[2][1] = rotVec[1]*rotVec[2]*(1-cos(rotAngle))+rotVec[0]*sin(rotAngle);
    rotationMatrix[2][2] = cos(rotAngle)+rotVec[2]*rotVec[2]*(1-cos(rotAngle));
  }
  
  /* rotation matrix
  given velocity vector and plane normal
  find cross product as rotation axis, and angle b/t vectors
  then
  R = ([cos(t)+ux^2*(1-cos(t)), ux*uy*(1-cos(t))-uz*sin(t), ux*uz*(1-cos(t))+uy*sin(t);
        uy*ux*(1-cos(t))+uz*sin(t), cos(t)+uy^2*(1-cos(t)), uy*uz*(1-cos(t))-ux*sin(t);
        uz*ux*(1-cos(t))-uy*sin(t), uz*uy*(1-cos(t))+ux*sin(t), cos(t) + uz^2*(1-cos(t))]);
  if angle  = 0, R = I
  */
  
  //TEST BLOCK for rotation
  double tempX = 0.0; 
  double tempY = 0.0;
  double tempZ = 0.0;
  if (faceSide=="x-" ) {
    tempX = magVelocity;
  } else if (faceSide=="x+") {
    tempX = -magVelocity;
  } else if (faceSide=="y-") {
    tempY = magVelocity;
  } else if (faceSide=="y+") {
    tempY = -magVelocity;
  } else if (faceSide=="z-"){
    tempZ = magVelocity;
  } else {
    tempZ = -magVelocity;
  }
  vector<double> testRotation (3);
  testRotation[0] = rotationMatrix[0][0] * tempX + rotationMatrix[1][0]*tempY + rotationMatrix[2][0]*tempZ;
  testRotation[1] = rotationMatrix[0][1] * tempX + rotationMatrix[1][1]*tempY + rotationMatrix[2][1]*tempZ;
  testRotation[2] =  rotationMatrix[0][2] * tempX + rotationMatrix[1][2]*tempY + rotationMatrix[2][2]*tempZ;
  
  cout << "RotationCheck (should equal input velocity vector if angled): " << testRotation[0] << " " << testRotation[1] << " " << testRotation[2] << endl;
  //END TEST BLOCK
  
  //take the total size of the inlet, from non-normal sizes
  int jSize = 0;
  int kSize = 0; //will chaneg based on face
  if (faceSide=="x-" || faceSide =="x+") {
    jSize = M_y;
    kSize = M_z;
  } else if (faceSide=="y-" || faceSide =="y+") {
    jSize = M_x;
    kSize = M_z;
  } else if (faceSide=="z-" || faceSide =="z+") {
    jSize = M_x;
    kSize = M_y;
  }
  
  string varySpace;
  varySpace = getString(gzFp);
  
  if (varySpace == "n" || varySpace == "no") {
    gzclose(gzFp);
    vector< vector <vector<double> > > filterCoefficients( filterSize[0], vector< vector<double> > (filterSize[1], vector<double> (filterSize[2])));
    
    for (int i = 0; i<filterSize[0]; i++) {
      for (int j = 0; j<filterSize[1]; j++) {
        for (int k = 0; k<filterSize[2]; k++) {
          filterCoefficients[i][j][k] = BSum(i, n_a[0], N_a[0]) * BSum(j, n_a[1], N_a[1]) * BSum(k, n_a[2], N_a[2]); 
        }
      }
    }
    
    spatiallyConstant( outputfile,
                      faceSide,
                      randomFieldx,
                      randomFieldy,
                      randomFieldz,
                      filterCoefficients,
                      Aij,
                      filterSize,
                      aveU,
                      NT, jSize, kSize, 
                      angleVelocity, rotationMatrix, 
                      minCell);
    
  } else if (varySpace =="yes" || varySpace == "y") {
    // for a spatially variant case, calculate velocity fluctuations based on the magnitude & normal flow direction
    // then apply the rotatin matrix (if needed ) at the end
    string variationDirection;
    variationDirection = getString(gzFp);
    
    //find normalized distance of each point from origin or plane of inlet
    // 0 = at origin, 1 = at wall
    vector<vector <double> > normDistance;
    if (faceShape == "circle" ) {
      normDistance = setCircleD( jSize, kSize, faceSide, charDim, origin, Dx, minCell, gridLoPts);
    } else if (faceShape == "ellipse" ) {
      normDistance = setEllipseD( jSize, kSize, faceSide, minor_radius, major_radius, angle, origin,  Dx, minCell, gridLoPts, ptList, resolution);
    } else if (faceShape == "box" || faceShape == "rectangle") {
      normDistance = setPlaneD( jSize, kSize, faceSide, charDim, origin, Dx, minCell, gridLoPts, variationDirection);
    } else if (faceShape =="annulus" ) {
      normDistance = setAnnulusD( jSize, kSize, faceSide, minor_radius, major_radius, origin, Dx, minCell, gridLoPts); 
    }

    vector<vector <vector <double> > > velocityProfile;
    vector<vector <vector <double> > > reynoldsStressProfile;
    vector<vector <vector <double> > > lengthScaleProfile;
    
    //set velocity profile
    string velocityType;
    velocityType = getString(gzFp);
    if (velocityType == "constant") {
      velocityProfile = setConstV( magVelocity, jSize, kSize, faceSide) ;
    } else if (velocityType == "laminar") {
      velocityProfile = setLaminarV( magVelocity, jSize, kSize, faceSide, normDistance);
    } else if (velocityType == "tanh") {
      velocityProfile = setTanhV( magVelocity, jSize, kSize, faceSide, normDistance);
    } else if (velocityType == "power") {
      velocityProfile = setPowerV( magVelocity, jSize, kSize, faceSide, normDistance);
    } else {
      cout << "Specify a valid velocity profile: constant/laminar/tanh/power" << endl;
      exit(1);
    }

    //set stress profile
    string stressType;
    stressType = getString(gzFp);
    if (stressType == "constant" ) {
      reynoldsStressProfile = setConstAij( Aij, jSize, kSize );
    } else if (stressType == "jet" ) {
      //use for open flow
      reynoldsStressProfile = setJetAij(magVelocity, jSize, kSize, faceSide, normDistance);
    } else if (stressType == "channel" || stressType == "pipe") {
      //use for confined near wall effects
      reynoldsStressProfile = setChannelAij(magVelocity, jSize, kSize, faceSide, normDistance);
    } else {
      cout << "Specify a valid reynolds stress profile: constant/jet/channel/pipe" << endl;
      exit(1);
    }
    
    //set length scale profile
    string lengthScaleType;
    lengthScaleType = getString(gzFp);
    gzclose(gzFp);
    
    if (lengthScaleType == "constant") {
      lengthScaleProfile = setConstL( L_a, jSize, kSize );
    } else if (lengthScaleType == "jet" ) {
      lengthScaleProfile = setChannelL( L_a, jSize, kSize, faceSide, normDistance);
    } else if (lengthScaleType == "channel" || lengthScaleType == "pipe") {
      lengthScaleProfile = setChannelL( L_a, jSize, kSize, faceSide, normDistance); 
    } else {
      cout << "Specify a valid length scale profile: constant/jet/channel/pipe" << endl;
      exit(1);
    }

    spatiallyVariable( outputfile,
                      faceSide,
                      randomFieldx,
                      randomFieldy,
                      randomFieldz,
                      velocityProfile,
                      reynoldsStressProfile,
                      lengthScaleProfile,
                      NT, jSize, kSize,
                      angleVelocity, rotationMatrix,
                      Dx, AccuracyCoef,
                      minCell); 
    
  }
  
  return 0;
}

