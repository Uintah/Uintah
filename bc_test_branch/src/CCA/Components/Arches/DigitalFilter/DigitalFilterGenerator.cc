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
  for (int j=-N; j<=N; j++) {  
    SumTerm	= SumTerm + bTilde(j, n) * bTilde(j, n);
  }
  SumTerm = sqrt(SumTerm);
  double BK = bTilde(k, n)/SumTerm;
  return(BK);
}

/***********************************************************************
 interpolation class
 ************************************************************************/

class Interpolator {
public:
  Interpolator(std::vector<double> & x, std::vector<std::vector<double> >& y){
    xVals = x;
    yVals = y;
  };
  ~Interpolator(){};
  
  double returnValue( double xi, int var_index ) {     
    int lo = 0; int mid = 0;
    int hi = (int)xVals.size() - 1;
    double var_val;
    
    if ( xVals[lo] < xi && xVals[hi] > xi) {
      while ( (hi - lo) > 1) {
        mid = (lo+hi)/2;
        if (xVals[mid] > xi ) {
          hi = mid;
        } else if (xVals[mid] < xi) {
          lo = mid;
        } else {
          //if (i1[i1dep_ind][mid] == iv[0])
          lo = mid;											
          hi = mid;
        } 
      }
    } else if ( xVals[lo] == xi) {
      hi = 1;
    } else if (xVals[hi] == xi) {
      lo = hi - 1;   
    } else if (xi < xVals[lo]) {
      var_val = yVals[lo][var_index];
      return var_val;
    } else if (xi > xVals[hi]) {
      var_val = yVals[hi][var_index];
      return var_val;
    }

    var_val = (yVals[hi][var_index]-yVals[lo][var_index])/(xVals[hi]-xVals[lo])*(xi-xVals[lo])+ yVals[lo][var_index];
    return var_val;
  };
  
private:
  std::vector<double> xVals;
  std::vector<std::vector<double> > yVals;
};

/***********************************************************************
 Structure to hold information needed for setting up profiles
 ************************************************************************/
struct InletInfo {
  int jSize;
  int kSize;
  string faceSide;
  
  //profiles
  std::vector<std::vector<double> > distanceProfile;
  std::vector<std::vector<std::vector<double> > > velocityProfile;
  std::vector<double> constVelocity;
  double velocityMagnitude;
  std::vector<std::vector<std::vector<double> > > stressProfile;
  std::vector<double> constStress;
  std::vector<std::vector<std::vector<double> > > lengthScaleProfile;
  std::vector<double> maxLengthScale;
};
  
/***********************************************************************
 Wall Distance class
 ************************************************************************/
class BaseDistance {
public:
  BaseDistance(){};
  virtual ~BaseDistance(){};
public:
  virtual inline void setDistance( InletInfo& inlet ){};
protected:
  std::vector<double> gridLoPts_;
  std::vector<double> Dx_;
  std::vector<int> minCell_;
  std::vector<double> origin_;
  double charDim_;
  double middleRadius_;
  double minorRadius_;
  double majorRadius_;
  double angle_;
  std::string varyDim_;
};
  
class circleD : public BaseDistance {
public: 
  circleD( vector<double>& gridLoPts, vector<double>& origin, vector<int>& minCell, vector<double>& dx, double& charDim){
    gridLoPts_ = gridLoPts;
    origin_ = origin;
    minCell_ = minCell;
    charDim_ = charDim;
    Dx_ = dx;
  };
  ~circleD(){}; 
  inline void setDistance( InletInfo& inlet ){
    
    double isPos; //multiply either 1 or -1 by distance, convention is that "j-dir" denotes which way
    int jj = 0, kk = 0;
    //set indicies to use, based on the face side
    if (inlet.faceSide=="x-" || inlet.faceSide=="x+") {
      jj = 1; kk = 2;
    } else if (inlet.faceSide=="y-" || inlet.faceSide=="y+") {
      jj = 0; kk = 2;
    } else if (inlet.faceSide=="z-" || inlet.faceSide=="z+") {
      jj = 0; kk = 1;
    } 
    
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {

        inlet.distanceProfile[j][k] = sqrt( ((minCell_[jj]+j+0.5)*Dx_[jj] +gridLoPts_[jj] - origin_[jj]) * ((minCell_[jj]+j+0.5)*Dx_[jj] +gridLoPts_[jj] - origin_[jj]) +
                                             ((minCell_[kk]+k+0.5)*Dx_[kk] +gridLoPts_[kk] - origin_[kk]) * ((minCell_[kk]+k+0.5)*Dx_[kk] +gridLoPts_[kk] - origin_[kk]) ); 
        
        if ( ((minCell_[jj]+j+0.5)*Dx_[jj] +gridLoPts_[jj] - origin_[jj]) >= 0.0 ) {
          isPos = 1.0;
        } else {
          isPos = -1.0;
        }
        
        inlet.distanceProfile[j][k] = isPos*inlet.distanceProfile[j][k]/charDim_*2.0; //normalize distance
      }
    }
  };
};

class annulusD : public BaseDistance {
public:
  annulusD(vector<double>& gridLoPts, vector<double>& origin, vector<int>& minCell, vector<double>& dx, double& minorRadius, double& majorRadius){
    gridLoPts_ = gridLoPts;
    origin_ = origin;
    minCell_ = minCell;
    middleRadius_ = (minorRadius+ majorRadius) * 0.5;
    charDim_ = middleRadius_ - minorRadius;
    Dx_ = dx;  
  };
  ~annulusD(){};
  inline void setDistance( InletInfo& inlet) {
    int jj = 0, kk = 0;
    //set indicies to use, based on the face side
    if (inlet.faceSide=="x-" || inlet.faceSide=="x+") {
      jj = 1; kk = 2;
    } else if (inlet.faceSide=="y-" || inlet.faceSide=="y+") {
      jj = 0; kk = 2;
    } else if (inlet.faceSide=="z-" || inlet.faceSide=="z+") {
      jj = 0; kk = 1;
    }

    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        //get distance from origin at each pt
        inlet.distanceProfile[j][k] = sqrt( ((minCell_[jj]+j+0.5)*Dx_[jj] +gridLoPts_[jj] - origin_[jj]) * ((minCell_[jj]+j+0.5)*Dx_[jj] +gridLoPts_[jj] - origin_[jj]) +
                                             ((minCell_[kk]+k+0.5)*Dx_[kk] +gridLoPts_[kk] - origin_[kk]) * ((minCell_[kk]+k+0.5)*Dx_[kk] +gridLoPts_[kk] - origin_[kk]) );
        // neg D = inner, pos D = outer
        inlet.distanceProfile[j][k] = (inlet.distanceProfile[j][k] - middleRadius_)/charDim_; //normalize distance
      }
    }
    
  };
};
  
class ellipseD : public BaseDistance {
public:
  ellipseD(vector<double>& gridLoPts, vector<double>& origin, vector<int>& minCell, vector<double>& dx, double& minorRadius, double& majorRadius, double& angleRad){
    gridLoPts_ = gridLoPts;
    origin_ = origin;
    minCell_ = minCell;
    Dx_ = dx;
    minorRadius_ = minorRadius;
    majorRadius_ = majorRadius;
    angle_ = angleRad*PI/180.0;
  };
  ~ellipseD(){};
  inline void setDistance( InletInfo& inlet) {

    double x1 = 0.0;
    double x2 = 0.0;
    vector<double> rotation (4);
    double r1 = 0.0;
    double r2 = 0.0;

    //r11 r12 r21 r22
    rotation[0] = cos(angle_); rotation[3] = cos(angle_); //rotation matrix on - faces
    rotation[1] = sin(angle_); rotation[2] = -sin(angle_);
    if (inlet.faceSide=="x+" || inlet.faceSide=="y+" || inlet.faceSide=="z+") {
      rotation[1] = -rotation[1]; rotation[2] = -rotation[2]; //only need to swap sign for these two on + faces
    }
    
    int jj = 0, kk = 0;
    double isPos;
    //set indicies to use, based on the face side
    if (inlet.faceSide=="x-" || inlet.faceSide=="x+") {
      jj = 1; kk = 2;
    } else if (inlet.faceSide=="y-" || inlet.faceSide=="y+") {
      jj = 2; kk = 0;  //this is opposite other geometry, due to way uintah rotation is set
    } else if (inlet.faceSide=="z-" || inlet.faceSide=="z+") {
      jj = 0; kk = 1;
    }
      
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        x1 = (minCell_[jj] + j +0.5)*Dx_[jj] + gridLoPts_[jj] - origin_[jj]; //jdir distance
        x2 = (minCell_[kk] + k +0.5)*Dx_[kk] + gridLoPts_[kk] - origin_[kk]; //kdir distance
        r1 = x1*rotation[0] + x2*rotation[2]; //rotate distacnes
        r2 = x1*rotation[1] + x2*rotation[3];
        
        if (x1 >= 0) {
          isPos = 1.0;
        } else {
          isPos = -1.0;
        }
        
        inlet.distanceProfile[j][k] = isPos*sqrt(  r1/majorRadius_*r1/majorRadius_ + r2/minorRadius_*r2/minorRadius_ ) ; 
      }
    }
    
  };
};
  
class boxD : public BaseDistance {
public:
  boxD(vector<double>& gridLoPts, vector<double>& origin, vector<int>& minCell, vector<double>& dx, double& charDim, string& varyDim) {
    gridLoPts_ = gridLoPts;
    origin_ = origin;
    minCell_ = minCell;
    Dx_ = dx;
    charDim_ = charDim;
    varyDim_ = varyDim;
  };
  ~boxD(){};
  inline void setDistance( InletInfo& inlet) {
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        if (varyDim_ == "x") {
          inlet.distanceProfile[j][k] = (minCell_[0] + j+ 0.5)*Dx_[0] + gridLoPts_[0] - origin_[0];
        } else if (varyDim_ == "y") {
          if (inlet.faceSide == "x-" || inlet.faceSide == "x+" ) {
            inlet.distanceProfile[j][k] = (minCell_[1] + j+ 0.5)*Dx_[1] + gridLoPts_[1] - origin_[1];
          } else {
            inlet.distanceProfile[j][k] = (minCell_[1] + k+ 0.5)*Dx_[1] + gridLoPts_[1] - origin_[1];
          }
        } else if (varyDim_ == "z") {
          inlet.distanceProfile[j][k] = (minCell_[2] + k+ 0.5)*Dx_[2] + gridLoPts_[2] - origin_[2];
        } else {
          cout << "Enter Valid Variation Direction (x/y/z) " << endl;
          exit(1);
        }
        
        inlet.distanceProfile[j][k] = inlet.distanceProfile[j][k]/charDim_*2.0; //use center as char distance, rather than whole channel width
      }
    }
  };
};
  
/***********************************************************************
 Velocity class
 ************************************************************************/
class BaseVelocity {
public:
  BaseVelocity(){};
  virtual ~BaseVelocity(){};
  virtual inline void setVelocity( InletInfo& inlet) {};
};

class tanhV : public BaseVelocity {
public:
  tanhV(){};
  ~tanhV(){};
  inline void setVelocity( InletInfo& inlet){
    double distance;
    double Vmax = inlet.velocityMagnitude;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = abs(inlet.distanceProfile[j][k]);
        if (inlet.faceSide == "x-" ) {
          inlet.velocityProfile[j][k][0] = Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) );
        } else if (inlet.faceSide == "x+") {
          inlet.velocityProfile[j][k][0] = -(Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) ) );
        } else if (inlet.faceSide == "y-") {
          inlet.velocityProfile[j][k][1] = Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) );
        } else if (inlet.faceSide == "y+") {
          inlet.velocityProfile[j][k][1] = -( Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) ) );
        } else if (inlet.faceSide == "z-") {
          inlet.velocityProfile[j][k][2] = Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) );
        } else if (inlet.faceSide == "z+") {
          inlet.velocityProfile[j][k][2] = -( Vmax* 0.5 + Vmax* 0.5 * tanh(10.0 * (1.0-distance) ) );
        }
      }
    }
  };
};

class laminarV : public BaseVelocity {
public:
  laminarV(){};
  ~laminarV(){};
  inline void setVelocity( InletInfo& inlet) {
    double distance;
    double Vmax = inlet.velocityMagnitude;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = abs( inlet.distanceProfile[j][k] ); 
        if (inlet.faceSide == "x-" ) {
          inlet.velocityProfile[j][k][0] = Vmax* (1.0 - distance*distance);
        } else if (inlet.faceSide == "x+") {
          inlet.velocityProfile[j][k][0] = -(Vmax* (1.0 - distance*distance) );
        } else if (inlet.faceSide == "y-") {
          inlet.velocityProfile[j][k][1] =  Vmax* (1.0 - distance*distance);
        } else if (inlet.faceSide == "y+") {
          inlet.velocityProfile[j][k][1] = -(Vmax* (1.0  - distance*distance) );
        } else if (inlet.faceSide == "z-") {
          inlet.velocityProfile[j][k][2] =  Vmax* (1.0 - distance*distance);
        } else if (inlet.faceSide == "z+") {
          inlet.velocityProfile[j][k][2] = -(Vmax* (1.0 - distance*distance) );
        }
      }
    }
  };
};

class powerV : public BaseVelocity {
public:
  powerV(){};
  ~powerV(){};
  inline void setVelocity( InletInfo& inlet) {
    double distance;
    double Vmax = inlet.velocityMagnitude;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = inlet.distanceProfile[j][k];
        if (distance < 1.0) {
          if (inlet.faceSide == "x-" ) {
            inlet.velocityProfile[j][k][0] = Vmax* pow( (1.0 - distance), 1.0/7.0 );
          } else if (inlet.faceSide == "x+") {
            inlet.velocityProfile[j][k][0] = -( Vmax* pow( (1.0 - distance), 1.0/7.0 ) );
          } else if (inlet.faceSide == "y-") {
            inlet.velocityProfile[j][k][1] =  Vmax* pow( (1.0 - distance), 1.0/7.0 );
          } else if (inlet.faceSide == "y+") {
            inlet.velocityProfile[j][k][1] = -( Vmax* pow( (1.0 - distance), 1.0/7.0 ) );
          } else if (inlet.faceSide == "z-") {
            inlet.velocityProfile[j][k][2] =  Vmax* pow( (1.0 - distance), 1.0/7.0 );
          } else if (inlet.faceSide == "z+") {
            inlet.velocityProfile[j][k][2] = -( Vmax* pow( (1.0 - distance), 1.0/7.0 ) );
          }
        } 
      }
    }
  };
};

class specifiedV : public BaseVelocity {
public:
  specifiedV( std::vector<double>& specDistance, std::vector<std::vector<double> >& specVelocity, bool isSymmetric){
    interpV = new Interpolator( specDistance, specVelocity);
		isSymmetric_ = isSymmetric;
  };
  ~specifiedV(){};
  inline void setVelocity( InletInfo& inlet) {
    double distance;
    double U,V,W;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = inlet.distanceProfile[j][k];
        if (isSymmetric_)
          distance = abs(distance);
        U = interpV->returnValue(distance, 0);
        V = interpV->returnValue(distance, 1);
        W = interpV->returnValue(distance, 2);
        inlet.velocityProfile[j][k][0] = U;
        inlet.velocityProfile[j][k][1] = V;
        inlet.velocityProfile[j][k][2] = W;
      }
    }
  };
private:
  Interpolator * interpV;
  bool isSymmetric_;
};

class constV : public BaseVelocity {
public:
  constV(){};
  ~constV(){};
};

/***********************************************************************
 Stress Class
 ************************************************************************/

class BaseStress {
public:
  BaseStress(){};
  virtual ~BaseStress(){};
  virtual inline void setStress( InletInfo& inlet) {};
};

class jetStress : public BaseStress {
public:
  jetStress(){};
  ~jetStress(){};
  inline void setStress( InletInfo& inlet) {
    double distance;
    double r11=0.0,r21=0.0,r22=0.0,r31=0.0,r32=0.0,r33=0.0;

    //rough correlation of shear layer in jet flow from Pope
    double USq = inlet.velocityMagnitude * inlet.velocityMagnitude ;
    
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = inlet.distanceProfile[j][k];
        if (inlet.faceSide=="x-" || inlet.faceSide=="x+") {
          r11 = (- 0.03 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.035) * USq;  //uu
          r21 = (- 0.01 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.011) * USq;  //uv
          r22 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //vv
          r31 = (- 0.01 * (1.0 - abs(distance)) * (1.0 - abs(distance) ) + 0.011) * USq;  //uw
          r32 = (- 0.001 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.001) * USq; //vw
          r33 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //ww
          
        } else if (inlet.faceSide=="y-" || inlet.faceSide=="y+") {
          r11 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //uu
          r21 = (- 0.01 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.011) * USq;  //uv
          r22 = (- 0.03 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.035) * USq;  //vv
          r31 = (- 0.001 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.001) * USq; //uw
          r32 = (- 0.01 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.011) * USq;  //vw
          r33 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //ww
          
        } else if (inlet.faceSide=="z-" || inlet.faceSide=="z+") {
          r11 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //uu
          r21 = (- 0.001 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.001) * USq; //uv
          r22 = (- 0.015 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.016) * USq; //vv
          r31 = (- 0.01 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.011) * USq;  //uw
          r32 = (- 0.01 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.011) * USq;  //vw
          r33 = (- 0.03 * (1.0 - abs(distance) ) * (1.0 - abs(distance) ) + 0.035) * USq;  //ww
        }
        
        if ( distance < 0 ) {  //make the cross correlations anti-symmetric
          r21 = -r21;  
          r31 = -r31;
          r32 = -r32;
        }
        
        //Lund transform of stress tensor
        inlet.stressProfile[j][k][0] = sqrt( r11 );
        inlet.stressProfile[j][k][1] = r21/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][2] = sqrt(r22-inlet.stressProfile[j][k][1]*inlet.stressProfile[j][k][1]);
        inlet.stressProfile[j][k][3] = r31/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][4] = ( r32 - inlet.stressProfile[j][k][2]*inlet.stressProfile[j][k][3] ) / inlet.stressProfile[j][k][2];
        inlet.stressProfile[j][k][5] = sqrt(r33 - inlet.stressProfile[j][k][3] * inlet.stressProfile[j][k][3] - 
                                             inlet.stressProfile[j][k][4] * inlet.stressProfile[j][k][4] );
        
      }
    } 
  };
};

class channelStress : public BaseStress {
  //piece wise correlation to match some dns results in fully developed flow
public:
  channelStress(){};
  ~channelStress(){};
  inline void setStress( InletInfo& inlet) {
    double USq = inlet.velocityMagnitude * inlet.velocityMagnitude;
    double wallDist, distance;
    double r11=0.0,r21=0.0,r22=0.0,r31=0.0,r32=0.0,r33=0.0;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = inlet.distanceProfile[j][k];
        wallDist = 1.0 - abs(inlet.distanceProfile[j][k]);
        if (inlet.faceSide=="x-" || inlet.faceSide=="x+") {
          if ( wallDist <= 0.1) {
            r11 = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;   //uu  
            r21 = 0.02 * wallDist * USq; //uv
            r22 = 0.6*wallDist*USq;      //vv
            r31 = 0.02 * wallDist * USq; //uw
            r32 = 0.001*USq;             //vw
            r33 = 0.6*wallDist*USq;      //ww
          } else if ( wallDist <= 0.3 ) {
            r11 = 0.014 / sqrt(wallDist);               //uu
            r21 = (-0.00225*wallDist + .00225 ) * USq;  //uv
            r22 = (-0.0444*wallDist + 0.0644) *USq;     //vv
            r31 = (-0.00225*wallDist + .00225 ) * USq;  //uw
            r32 = 0.001*USq;                            //vw
            r33 = (-0.0444*wallDist + 0.0644) *USq;     //ww
          } else { //bulk
            r11 = 0.02*USq;                            //uu
            r21 = (-0.00225*wallDist + .00225 ) * USq; //uv
            r22 = 0.02*USq;                            //vv
            r31 = (-0.00225*wallDist + .00225 ) * USq; //uw
            r32 = 0.001*USq;                           //vw
            r33 = 0.02*USq;                            //ww
          }
          
        } else if (inlet.faceSide=="y-" || inlet.faceSide=="y+") {
          if ( wallDist <= 0.1) {
            r11 = 0.6*wallDist*USq;      //uu  
            r21 = 0.02 * wallDist * USq; //uv
            r22 = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;      //vv
            r31 = 0.001*USq;             //uw
            r32 = 0.02 * wallDist * USq; //vw
            r33 = 0.6*wallDist*USq;      //ww
          } else if ( wallDist <= 0.3 ) {
            r11 = (-0.0444*wallDist + 0.0644) *USq;     //uu
            r21 = (-0.00225*wallDist + .00225 ) * USq;  //uv
            r22 = 0.014 / sqrt(wallDist);               //vv
            r31 = 0.001*USq;                            //uw
            r32 = (-0.00225*wallDist + .00225 ) * USq;  //vw
            r33 = (-0.0444*wallDist + 0.0644) *USq;     //ww
          } else { //bulk
            r11 = 0.02*USq;                            //uu
            r21 = (-0.00225*wallDist + .00225 ) * USq; //uv
            r22 = 0.02*USq;                            //vv
            r31 = 0.001*USq;                           //uw
            r32 = (-0.00225*wallDist + .00225 ) * USq; //vw
            r33 = 0.02*USq;                            //ww
          }
          
        } else if (inlet.faceSide=="z-" || inlet.faceSide=="z+") {
          if ( wallDist <= 0.1) {
            r11 = 0.6*wallDist*USq;      //uu  
            r21 = 0.001*USq;             //uv
            r22 = 0.6*wallDist*USq;      //vv
            r31 = 0.02 * wallDist * USq; //uw
            r32 = 0.02 * wallDist * USq; //vw
            r33 = (-20.0 * ( wallDist - 0.05 ) * (wallDist - 0.05 ) +0.12) * USq;      //ww
          } else if ( wallDist <= 0.3 ) {
            r11 = (-0.0444*wallDist + 0.0644) *USq;     //uu
            r21 = 0.001*USq;                            //uv
            r22 = (-0.0444*wallDist + 0.0644) *USq;     //vv
            r31 = (-0.00225*wallDist + .00225 ) * USq;  //uw
            r32 = (-0.00225*wallDist + .00225 ) * USq;  //vw
            r33 = 0.014 / sqrt(wallDist);               //ww
          } else { //bulk
            r11 = 0.02*USq;                            //uu
            r21 = 0.001*USq;                           //uv
            r22 = 0.02*USq;                            //vv
            r31 = (-0.00225*wallDist + .00225 ) * USq; //uw
            r32 = (-0.00225*wallDist + .00225 ) * USq; //vw
            r33 = 0.02*USq;                            //ww
          }
        }
        
        if ( distance < 0 ) {  //make the cross correlations anti-symmetric
          r21 = -r21;  
          r31 = -r31;
          r32 = -r32;
        }
      
        //Lund transform of stress tensor
        inlet.stressProfile[j][k][0] = sqrt( r11 );
        inlet.stressProfile[j][k][1] = r21/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][2] = sqrt(r22-inlet.stressProfile[j][k][1]*inlet.stressProfile[j][k][1]);
        inlet.stressProfile[j][k][3] = r31/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][4] = ( r32 - inlet.stressProfile[j][k][2]*inlet.stressProfile[j][k][3] ) / inlet.stressProfile[j][k][2];
        inlet.stressProfile[j][k][5] = sqrt(r33 - inlet.stressProfile[j][k][3] * inlet.stressProfile[j][k][3] - 
                                              inlet.stressProfile[j][k][4] * inlet.stressProfile[j][k][4] );
      }
    }
  };
};

class specifiedS : public BaseStress {
public:
  specifiedS( std::vector<double>& specDistance, std::vector<std::vector<double> >& specStress, bool isSymmetric ) {
    interpS = new Interpolator( specDistance, specStress );
    isSymmetric_ = isSymmetric;
  };
  ~specifiedS(){};
  inline void setStress( InletInfo& inlet) {
    double distance, distanceTmp;
    double r11,r21,r22,r31,r32,r33;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        
        distance = inlet.distanceProfile[j][k];
        distanceTmp = distance;
        if (isSymmetric_)
          distance = abs(distance);

        r11 = interpS->returnValue(distance, 0);
        r21 = interpS->returnValue(distance, 1);
        r22 = interpS->returnValue(distance, 2);
        r31 = interpS->returnValue(distance, 3);
        r32 = interpS->returnValue(distance, 4);
        r33 = interpS->returnValue(distance, 5);
        
        if ( distanceTmp < 0 && isSymmetric_) {  //make the cross correlations anti-symmetric
          r21 = -r21;  
          r31 = -r31;
          r32 = -r32;
        }
          
        //Lund transform of stress tensor
        inlet.stressProfile[j][k][0] = sqrt( r11 );
        inlet.stressProfile[j][k][1] = r21/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][2] = sqrt(r22-inlet.stressProfile[j][k][1]*inlet.stressProfile[j][k][1]);
        inlet.stressProfile[j][k][3] = r31/inlet.stressProfile[j][k][0];
        inlet.stressProfile[j][k][4] = ( r32 - inlet.stressProfile[j][k][1]*inlet.stressProfile[j][k][3] ) / inlet.stressProfile[j][k][2];
        inlet.stressProfile[j][k][5] = sqrt(r33 - inlet.stressProfile[j][k][3] * inlet.stressProfile[j][k][3] - 
                                                   inlet.stressProfile[j][k][4] * inlet.stressProfile[j][k][4] );
      }
    }
  };
private:
  bool isSymmetric_;
  Interpolator * interpS;
};

class constS : public BaseStress {
public:
  constS(){};
  ~constS(){};
};

/***********************************************************************
 Length Scale Class
 ************************************************************************/

class BaseLengthScale {
public:
  BaseLengthScale(){};
  virtual ~BaseLengthScale(){};
  virtual inline void setLengthScale( InletInfo& inlet) {};
};

class channelLength : public BaseLengthScale {
public:
  channelLength(){};
  ~channelLength(){};
  inline void setLengthScale( InletInfo& inlet) {
    double distance;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = abs(inlet.distanceProfile[j][k]); //this dist from center
        
        //use smaller length scale near walls
        if (distance > 0.95) {
          inlet.lengthScaleProfile[j][k][0] = inlet.maxLengthScale[0]/4.0;
          inlet.lengthScaleProfile[j][k][1] = inlet.maxLengthScale[1]/4.0;
          inlet.lengthScaleProfile[j][k][2] = inlet.maxLengthScale[2]/4.0;
        } else if (distance > 0.90) {
          inlet.lengthScaleProfile[j][k][0] = inlet.maxLengthScale[0]/2.0;
          inlet.lengthScaleProfile[j][k][1] = inlet.maxLengthScale[1]/2.0;
          inlet.lengthScaleProfile[j][k][2] = inlet.maxLengthScale[2]/2.0;
        } else {
          inlet.lengthScaleProfile[j][k][0] = inlet.maxLengthScale[0];
          inlet.lengthScaleProfile[j][k][1] = inlet.maxLengthScale[1];
          inlet.lengthScaleProfile[j][k][2] = inlet.maxLengthScale[2];
        }

      }
    }
  
  };
};

class specifiedL : public BaseLengthScale {
public:
  specifiedL( std::vector<double> & specDistance, std::vector<std::vector<double> >& specLengthScale, bool isSymmetric){
    interpL = new Interpolator(specDistance, specLengthScale);
    isSymmetric_ = isSymmetric;
  };
  ~specifiedL(){};
  inline void setLengthScale( InletInfo& inlet) {
    double distance;
    for (int j = 0; j< inlet.jSize; j++) {
      for (int k = 0; k< inlet.kSize; k++) {
        distance = inlet.distanceProfile[j][k];
        if (isSymmetric_)
          distance = abs(distance);
        
        inlet.lengthScaleProfile[j][k][0] = interpL->returnValue(distance, 0);
        inlet.lengthScaleProfile[j][k][1] = interpL->returnValue(distance, 1);
        inlet.lengthScaleProfile[j][k][2] = interpL->returnValue(distance, 2);
      
        //replace maximum length scale if needed
        if ( inlet.lengthScaleProfile[j][k][0] > inlet.maxLengthScale[0] )
          inlet.maxLengthScale[0] = inlet.lengthScaleProfile[j][k][0];
        if ( inlet.lengthScaleProfile[j][k][1] > inlet.maxLengthScale[1] )
          inlet.maxLengthScale[1] = inlet.lengthScaleProfile[j][k][2];
        if ( inlet.lengthScaleProfile[j][k][2] > inlet.maxLengthScale[2] )
          inlet.maxLengthScale[2] = inlet.lengthScaleProfile[j][k][2];
      }
    }
  };
  
private:
  Interpolator * interpL;
  bool isSymmetric_;
};

class constL : public BaseLengthScale {
public:
  constL(){};
  ~constL(){};
};

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
  
  //______________________
  //Create inlet to hold relevant data, makes other functions cleaner
  InletInfo newTurbInlet;
  
  //__________________________
  //Parsing the Input Information
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
  
  //____________________________
  // Parse the face inlet
  string faceSide  = getString( gzFp );
  string faceShape = getString( gzFp );

  Patch::FaceType face = Patch::stringToFaceType( faceSide );
  
  vector<double> lower (3); vector<double> upper (3);
  vector<double> origin (3);
  double radius;
  double majorRadius = 0.0;
  double minorRadius = 0.0;
  double angle = 0.0;
  double charDim = 0.0; //charateristic dimension needed for spatial correlations
  BCGeomBase* bcGeom; //face inlet
  
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
    bcGeom = new RectangleBCData( l, u, "DF Rectangle", face );
  }
  else if (faceShape=="circle") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    radius = getDouble(gzFp);
    charDim = 2*radius;
    Point p(origin[0], origin[1], origin[2]);
    bcGeom = new CircleBCData( p, radius, "DF Circle", face );
  }
  else if (faceShape=="ellipse") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    majorRadius = getDouble(gzFp);
    minorRadius = getDouble(gzFp);
    charDim = 2*minorRadius;
    angle = getDouble(gzFp);
    Point p(origin[0], origin[1], origin[2]);    
    bcGeom = new EllipseBCData( p, minorRadius, majorRadius, angle, faceSide, face );
    cout << "made ellipse" <<endl;
  }
  else if (faceShape=="annulus") {
    origin[0] = getDouble(gzFp);
    origin[1] = getDouble(gzFp);
    origin[2] = getDouble(gzFp);
    
    majorRadius = getDouble(gzFp);
    minorRadius = getDouble(gzFp);  
    charDim = majorRadius - minorRadius;
    Point p(origin[0], origin[1], origin[2]);
    bcGeom = new AnnulusBCData( p, minorRadius, majorRadius, "DF Annulus", face );
  }
  else {
    cout << "Specify a valid BC shape" << endl;
    exit(1);
  }
  
  //_________________________
  //Parsing other info
  int NT; //number of timesteps before repeat
  NT = getInt( gzFp );
  
  vector<double> L_a (3);
  L_a[0] = getDouble(gzFp);
  L_a[1] = getDouble(gzFp);
  L_a[2] = getDouble(gzFp);
  newTurbInlet.maxLengthScale = L_a;
  
  double AccuracyCoef;
  AccuracyCoef = getDouble(gzFp);
  if (AccuracyCoef < 2.0)
    AccuracyCoef = 2.0;  //2 is min rqmt for algorithm accuracy
  
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
  
  // options for profiles
  string varySpace;
  varySpace = getString(gzFp);
  string variationDirection;
  variationDirection = getString(gzFp);
  string velocityType;
  velocityType = getString(gzFp);
  string stressType;
  stressType = getString(gzFp);
  string lengthScaleType;
  lengthScaleType = getString(gzFp);
  
  //Parse expt or other user defined correlations
  int nPts = 0;
  bool isSymmetric= true; //boolean to copy all data in positive direction to negative
  if (velocityType == "specified" || stressType == "specified" || lengthScaleType=="specified")
    nPts = getInt(gzFp); //set number of data points in inlet
  
  vector<double> specDistance (nPts, 0.0);
  vector< vector<double> > specVelocity (nPts, vector<double> (3,0.0) );	
  vector< vector<double> > specStress (nPts, vector<double> (6,0.0) );	
  vector< vector<double> > specLengthScale (nPts, vector<double> (3,0.0) );	
  
  if (velocityType == "specified" || stressType == "specified" || lengthScaleType=="specified") {
    cout << "Reading Specified Values " << endl;
    cout << "NP: " << nPts << endl;
    for (int d = 0; d < nPts; d++ ) {
      specDistance[d] = getDouble(gzFp);
      if ( velocityType=="specified" ) { 
        for (int i = 0; i< 3; i++) {
          specVelocity[d][i] = getDouble(gzFp);  
        }
      }
      if ( stressType=="specified" ) { 
        for (int i = 0; i< 6; i++) {
          specStress[d][i] = getDouble(gzFp);  
        }
      }
      if ( lengthScaleType=="specified" ) { 
        for (int i = 0; i< 3; i++) {
          specLengthScale[d][i] = getDouble(gzFp);  
        }
      }
    }
    if ( specDistance[0] < 0.0 )
      isSymmetric = false; 
    
    //Debug cout statements
    for (int d = 0; d<nPts; d++) {
      for (int i = 0; i< 3; i++) {
        cout << " u[" << i << "] " << specVelocity[d][i] ;
      }
      cout << endl;
    }  
    
    for (int d = 0; d<nPts; d++) {
      for (int i = 0; i< 3; i++) {
        cout << " L[" << i << "] " << specLengthScale[d][i] ;
      }
      cout << endl;
    }  
    
    for (int d = 0; d<nPts; d++) {
      for (int i = 0; i< 6; i++) {
        cout << " Rij[" << i << "] " << specStress[d][i] ;
      }
      cout << endl;
    } 
  }
  
  gzclose(gzFp);
  
  //_________________
  //Done Parsing, finding size of inlet
  // create a list of int vector points for the face
  // to find the minimum cell value later
  IntVector C;
  Point pointTest;
  vector< IntVector > ptList;
  cout << "Getting Inlet Size" << endl;
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

  if (ptList.empty() ) { //make sure inlet is properly specified
    cout << "No points found for the geom inlet on this face, check input file" << endl;
    exit(1);
  }  
  
  vector<int> minCell (3,0);
  vector<int> maxCell (3,0);
  //find bounding box
  minCell[0] = ptList[0].x(); minCell[1] = ptList[0].y(); minCell[2] = ptList[0].z();
  for (int i = 0; i< (int)ptList.size(); i++) {
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
  
  int M_x, M_y, M_z; //resolution size across inlet
  M_x = maxCell[0] - minCell[0] + 1; //add one since need cell inclusion from the min/max list
  M_y = maxCell[1] - minCell[1] + 1;
  M_z = maxCell[2] - minCell[2] + 1;
  cout << "Inlet Dimen: " << M_x << " " << M_y << " " << M_z << endl;
  
  int jSize = 0;
  int kSize = 0; //will change based on face
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
  
  //Aij order A11 A21 A22 A31 A32 A33
  //find transformation matrix
  vector<double> Aij (6,0.0 );
  Aij[0] = sqrt(Rij[0]);
  Aij[1] = Rij[1]/Aij[0];
  Aij[2] = sqrt(Rij[2]-Aij[1]*Aij[1]);
  Aij[3] = Rij[3]/Aij[0];
  Aij[4] = (Rij[4] - Aij[1] * Aij[3]) / Aij[2];
  Aij[5] = sqrt(Rij[5] - Aij[3]*Aij[3] - Aij[4] * Aij[4] );
  
  //_____________________________________________
  // Allocate Variables to be used
  newTurbInlet.jSize = jSize; newTurbInlet.kSize = kSize;
  newTurbInlet.faceSide = faceSide;
  newTurbInlet.constVelocity = aveU;
  newTurbInlet.constStress = Aij;
  
  vector<vector <double> > dProf (jSize, vector<double> (kSize, 0.0) );
  newTurbInlet.distanceProfile = dProf;
  vector< vector<vector <double> > > vProf (jSize, vector<vector<double> > (kSize, vector<double> (3, 0.0) ) );
  newTurbInlet.velocityProfile = vProf;
  vector< vector<vector <double> > > AijProf (jSize, vector<vector<double> > (kSize, vector<double> (6, 0.0) ) );
  newTurbInlet.stressProfile = AijProf;
  vector< vector<vector <double> > > lProf (jSize, vector<vector<double> > (kSize, vector<double> (3, 0.0) ) );
  newTurbInlet.lengthScaleProfile = lProf;
  
  //_________________________________________________
  //Set up normalized distance profile
  BaseDistance* dist;
  if ( faceShape =="circle" ) {
    dist = new circleD(  gridLoPts,  origin,  minCell, Dx, charDim );
  } else if ( faceShape == "box" || faceShape=="rectangle" ) {
    dist = new boxD(gridLoPts, origin, minCell, Dx, charDim, variationDirection );
  } else if (faceShape == "ellipse") { 
    dist = new ellipseD( gridLoPts, origin, minCell, Dx, minorRadius, majorRadius, angle);
  } else if (faceShape == "annulus") { 
    dist = new annulusD( gridLoPts, origin, minCell, Dx, minorRadius, majorRadius);
  } else {
    exit(1);
  }

    
  dist->setDistance( newTurbInlet );
  cout << "Distance Profile set up" << endl;
  
  //set up velcoity profile
  double magVelocity;
  magVelocity = sqrt(aveU[0]*aveU[0] + aveU[1]*aveU[1] + aveU[2]*aveU[2]);
  newTurbInlet.velocityMagnitude = magVelocity;
  cout << "Average Velocity Magnitude: " << magVelocity << endl;
  
  BaseVelocity* vel;
  bool userV = true;
  if (velocityType == "laminar") {
    vel = new laminarV();
  } else if (velocityType == "tanh") {
    vel = new tanhV();
  } else if (velocityType == "power") {
    vel = new powerV();
  } else if (velocityType == "specified") {
    vel = new specifiedV( specDistance, specVelocity, isSymmetric);
  } else if (velocityType == "constant") {
    userV = false;
    vel = new constV();
  } else {
    cout << "Specify a valid velocity profile: constant/laminar/tanh/power/specified" << endl;
    exit(1);
  }
  vel->setVelocity( newTurbInlet );
  cout << "Velocity Profile Set up" << endl;
  
  //set up stress profile
  BaseStress * stress;
  bool userS = true;
  if (stressType == "jet") {
    stress = new jetStress();
  } else if (stressType == "channel" || stressType=="pipe") {
    stress = new channelStress();
  } else if (stressType == "specified") {
    stress = new specifiedS( specDistance, specStress, isSymmetric );
  } else if (stressType == "constant") {
    userS = false;
    stress = new constS();
  } else {
    cout << "Specifiy a valid stress profile: constant/channel/pipe/jet/specified" << endl;
    exit(1);
  }
  stress->setStress(newTurbInlet);
  cout << "Stress Profile set up" << endl;
  
  //set up length scale profile
  BaseLengthScale * length;
  bool userL = true;
  if (lengthScaleType == "channel" || lengthScaleType == "pipe") {
    length = new channelLength();
  } else if (lengthScaleType == "specified" ) {
    //this also adjusts maxL, if needed
    length = new specifiedL( specDistance, specLengthScale, isSymmetric );
  } else if (lengthScaleType == "constant") {
    userL = false;
    length = new constL();
  } else {
    cout << "Specify a valid length scale profile: constant/channel/pipe/jet/specified" << endl;
    exit(1);
  }
  length->setLengthScale(newTurbInlet);
  cout << "Length Scale Profile set up" << endl;
  
  //______________________________________________
  //find sizes to use for filter support, 
  
  vector<double> n_a (3); //multiple of length scale
  vector<int> N_a (3);    //Required support filter
  
  vector<int> filterSize (3);
  //set up lengths of filter coefficients
  for (int i = 0; i<3; i++) {
    n_a[i] = newTurbInlet.maxLengthScale[i]/Dx[i];
    N_a[i] = (int) ceil(n_a[i] * AccuracyCoef);
    filterSize[i] = 2*N_a[i] + 1;
  }
  
  //Print input params
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
  srand( time(NULL) ); //seed generator
  
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
  
  //_____________________________________________________________
  //check if the velocity vector is at an angle from normal plane
  /* rotation matrix
   given velocity vector and plane normal
   find cross product as rotation axis, and angle b/t vectors
   then
   R = ([cos(t)+ux^2*(1-cos(t)), ux*uy*(1-cos(t))-uz*sin(t), ux*uz*(1-cos(t))+uy*sin(t);
   uy*ux*(1-cos(t))+uz*sin(t), cos(t)+uy^2*(1-cos(t)), uy*uz*(1-cos(t))-ux*sin(t);
   uz*ux*(1-cos(t))-uy*sin(t), uz*uy*(1-cos(t))+ux*sin(t), cos(t) + uz^2*(1-cos(t))]);
   if angle  = 0, R = I
   */
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
  testRotation[2] = rotationMatrix[0][2] * tempX + rotationMatrix[1][2]*tempY + rotationMatrix[2][2]*tempZ;
  
  cout << "Rotation Check (should equal input velocity vector if angled): " << testRotation[0] << " " << testRotation[1] << " " << testRotation[2] << endl;
  //END TEST BLOCK
  
  //__________________________________________
  // Calculate the filter coefficients 
  // If length scale variations are present, these differ at each grid pt
  
  vector< vector <vector<double> > > filterCoefficients( filterSize[0], vector< vector<double> > (filterSize[1], vector<double> (filterSize[2])));
  vector< vector <vector<vector<vector <double> > > > > variableFilterCoefficients ( jSize, vector<vector<vector<vector<double> > > > 
                                                                                   ( kSize, vector<vector<vector<double> > >
                                                                                    ( filterSize[0], vector<vector<double> >
                                                                                     ( filterSize[1], vector<double>
                                                                                      ( filterSize[2]) ) ) ) ) ;
  vector< vector<vector <int> > > variableFilterSize (jSize, vector<vector<int> > (kSize, vector<int> (3,0) ) );
  
  if ( !userL ) {    
    for (int i = 0; i<filterSize[0]; i++) {
      for (int j = 0; j<filterSize[1]; j++) {
        for (int k = 0; k<filterSize[2]; k++) {
          filterCoefficients[i][j][k] = BSum(i-N_a[0], n_a[0], N_a[0]) * BSum(j-N_a[1], n_a[1], N_a[1]) * BSum(k-N_a[2], n_a[2], N_a[2]); 
        }
      }
    }
  } else {
    vector< vector<vector <double> > > variable_n_a (jSize, vector<vector<double> > (kSize, vector<double> (3,0) ) );
    vector< vector<vector <int> > > variable_N_a (jSize, vector<vector<int> > (kSize, vector<int> (3,0) ) );
        
    //set up lengths of filter coefficients
    for ( int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++ ) {
        for (int i =0; i<3; i++) {
          variable_n_a[j][k][i] = newTurbInlet.lengthScaleProfile[j][k][i]/Dx[i];
          variable_N_a[j][k][i] = (int) ceil(variable_n_a[j][k][i] * AccuracyCoef);
          variableFilterSize[j][k][i] = 2*variable_N_a[j][k][i] + 1;
        }
      }
    }
 
    for (int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++) {
        //calc filter coeff at each pt
        for (int iprime = 0; iprime < variableFilterSize[j][k][0]; iprime++) {
          for (int jprime = 0; jprime < variableFilterSize[j][k][1]; jprime++) {
            for (int kprime = 0; kprime < variableFilterSize[j][k][2]; kprime++) {
              variableFilterCoefficients[j][k][iprime][jprime][kprime] = BSum(iprime-variable_N_a[j][k][0], variable_n_a[j][k][0], variable_N_a[j][k][0]) *
                                                                         BSum(jprime-variable_N_a[j][k][1], variable_n_a[j][k][1], variable_N_a[j][k][1]) *
                                                                         BSum(kprime-variable_N_a[j][k][2], variable_n_a[j][k][2], variable_N_a[j][k][2]);
            }
          }
        } //end iprime
      }
      cout << "Filter Coefficient done for j = " << j << endl;
    }
  }
   
  double infinity = 1.0e10;
  if (userS) { //check that Aij matrix is not nan/inf
    for (int i = 0; i<6; i++) {
      for (int j = 0; j<jSize; j++) {
        for (int k = 0; k<kSize; k++ ) {
          if (newTurbInlet.stressProfile[j][k][i] != newTurbInlet.stressProfile[j][k][i] || abs(newTurbInlet.stressProfile[j][k][i]) > infinity) 
            newTurbInlet.stressProfile[j][k][i] = 0.0;
        }
      }
    }
  } else {
    for (int i = 0; i<6; i++) {
      if (newTurbInlet.constStress[i] != newTurbInlet.constStress[i] || abs(newTurbInlet.constStress[i]) > infinity) 
        newTurbInlet.constStress[i] = 0.0; 
    }
  }

  //allocate velocity vectors
  vector <vector<double> > uBaseX(jSize, vector<double>(kSize));
  vector <vector<double> > uBaseY(jSize, vector<double>(kSize));
  vector <vector<double> > uBaseZ(jSize, vector<double>(kSize));
  
  vector <vector <vector<double> > > uFluctX( NT+1, vector<vector<double> >(jSize, vector<double>(kSize, 0.0)));
  vector <vector <vector<double> > > uFluctY( NT+1, vector<vector<double> >(jSize, vector<double>(kSize, 0.0)));
  vector <vector <vector<double> > > uFluctZ( NT+1, vector<vector<double> >(jSize, vector<double>(kSize, 0.0))); 

  //some temp vars  
  double bijk;
  double u, v, w;
  double a11, a21, a22, a31, a32, a33;
  int sizeX, sizeY, sizeZ;
  
  for (int t = 0; t<NT+1; t++) {
    //each timestep
    cout << "Writing TimeStep: " << t << endl;
    for (int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++) {
        uBaseX[j][k] = 0.0;
        uBaseY[j][k] = 0.0;
        uBaseZ[j][k] = 0.0;
        
        if ( !userL) {
          sizeX = filterSize[0];
          sizeY = filterSize[1];
          sizeZ = filterSize[2];
        } else {
          sizeX = variableFilterSize[j][k][0];
          sizeY = variableFilterSize[j][k][1];
          sizeZ = variableFilterSize[j][k][2];
        }
        
        for (int iprime = 0; iprime<sizeX; iprime++) {
          for (int jprime = 0; jprime<sizeY; jprime++) {
            for (int kprime = 0; kprime<sizeZ; kprime++) {

              if (!userL) {
                bijk = filterCoefficients[iprime][jprime][kprime];
              } else {
                bijk = variableFilterCoefficients[j][k][iprime][jprime][kprime];
              }
              
              if (faceSide == "x-" || faceSide=="x+") {
                uBaseX[j][k] += bijk * randomFieldx[iprime+t][j+jprime][k+kprime];
                uBaseY[j][k] += bijk * randomFieldy[iprime+t][j+jprime][k+kprime];
                uBaseZ[j][k] += bijk * randomFieldz[iprime+t][j+jprime][k+kprime];
              } else if (faceSide=="y-" || faceSide=="y+") {
                uBaseX[j][k] += bijk * randomFieldx[iprime+j][jprime+t][k+kprime];
                uBaseY[j][k] += bijk * randomFieldy[iprime+j][jprime+t][k+kprime];
                uBaseZ[j][k] += bijk * randomFieldz[iprime+j][jprime+t][k+kprime];
              } else if (faceSide=="z-" || faceSide=="z+") {
                uBaseX[j][k] += bijk * randomFieldx[iprime+j][k+jprime][kprime+t];
                uBaseY[j][k] += bijk * randomFieldy[iprime+j][k+jprime][kprime+t];
                uBaseZ[j][k] += bijk * randomFieldz[iprime+j][k+jprime][kprime+t];
              }
              
            }
          }
        } //end iprime
        
        if (!userV) {
          u = aveU[0];
          v = aveU[1];
          w = aveU[2];
        } else {
          u = newTurbInlet.velocityProfile[j][k][0];
          v = newTurbInlet.velocityProfile[j][k][1];
          w = newTurbInlet.velocityProfile[j][k][2];
        }
        
        if (!userS) {
          a11 = newTurbInlet.constStress[0];
          a21 = newTurbInlet.constStress[1];
          a22 = newTurbInlet.constStress[2];
          a31 = newTurbInlet.constStress[3];
          a32 = newTurbInlet.constStress[4];
          a33 = newTurbInlet.constStress[5];
        } else {
          a11 = newTurbInlet.stressProfile[j][k][0];
          a21 = newTurbInlet.stressProfile[j][k][1];
          a22 = newTurbInlet.stressProfile[j][k][2];
          a31 = newTurbInlet.stressProfile[j][k][3];
          a32 = newTurbInlet.stressProfile[j][k][4];
          a33 = newTurbInlet.stressProfile[j][k][5];
        }

        if (!angleVelocity) {
          uFluctX[t][j][k] = u + a11 * uBaseX[j][k];
          uFluctY[t][j][k] = v + a21 * uBaseX[j][k] + a22*uBaseY[j][k];
          uFluctZ[t][j][k] = w + a31 * uBaseX[j][k] + a32*uBaseY[j][k] + a33*uBaseZ[j][k];
        } else {
          double tempX, tempY, tempZ;
          tempX = u + a11 * uBaseX[j][k];
          tempY = v + a21 * uBaseX[j][k] + a22*uBaseY[j][k];
          tempZ = w + a31 * uBaseX[j][k] + a32*uBaseY[j][k] + a33*uBaseZ[j][k];
          
          uFluctX[t][j][k] = rotationMatrix[0][0] * tempX + rotationMatrix[1][0]*tempY + rotationMatrix[2][0]*tempZ;
          uFluctY[t][j][k] = rotationMatrix[0][1] * tempX + rotationMatrix[1][1]*tempY + rotationMatrix[2][1]*tempZ;
          uFluctZ[t][j][k] = rotationMatrix[0][2] * tempX + rotationMatrix[1][2]*tempY + rotationMatrix[2][2]*tempZ;
        }
        
        //prevent negative flow on velocity normal to the inlet
        if ( faceSide=="x-" && uFluctX[t][j][k] < 0.0) {
          uFluctX[t][j][k] = 0.0;
        } else if ( faceSide=="x+" && uFluctX[t][j][k] > 0.0 ) {
          uFluctX[t][j][k] = 0.0;
        } else if ( faceSide=="y-" && uFluctY[t][j][k] < 0.0 ) {
          uFluctY[t][j][k] = 0.0;
        } else if ( faceSide=="y+" && uFluctY[t][j][k] > 0.0 ) {
          uFluctY[t][j][k] = 0.0;
        } else if ( faceSide=="z-" && uFluctZ[t][j][k] < 0.0 ) {
          uFluctZ[t][j][k] = 0.0;
        } else if ( faceSide=="z+" && uFluctZ[t][j][k] > 0.0 ) {
          uFluctZ[t][j][k] = 0.0; 
        }
        
      }
    } //end j
  } //end t
  
  //open file to write out
  ofstream myfile;
  myfile.open( outputfile.c_str(), ios::out );
  //Some indentifing header info
  myfile << "#Resolution: " << resolution[0] << " " << resolution[1] << " " << resolution[2] << endl;
  myfile << "#Velocity Prof: " << velocityType << endl;
  myfile << "#Stress Prof: " << stressType << endl;
  myfile << "#Length Prof: " << lengthScaleType << endl;
  myfile << "#Ave Length Scale: " << L_a[0] << " " << L_a[1] << " " << L_a[2] << endl;
  
  myfile << "#Table Dimensions" << endl;
  myfile << NT+1 << " " << jSize << " " << kSize << endl; 
  myfile << "#Minimum Cell Index" << endl;
  myfile << minCell[0] << " " << minCell[1] << " " << minCell[2] << endl;
  myfile.precision(15);
  for (int t=0; t<NT+1; t++) {
    for (int j=0; j<jSize; j++) {
      for (int k=0; k<kSize; k++) {
        myfile << t << " " << j << " " << k << "  "; 
        myfile << uFluctX[t][j][k] << "  " << uFluctY[t][j][k] << "  " << uFluctZ[t][j][k] << "\n";
      }
    }
  }
  myfile.close();
  cout << "File written and saved as: " << outputfile << endl;
  
  return 0;
}
