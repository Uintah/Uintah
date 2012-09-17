/*
 *  DigitalFilterInlet.cc
 *  \author Alex Abboud
 *  \date August 2012
 *  \brief implements a turbulent inlet that is derived from a digital filter method
 *  -have a prespecified table with t,j,k points listed with u,v,w vectors
 *  j,k will change based on the normal face (xface y/z, yface x/z, zface x/y)
 *  -create this table with DigitalFilterGenerator prior to simulation
 *  this will effectively convect a volume of fluctuations through the inlet, and repeat
 *  if necessary
 *  -In the future adding a table generation to .ups file might be done,
 *  but dealing with a inlet generation across multiple patches 
 */

#include <CCA/Components/Arches/DigitalFilter/DigitalFilterInlet.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MiscMath.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
using namespace std;

//Constructor
DigitalFilterInlet::DigitalFilterInlet()
{
  t_index = 0; //initialize time index
}

//Destructor
DigitalFilterInlet::~DigitalFilterInlet( ) 
{}

//____________________________________________
//keeps track of time index of the table
//____________________________________________

int DigitalFilterInlet::getTimeIndex( int timestep, double elapTime )
{
  //make sure if end of table is reached, it is repeated
  //work for either a specified time period, or a number of steps
  if (period != 0) {
    while (timestep >= NT*period) {
      timestep -= NT*period;
    }
    t_index = timestep/period;
  
  } else {
    while (elapTime >= NT*timePeriod) {
      elapTime -= NT*timePeriod;
    }
    t_index = (int)(elapTime/timePeriod);
  }
    
  return t_index;
}

//____________________________________________
//Problem Setup - parses table
//____________________________________________
void DigitalFilterInlet::problemSetup( const ProblemSpecP& params )
{
  string fileName;
  string timeIntegrator;
  
  ProblemSpecP bcParams = params;
  bcParams->getWithDefault("period",period,1);
  
  if (bcParams->findBlock("timeperiod") ) {
    bcParams->get("timeperiod",timePeriod);
    period = 0;
  }

  bcParams->get("inputfile",fileName);
  
  gzFile gzFp = gzopen( fileName.c_str(), "r" );
  
  if( gzFp == NULL ) {
    proc0cout << "Error opening file for Turbulent Inlet: " << fileName << endl;
    throw ProblemSetupException("Unable to open the given input file: " + fileName, __FILE__, __LINE__);
  }
  
  NT = getInt(gzFp);
  jSize = getInt(gzFp);
  kSize = getInt(gzFp);

  minC.resize(3);
  minC[0] = getInt(gzFp);
  minC[1] = getInt(gzFp);
  minC[2] = getInt(gzFp);
  
  uFluct.resize(NT); vFluct.resize(NT); wFluct.resize(NT);
  for (int i=0; i<NT; ++i) {
    uFluct[i].resize(jSize); vFluct[i].resize(jSize); wFluct[i].resize(jSize);
    
    for (int j=0; j<jSize; ++j) {
      uFluct[i][j].resize(kSize);  
      vFluct[i][j].resize(kSize);
      wFluct[i][j].resize(kSize);
    }
  }
  
  for (int t = 0; t<NT; t++) {
    for(int j = 0; j<jSize; j++) {
      for (int k = 0; k<kSize; k++) {
        int tt = getInt(gzFp); int jj = getInt(gzFp); int kk = getInt(gzFp);
        tt +=0; jj +=0; kk +=0; //silence compiler warning
        double u = getDouble(gzFp);
        double v = getDouble(gzFp);
        double w = getDouble(gzFp);
        uFluct[t][j][k] = u;
        vFluct[t][j][k] = v;
        wFluct[t][j][k] = w;
      }
    }
  }
  
  if (period != 0) {
    proc0cout << "Digital filter setup period = " << period << endl;
  } else {
    proc0cout << "Digital filter setup period = " << timePeriod << endl;
  }
  proc0cout << "Total Realizations: " << NT << endl;
  proc0cout << "Geom Size " << jSize << " " << kSize << endl;
}

//____________________________________________
//Return Velocity when called by BC code
//____________________________________________
vector<double> DigitalFilterInlet::getVelocityVector( int t, int j, int k )
{
 //need to subtract offset vector based on face side
  
  vector<double> velocity (3, 0);
  if (j>=0 && k >=0 && j<jSize && k<kSize) {
    //make sure within the bounds of inlet (required to check this due to extra cells if inlet is a face side)
    velocity[0] = uFluct[t][j][k];
    velocity[1] = vFluct[t][j][k];
    velocity[2] = wFluct[t][j][k];
  }
  return velocity;
}

//______________________
// find smallest value in the bc ptr to shift the lookup - debug/sanity check
//______________________
void DigitalFilterInlet::findOffsetVector( const Patch* patch, const Patch::FaceType& face, 
                                              Iterator bound_ptr )
{
  vector<int> minCellTest (3);
  IntVector c = *bound_ptr;
  minCellTest[0] = c.x(); 
  minCellTest[1] = c.y();
  minCellTest[2] = c.z();
  
  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
    IntVector c = *bound_ptr;
    
    if (c.x() < minCellTest[0] ) 
      minCellTest[0] = c.x();
    if (c.y() < minCellTest[1] ) 
      minCellTest[1] = c.y();
    if (c.z() < minCellTest[2] ) 
      minCellTest[2] = c.z();
  }
  c = IntVector(minCellTest[0], minCellTest[1], minCellTest[2] );
  cout << "min vec " << c << "  checking boundary cell layout- smallest value should match table header value" << endl;
}

//___________ 
// return the int vector
IntVector DigitalFilterInlet::getOffsetVector( )
{
  IntVector c;
  c = IntVector(minC[0], minC[1], minC[2] );
  return c;
}
