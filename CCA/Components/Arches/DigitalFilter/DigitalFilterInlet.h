/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Uintah_Components_Arches_DigitalFilterInlet_h
#define Uintah_Components_Arches_DigitalFilterInlet_h


#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/LevelP.h>

#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <vector>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

namespace Uintah {
  using namespace SCIRun;
  class ArchesVariables;
  class ArchesConstVariables;
  class CellInformation;
  class VarLabel;
  class PhysicalConstants;
  class Properties;
  class Stream;
  class InletStream;
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class DataWarehouse;
  class TimeIntegratorLabel;
  class IntrusionBC;
  class BoundaryCondition_new; 
  
  class DigitalFilterInlet {
    public:
    //Default Constructor
    DigitalFilterInlet();
    
    // Destructor
    ~DigitalFilterInlet();
    
    // GROUP: Problem Setup:
    // Details here
    void problemSetup( const ProblemSpecP& params );
    
    int getTimeIndex( int timestep, double elapTime);
    
    vector<double> getVelocityVector( int t, int j, int k );
    
    void findOffsetVector(const Patch* patch, const Patch::FaceType& face, 
                                Iterator bound_ptr );
    
    IntVector getOffsetVector( );
    
    private:
    int t_index;         //count time index from input file
    int period;          //steps before incrementing t_index
    double timePeriod;   //time before incrementing t_index
    int NT, jSize, kSize; //time and spatial table dimensions
    
    //final velocities
    vector<vector<vector<double> > > uFluct;
    vector<vector<vector<double> > > vFluct;
    vector<vector<vector<double> > > wFluct;
    
    vector<int> minC; //store indicies of lowest corner value of bounding box aroudn inlet
    
  }; //end class DigitalFilterInlet
  
} //end Uintah namespace
#endif
