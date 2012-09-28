/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <StandAlone/tools/puda/POL.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Datatypes/Matrix.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

//////////////////////////////////////////////////////////////////////////
//              P O L (Particle On Line)  O P T I O N
//  Takes an axis and two coordinate values orthogonal to that axis
//   and extracts the specified particle variable along the line.  
//   The next paramter is a boolean that if set to true will instead of
//   average each particle in a cell, will extract 1 particle per cell,
//   for whichever particle it first encounters in the cell.
//   It also has the special function specified by the boolean final 
//   parameter.  If this final parameter is set, the average effective and 
//   hydrostatic pressure of the material will be output per cell.
//

void
Uintah::POL( DataArchive * da, CommandLineFlags & clf, char axis, int ortho1, int ortho2, bool average = true, bool stresses = false)
{
  cout << "Attempting to extract down axis: " << axis << " at point: (" << ortho1 << "," << ortho2 << ") for variable: " << clf.particleVariable << endl;
  // Print out all the variables to console
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++)
    cout << vars[i] << ": " << types[i]->getName() << endl;
     
  // Print the timesteps to console
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
  
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
  

  // get the axis and world extents
  int lineStart = 0;
  int lineEnd   = 0;
 
  // Determine the size of the domain.
  IntVector domainLo, domainHi;

  da->queryGrid(0)->getLevel(0)->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells

  if(axis == 'x')
  {
    lineEnd = domainHi.x();

    if(ortho1 > domainHi.y() || ortho1 < 0)
    {
      cout << "ERROR: Incorrect y specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
    if(ortho2 > domainHi.z() || ortho2 < 0)
    {
      cout << "ERROR: Incorrect z specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
  } else if(axis == 'y') 
  {
    lineEnd = domainHi.y();

    if(ortho1 > domainHi.x() || ortho1 < 0)
    {
      cout << "ERROR: Incorrect x specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
    if(ortho2 > domainHi.z() || ortho2 < 0)
    {
      cout << "ERROR: Incorrect z specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
  } else if(axis == 'z')
  {
    lineEnd = domainHi.z();

    if(ortho1 > domainHi.x() || ortho1 < 0)
    {
      cout << "ERROR: Incorrect x specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
    if(ortho2 > domainHi.y() || ortho2 < 0)
    {
      cout << "ERROR: Incorrect y specified in 'pol' option: " << ortho1 << endl;
      exit(1);
    }
  } else {
    cout << "ERROR: Incorrect axis specified in 'pol' option: " << axis << ".  Must be x, y or z." << endl;
    exit(1);
  } 

  // begin looping through cells
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    
    // set up the output file
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    ostringstream fnum;
    string filename;
    fnum << "axis." << axis << ".at." << ortho1 << "." << ortho2 << ".time." << setw(4) << setfill('0') << t/clf.time_step_inc;
    string partroot;
    if(average)
      partroot = string("particleAverage.");
    else
      partroot = string("particleSingle.");
    filename = partroot + fnum.str();
    ofstream partfile(filename.c_str());


    // Cell Loop
    for(int i = lineStart; i < lineEnd; i++)
    { 
      IntVector cell;
      // advance cell in the specirfied line
      if(axis == 'x')
      {
         cell = IntVector(i, ortho1, ortho2);
      } else if(axis == 'y')
      {
         cell = IntVector(ortho1, i, ortho2);
      } else if(axis == 'z')
      {
         cell = IntVector(ortho1, ortho2, i);
      }
      

      for(int l=0;l<grid->numLevels();l++){
        LevelP level = grid->getLevel(l);

        // Find the patch on which the cell lives
        // don't include extra cells -> false
        const Patch * patch = level->getPatchFromIndex(cell, true);

        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<long64> value_pID;
        ParticleVariable<Point> value_pos;

        // Have different types for different types of output variables
        ParticleVariable<double> valuePoint;
        ParticleVariable<Vector> valueVector;
        ParticleVariable<Matrix3> valueMatrix;
        da->query(value_pos,       "p.x",          matl, patch, t);
        // find all or just the first particle on that patch
        vector<long64> particlesToGrab;
        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            IntVector ci;
            patch->findCell(value_pos[*iter], ci);

            if(ci.x() == cell.x() && ci.y() == cell.y() && ci.z() == cell.z())
            {
               if(average)
               {
                  // save all the particles
                  particlesToGrab.push_back(*iter);
               } else {
                  cout << "Adding particle: " << *iter << endl;
                  particlesToGrab.push_back(*iter);
                  break;
               }
            }
          } // for
        } //if pset > 0
        
        // find variables and output
        if(clf.particleVariable.compare("p.temperature") == 0 ||
           clf.particleVariable.compare("p.mass") == 0)
        {
           da->query(valuePoint,  clf.particleVariable,  matl, patch, t);
           double total = 0.0;

           for(unsigned int i = 0; i < particlesToGrab.size(); i++)
           {
              total += valuePoint[particlesToGrab[i]];
           }

           if(particlesToGrab.size() > 0)
             partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << total / particlesToGrab.size() << endl;
           else 
             partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << 0 << endl;
             
        } else if( clf.particleVariable.compare("p.velocity") == 0) {
           da->query(valueVector,  clf.particleVariable,  matl, patch, t);
           Vector total = Vector(0.0,0.0,0.0);

           for(unsigned int i = 0; i < particlesToGrab.size(); i++)
           {
              total += valueVector[particlesToGrab[i]];
           }

           if(particlesToGrab.size() > 0)
             partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << total / particlesToGrab.size() << endl;
           else 
             partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << 0 << endl;

        } else if( clf.particleVariable.compare("p.stress") == 0) {
           da->query(valueMatrix,  clf.particleVariable,  matl, patch, t);
           Matrix3 total = Matrix3(0.0);

           for(unsigned int i = 0; i < particlesToGrab.size(); i++)
           {
              total += valueMatrix[particlesToGrab[i]];
           }

           if(stresses)
           {
              Matrix3 one; 
              one.Identity();
              Matrix3 Mdev = total/particlesToGrab.size() - one*((total/particlesToGrab.size()).Trace()/3.0);
              double eq_stress=sqrt(Mdev.NormSquared()*1.5);

             if(particlesToGrab.size() > 0)
               partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << -1./3.0*(total / particlesToGrab.size()).Trace() << "\t" << eq_stress << endl;
             else 
               partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << 0 << "\t" << 0 << endl;
           } else {
             if(particlesToGrab.size() > 0)
               partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << total / particlesToGrab.size() << endl;
             else 
               partfile << cell.x() << "\t" << cell.y() << "\t" << cell.z() << "\t" << 0 << endl;
           }
        }

      } // for levels
    } // for cells
  }
} // end POL()
