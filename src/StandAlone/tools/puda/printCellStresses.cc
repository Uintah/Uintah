
/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <StandAlone/tools/puda/printCellStresses.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////////
//
// Print ParticleVariable
//
void
Uintah::printCellStresses( DataArchive      * da, 
                           CommandLineFlags & clf,
                           int material_of_interest )                          
{
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
  ASSERTEQ(vars.size(), types.size());

  cout << "There are " << vars.size() << " variables:\n";
  vector<int> index;
  vector<double> times;
  da->queryTimesteps( index, times );
  ASSERTEQ(index.size(), times.size());

  cout << "There are " << index.size() << " timesteps:\n";

  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set,times, clf.time_step_lower, clf.time_step_upper );

  // obtain the desired timesteps
  unsigned long t = 0, start_time, stop_time;

  cout << "Time Step       Value\n";

  for(t = clf.time_step_lower; t <= clf.time_step_upper; t++){
    double time = times[t];
    cout << "    " << t + 1 << "        "  << time << "\n";
  }
  cout << "\n";
  if (t != (clf.time_step_lower +1)){
    cout << "Enter start time-step (1 - " << t << "): ";
    cin >> start_time;
    start_time--;
    cout << "Enter stop  time-step (1 - " << t << "): ";
    cin >> stop_time;
    stop_time--;
  }
  else 
    if(t == (clf.time_step_lower + 1)){
      start_time = t-1;
      stop_time  = t-1;
    }
  // end of timestep acquisition

  for(t=start_time;t<=stop_time;t++){

    double time = times[t];
    cout << "time = " << time << "\n";
    GridP grid = da->queryGrid( t );
    for(int v=0;v<(int)vars.size();v++){
      std::string var = vars[ v ];

      // Only dumps out data if it is variable g.stressFS
      if (var == "g.stressFS"){
        const Uintah::TypeDescription* td = types[v];
        const Uintah::TypeDescription* subtype = td->getSubType();
        cout << "\tVariable: " << var << ", type " << td->getName() << "\n";
        for(int l=0;l<grid->numLevels();l++){
          LevelP level = grid->getLevel(l);
          for( Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++ ){
            const Patch* patch = *iter;
            cout << "\t\tPatch: " << patch->getID() << "\n";
            ConsecutiveRangeSet matls = da->queryMaterials( var, patch, t );

            // Loop over materials:
            for( ConsecutiveRangeSet::iterator matlIter = matls.begin(); matlIter != matls.end(); matlIter++ ){
              int matl = *matlIter;
              if( material_of_interest != -1 && matl != material_of_interest ) {
                continue;
              }

              // dumps header and variable info to file
              ostringstream fnum, pnum, matnum; 
              string filename;
              unsigned long timestepnum=t+1;
              fnum << setw(4) << setfill('0') << timestepnum;
              pnum << setw(4) << setfill('0') << patch->getID();
              matnum << setw(4) << setfill('0') << matl;

              string partroot("stress.t");
              string partextp(".p"); 
              string partextm(".m");

              filename = partroot+fnum.str()+partextp+pnum.str()+partextm+matnum.str();
              ofstream partfile(filename.c_str());
              partfile << "# x, y, z, st11, st12, st13, st21, st22, st23, st31, st32, st33\n";

              cout << "\t\t\tMaterial: " << matl << "\n";
              switch(td->getType()){
              case Uintah::TypeDescription::NCVariable:
                switch(subtype->getType()){
                case Uintah::TypeDescription::Matrix3:{
                  NCVariable<Matrix3> value;
                  da->query( value, var, matl, patch, t );
                  cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex()
                       << " to " << value.getHighIndex() << "\n";
                  IntVector dx(value.getHighIndex()-value.getLowIndex());
                  if(dx.x() && dx.y() && dx.z()){
                    NodeIterator iter = patch->getNodeIterator();
                    for(;!iter.done(); iter++){
                      partfile << (*iter).x() << " " << (*iter).y() << " " << (*iter).z()
                               << " " << (value[*iter])(0,0) << " " << (value[*iter])(0,1) << " " 
                               << (value[*iter])(0,2) << " " << (value[*iter])(1,0) << " "
                               << (value[*iter])(1,1) << " " << (value[*iter])(1,2) << " "
                               << (value[*iter])(2,0) << " " << (value[*iter])(2,1) << " "
                               << (value[*iter])(2,2) << "\n";
                    }
                  }
                }
                  break;
                default:
                  cerr << "No Matrix3 Subclass avaliable." << subtype->getType() << "\n";
                  break;
                }
                break;
              default:
                cerr << "No NC Variables avaliable." << td->getType() << "\n";
                break;
              }
            }
          }
        }
      }
      else
        cout << "No g.stressFS variables avaliable at time " << t << ".\n";
    }
    if (start_time == stop_time)
      t++;   
  }
} // end do_cell_stresses
