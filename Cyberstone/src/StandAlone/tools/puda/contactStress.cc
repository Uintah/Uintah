/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <StandAlone/tools/puda/contactStress.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////

void
Uintah::contactStress( DataArchive * da, CommandLineFlags & clf )
{
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());

    vector<int> index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    cout << "There are " << index.size() << " timesteps:\n";
    for( int i = 0; i < (int)index.size(); i++ ) {
        cout << index[i] << ": " << times[i] << endl;
    }
    
    findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, 
                             clf.time_step_lower, clf.time_step_upper);

    int maxMatl = clf.matl;
    int m_all = maxMatl+1;

    string fileroot("contactsWGroup");

    for(int mthis= 0; mthis<=maxMatl; mthis++){
      ostringstream fnum;
      fnum << mthis;
//      string filename = fileroot + fnum.str();
//      ofstream out(filename.c_str());
//      out.setf(ios::scientific,ios::floatfield);
//      out.precision(8);

      cout << "mthis = " << mthis << endl;
      int mat2=-999;
      double val2, val3;

      for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;
                                              t+=clf.time_step_inc){
        double time = times[t];
        cout << "Working on time = " << t << endl;

        ostringstream tnum;
        tnum << setw(3) << setfill('0') << t/clf.time_step_inc;
        string filename = fileroot + fnum.str() + '.' + tnum.str();
        ofstream out(filename.c_str());
        out.setf(ios::scientific,ios::floatfield);
        out.precision(8);

        out << "%outputting for time["<< t <<"] = " << time << endl;
        out << "material_1  material_2 nodePos_x nodePos_y nodePos_z nodeIdxI nodeIdxJ nodeIdxK mass_1 mass_2 pressure_1 pressure_2 pressure_N eqStress1 eqStress2 eqStressN color_1 color_2" << endl;
        GridP grid = da->queryGrid(t);
        LevelP level = grid->getLevel(grid->numLevels()-1);
        for(Level::const_patch_iterator iter = level->patchesBegin();
                                        iter != level->patchesEnd(); iter++){
          const Patch* patch = *iter;

          // Read in data for the primary material that we're contacting WITH
          NCVariable<double>  massthis, colorthis; //, mass_all;
          NCVariable<Matrix3> stressthis, stress_all;
          da->query(massthis,   "g.mass",        mthis, patch, t);
          da->query(colorthis,  "g.color",       mthis, patch, t);
          da->query(stressthis, "g.stressFS",    mthis, patch, t);

          // Read in data for the all-in-one material
          da->query(stress_all, "g.stressFS",    m_all, patch, t);
//        da->query(mass_all,   "g.mass",        m_all, patch, t);

          // Read in data for the other materials
          vector<NCVariable<double> > massother(maxMatl+1);
          vector<NCVariable<double> > colorother(maxMatl+1);
          vector<NCVariable<Matrix3> > stressother(maxMatl+1);
          for(int mother= mthis+1; mother<=maxMatl; mother++){
            da->query(massother[mother],   "g.mass",          mother, patch, t);
            da->query(colorother[mother],  "g.color",         mother, patch, t);
            da->query(stressother[mother], "g.stressFS",      mother, patch, t);
          } // mother

          for (NodeIterator iter =patch->getNodeIterator();!iter.done();iter++){
            IntVector c = *iter;

            if(massthis[c]>1.e-6){
              val3=0.0;
              for(int mother= mthis+1; mother<=maxMatl; mother++){
                val2=massother[mother][c];
                if(val2>1.e-6){
                 Point point = level->getNodePosition(c);
                 if(val2>val3){
                   mat2=mother;
                   val3=val2;
                 }
                 Matrix3 sig1=stressthis[c];
                 Matrix3 sig2=stressother[mat2][c];
                 Matrix3 sigN=stress_all[c];
                 double p1 = (-1.0/3.0)*sig1.Trace();
                 double p2 = (-1.0/3.0)*sig2.Trace();
                 double pN = (-1.0/3.0)*sigN.Trace();
                 double col1 = colorthis[c];
                 double col2 = colorother[mat2][c];
                 double eqStr1=sqrt(0.5*((sig1(0,0)-sig1(1,1))*(sig1(0,0)-sig1(1,1))
                                +        (sig1(1,1)-sig1(2,2))*(sig1(1,1)-sig1(2,2))
                                +        (sig1(2,2)-sig1(0,0))*(sig1(2,2)-sig1(0,0))
                                +    6.0*(sig1(0,1)*sig1(0,1) + sig1(1,2)*sig1(1,2)
                                        + sig1(2,0)*sig1(2,0))));
                 double eqStr2=sqrt(0.5*((sig2(0,0)-sig2(1,1))*(sig2(0,0)-sig2(1,1))
                                +        (sig2(1,1)-sig2(2,2))*(sig2(1,1)-sig2(2,2))
                                +        (sig2(2,2)-sig2(0,0))*(sig2(2,2)-sig2(0,0))
                                +    6.0*(sig2(0,1)*sig2(0,1) + sig2(1,2)*sig2(1,2)
                                        + sig2(2,0)*sig2(2,0))));
                 double eqStrN=sqrt(0.5*((sigN(0,0)-sigN(1,1))*(sigN(0,0)-sigN(1,1))
                                +        (sigN(1,1)-sigN(2,2))*(sigN(1,1)-sigN(2,2))
                                +        (sigN(2,2)-sigN(0,0))*(sigN(2,2)-sigN(0,0))
                                +    6.0*(sigN(0,1)*sigN(0,1) + sigN(1,2)*sigN(1,2)
                                        + sigN(2,0)*sigN(2,0))));
                 out << mthis <<     " " << mat2 << " "
                     << point.x() << " " << point.y() << " " << point.z() << " "
                     << c.x()     << " " << c.y()     << " " << c.z() << " "
                     << massthis[c] << " " << val3    << " " 
                     << p1          << " " << p2      << " " << pN     << " "
                     << eqStr1      << " " << eqStr2  << " " << eqStrN << " "
                     << col1 << " " << col2    << " " 
                     << endl;
                } // if massother
              } // mother
            } // if massthis
          } // cells
        } // patches
        out << endl;
      }  // timesteps
    } // mthis
} // end contactStress
