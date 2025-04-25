/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

////////////////////////////////////////////////////////////////////////////

// USING THE puda -contactStress OPTION

// This code exists to extract the contact pairs at each node, and to output
// the material pair, node position and indices, mass of each material in the
// pair, as well as the color for each material in the pair,

// Usage looks like the following, assuming that one is already inside of
// the uda from which data is to be extracted:

// > /path/to/puda -contactStress -matl M -sepfac X.YZ -timesteplow TL -timestephigh TH .

// where TL and TH are the low and high output timesteps to be analyzed,
// M is the maximum material number with which contacts will be sought, 
// sepfac should be the same value used in the friction contact specification,
// e.g., <separation_factor>0.85</separation_factor>.  So, in the above,
// X.YZ = 0.85

// Output from this will be a series of files of the format contactsWGroupN.YYY
// where N is the material number with which other materials will contact,
// and YYY is the timestep number.

// A header at the top of each file describes the output columns:

// %outputting for time[1] = 1.00025460e+00
// %material_1  material_2 nodePos_x nodePos_y nodePos_z nodeIdxI nodeIdxJ nodeIdxK mass_1 mass_2 color_1 color_2 volume_1 volume_2 gIF1x gIF1y gIF1z gIF2x gIF2y gIF2z

// In addition, the data in that file is consolidated by contact pairs, and is
// placed in files named contactPairs.X.YYY.  There again, according to the
// file header the columns in those files are:

// GrainPair grain1 grain2 ContactArea ContactTraction

////////////////////////////////////////////////////////////////////////////

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
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, num_matls, types);
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
  double sepfac = clf.sepfac;

  string fileroot("contactsWGroup");
  string pout_filename;
  string pairOutRoot("contactPairs.");

  for(int mthis= 0; mthis<maxMatl; mthis++){
    ostringstream gnum;
    gnum << mthis;

    int mat2=-999;
    double val2, val3;

    for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;
                                            t+=clf.time_step_inc){
      double time = times[t];
      cout << "Working on time = " << t << endl;
      int numLines=0;
      double dx=0.;

      // Grid variables stored in std:vectors
      vector<int>    m1, m2, nIx, nIy, nIz;
      vector<double> nPx, nPy, nPz, mass1, mass2;
      vector<double> cl1, cl2, vl1, vl2;
      vector<double> gIFx1, gIFy1, gIFz1, gIFx2, gIFy2, gIFz2;

      ostringstream fnum;
      fnum << setw(3) << setfill('0') << t/clf.time_step_inc;

      string pout_filename;
      string pairOutRoot("contactPairs.");
      pout_filename = pairOutRoot + gnum.str() + "." + fnum.str();
      ofstream pout(pout_filename.c_str());
      if(!pout){
        cerr << "File not opened, exiting" << endl;
        cerr << pout_filename << endl;
        exit(1);
      }
      string filename = fileroot + gnum.str() + '.' + fnum.str();
      ofstream out(filename.c_str());
      out.setf(ios::scientific,ios::floatfield);
      out.precision(8);

      out << "%outputting for time["<< t <<"] = " << time << endl;
      out << "%material_1  material_2 nodePos_x nodePos_y nodePos_z nodeIdxI nodeIdxJ nodeIdxK mass_1 mass_2 color_1 color_2 volume_1 volume_2 gIF1x gIF1y gIF1z gIF2x gIF2y gIF2z" << endl;

      GridP grid = da->queryGrid(t);
      LevelP level = grid->getLevel(grid->numLevels()-1);
      for(Level::const_patch_iterator iter = level->patchesBegin();
                                      iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;

        Vector DX = patch->dCell();
        dx = DX.x();

        // Read in data for the primary material that we're contacting WITH
        NCVariable<double>  massthis, colorthis, volumethis; //, mass_all;
        NCVariable<Point>   posthis;
        NCVariable<Vector>  normthis, IFthis;
        da->query(massthis,   "g.mass",        mthis, patch, t);
        da->query(colorthis,  "g.color",       mthis, patch, t);
        da->query(volumethis, "g.volume",      mthis, patch, t);
        da->query(normthis,   "g.surfnorm",    mthis, patch, t);
        da->query(posthis,    "g.position",    mthis, patch, t);
        da->query(IFthis,     "g.internalforce",mthis,patch, t);

        // Read in data for the other materials
        vector<NCVariable<double> >  massother(maxMatl+1);
        vector<NCVariable<double> >  colorother(maxMatl+1);
        vector<NCVariable<double> >  volumeother(maxMatl+1);
        vector<NCVariable<Matrix3> > stressother(maxMatl+1);
        vector<NCVariable<Point> >   posother(maxMatl+1);
        vector<NCVariable<Vector> >  normother(maxMatl+1),IFother(maxMatl+1);
        for(int mother= mthis+1; mother<=maxMatl; mother++){
            da->query(massother[mother],   "g.mass",          mother, patch, t);
            da->query(colorother[mother],  "g.color",         mother, patch, t);
            da->query(volumeother[mother], "g.volume",        mother, patch, t);
            da->query(normother[mother],   "g.surfnorm",      mother, patch, t);
            da->query(IFother[mother],     "g.internalforce", mother, patch, t);
            da->query(posother[mother],    "g.position",      mother, patch, t);
        } // mother

        double sepDis = sepfac*dx;
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
               Point centerOfMassPos;
               Vector sepvec1, sepvec2;
               double centerOfMassMass=(massthis[c] + massother[mat2][c]);
               centerOfMassPos = (massthis[c] *posthis[c]+
                                  massother[mat2][c]*posother[mat2][c])
                                        /centerOfMassMass;
               sepvec1=(centerOfMassMass/(centerOfMassMass-massthis[c]))*
                         (centerOfMassPos - posthis[c]);
               sepvec2=(centerOfMassMass/(centerOfMassMass-massother[mat2][c]))*
                         (centerOfMassPos - posother[mat2][c]);

               double sepvecL1 = sepvec1.length();
               double sepvecL2 = sepvec2.length();

               if((sepvecL1/sepDis <= 1.0 && sepvecL2/sepDis <= 1.0)){
                 double col1 = colorthis[c];
                 double col2 = colorother[mat2][c];
                 double vol1 = volumethis[c];
                 double vol2 = volumeother[mat2][c];
                 Vector gIF1 = IFthis[c];
                 Vector gIF2 = IFother[mat2][c];
                 out << mthis <<     " " << mat2 << " "
                     << point.x() << " " << point.y() << " " << point.z() << " "
                     << c.x()     << " " << c.y()     << " " << c.z() << " "
                     << massthis[c] << " " << val3    << " " 
                     << col1 << " " << col2    << " " 
                     << vol1 << " " << vol2    << " " 
                     << gIF1.x() << " " << gIF1.y() << " " << gIF1.z() << " "
                     << gIF2.x() << " " << gIF2.y() << " " << gIF2.z() << " "
                     << endl;
                 numLines++;
                 m1.push_back(mthis);
                 m2.push_back(mat2);
                 nPx.push_back(point.x());
                 nPy.push_back(point.y());
                 nPz.push_back(point.z());
                 nIx.push_back(c.x());
                 nIy.push_back(c.x());
                 nIz.push_back(c.x());
                 mass1.push_back(massthis[c]);
                 mass2.push_back(val3);
                 cl1.push_back(col1);
                 cl2.push_back(col2);
                 vl1.push_back(vol1);
                 vl2.push_back(vol2);
                 gIFx1.push_back(gIF1.x());
                 gIFy1.push_back(gIF1.y());
                 gIFz1.push_back(gIF1.z());
                 gIFx2.push_back(gIF2.x());
                 gIFy2.push_back(gIF2.y());
                 gIFz2.push_back(gIF2.z());
               }
              } // if massother
            } // mother
          } // if massthis
        } // cells
      } // patches
      out << endl;

      if(numLines>0){
        // Determine which contact groups are in contact with this material
        vector<int> inContactMatls;
        inContactMatls.push_back(m2[0]);
  
        for(int i=1;i<numLines;i++){
          bool alreadyHaveIt=false;
          for(unsigned int j = 0;j<inContactMatls.size();j++){
            if(m2[i]==inContactMatls[j]){
              alreadyHaveIt=true;
            }
          }
          if(!alreadyHaveIt){
            inContactMatls.push_back(m2[i]);
          }
        }

        pout << "%outputting for time["<< t <<"] = " << time << endl;
        pout << "%Material " << m1[0] << " is in contact with these materials:";
        for(unsigned int j = 0;j<inContactMatls.size();j++){
          pout << " " << inContactMatls[j];
        }
        pout << endl;
        pout << "%GrainPair grain1 grain2 ContactArea ContactTraction" << endl;


        // Find grain contact pairs
        pair <double, double> grainPair;
        vector<pair<double, double> > grainPairs;
        grainPair.first=cl1[0];
        grainPair.second=cl2[0];
        grainPairs.push_back(grainPair);

        set<pair<double, double> > grainPairSet;
        grainPairSet.insert(grainPair);

        for(int i=1;i<numLines;i++){
          bool alreadyHaveIt=false;
          grainPair.first =cl1[i];
          grainPair.second=cl2[i];
          grainPairSet.insert(grainPair);
          for(unsigned int j = 0;j<grainPairs.size();j++){
            if(fabs(grainPair.first - grainPairs[j].first) < 1.e-12 && 
               fabs(grainPair.second- grainPairs[j].second) < 1.e-12){
              alreadyHaveIt=true;
            }
          }
          if(!alreadyHaveIt){
            grainPairs.push_back(grainPair);
          }
        }

        pout.precision(10);

        // Find the average stresses across each contact pair
        vector<double> meanT1x, meanT1y, meanT1z;
        vector<double> meanT2x, meanT2y, meanT2z;
        vector<double> meanTx,  meanTy,  meanTz;
        vector<double> volCP1,  volCP2, volCP;
        vector<double> areaCP;

        for(unsigned int j = 0;j<grainPairs.size();j++){
          meanT1x.push_back(0.);
          meanT1y.push_back(0.);
          meanT1z.push_back(0.);
          meanT2x.push_back(0.);
          meanT2y.push_back(0.);
          meanT2z.push_back(0.);
          meanTx.push_back(0.);
          meanTy.push_back(0.);
          meanTz.push_back(0.);
          volCP1.push_back(0.);
          volCP2.push_back(0.);
          volCP.push_back(0.);
          areaCP.push_back(0.);
          int numNodesThisPair=0;

          for(int i=0;i<numLines;i++){
           pair <double, double> grainPair;
           grainPair.first =cl1[i];
           grainPair.second=cl2[i];
           if(fabs(grainPair.first - grainPairs[j].first) < 1.e-12 && 
              fabs(grainPair.second- grainPairs[j].second) < 1.e-12){
             numNodesThisPair++;
             meanT1x[j]+=gIFx1[i];
             meanT1y[j]+=gIFy1[i];
             meanT1z[j]+=gIFz1[i];
             volCP1[j] +=vl1[i];

             meanT2x[j]+=gIFx2[i];
             meanT2y[j]+=gIFy2[i];
             meanT2z[j]+=gIFz2[i];
             volCP2[j] +=vl2[i];
            } // if the current line's grain pair is the same as the pair
          }  // Loop over lines in the contactStressFile
          volCP[j]  = 0.5*(volCP1[j] + volCP2[j]);
          // volCP/dx is an approximation to contact area
          areaCP[j] = volCP[j]/dx;
          meanTx[j] = 0.5*(fabs(meanT1x[j]) + fabs(meanT2x[j]))/areaCP[j];
          meanTy[j] = 0.5*(fabs(meanT1y[j]) + fabs(meanT2y[j]))/areaCP[j];
          meanTz[j] = 0.5*(fabs(meanT1z[j]) + fabs(meanT2z[j]))/areaCP[j];
  
          pout << j << " "
               << grainPairs[j].first << "  " << grainPairs[j].second << " "
               << areaCP[j] << " "
               << meanTx[j] << " " << meanTy[j] << " " << meanTz[j]  << endl;
        } // Loop over all the contacting grain pairs
      } else {
        pout << "%No materials  of higher material number in contact with this material at this time";
      }
    }  // timesteps
  } // mthis
} // end contactStress
