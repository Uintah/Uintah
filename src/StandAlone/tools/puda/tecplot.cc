/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include <StandAlone/tools/puda/tecplot.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>

#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

void
tecplot( DataArchive *   da,
         const bool      tslow_set, 
         const bool      tsup_set,
         unsigned long & time_step_lower,
         unsigned long & time_step_upper,
         bool            do_all_ccvars,
         const string &  ccVarInput,
         const string &  i_xd,
         int             tskip )

{
  string ccVariable;
  bool ccVarFound = false;
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "There are " << vars.size() << " variables:\n";
  
  const Uintah::TypeDescription* td;
  const Uintah::TypeDescription* subtype;
  if(!do_all_ccvars) {
    for(int i=0;i<(int)vars.size();i++){
      cout << vars[i] << ": " << types[i]->getName() << endl;
      if(vars[i] == ccVarInput) {
        ccVarFound = true;
      }
    }
    if(!ccVarFound) {
      cerr << "the input ccVariable for tecplot is not storaged in the Dada Archive" << endl;
      abort();
    }
  } // end of (!do_all_ccvars)

  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for(int i=0;i<(int)index.size();i++) {
    cout << index[i] << ": " << times[i] << endl;
  }
  
  if (!tslow_set)
    time_step_lower =0;
  else if (time_step_lower >= times.size()) {
    cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
    abort();
  }
  if (!tsup_set)
    time_step_upper = times.size()-1;
  else if (time_step_upper >= times.size()) {
    cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
    abort();
  }
              
  for(int i=0;i<(int)vars.size();i++){ //for loop over all the variables: 2
    cout << vars[i] << ": " << types[i]->getName() << endl;
    if(do_all_ccvars || ((!do_all_ccvars) && (vars[i] == ccVarInput))){ // check if do all CCVariables 
      // or just do one variable: 3  
      td = types[i];
      subtype = td->getSubType();
      ccVariable = vars[i];
      switch(td->getType()){ //switch to get data type: 4
    
        //___________________________________________
        //  Curently C C  V A R I A B L E S Only
        //
      case Uintah::TypeDescription::CCVariable:
        { //CCVariable case: 5 
          //    generate the name of the output file;
          string filename;
          string fileroot("tec.");
          string filetype(".dat");
          filename = fileroot + ccVariable;
          filename = filename + filetype;
          ofstream outfile(filename.c_str());
          outfile.setf(ios::scientific,ios::floatfield);
          outfile.precision(20);

          //    print out the Title of the output file according to the subtype of the CCVariables 
		       
          outfile << "TITLE = " << "\"" << ccVariable << " tecplot data file" << "\"" << endl;

          if(i_xd == "i_3d") {
            if(subtype->getType() == Uintah::TypeDescription::double_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
                      << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::float_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
                      << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
                      << "\"" << ccVariable << ".X" << "\", " << "\"" << ccVariable << ".Y" << "\", " 
                      << "\"" << ccVariable << ".Z" << "\"";
            }
            if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", " 
                      << "\"" << ccVariable  << ".1.1\"" << ", \"" << ccVariable << ".1.2\"" << ", \"" << ccVariable << ".1.3\""
                      << ", \"" << ccVariable << ".2.1\"" << ", \"" << ccVariable << ".2.2\"" << ", \"" << ccVariable << ".2.3\""
                      << ", \"" << ccVariable << ".3.1\"" << ", \"" << ccVariable << ".3.2\"" << ", \"" << ccVariable << ".3.3\"";
            }
            outfile << endl;
          } else if(i_xd == "i_2d") {
            if(subtype->getType() == Uintah::TypeDescription::double_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " 
                      << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::float_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " 
                      << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", "
                      << "\"" << ccVariable << ".X" << "\", " << "\"" << ccVariable << ".Y" << "\"";
            }
            if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " 
                      << "\"" << ccVariable  << ".1.1\"" << ", \"" << ccVariable << ".1.2\"" 
                      << ", \"" << ccVariable << ".2.1\"" << ", \"" << ccVariable << ".2.2\""; 
            }
            outfile << endl;
          } else if(i_xd == "i_1d") {
            if(subtype->getType() == Uintah::TypeDescription::double_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::float_type) {
              outfile << "VARIABLES = " << "\"X" << "\", " << "\"" << ccVariable << "\""; 
            }
            if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"" << ccVariable << ".X" << "\" ";
            }
            if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
              outfile << "VARIABLES =" << "\"X" << "\", " << "\"" << ccVariable  << ".1.1\"";
            }
            outfile << endl;
          }

          //loop over the time
          for(unsigned long t=time_step_lower;t<=time_step_upper;t=t+tskip){  //time loop: 6
            double time = times[t];
            cout << "time = " << time << endl;
	
            /////////////////////////////////////////////////////////////////
            // find index ranges for current grid level
            ////////////////////////////////////////////////////////////////

            GridP grid = da->queryGrid(t);
            for(int l=0;l<grid->numLevels();l++){  //level loop: 7
              LevelP level = grid->getLevel(l);
              cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;

              //		  int numNode,numPatch;
              int numMatl;
              int Imax,Jmax,Kmax,Imin,Jmin,Kmin;
              int Irange, Jrange, Krange;
              int indexI, indexJ, indexK;
              IntVector lo,hi;
              numMatl = 0;
              Imax = 0;
              Imin = 0;
              Jmax = 0;
              Jmin = 0;
              Kmax = 0;
              Kmin = 0;
              Irange = 0;
              Jrange = 0;
              Krange = 0;
              for(Level::const_patchIterator iter = level->patchesBegin();
                  iter != level->patchesEnd(); iter++){ // patch loop
                const Patch* patch = *iter;
                lo = patch->getExtraCellLowIndex();
                hi = patch->getExtraCellHighIndex();
                cout << "\t\tPatch: " << patch->getID() << " Over: " << lo << " to " << hi << endl;
                int matlNum = da->queryNumMaterials(patch, t);
                if(numMatl < matlNum) numMatl = matlNum;
                if(Imax < hi.x()) Imax = hi.x();
                if(Jmax < hi.y()) Jmax = hi.y();
                if(Kmax < hi.z()) Kmax = hi.z();
                if(Imin > lo.x()) Imin = lo.x();
                if(Jmin > lo.y()) Jmin = lo.y();
                if(Kmin > lo.z()) Kmin = lo.z();
              } //patch loop
	    
              Irange = Imax - Imin;             
              Jrange = Jmax - Jmin;
              Krange = Kmax - Kmin;

              for(int matlsIndex = 0; matlsIndex < numMatl; matlsIndex++){ //matls loop: 8
                //         write each Zone for diferent material at different time step for all patches for one of the levels
                if((ccVariable != "delP_Dilatate" && ccVariable != "press_equil_CC" && ccVariable != "mach") ||
                   ((ccVariable == "delP_Dilatate" || ccVariable == "press_equil_CC") && matlsIndex == 0) ||
                   (ccVariable == "mach" && ( matlsIndex > 0))) {

                  if(i_xd == "i_3d"){ 
                    outfile << "ZONE T =  " << "\"T:" << time << "," <<"M:" << matlsIndex << "," << "L:" << l << "," << "\"," 
                            << "N = " << Irange*Jrange*Krange << "," << "E = " << (Irange-1)*(Jrange-1)*(Krange-1) << "," 
                            << "F = " << "\"FEPOINT\"" << "," << "ET = " << "\"BRICK\"" << endl;
                  } else if(i_xd == "i_2d") {
                    outfile << "ZONE T =  " <<"\"T:"  << time << "," <<"M:" << matlsIndex << ","  << "L:" << l << "\"," 
                            << "N = " << Irange*Jrange << "," << "E = " << (Irange-1)*(Jrange-1) << "," 
                            << "F = " << "\"FEPOINT\"" << "," << "ET = " << "\"QUADRILATERAL\"" << endl;
                  } else if(i_xd == "i_1d"){
                    outfile << "ZONE T =  " <<"\"T:"  << time << "," <<"M:" << matlsIndex << ","  << "L:" << l << "\","
                            << "I = " << Irange << "," << "F = " << "\"POINT\"" << endl;
                  }

                  SCIRun::Array3<int> nodeIndex(Imax-Imin,Jmax-Jmin,Kmax-Kmin);
                  nodeIndex.initialize(0);

                  int totalNode = 0;
                  ////////////////////////////////////////////
                  // Write values of variable in current Zone
                  // /////////////////////////////////////////

                  for(Level::const_patchIterator iter = level->patchesBegin();
                      iter != level->patchesEnd(); iter++){ // patch loop: 9
                    const Patch* patch = *iter;
                    // get anchor, spacing for current level and patch
                    Point start = level->getAnchor();
                    Vector dx = patch->dCell();
                    IntVector lo = patch->getExtraCellLowIndex();
                    IntVector hi = patch->getExtraCellHighIndex();
                    cout << "\t\tPatch: " << patch->getID() << " Over: " << lo << " to " << hi << endl;
                    ConsecutiveRangeSet matls = da->queryMaterials(ccVariable, patch, t);
                    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
                        matlIter != matls.end(); matlIter++){ //material loop: 10
                      int matl = *matlIter;
                      if(matlsIndex == matl) { // if(matlsIndex == matl): 11
                        switch(subtype->getType()){ //switch to get data subtype: 12
                        case Uintah::TypeDescription::double_type:
                          {
                            CCVariable<double> value;
                            da->query(value, ccVariable, matl, patch, t);
                            if(i_xd == "i_3d") {
                              for(indexK = lo.z(); indexK < hi.z(); ++indexK){
                                for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                    ++totalNode;
                                    nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
                                    IntVector cellIndex(indexI, indexJ, indexK);
                                    outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                            << start.y() + dx.y()*(indexJ + 1) << " "   
                                            << start.z() + dx.z()*(indexK + 1) << " "  
                                            << value[cellIndex] << endl;
                                  }
                                }
                              } 
                            } //(i_xd == "i_3d") 

                            else if(i_xd == "i_2d") {
                              for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                  ++totalNode;
                                  nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
                                  IntVector cellIndex(indexI, indexJ, 0);
                                  outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                          << start.y() + dx.y()*(indexJ + 1) << " "   
                                          << value[cellIndex] << endl;
                                }
                              }
                            } //end of if(i_xd == "i_2d")

                            else if(i_xd == "i_1d") {
                              for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                ++totalNode;
                                nodeIndex(indexI-Imin,0,0) = totalNode;
                                IntVector cellIndex(indexI, 0, 0);
                                outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                        << value[cellIndex] << endl;
                              }
                            }//end of if(i_xd == "i_1d") 
                          }
                        break;
                        case Uintah::TypeDescription::float_type:
                          {
                            CCVariable<float> value;
                            da->query(value, ccVariable, matl, patch, t);
                            if(i_xd == "i_3d") {
                              for(indexK = lo.z(); indexK < hi.z(); ++indexK){
                                for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                    ++totalNode;
                                    nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
                                    IntVector cellIndex(indexI, indexJ, indexK);
                                    outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                            << start.y() + dx.y()*(indexJ + 1) << " "   
                                            << start.z() + dx.z()*(indexK + 1) << " "  
                                            << value[cellIndex] << endl;
                                  }
                                }
                              } 
                            } //(i_xd == "i_3d") 

                            else if(i_xd == "i_2d") {
                              for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                  ++totalNode;
                                  nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
                                  IntVector cellIndex(indexI, indexJ, 0);
                                  outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                          << start.y() + dx.y()*(indexJ + 1) << " "   
                                          << value[cellIndex] << endl;
                                }
                              }
                            } //end of if(i_xd == "i_2d")

                            else if(i_xd == "i_1d") {
                              for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                ++totalNode;
                                nodeIndex(indexI-Imin,0,0) = totalNode;
                                IntVector cellIndex(indexI, 0, 0);
                                outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
                                        << value[cellIndex] << endl;
                              }
                            }//end of if(i_xd == "i_1d") 
                          }
                        break;
                        case Uintah::TypeDescription::Vector:
                          {
                            CCVariable<Vector> value;
                            da->query(value, ccVariable, matl, patch, t);
                            if(i_xd == "i_3d") {
                              for(indexK = lo.z(); indexK < hi.z(); ++indexK){
                                for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                    IntVector cellIndex(indexI, indexJ, indexK);
                                    ++totalNode;
                                    nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
                                    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                            << start.y() + dx.y()*(indexJ + 1) << " "
                                            << start.z() + dx.z()*(indexK + 1) << " " 
                                            << value[cellIndex].x() << " " << value[cellIndex].y() << " "
                                            << value[cellIndex].z() << endl;  
                                  }
                                }
                              }
                            } // end of if(i_xd == "i_3d") 
                            else if(i_xd == "i_2d"){
                              for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                  IntVector cellIndex(indexI, indexJ, 0);
                                  ++totalNode;
                                  nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
                                  outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                          << start.y() + dx.y()*(indexJ + 1) << " "
                                          << value[cellIndex].x() << " " << value[cellIndex].y() << endl;
                                }
                              }
                            } //end of if(i_xd == "i_2d")

                            else if(i_xd == "i_1d") {
                              for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                IntVector cellIndex(indexI, 0, 0);
                                ++totalNode;
                                nodeIndex(indexI-Imin,0,0) = totalNode;
                                outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                        << value[cellIndex].x() << endl;
                              }
                            } //end of if(i_xd == "i_1d")
                          }
                        break;
                        case Uintah::TypeDescription::Point:
                          {
                            CCVariable<Point> value;
                            da->query(value, ccVariable, matl, patch, t);
                            if(i_xd == "i_3d") {
                              for(indexK = lo.z(); indexK < hi.z(); ++indexK){
                                for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                    IntVector cellIndex(indexI, indexJ, indexK);
                                    ++totalNode;
                                    nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
                                    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                            << start.y() + dx.y()*(indexJ + 1) << " "
                                            << start.z() + dx.z()*(indexK + 1) << " " 
                                            << value[cellIndex].x() << " " << value[cellIndex].y() << " "
                                            << value[cellIndex].z() << endl;  
                                  }
                                }
                              }
                            } // end of if(i_xd == "i_3d") 
			   
                            else if(i_xd == "i_2d") {
                              for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                  IntVector cellIndex(indexI, indexJ, 0);
                                  ++totalNode;
                                  nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
                                  outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                          << start.y() + dx.y()*(indexJ + 1) << " "
                                          << value[cellIndex].x() << " " << value[cellIndex].y() << endl;
                                }
                              }
                            } //end of if(i_xd == "i_2d")

                            else if(i_xd == "i_1d") {
                              for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                IntVector cellIndex(indexI, 0, 0);
                                ++totalNode;
                                nodeIndex(indexI-Imin,0,0) = totalNode;
                                outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                        << value[cellIndex].x() << endl;
                              }
                            } //end of if(i_xd == "i_1d")
                          }
                        break;

                        case Uintah::TypeDescription::Matrix3:
                          {
                            CCVariable<Matrix3> value;
                            da->query(value, ccVariable, matl, patch, t);
                            if(i_xd == "i_3d") {
                              for(indexK = lo.z(); indexK < hi.z(); ++indexK){
                                for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                    ++totalNode;
                                    nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
                                    IntVector cellIndex(indexI, indexJ, indexK);
                                    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                            << start.y() + dx.y()*(indexJ + 1) << " " 
                                            << start.z() + dx.z()*(indexK + 1) << " "   
                                            << (value[cellIndex])(0,0) << " " << (value[cellIndex])(0,1) << " " 
                                            << (value[cellIndex])(0,2) << " " 
                                            << (value[cellIndex])(1,0) << " " << (value[cellIndex])(1,1) << " "  
                                            << (value[cellIndex])(1,2) << " " 
                                            << (value[cellIndex])(2,0) << " " << (value[cellIndex])(2,1) << " " 
                                            << (value[cellIndex])(2,2) << endl;  
                                  }
                                }
                              }
                            }//end of if(i_xd == "i_3d")
		      
                            else if(i_xd == "i_2d"){
                              for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
                                for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                  ++totalNode;
                                  nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
                                  IntVector cellIndex(indexI, indexJ, 0);
                                  outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                          << start.y() + dx.y()*(indexJ + 1) << " "  
                                          << (value[cellIndex])(0,0) << " " << (value[cellIndex])(0,1) << " "
                                          << (value[cellIndex])(1,0) << " " << (value[cellIndex])(1,1) << " "
                                          << endl;
                                }
                              }
                            }//end of if(i_xd == "i_2d")

                            else if(i_xd == "i_1d"){
                              for(indexI = lo.x(); indexI < hi.x(); ++indexI){
                                ++totalNode;
                                nodeIndex(indexI-Imin,0,0) = totalNode;
                                IntVector cellIndex(indexI, 0, 0);
                                outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
                                        << (value[cellIndex])(0,0) << endl; 
                              }
                            }//end of if(i_xd == "i_1d") 
                          }
                        break;
                        default:
                          cerr << "CC Variable of unknown type: " << subtype->getName() << endl;
                          break;
                        } //end of switch (subtype->getType()): 12
                      } //end of if(matlsIndex == matl): 11
                    } //end of matls loop: 10
                  } // end of loop over patches: 9
	    
                  //////////////////////////////////////////////////////////////////////////////////////////////
                  // Write connectivity list in current Zone
                  /////////////////////////////////////////////////////////////////////////////////////////////
                  if(i_xd == "i_3d"){
                    for(indexK = Kmin; indexK < Kmax-1; ++indexK){
                      for(indexJ = Jmin; indexJ < Jmax-1; ++indexJ){
                        for(indexI = Imin; indexI < Imax-1; ++indexI){
                          outfile << nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) << " "  
                                  << nodeIndex(indexI+1-Imin,indexJ-Jmin,indexK-Kmin) << " "  
                                  << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
                                  << nodeIndex(indexI-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
                                  << nodeIndex(indexI-Imin,indexJ-Jmin,indexK+1-Kmin) << " "  
                                  << nodeIndex(indexI+1-Imin,indexJ-Jmin,indexK+1-Kmin) << " "  
                                  << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
                                  << nodeIndex(indexI-Imin,indexJ+1-Jmin,indexK+1-Kmin) << endl;
                        } //end of loop over indexI
                      } //end of loop over indexJ
                    } //end of loop over indexK
                  }//end of if(i_xd == "i_3d") 
			
                  else if(i_xd == "i_2d"){
                    for(indexJ = Jmin; indexJ < Jmax-1; ++indexJ){
                      for(indexI = Imin; indexI < Imax-1; ++indexI){
                        outfile << nodeIndex(indexI-Imin,indexJ-Jmin,0) << " "  
                                << nodeIndex(indexI+1-Imin,indexJ-Jmin,0) << " "  
                                << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,0) << " "  
                                << nodeIndex(indexI-Imin,indexJ+1-Jmin,0) << " "  
                                << endl;
                      } //end of loop over indexI
                    } //end of loop over indexJ
                  } //end of if(i_xd == "i_2d") 
                } // end of pressure ccVariable if: 8'9
              } // end of loop over matlsIndex: 8
            } // end of loop over levels: 7
          } // end of loop over times: 6
        }//end of CCVariable case: 5
      break;
      default:
        // for other variables in the future
        break;
      } // end switch( td->getType() ): 4
    } // end of if block (do_all_ccvars || ...): 3
  } // end of loop over variables: 2

} // end tecplot()
