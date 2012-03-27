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


#pragma once

#include <string>
#include <vector>

class PatchInfo {
public:

  void getLow(bool use_nc[3], int out[3]) {
    for (int i=0; i<3; i++)
      out[i] = use_nc[i] ? nc_low[i] : cc_low[i];
  }

  void getHigh(bool use_nc[3], int out[3]) {
    for (int i=0; i<3; i++)
      out[i] = use_nc[i] ? nc_high[i] : cc_high[i];
  }
  
  void getExtraLow(bool use_nc[3], int out[3]) {
    for (int i=0; i<3; i++)
      out[i] = use_nc[i] ? nc_extra_low[i] : cc_extra_low[i];
  }

  void getExtraHigh(bool use_nc[3], int out[3]) {
    for (int i=0; i<3; i++)
      out[i] = use_nc[i] ? nc_extra_high[i] : cc_extra_high[i];
  }
  
  // cell centered indices
  int cc_low[3];
  int cc_high[3];
  int cc_extra_low[3];
  int cc_extra_high[3];

  // node centered indices
  int nc_low[3];
  int nc_high[3];
  int nc_extra_low[3];
  int nc_extra_high[3];

  // sfcx indices
  int sfcx_low[3];
  int sfcx_high[3];
  int sfcx_extra_low[3];
  int sfcx_extra_high[3];

  // sfcy centered indices
  int sfcy_low[3];
  int sfcy_high[3];
  int sfcy_extra_low[3];
  int sfcy_extra_high[3];

  // sfcz centered indices
  int sfcz_low[3];
  int sfcz_high[3];
  int sfcz_extra_low[3];
  int sfcz_extra_high[3];

  int proc_id;
};


class LevelInfo {
public:
  std::vector<PatchInfo> patchInfo;
  int refinementRatio[3];
  int extraCells[3];
  double spacing[3];
  double anchor[3];
  int periodic[3];
};



class VariableInfo {
public:
  std::string name;
  std::string type;
  std::vector<int> materials;
};


class TimeStepInfo {
public:
  std::vector<LevelInfo> levelInfo;
  std::vector<VariableInfo> varInfo;
};




class GridDataRaw {
public:
  // Low and high indexes of the data that was read.
  // They SHOULD match what we're expecting, but may not 
  // if there is a boundary layer for the variable.
  int low[3];
  int high[3];
  int components;

  double *data;
};


class ParticleDataRaw {
public:
  int num;
  int components;
  double *data;
};

