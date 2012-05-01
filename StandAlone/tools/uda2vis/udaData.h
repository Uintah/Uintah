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
#include <sstream>
#include <vector>


class PatchInfo {
public:

  //GridType
  enum GridType { UNKNOWN=-1, CC, NC, SFCX, SFCY, SFCZ };

  //str2GridType
  static GridType str2GridType(const std::string &type)
  {
    if (type.find("SFCX")!=std::string::npos) return SFCX;
    if (type.find("SFCY")!=std::string::npos) return SFCY;
    if (type.find("SFCZ")!=std::string::npos) return SFCZ;
    if (type.find("CC")  !=std::string::npos) return CC;
    if (type.find("NC")  !=std::string::npos) return NC;
    return UNKNOWN;
  }

  //getBounds
  void getBounds(int low[3],int high[3],const std::string &typestr) const 
  {
    GridType type=str2GridType(typestr);
    getLow(low,type);
    getHigh(high,type);
  } 

  //getLow
  void getLow(int low[3],GridType type) const
  {
    switch (type)
    {
    case CC:
      low[0]=cc_low[0];
      low[1]=cc_low[1];
      low[2]=cc_low[2];
      break;

    case NC:
      low[0]=nc_low[0];
      low[1]=nc_low[1];
      low[2]=nc_low[2];
      break;

    case SFCX:
      low[0]=sfcx_low[0];
      low[1]=sfcx_low[1];
      low[2]=sfcx_low[2];
      break;

    case SFCY:
      low[0]=sfcy_low[0];
      low[1]=sfcy_low[1];
      low[2]=sfcy_low[2];
      break;

    case SFCZ:
      low[0]=sfcz_low[0];
      low[1]=sfcz_low[1];
      low[2]=sfcz_low[2];
      break;

    default:
      low[0]=low[1]=low[2]=-1E6;
    }
  }

  //getHigh
  void getHigh(int high[3],GridType type) const
  {
    switch (type)
    {
    case CC:
      high[0]=cc_high[0];
      high[1]=cc_high[1];
      high[2]=cc_high[2];
      break;

    case NC:
      high[0]=nc_high[0];
      high[1]=nc_high[1];
      high[2]=nc_high[2];
      break;

    case SFCX:
      high[0]=sfcx_high[0];
      high[1]=sfcx_high[1];
      high[2]=sfcx_high[2];
      break;

    case SFCY:
      high[0]=sfcy_high[0];
      high[1]=sfcy_high[1];
      high[2]=sfcy_high[2];
      break;

    case SFCZ:
      high[0]=sfcz_high[0];
      high[1]=sfcz_high[1];
      high[2]=sfcz_high[2];
      break;

    default:
      high[0]=high[1]=high[2]=-1E6;
    }
  }

  //getProcId
  int getProcId() const
  {
    return proc_id;
  }

  //setProcId
  void setProcId(const int new_proc_id)
  {
    proc_id=new_proc_id;
  }

  //setBounds
  bool setBounds(const int low[3],const int high[3],const std::string &typestr)
  {
    GridType type=str2GridType(typestr);
    switch (type)
    {
    case CC:
      cc_low[0]=low[0];
      cc_low[1]=low[1];
      cc_low[2]=low[2];
      cc_high[0]=high[0];
      cc_high[1]=high[1];
      cc_high[2]=high[2];
      break;

    case NC:
      nc_low[0]=low[0];
      nc_low[1]=low[1];
      nc_low[2]=low[2];
      nc_high[0]=high[0];
      nc_high[1]=high[1];
      nc_high[2]=high[2];
      break;

    case SFCX:
      sfcx_low[0]=low[0];
      sfcx_low[1]=low[1];
      sfcx_low[2]=low[2];
      sfcx_high[0]=high[0];
      sfcx_high[1]=high[1];
      sfcx_high[2]=high[2];
      break;

    case SFCY:
      sfcy_low[0]=low[0];
      sfcy_low[1]=low[1];
      sfcy_low[2]=low[2];
      sfcy_high[0]=high[0];
      sfcy_high[1]=high[1];
      sfcy_high[2]=high[2];
      break;

    case SFCZ:
      sfcz_low[0]=low[0];
      sfcz_low[1]=low[1];
      sfcz_low[2]=low[2];
      sfcz_high[0]=high[0];
      sfcz_high[1]=high[1];
      sfcz_high[2]=high[2];
      break;

    default:
      return false;
    }

    return true;
  }

  //toString
  std::string toString() const
  {
    std::ostringstream str;
    str<<"CC: <"<<cc_low[0]<<","<<cc_low[1]<<","<<cc_low[2]<<"> to <"<<cc_high[0]<<","<<cc_high[1]<<","<<cc_high[2]<<">"<<std::endl;
    str<<"NC: <"<<nc_low[0]<<","<<nc_low[1]<<","<<nc_low[2]<<"> to <"<<nc_high[0]<<","<<nc_high[1]<<","<<nc_high[2]<<">"<<std::endl;
    str<<"SFCX: <"<<sfcx_low[0]<<","<<sfcx_low[1]<<","<<sfcx_low[2]<<"> to <"<<sfcx_high[0]<<","<<sfcx_high[1]<<","<<sfcx_high[2]<<">"<<std::endl;
    str<<"SFCY: <"<<sfcy_low[0]<<","<<sfcy_low[1]<<","<<sfcy_low[2]<<"> to <"<<sfcy_high[0]<<","<<sfcy_high[1]<<","<<sfcy_high[2]<<">"<<std::endl;
    str<<"SFCZ: <"<<sfcz_low[0]<<","<<sfcz_low[1]<<","<<sfcz_low[2]<<"> to <"<<sfcz_high[0]<<","<<sfcz_high[1]<<","<<sfcz_high[2]<<">"<<std::endl;
    return str.str();
  }

private:
  // cell centered indices
  int cc_low[3];
  int cc_high[3];

  // node centered indices
  int nc_low[3];
  int nc_high[3];

  // sfcx indices
  int sfcx_low[3];
  int sfcx_high[3];

  // sfcy centered indices
  int sfcy_low[3];
  int sfcy_high[3];

  // sfcz centered indices
  int sfcz_low[3];
  int sfcz_high[3];

  int proc_id;
};


class LevelInfo {
public:
  std::vector<PatchInfo> patchInfo;
  int refinementRatio[3];
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

