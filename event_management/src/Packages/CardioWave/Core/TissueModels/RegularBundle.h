/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


#ifndef PACKAGES_CARDIOWAVE_CORE_TISSUEMODEL_REGULARBUNDLE_H
#define PACKAGES_CARDIOWAVE_CORE_TISSUEMODEL_REGULARBUNDLE_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace CardioWave {

using namespace SCIRun;

class TissueModel_RegularBundle : public AlgoLibrary {

public:
  TissueModel_RegularBundle(ProgressReporter* pr,int nx,int ny,int nz) :
    numcellsx_(nx), numcellsy_(ny), numcellsz_(nz),
    numelems_x_ics_(4), numelems_x_ecs_(2),
    numelems_y_ics_(4), numelems_y_ecs_(2),
    numelems_z_(8),
    numelems_bath_start_(6),
    numelems_bath_end_(6),
    cell_length_(100e-6),
    cell_crosssection_(300e-12),
    ics_vol_frac_(0.8)
  {
  }
  
  inline void set_numcellsx(int x) { numcellsx_ = x;}
  inline void set_numcellsy(int y) { numcellsy_ = y;}
  inline void set_numcellsz(int z) { numcellsz_ = z;}

  inline void set_numelems_x_ics(int x) { numelems_x_ics_ = x;}
  inline void set_numelems_y_ics(int y) { numelems_y_ics_ = y;}
  inline void set_numelems_x_ecs(int x) { numelems_x_ecs_ = x;}
  inline void set_numelems_y_ecs(int y) { numelems_y_ecs_ = y;}
  inline void set_numelems_z(int z) { numelems_z_ = z;}
  
  inline void set_numelems_bath_start(int z) { numelems_bath_start_ = z;}
  inline void set_numelems_bath_end(int z)   { numelems_bath_end_ = z;}
  
  inline void set_cell_length(double l) { cell_length_ = l;}
  inline void set_cell_crosssection(double l) { cell_crosssection_ = l;}
  
  inline void set_ics_vol_frac(double f) {ics_vol_frac_ = f;}
  inline void set_ecs_vol_frac(double f) {ics_vol_frac_ = 1.0-f;}

  bool create_mesh(FieldHandle& output);

private:
  int numcellsx_;
  int numcellsy_;
  int numcellsz_;
  
  int numelems_x_ics_;
  int numelems_x_ecs_;
  int numelems_y_ics_;
  int numelems_y_ecs_;
  int numelems_z_;
  int numelems_bath_start_;
  int numelems_bath_end_;
  
  double cell_length_;
  double cell_crosssection_;
  double ics_vol_frac_;  
};

}

#endif
