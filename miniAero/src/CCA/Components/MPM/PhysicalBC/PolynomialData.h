/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef UINTAH_MPM_POLYNOMIAL_H
#define UINTAH_MPM_POLYNOMIAL_H

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Geometry/Point.h>
#include <vector>
#include <deque>
#include <string>

namespace Uintah {

using namespace SCIRun;

/*****

CLASS
   PolynomialData

   Stores polynomial data for MPM


GENERAL INFORMATION

   PolynomialData.h


******/

 class PolynomialData {

 public:

   PolynomialData(ProblemSpecP& ps,const Point& bottom, const Point& top);
   ~PolynomialData();
   void outputProblemSpec(ProblemSpecP& ps);
   void loadData();
   double interpolateRadial(const int polyNum, const double theta);
   double interpolateValue(const Point& test_pt);

   std::string d_endCapName;
   double d_endCapLow, d_endCapHigh;   

 private:

   PolynomialData();
   PolynomialData(const PolynomialData&);
   PolynomialData& operator=(const PolynomialData&);
   

   std::vector<std::string> d_fileNames;
   std::deque<std::vector<double> > d_polyData;
   std::vector<double> d_polyRange;

   double d_length;
   Point d_top, d_bottom;
   double d_thetaBegin, d_thetaEnd;
 };

}
   

#endif
