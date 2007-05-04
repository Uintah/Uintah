#ifndef UINTAH_MPM_POLYNOMIAL_H
#define UINTAH_MPM_POLYNOMIAL_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Geometry/Point.h>
#include <vector>
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

 private:

   PolynomialData();
   PolynomialData(const PolynomialData&);
   PolynomialData& operator=(const PolynomialData&);
   

   std::vector<std::string> d_fileNames;
   std::vector<std::vector<double> > d_polyData;
   std::vector<double> d_polyRange;

   double d_length;
   Point d_top, d_bottom;
   double d_thetaBegin, d_thetaEnd;
 };

}
   

#endif
