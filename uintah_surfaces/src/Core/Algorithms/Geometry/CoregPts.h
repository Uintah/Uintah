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


/*
 *  CoregPtsAnalytic.h
 *
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Dec 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef CoregPtsAnalytic_h
#define CoregPtsAnalytic_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Datatypes/Field.h>
#include <Core/Math/MusilRNG.h>

namespace SCIRun {

class CoregPts 
{
protected:
  int allowScale_;
  int allowRotate_;
  int allowTranslate_;
  int validTransPtsA_;
  int validTrans_;
  Array1<Point> origPtsA_;
  Array1<Point> origPtsP_;
  Array1<Point> transPtsA_;
  Transform transform_;
public:
  virtual ~CoregPts();
  CoregPts(int allowScale=1, int allowRotate=1, int allowTranslate=1);
  void setOrigPtsA(Array1<Point> a);
  void setOrigPtsP(Array1<Point> p);
  int getTransPtsA(Array1<Point> &p);
  int getTrans(Transform &t);
  void invalidate();
  virtual int computeTrans()=0;
  virtual int computeTransPtsA();
  virtual int getMisfit(double &misfit);
};

class CoregPtsAnalytic : public CoregPts
{
public:
  CoregPtsAnalytic(int allowScale=1, int allowRotate=1, int allowTranslate=1);
  virtual ~CoregPtsAnalytic();
  virtual int computeTrans();
};
  
class CoregPtsProcrustes : public CoregPts
{
public:
  CoregPtsProcrustes(int allowScale=1,int allowRotate=1,int allowTranslate=1);
  virtual ~CoregPtsProcrustes();
  virtual int computeTrans();
};
  
class CoregPtsSimplexSearch : public CoregPts
{
private:
  int maxIters_;
  double misfitTol_;
  string trans_;
  int &abort_;
  ScalarFieldInterfaceHandle dField_;
  MusilRNG &mr_;
  Point origPtsCenter_;
  int NDIM_;
  int NSEEDS_;
  Array2<double> params_;
  Array1<double> misfit_;
  void compute_misfit(int idx);
  double simplex_step(Array1<double>&sum, double factor, int worst);
public:
  CoregPtsSimplexSearch(int maxIters, double misfitTol, int &abort,
			ScalarFieldInterfaceHandle dField, 
			MusilRNG &mr,
			int allowScale=1, int allowRotate=1,
			int allowTranslate=1);
  virtual ~CoregPtsSimplexSearch();
  virtual int computeTrans();
  virtual int getMisfit(double &misfit);
};
  
} // End namespace SCIRun

#endif // CoregAnalytic_H
