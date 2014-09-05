/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
