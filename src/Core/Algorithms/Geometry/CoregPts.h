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
 *  \author David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   \date Dec 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef CoregPtsAnalytic_h
#define CoregPtsAnalytic_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {

class CoregPts 
{
protected:
  int validTransPtsA_;
  int validTrans_;
  Array1<Point> origPtsA_;
  Array1<Point> origPtsP_;
  Array1<Point> transPtsA_;
  Transform transform_;
public:
  virtual ~CoregPts();
  void setOrigPtsA(Array1<Point> a);
  void setOrigPtsP(Array1<Point> p);
  int getTransPtsA(Array1<Point> &p);
  int getTrans(Transform &t);
  void invalidate();
  virtual int computeTrans()=0;
  virtual int computeTransPtsA();
  int getMisfit(double &misfit);
};

class CoregPtsAnalytic : public CoregPts
{
public:
  virtual ~CoregPtsAnalytic();
  virtual int computeTrans();
};
  
} // End namespace SCIRun

#endif // CoregAnalytic_H
