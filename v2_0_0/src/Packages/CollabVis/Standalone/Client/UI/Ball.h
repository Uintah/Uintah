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

#ifndef _H_Ball
#define _H_Ball

/*
 *  Ball.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */


/* smoosh ken shoemakes stuff into c++, PPS */

#include <UI/BallAux.h>

namespace SemotusVisum {

typedef enum AxisSet{NoAxes, CameraAxes, BodyAxes, OtherAxes, NSets} AxisSet;
typedef double *ConstraintSet;
class BallData {
public:
  BallData() { Init(); }
  void Init(void);
  void Place(HVect cntr, double rad) {
    center = cntr;
    radius = rad;
  }
  void Mouse(HVect vnow) { vNow = vnow; }
  void UseSet(AxisSet);
  void ShowResult(void);
  void HideResult(void);
  void Update(void);
  void Value(HMatrix&);
  void BeginDrag(void);
  void EndDrag(void);
  void Draw(void);
  void DValue(HMatrix&); // value at qDown
  void SetAngle(double); // works of normalized value...
  
  HVect center;
  double radius;
  Quat qNow, qDown, qDrag, qNorm;
  HVect vNow, vDown, vFrom, vTo, vrFrom, vrTo;
  HMatrix mNow, mDown;
  int showResult, dragging;
  ConstraintSet sets[NSets];
  int setSizes[NSets];
  AxisSet axisSet;
  int axisIndex;
};

/* Private routines */
void DrawAnyArc(HVect vFrom, HVect vTo);
void DrawHalfArc(HVect n);
void Ball_DrawConstraints(BallData *ball);
void Ball_DrawDragArc(BallData *ball);
void Ball_DrawResultArc(BallData *ball);

} // End namespace SCIRun




#endif

