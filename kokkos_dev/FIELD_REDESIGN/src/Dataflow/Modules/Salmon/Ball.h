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

#include "BallAux.h"

namespace PSECommon {
namespace Modules {

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:50  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:08  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//



#endif

