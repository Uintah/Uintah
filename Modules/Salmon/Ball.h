/***** Ball.h *****/
#ifndef _H_Ball
#define _H_Ball

/* smoosh ken shoemakes stuff into c++, PPS */

#include "BallAux.h"

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
    Bool showResult, dragging;
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
#endif
