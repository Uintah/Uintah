/***** Ball.c *****/
/* Ken Shoemake, 1993 */
#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <stdio.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#define LG_NSEGS 4
#define NSEGS (1<<LG_NSEGS)
#define RIMCOLOR()    RGBcolor(255, 255, 255)
#define FARCOLOR()    RGBcolor(195, 127, 31)
#define NEARCOLOR()   RGBcolor(255, 255, 63)
#define DRAGCOLOR()   RGBcolor(127, 255, 255)
#define RESCOLOR()    RGBcolor(195, 31, 31)

using namespace rtrt;

static HMatrix mId = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
static double otherAxis[][4] = {{-0.48, 0.80, 0.36, 1}};
static Quat qOne (0,0,0,1);

/* Establish reasonable initial values for controller. */
void BallData::Init(void)
{
    int i;
    center = qOne;
    radius = 1.0;
    vDown = vNow = qOne;
    qDown = qNow = qOne;
    for (i=15; i>=0; i--)
	((double *)mNow)[i] = ((double *)mDown)[i] = ((double *)mId)[i];
    showResult = dragging = 0;
    axisSet = NoAxes;
    sets[CameraAxes] = mId[X]; setSizes[CameraAxes] = 3;
    sets[BodyAxes] = mDown[X]; setSizes[BodyAxes] = 3;
    sets[OtherAxes] = otherAxis[X]; setSizes[OtherAxes] = 1;
}

/* Choose a constraint set, or none. */
void BallData::UseSet( AxisSet axisSet)
{
    if (!dragging) axisSet = axisSet;
}

/* Begin drawing arc for all drags combined. */
void BallData::ShowResult(void)
{
    showResult = 1;
}

/* Stop drawing arc for all drags combined. */
void BallData::HideResult(void)
{
    showResult = 0;
}

/* Using vDown, vNow, dragging, and axisSet, compute rotation etc. */
void BallData::Update(void)
{
    int setSize = setSizes[axisSet];
    HVect *set = (HVect *)(sets[axisSet]);
    vFrom = MouseOnSphere(vDown, center, radius);
    vTo = MouseOnSphere(vNow, center, radius);
    if (dragging) {
	if (axisSet!=NoAxes) {
	    vFrom = ConstrainToAxis(vFrom, set[axisIndex]);
	    vTo = ConstrainToAxis(vTo, set[axisIndex]);
	    printf("Constraint!\n");
	}
	qDrag = Qt_FromBallPoints(vFrom, vTo);
	qNow = qDrag*qDown;
    } else {
	if (axisSet!=NoAxes) {
	    axisIndex = NearestConstraintAxis(vTo, set, setSize);
	}
    }
    Qt_ToBallPoints(qDown, &vrFrom, &vrTo);
    (qNow.Conj()).ToMatrix(mNow);
    /* Gives transpose for GL. */
//    qNow.ToMatrix(mNow);
}

/* Return rotation matrix defined by controller use. */
void BallData::Value( HMatrix& mow)
{
    int i;
    for (i=15; i>=0; i--) ((double *)mow)[i] = ((double *)mNow)[i];
}


// assume that qNorm is normalized version of qNow.Conj(), you just need to 
// rotate around its axis by theta
void BallData::SetAngle(double theta)
{
    double c_theta = cos(theta/2.0); 
    double s_theta = sin(theta/2.0);

    qNow = qNorm;

    qNow.x *= s_theta; qNow.y *= s_theta; qNow.z *= s_theta;
    qNow.w = c_theta;
    qNow.ToMatrix(mNow);
}

/* Return rotation matrix defined by qDown only... */
void BallData::DValue( HMatrix& mow)
{
    int i;
    for (i=15; i>=0; i--) ((double *)mow)[i] = ((double *)mDown)[i];
}

/* Begin drag sequence. */
void BallData::BeginDrag(void)
{
    dragging = 1;
    vDown = vNow;
}

/* Stop drag sequence. */
void BallData::EndDrag(void)
{
    int i;
    dragging = 0;
    qDown = qNow;
    for (i=15; i>=0; i--)
	((double *)mDown)[i] = ((double *)mNow)[i];
}

/* Draw the controller with all its arcs. */
void BallData::Draw(void)
{
    double r = radius;
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(-1,1,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glScaled(r,r,r);
    GLUquadricObj *qobj = gluNewQuadric();
    gluQuadricDrawStyle(qobj, GLU_LINE);
    gluQuadricNormals(qobj, GLU_SMOOTH);
    glShadeModel(GL_SMOOTH);

    //set point light
    float light[4] = { 1.0, 1.0, 1.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light);
    float lightpos[4] = { 2.0, 2.0, 10.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
    glEnable(GL_LIGHT0);
    
    glEnable(GL_LIGHTING);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    //glDisable(GL_BLEND);
    //glColor4f(0.8,0.0,0.8,1.0);
    //Ball_DrawResultArc(this);
    //Ball_DrawDragArc(this);

    //block out the back side of the sphere
    glColorMask(0,0,0,0);
    glBegin(GL_POLYGON);
    glVertex3f(-1.0, -1.0, 0.0);
    glVertex3f(-1.0,  1.0, 0.0);
    glVertex3f( 1.0,  1.0, 0.0);
    glVertex3f( 1.0, -1.0, 0.0);
    glEnd();
    glColorMask(-1,-1,-1,-1);


    // now draw a "rotated" coordinate frame

    HMatrix m;
    (qNow.Conj()).ToMatrix(m);
    glLoadMatrixd(&m[0][0]);


    //draw the sphere wire ball controller
    gluSphere(qobj,1.0,20,20);
    glDisable(GL_LIGHTING);

    glLineWidth(4.0);
    glBegin(GL_LINES);
    glColor4f(1.0,0.0,0.0,1.0);
    glVertex3f(0.0,0.0,0.0);
    glVertex3f(1.0,0.0,0.0);
    glColor4f(0.0,1.0,0.0,1.0);
    glVertex3f(0.0,0.0,0.0);
    glVertex3f(0.0,1.0,0.0);
    glColor4f(0.0,0.0,1.0,1.0);
    glVertex3f(0.0,0.0,0.0);
    glVertex3f(0.0,0.0,1.0);
    glEnd();
    glDisable(GL_DEPTH_TEST);
    
    glPopMatrix();
    gluDeleteQuadric(qobj);
    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
}

/* Draw an arc defined by its ends. */
void DrawAnyArc(HVect vFrom, HVect vTo)
{
    int i;
    HVect pts[NSEGS+1];
    double dot;
    pts[0] = vFrom;
    pts[1] = pts[NSEGS] = vTo;
//    printf("%lf %lf %lf %lf -> %lf %lf %lf %lf\n",
//	   vFrom.x,vFrom.y,vFrom.z,vFrom.w,
//	   vTo.x,vTo.y,vTo.z,vTo.w);
	   
    for (i=0; i<LG_NSEGS; i++) pts[1] = V3_Bisect(pts[0], pts[1]);
    dot = 2.0*V3_Dot(pts[0], pts[1]);
    for (i=2; i<NSEGS; i++) {
	pts[i] = V3_Sub(V3_Scale(pts[i-1], dot), pts[i-2]);
    }
    glLineWidth(2.0);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINE_STRIP);
    for (i=0; i<=NSEGS; i++)
	glVertex3dv((double *)&pts[i]);
    glEnd();
    glEnable(GL_LIGHTING);
}

/* Draw the arc of a semi-circle defined by its axis. */
void DrawHalfArc(HVect n)
{
    HVect p, m;
    p.z = 0;
    if (n.z != 1.0) {
	p.x = n.y; p.y = -n.x;
	p = V3_Unit(p);
    } else {
	p.x = 0; p.y = 1;
    }
    m = V3_Cross(p, n);
    DrawAnyArc(p, m);
    DrawAnyArc(m, V3_Negate(p));
}

/* Draw all constraint arcs. */
void Ball_DrawConstraints(BallData *ball)
{
    ConstraintSet set;
    HVect axis;
    int axisI, setSize = ball->setSizes[ball->axisSet];
    if (ball->axisSet==NoAxes) return;
    set = ball->sets[ball->axisSet];
    for (axisI=0; axisI<setSize; axisI++) {
	if (ball->axisIndex!=axisI) {
	    if (ball->dragging) continue;
//	    FARCOLOR();
	} // else NEARCOLOR();
	axis = *(HVect *)&set[4*axisI];
	if (axis.z==1.0) {
//	    circ(0.0, 0.0, 1.0);
	} else {
	    DrawHalfArc(axis);
	}
    }
}

/* Draw "rubber band" arc during dragging. */
void Ball_DrawDragArc(BallData *ball)
{
//    DRAGCOLOR();
    if (ball->dragging) DrawAnyArc(ball->vFrom, ball->vTo);
}

/* Draw arc for result of all drags. */
void Ball_DrawResultArc(BallData *ball)
{
//    RESCOLOR();
    if (ball->showResult) DrawAnyArc(ball->vrFrom, ball->vrTo);
}
