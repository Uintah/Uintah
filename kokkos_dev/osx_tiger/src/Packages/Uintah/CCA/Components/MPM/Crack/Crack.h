/********************************************************************************
    Crack.h
    Created by Yajun Guo in 2002-2004.
********************************************************************************/
#ifndef UINTAH_HOMEBREW_CRACK_H
#define UNITAH_HOMEBREW_CRACK_H

#include <iomanip>
#include <iostream>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

namespace Uintah {
   using namespace SCIRun;
   using std::vector;
   using std::string;

   class DataWarehouse;
   class MPMLabel;
   class MPMFlags;
   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

class Crack
{
 public:
    // Constructor
    Crack(const ProblemSpecP& ps, SimulationStateP& d_sS,
                           Output* dataArchiver,
                           MPMLabel* lb,MPMFlags* MFlag);

    // Destructor
     ~Crack();

    // Crack methods
    void addComputesAndRequiresCrackDiscretization(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;  
    void CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresParticleVelocityField(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresAdjustCrackContactInterpolated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void AdjustCrackContactInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresAdjustCrackContactIntegrated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void AdjustCrackContactIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresGetNodalSolutions(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void GetNodalSolutions(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresCalculateFractureParameters(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CalculateFractureParameters(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresPropagateCrackFrontPoints(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void PropagateCrackFrontPoints(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresConstructNewCrackFrontElems(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void ConstructNewCrackFrontElems(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresCrackPointSubset(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CrackPointSubset(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresMoveCracks(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void MoveCracks(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresCrackFrontNodeSubset(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CrackFrontNodeSubset(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresRecollectCrackFrontSegments(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void RecollectCrackFrontSegments(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

 private:

    /************************ PRIVATE DATA MEMBERS *************************/
    MPI_Comm mpi_crack_comm;

    SimulationStateP d_sharedState;
    MPMFlags* flag;
    int n8or27;
    int NGP;
    int NGN;
    
    enum {NO=0,YES};               // No (NO=0) or Yes (YES=1)
    enum {R=0, L};                 // Right (R=0) or left (L=1)
    
    Output* dataArchiver;          // data archiving information
    string udaDir;                 // Base file directory
    bool outputJ;                  // Output J-integral (false by default) 
                                  
    string 
    GridBCType[Patch::numFaces];   // BC types of global grid
    Point GLP, GHP;                // Lowest and highest pt of real global grid 
    int    NJ;                     // rJ = NJ*min_dCell
    double rJ;                     // NJ = rJ/min_dCell
    double rdadx;                  // Ratio of crack growth to cell-size
    bool useVolumeIntegral;        // Flag if use volume intergarl in J calculation,
                                   // "no" by default
    bool saveCrackGeometry;        // Flag if save crack geometry for visulizatio,n
                                   // "yes" by default
    bool smoothCrackFront;         // Flag if smooth crack-front by cubic-spline,
                                   // "no" by default
    string d_calFractParameters;   // Flag if calculating fracture parameters
                                   // "no" by default
    string d_doCrackPropagation;   // Flag if doing crack propagation
                                   // "no" by default
    short calFractParameters;      // Flag if calculating fract para at this step
    short doCrackPropagation;      // Flag if doing crack propagation at this step
    double calFractParasInterval;  // Interval of calculating fracture parameters
                                   // zero by default (every time step) 
    double crackPropInterval;      // Interval of crack propagation
                                   // zero by default (every time step) 

    // Data of crack contact
    vector<string> crackType;      // Crack contact type
    vector<double> cmu;            // Crack surface frcition coefficients
    vector<double> separateVol;    // Critical separate volume
    vector<double> contactVol;     // Critical contact volume

    // Crack geometry
    // Quadrilateral segments of cracks
    vector<vector<vector<Point> > > rectangles;
    // Resolutions of quad cracks along p1-p2 and p2-p3 
    vector<vector<int> >  rectN12,rectN23;       
    // Sides at crack front
    vector<vector<vector<short> > > rectCrackSidesAtFront;
    
    // Trianglular segements of cracks 
    vector<vector<vector<Point> > > triangles;
    // Resolution of triangular cracks in all sides 
    vector<vector<int> > triNCells;
    // Sides at crack front
    vector<vector<vector<short> > > triCrackSidesAtFront;

    // Arc segements of cracks
    vector<vector<vector<Point> > > arcs;
    // Resolution of arc cracks on circumference
    vector<vector<int> > arcNCells;
    // Crack front segment ID
    vector<vector<int> > arcCrkFrtSegID;

    // Elliptical segments of cracks
    vector<vector<vector<Point> > > ellipses;
    // Resolution of arc cracks on circumference
    vector<vector<int> > ellipseNCells;
    // Crack front segment ID
    vector<vector<int> > ellipseCrkFrtSegID;

    // Partial elliptical segments of cracks
    vector<vector<vector<Point> > > pellipses;
    // Resolution of partial elliptic cracks on circumference 
    vector<vector<int> > pellipseNCells;
    // Crack front segment ID
    vector<vector<int> > pellipseCrkFrtSegID;
    // Extent of partial elliptic cracks
    vector<vector<string> > pellipseExtent;

    // Crack extent
    vector<Point> cmin,cmax;  

    // Crack data after mesh  
    vector<double>               css;     // Average length of crack-front segs
    vector<double>               csa;     // Average angle between adjacent crack-front segs  
    vector<vector<Point> >        cx;     // Crack points
    vector<vector<IntVector> >    ce;     // Crack elems
    vector<vector<int> >  cfSegNodes;     // Crack-front nodes, stored segment by segment 
    vector<vector<double> > cfSegVel;     // Velocity of crack-front nodes
    vector<vector<double> >cfSegTime;     // Time instant of crack propagation
    vector<vector<double> >cfSegDis;      // Crack incremental 
    vector<vector<int> > cfSegPreIdx;     // node[i]=node[preIdx]
    vector<vector<int> > cfSegMinIdx;     // Min node-index of the sub-crack
    vector<vector<int> > cfSegMaxIdx;     // Max node-index of the sub-crack
    vector<vector<Point> > cfSegPtsT;     // Crack-front points after propagation
    vector<vector<Vector> >  cfSegV1;     // Bi-normals at crack-front nodes
    vector<vector<Vector> >  cfSegV2;     // Normals at crack-front nodes
    vector<vector<Vector> >  cfSegV3;     // Tangential normals at crack-front nodes
    vector<vector<Vector> >   cfSegJ;     // J-integral at crack-front nodes
    vector<vector<Vector> >   cfSegK;     // SIF at crack-front nodes
    vector<vector<vector<int> > > cnset;  // Crack-node subset in each patch
    vector<vector<vector<int> > > cfnset; // Crack-front-node index subset
    vector<vector<vector<int> > > cfsset; // Crack-front-seg subset in each patch


    /*************************** PRIVATE METHODS ***************************/

    // Detect if doing fracture analysis at this time step
    void DetectIfDoingFractureAnalysisAtThisTimeStep(double);
    
    // Calculate normal of a triangle
    Vector TriangleNormal(const Point&,const Point&,const Point&);

    // Detect position relation between particle, node and crack plane
    short ParticleNodeCrackPLaneRelation(const Point&,const Point&,const 
                                      Point&,const Point&,const Point&);

    // Calculate signed volume of a tetrahedron
    double Volume(const Point&,const Point&,const Point&,const Point&);

    // A private function
    IntVector CellOffset(const Point&, const Point&, Vector);

    // Detect if a line is included in another line
    short TwoLinesDuplicate(const Point&,const Point&,const Point&,const Point&);

    // Direction consine of two points
    Vector TwoPtsDirCos(const Point&,const Point&);

    // Find the equation of a plane which is composed of three points
    void FindPlaneEquation(const Point&,const Point&,const Point&,
                           double&,double&,double&,double&);

    // Detect if a point is within a triangle
    short PointInTriangle(const Point&,const Point&,const Point&,const Point&); 

    // Find the parameters of J-contour circle
    void FindJPathCircle(const Point&,const Vector&,const Vector&,
                         const Vector&,double []);

    // Find intersection between J-contour and crack plane
    bool FindIntersectionOfJPathAndCrackPlane(const int&,const double& r,
                                              const double [],Point&); 

    // Find the segment(s) connected by the node
    void FindSegsFromNode(const int&,const int&, int []);

    // Find the min and max node-indexes of the sub-crack
    void FindCrackFrontNodeIndexes(const int&);

    // Detect if a point is within or around the real global grid
    short PhysicalGlobalGridContainsPoint(const double&,const Point&);

    // Trim line-segment with a box
    void TrimLineSegmentWithBox(const Point&, Point&,
	 	                 const Point&, const Point&);

    // Apply symmetric boundary condition to crack points
    void ApplySymmetricBCsToCrackPoints(const Vector&,const Point&,Point&); 

    // Output crack-front position and fracture parameters
    void OutputCrackFrontResults(const int&); 

    // Read in crack geometries
    void ReadRectangularCracks(const int&,const ProblemSpecP&);
    void ReadTriangularCracks(const int&,const ProblemSpecP&); 
    void ReadArcCracks(const int&,const ProblemSpecP&); 
    void ReadEllipticCracks(const int&,const ProblemSpecP&); 
    void ReadPartialEllipticCracks(const int&,const ProblemSpecP&); 
    void OutputInitialCrackPlane(const int&);

    // Discretize crack plane
    void DiscretizeRectangularCracks(const int&,int&);
    void DiscretizeTriangularCracks(const int&,int&);
    void DiscretizeArcCracks(const int&,int&);
    void DiscretizeEllipticCracks(const int&,int&);
    void DiscretizePartialEllipticCracks(const int&,int&);
    void OutputInitialCrackMesh(const int&);
 
    // Prune crack front points if the angle of the two segments of 
    // any point connected by it is larger than a critical angle (ca)
    void PruneCrackFrontAfterPropagation(const int& m, const double& ca);
    
    // Calculate crack-front normals, tangential normals and bi-normals
    void CalculateCrackFrontNormals(const int& m);

    // Smooth crack-front and then calculate crack-front normals
    short SmoothCrackFrontAndCalculateNormals(const int& m);

    // Cubic-spline fitting function
    short CubicSpline(const int& n, const int& m, const int& n1,
                      double [], double [], double [],
                      int [], double [], const double&);
                                                         
    // Output crack elems and crack points for visualization with scirun
    void OutputCrackGeometry(const int&, const int&);

 protected:
     
     MPMLabel* lb;
};

}//End namespace Uintah

#endif  /* __CRACK_H__*/
