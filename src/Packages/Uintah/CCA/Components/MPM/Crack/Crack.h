// Crack.h

#ifndef UINTAH_HOMEBREW_CRACK_H
#define UNITAH_HOMEBREW_CRACK_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>

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

#define MNM 10   // maximum number of materials

namespace Uintah {
using namespace SCIRun;
using std::vector;
using std::string;

   class DataWarehouse;
   class MPMLabel;
   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

class Crack
{
 public:

    double d_outputInterval;  

    // Constructor
    Crack(const ProblemSpecP& ps, SimulationStateP& d_sS,
                           MPMLabel* lb,int n8or27);

    // Destructor
     ~Crack();

    // Basic crack methods
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

    void addComputesAndRequiresCrackFrontSegSubset(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CrackFrontSegSubset(const ProcessorGroup*,
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

    void addComputesAndRequiresPropagateCrackFrontNodes(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void PropagateCrackFrontNodes(const ProcessorGroup*,
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

    void addComputesAndRequiresUpdateCrackExtentAndNormals(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void UpdateCrackExtentAndNormals(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

 private:

     // Prevent copying of this class
     // copy constructor
     Crack(const Crack &con);
     Crack& operator=(const Crack &con);

     // Data members
     SimulationStateP d_sharedState;

     int d_8or27;
     int NGP;
     int NGN;
     double d_SMALL_NUM_MPM;

     string GridBCType[Patch::numFaces];  // Boundary condition of the global grid
     Point GLP, GHP;                // the lowest and highest point of the real global grid 
     int NJ;                        // rJ = NJ*min_dCell
     double rJ;                     // NJ = rJ/min_dCell
     int mS;                        // matID of saving J-integral
     double delta;                  // Ratio of crack increment of every time to cell-size
     bool d_useSecondTerm;          // If use the second term to calculate J-integral

     string d_calFractParameters;   // Flag if calculating fracture parameters
     string d_doCrackPropagation;   // Flag if doing crack propagation
     string d_outputCrackResults;   // Flag if outputting crack front geometry
     short calFractParameters;      // Flag if calculating fracture parameters at this step
     short doCrackPropagation;      // Flag if doing crack propagation at this step

     // Data members of cracks
     string crackType[MNM];         // Crack contact type
     double c_mu[MNM];              // Frcition coefficients
     double separateVol[MNM];       // Critical separate volume
     double contactVol[MNM];        // Critical contact volume

     // Crack geometry
       // Quadrilateral segments of cracks
     vector<vector<Point> > rectangles[MNM];
       // Resolution of quadrilateral cracks in 1-2 & 2-3 directions 
     vector<int>  rectN12[MNM],rectN23[MNM];       
       // Sides at crack front
     vector<vector<short> > rectCrackSidesAtFront[MNM];
     
       // Trianglular segements of cracks 
     vector<vector<Point> > triangles[MNM];
       // Resolution of triangular cracks in all sides 
     vector<int>  triNCells[MNM];
       // Sides at crack front
     vector<vector<short> > triCrackSidesAtFront[MNM];

       // Arc segements of cracks
     vector<vector<Point> > arcs[MNM];
       // Resolution of arc cracks on circumference
     vector<int>  arcNCells[MNM];
       // Crack front segment ID
     vector<int> arcCrkFrtSegID[MNM];

       // Elliptical segments of cracks
     vector<vector<Point> > ellipses[MNM];
       // Resolution of arc cracks on circumference
     vector<int>  ellipseNCells[MNM];
       // Crack front segment ID
     vector<int> ellipseCrkFrtSegID[MNM];

       // Partial elliptical segments of cracks
     vector<vector<Point> > pellipses[MNM];
       // Resolution of partial elliptic cracks on circumference 
     vector<int>  pellipseNCells[MNM];
       // Crack front segment ID
     vector<int> pellipseCrkFrtSegID[MNM];
       // Extent of partial elliptic cracks
     vector<string> pellipseExtent[MNM];

       // Crack extent
     Point cmin[MNM],cmax[MNM];  

     // Crack data after mesh  
     int               cnumElems[MNM];      // Number of carck elements
     int               cnumNodes[MNM];      // Number of crack points
     int               cnumFrontSegs[MNM];  // Number of segments at crack front
     double            cDELT0[MNM];         // Average length of crack front segments
     vector<Point>     cx[MNM];             // Crack node position
     vector<IntVector> cElemNodes[MNM];     // Nodes of crack elements
     vector<Vector>    cElemNorm[MNM];      // Crack element normals
     vector<int>       cfSegNodes[MNM];     // Crack front segment nodes
     vector<int>       cfSegNodesT[MNM];    // Crack front nodes after propagation 
     vector<Point>     cfSegPts[MNM];       // Crack front points after propagation
     vector<Vector>    cfSegV3[MNM];        // Tangential direction of crack-front
     vector<Vector>    cfSegNorms[MNM];     // Crack front segment normals
     vector<Vector>    cfSegJ[MNM];         // J integral of crack front segments
     vector<Vector>    cfSegK[MNM];         // SIF of crack front segments
     vector<int>       cnset[MNM][128];     // Crack node subset for each patch
     vector<int>       cfnset[MNM][128];    // Crack-front node index subset for each patch 
     vector<int>       cfsset[MNM][128];    // Crack-front seg subset for each patch
     vector<short>     cfSegNodesInMat[MNM];// Flag if crack front nodes inside material
     vector<short>     cfSegCenterInMat[MNM]; // Flag if crack-front seg center in mat

     // --- private methods ---  

     // Detect if doing fracture analysis
     void FindTimeStepForFractureAnalysis(double);
     // Calculate normal of a triangle
     Vector TriangleNormal(const Point&,const Point&,const Point&);
     // Detect if particle and node in same side of a plane
     short Location(const Point&,const Point&,
                       const Point&,const Point&,const Point&);
     // Compute signed volume of a tetrahedron
     double Volume(const Point&,const Point&,const Point&,const Point&);
     // A private function
     IntVector CellOffset(const Point&, const Point&, Vector);
     // Detect if a line is in another line
     short TwoLinesDuplicate(const Point&,const Point&,const Point&,
                                                       const Point&);
     // Direction consines of two points
     Vector TwoPtsDirCos(const Point&,const Point&);
     // Find the equation of a plane with three points on it
     void FindPlaneEquation(const Point&,const Point&,const Point&,
                            double&,double&,double&,double&);
     // Detect if a point is within a triangle
     short PointInTriangle(const Point&,const Point&, 
                          const Point&,const Point&); 
     // Find parameters of J-patch circle
     void FindJPathCircle(const Point&,const Vector&,
              const Vector&,const Vector&,double []);
     // Find intersection between J-patch and crack plane
     bool FindIntersectionOfJPathAndCrackPlane(const int&,
                   const double& r,const double [],Point&); 
     // Find the segment(s) connected by the node
     void FindSegsFromNode(const int&,const int&, int []);
     // Detect if a node is within or around the real global grid
     short RealGlobalGridContainsNode(const double&,const Point&);

     // Find the intersection between a line and the global grid boundary
     void FindIntersectionLineAndGridBoundary(const Point&, Point&);
     void ApplyBCsForCrackPoints(const Vector&,const Point&,Point&); 
     void OutputCrackFrontResults(const int&); 

     // Read in crack geometries
     void ReadRectangularCracks(const int&,const ProblemSpecP&);
     void ReadTriangularCracks(const int&,const ProblemSpecP&); 
     void ReadArcCracks(const int&,const ProblemSpecP&); 
     void ReadEllipticCracks(const int&,const ProblemSpecP&); 
     void ReadPartialEllipticCracks(const int&,const ProblemSpecP&); 
     void OutputInitialCracks(const int&);

     // Discretize crack plane
     void DiscretizeRectangularCracks(const int&,int&);
     void DiscretizeTriangularCracks(const int&,int&);
     void DiscretizeArcCracks(const int&,int&);
     void DiscretizeEllipticCracks(const int&,int&);
     void DiscretizePartialEllipticCracks(const int&,int&);
     void OutputCrackPlaneMesh(const int&);
     void CheckCrackFrontSegments(const int&);

     // Smooth crack front 
     short SmoothCrackFront(const int&);
     
     // Cubic spline fitting
     short CubicSpline(const int& n, const int& m, const int& n1,
                       double [], double [], double [],
                       int [], double [], const double&);

 protected:
     
     MPMLabel* lb;
};

}//End namespace Uintah

#endif  /* __CRACK_H__*/
