// Crack.h

#ifndef UINTAH_HOMEBREW_CRACK_H
#define UNITAH_HOMEBREW_CRACK_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>

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

    void addComputesAndRequiresCalculateFractureParameters(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CalculateFractureParameters(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresPropagateCracks(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void PropagateCracks(const ProcessorGroup*,
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

     int NJ;                               // rJ = NJ*min_dCell
     double rJ;                            // NJ = rJ/min_dCell
     int mS,iS;                            // matID & crkFrtSegID for saving J-integral

     string d_calFractParameters;   // Flag if calculating fracture parameters
     string d_doCrackPropagation;   // Flag if doing crack propagation
     short calFractParameters;      // Flag if calculating fracture parameters at this step
     short doCrackPropagation;      // Flag if doing crack propagation at this step

     // Data members of cracks
     string crackType[MNM];                // Crack contact type
     double c_mu[MNM];                     // Frcition coefficients
     double separateVol[MNM];              // Critical separate volume
     double contactVol[MNM];               // Critical contact volume

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
     vector<Point>     cx[MNM];             // Crack node position
     vector<IntVector> cElemNodes[MNM];     // Nodes of crack elements
     vector<Vector>    cElemNorm[MNM];      // Crack element normals
     vector<Point>     cFrontSegPts[MNM];   // Crack front segments
     vector<Vector>    cFrontSegNorm[MNM];  // Crack front segment normals
     vector<Vector>    cFrontSegJ[MNM];     // J integral of crack front segments
     vector<Vector>    cFrontSegK[MNM];     // SIF of crack front segments
     vector<int>       cpset[128][MNM];     //crack point sunset, 128=64nodes*2procs/node

     // private methods  
     // Detect if a node is within the whole grid 
     short NodeWithinGrid(const IntVector&,const IntVector&,const IntVector&);
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
     // Find plane equation by three points
     void FindPlaneEquation(const Point&,const Point&,const Point&,
                            double&,double&,double&,double&);
     // Detect if a point is within a triangle
     short PointInTriangle(const Point&,const Point&, 
                          const Point&,const Point&); 
     // Find parameters of J-patch circle
     void FindJPathCircle(const Point&,const Vector&,
              const Vector&,const Vector&,double []);
     // Find intersection between J-patch and crack plane
     void FindIntersectionOfJPathAndCrackPlane(const int&,
                   const double& r,const double [],Point&); 
    // Detect if doing fracture analysis 
     void FindTimeStepForFractureAnalysis(double);

 protected:
     
     MPMLabel* lb;
};

}//End namespace Uintah

#endif  /* __CRACK_H__*/
