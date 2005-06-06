/********************************************************************************
    Crack.h
    Created by Yajun Guo in 2002-2005.
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

    // Public methods in ReadAndDiscretizeCracks.cc
    void addComputesAndRequiresCrackDiscretization(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;  
    void CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    // Public methods in ParticleNodePairVelocityField.cc
    void addComputesAndRequiresParticleVelocityField(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    // Public methods in CrackSurfaceContact.cc
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

    // Public methods in FractureParametersCalculation.cc
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

    // Public methods in CrackPropagation.cc
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

    // Public methods in MoveCracks.cc
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
    
    // Public methods in UpdateCrackFront.cc
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

    // PRIVATE DATA MEMBERS
    MPI_Comm mpi_crack_comm;
    SimulationStateP d_sharedState;
    MPMFlags* flag;
    int n8or27;
    int NGP;
    int NGN;
    enum {NO=0,YES};               // No (NO=0) or Yes (YES=1)
    enum {R=0, L};                 // Right (R=0) or left (L=1)
    Output* dataArchiver;          // Data archiving information
    string udaDir;                 // Base file directory
    bool outputJ;                  // Output J-integral (false by default) 
    string 
    GridBCType[Patch::numFaces];   // BC types of global grid
    Point GLP, GHP;                // Lowest and highest pt of real global grid 
    int    NJ;                     // rJ = NJ*min_dCell
    double rJ;                     // NJ = rJ/min_dCell
    double rdadx;                  // Ratio of crack incremental to cell-size
    bool useVolumeIntegral;        // Use volume integral in J, "no" by default
    bool saveCrackGeometry;        // Save crack geometry, "yes" by default
    bool smoothCrackFront;         // Smoothe crack-front, "no" by default
    string d_calFractParameters;   // Calculate J or K, "no" by default
    string d_doCrackPropagation;   // Do crack propagation, "no" by default
    short calFractParameters;      // Calculate J or K at this step
    short doCrackPropagation;      // Do crack propagation at this step
    double calFractParasInterval;  // Interval of calculating fracture parameters
                                   // zero by default (every time step) 
    double crackPropInterval;      // Interval of crack propagation
                                   // zero by default (every time step) 

    // Physical parameters of cracks
    vector<string> crackType;      // Crack contact type
    vector<double> cmu;            // Crack surface frcition coefficients

    // Geometrical parameters of crack segments
    vector<vector<vector<Point> > > quads;
    vector<vector<int> >            quadN12,quadN23;       
    vector<vector<vector<short> > > quadCrackSidesAtFront;
    vector<vector<vector<Point> > > cquads;
    vector<vector<int> >            cquadNStraightSides;
    vector<vector<vector<Point> > > cquadPtsSide2;
    vector<vector<vector<Point> > > cquadPtsSide4;
    vector<vector<vector<short> > > cquadCrackSidesAtFront;
    vector<vector<vector<Point> > > triangles;
    vector<vector<int> >            triNCells;
    vector<vector<vector<short> > > triCrackSidesAtFront;
    vector<vector<vector<Point> > > arcs;
    vector<vector<int> >            arcNCells;
    vector<vector<int> >            arcCrkFrtSegID;
    vector<vector<vector<Point> > > ellipses;
    vector<vector<int> >            ellipseNCells;
    vector<vector<int> >            ellipseCrkFrtSegID;
    vector<vector<vector<Point> > > pellipses;
    vector<vector<int> >            pellipseNCells;
    vector<vector<int> >            pellipseCrkFrtSegID;
    vector<vector<string> >         pellipseExtent;
    vector<Point>                   cmin,cmax;  

    // Crack data after mesh  
    vector<double>               css;  // Average length of crack-front segments
    vector<double>               csa;  // Average angle of crack-front segments  
    vector<vector<Point> >        cx;  // Coordinates of crack nodes
    vector<vector<IntVector> >    ce;  // Crack elements
    vector<vector<int> >  cfSegNodes;  // Crack-front nodes 
    vector<vector<double> > cfSegVel;  // Velocity of crack-front nodes
    vector<vector<double> >cfSegTime;  // Time instant of crack propagation
    vector<vector<double> >cfSegDis;   // Crack incremental 
    vector<vector<int> > cfSegPreIdx;  // node[i]=node[preIdx]
    vector<vector<int> > cfSegMinIdx;  // Minimum node-index of the sub-crack
    vector<vector<int> > cfSegMaxIdx;  // Maximum node-index of the sub-crack
    vector<vector<Point> > cfSegPtsT;  // Crack-front points after propagation
    vector<vector<Vector> >  cfSegV1;  // Bi-normals at crack-front nodes
    vector<vector<Vector> >  cfSegV2;  // Outer normals at crack-front nodes
    vector<vector<Vector> >  cfSegV3;  // Tangential normals at crack-front nodes
    vector<vector<Vector> >   cfSegJ;  // J-integral at crack-front nodes
    vector<vector<Vector> >   cfSegK;  // SIF at crack-front nodes
    vector<vector<vector<int> > > cnset;  // Crack-node subset in each patch
    vector<vector<vector<int> > > cfnset; // Crack-front-node index subset
    vector<vector<vector<int> > > cfsset; // Crack-front-seg subset in each patch


    // PRIVATE METHODS
           // Private methods in ReadAndDiscretizeCracks.cc
    void   ReadQuadCracks(const int&,const ProblemSpecP&);
    void   ReadCurvedQuadCracks(const int&,const ProblemSpecP&);
    void   ReadTriangularCracks(const int&,const ProblemSpecP&);
    void   ReadArcCracks(const int&,const ProblemSpecP&);
    void   ReadEllipticCracks(const int&,const ProblemSpecP&);
    void   ReadPartialEllipticCracks(const int&,const ProblemSpecP&);
    void   OutputInitialCrackPlane(const int&);
    void   DiscretizeQuadCracks(const int&,int&);
    void   DiscretizeCurvedQuadCracks(const int&,int&);
    void   DiscretizeTriangularCracks(const int&,int&);
    void   DiscretizeArcCracks(const int&,int&);
    void   DiscretizeEllipticCracks(const int&,int&);
    void   DiscretizePartialEllipticCracks(const int&,int&);
    void   CombineCrackSegments(const int&);
    short  TwoPointsCoincide(const Point&,const Point&);
    short  TwoDoublesEqual(const double&, const double&);
    void   ResetCrackNodes(const int&, const int&, const int&);
    void   ResetCrackElements(const int&, const int&, const int&);
    void   ResetCrackFrontNodes(const int&, const int&, const int&);
    void   ReorderCrackFrontNodes(const int&);
    void   FindCrackFrontNodeIndexes(const int&);   
    short  SmoothCrackFrontAndCalculateNormals(const int& m);    
    short  CubicSpline(const int& n, const int& m, const int& n1,
                       double [], double [], double [],
                       int [], double [], const double&);    
    void   CalculateCrackFrontNormals(const int& m);
    void   FindSegsFromNode(const int&,const int&, int []);
    void   OutputInitialCrackMesh(const int&);    
    short  TwoLinesCoincide(const Point&,const Point&,const Point&,const Point&);
    Vector TwoPtsDirCos(const Point&,const Point&);
    Vector TriangleNormal(const Point&,const Point&,const Point&);
    
           // Private methods in ParticleNodePairVelocityField.cc
    IntVector CellOffset(const Point&, const Point&, Vector);    
    short  ParticleNodeCrackPLaneRelation(const Point&,const Point&,
	 	            const Point&,const Point&,const Point&);	    
    double Volume(const Point&,const Point&,const Point&,const Point&);
    
           // Private methods in FractureParametersCalculation.cc
    void   DetectIfDoingFractureAnalysisAtThisTimeStep(double);
    void   FindJIntegralPath(const Point&,const Vector&,const Vector&,
	  	             const Vector&,double []);    
    bool   FindIntersectionJPathAndCrackPlane(const int&,const double& r,
		                              const double [],Point&);    
    void   FindPlaneEquation(const Point&,const Point&,const Point&,
                             double&,double&,double&,double&);
    short  PointInTriangle(const Point&,const Point&,const Point&,const Point&); 
    void   OutputCrackFrontResults(const int&);    

           // Private methods in CrackPropagation.cc
    void   TrimLineSegmentWithBox(const Point&,Point&,const Point&,const Point&);
    void   PruneCrackFrontAfterPropagation(const int& m,const double& ca);
    
           // Private methods in MoveCracks.cc
    short  PhysicalGlobalGridContainsPoint(const double&,const Point&);
    void   ApplySymmetricBCsToCrackPoints(const Vector&,const Point&,Point&); 

           // Private methods in UpdateCrackFront.cc
    void   OutputCrackGeometry(const int&,const int&);

 protected:
     MPMLabel* lb;
};

}

#endif  /* __CRACK_H__*/
