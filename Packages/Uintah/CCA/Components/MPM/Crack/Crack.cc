// Crack.cc 
#include "Crack.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#define MAX_BASIS 27

MPI_Comm mpi_crack_comm;

Crack::Crack(const ProblemSpecP& ps,SimulationStateP& d_sS,
                           MPMLabel* Mlb,int n8or27)
{
  MPI_Comm_dup( MPI_COMM_WORLD, & mpi_crack_comm );

  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  d_8or27=n8or27;
  if(d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(d_8or27==MAX_BASIS){
    NGP=2;
    NGN=2;
  }
  rJ=0.0;          // Radius of J-path circle
  NJ=2;            // Number of cells J-path away from crack tip 
  iS=0;            // Crack front segment ID for saving J integral
  mS=0;            // MatErial ID for saving J integral 
  d_calFractParameters = "false"; // Flag for calculatinf J
  d_doCrackPropagation = "false"; // Flag for doing crack propagation

  // Read in MPM parameters related to fracture analysis
  ProblemSpecP mpm_soln_ps = ps->findBlock("MPM");
  if(mpm_soln_ps) {
     mpm_soln_ps->get("calculate_fracture_parameters", d_calFractParameters);
     mpm_soln_ps->get("do_crack_propagation", d_doCrackPropagation);
     mpm_soln_ps->get("J_radius", rJ);
     mpm_soln_ps->get("save_J_matID", mS);
     mpm_soln_ps->get("save_J_crkFrtSegID", iS);
  }

  // Read in crack parameters, which are placed in element "material" 
  ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

  int m=0; // current material ID 
  for( ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                   mat_ps=mat_ps->findNextBlock("material") ) {

    ProblemSpecP crk_ps=mat_ps->findBlock("crack");
 
    if(crk_ps==0) crackType[m]="NO_CRACK";
 
    if(crk_ps!=0) { 
       /* Read in crack contact type, frictional coefficient if any, and
          critical contact and separate volumes
       */
       crk_ps->require("type",crackType[m]);
     
       /* Default values of critical volumes are set to be -1.0. Don't input them 
          if using displacement criterion for crack contact check
       */   
       separateVol[m] = -1.;
       contactVol[m]  = -1.;
       if(crackType[m]=="frictional" || crackType[m]=="stick") {
          crk_ps->get("separate_volume",separateVol[m]);
          crk_ps->get("contact_volume",contactVol[m]);
          if(crackType[m]=="frictional") 
             crk_ps->require("mu",c_mu[m]);
          else 
             c_mu[m]=0.0;
       }
       else if(crackType[m]=="null") {
          c_mu[m]=0.;
       }
       else {
          cout << "Unkown crack contact type in subroutine Crack::Crack: " 
               << crackType[m] << endl;
          exit(1);
       }
        
       /* Initialize the arries related to cracks
       */
       rectangles[m].clear();
       rectN12[m].clear();
       rectN23[m].clear();
       rectCrackSidesAtFront[m].clear();

       triangles[m].clear();
       triNCells[m].clear();
       triCrackSidesAtFront[m].clear();

       arcs[m].clear();
       arcNCells[m].clear();
       arcCrkFrtSegID[m].clear();

       ellipses[m].clear();
       ellipseNCells[m].clear();
       ellipseCrkFrtSegID[m].clear();

       pellipses[m].clear();
       pellipseExtent[m].clear();
       pellipseNCells[m].clear();
       pellipseCrkFrtSegID[m].clear();

       /* Read in crack geometry
       */
       ProblemSpecP geom_ps=crk_ps->findBlock("crack_segments");
       // 1-- Read in quadrilaterals
       for(ProblemSpecP quad_ps=geom_ps->findBlock("quadrilateral");
          quad_ps!=0; quad_ps=quad_ps->findNextBlock("quadrilateral")) {
          int n12=1,n23=1;
          Point p;   
          vector<Point> thisRectPts;
          vector<short> thisRectCrackSidesAtFront;

          // Four vertices of the quadrilateral
          quad_ps->require("p1",p);
          thisRectPts.push_back(p);
          quad_ps->require("p2",p);
          thisRectPts.push_back(p);
          quad_ps->require("p3",p);
          thisRectPts.push_back(p);
          quad_ps->require("p4",p);
          thisRectPts.push_back(p);
          rectangles[m].push_back(thisRectPts);
          thisRectPts.clear();
          
          // Mesh resolution  
          quad_ps->get("resolution_p1_p2",n12); 
          rectN12[m].push_back(n12);
          quad_ps->get("resolution_p2_p3",n23);
          rectN23[m].push_back(n23);

          // Crack front
          short Side;
          string cfsides;
          quad_ps->get("crack_front_sides",cfsides);
          if(cfsides.length()==4) {
            for(string::const_iterator iter=cfsides.begin();
                                     iter!=cfsides.end(); iter++) {
              if( *iter=='Y' || *iter=='y')
                Side=1;
              else if(*iter=='N' || *iter=='n')
                Side=0;
              else { 
                cout << " Wrong specification for crack_front_sides." << endl;
                exit(1);
              }
              thisRectCrackSidesAtFront.push_back(Side);
            }
          }
          else if(cfsides.length()==0) {
            thisRectCrackSidesAtFront.push_back(0);
            thisRectCrackSidesAtFront.push_back(0);
            thisRectCrackSidesAtFront.push_back(0);
            thisRectCrackSidesAtFront.push_back(0);
          } 
          else { 
            cout << " The length of string crack_front_sides for "
                 << "quadrilaterals should be 4." << endl;
            exit(1);
          }
          rectCrackSidesAtFront[m].push_back(thisRectCrackSidesAtFront);
          thisRectCrackSidesAtFront.clear();
       } // End of quadrilateral
 
       // 2-- Read in triangles         
       for(ProblemSpecP tri_ps=geom_ps->findBlock("triangle");
             tri_ps!=0; tri_ps=tri_ps->findNextBlock("triangle")) {
          int n=1;
          Point p;
          vector<Point> thisTriPts;
          vector<short> thisTriCrackSidesAtFront;

          // Three vertices of the triangle
          tri_ps->require("p1",p);
          thisTriPts.push_back(p);
          tri_ps->require("p2",p);
          thisTriPts.push_back(p);
          tri_ps->require("p3",p);
          thisTriPts.push_back(p);
          triangles[m].push_back(thisTriPts);
          thisTriPts.clear();
          tri_ps->get("resolution",n);
          triNCells[m].push_back(n);

          // Crack front
          short Side;
          string cfsides;
          tri_ps->get("crack_front_sides",cfsides);
          if(cfsides.length()==3) {
            for(string::const_iterator iter=cfsides.begin();
                                     iter!=cfsides.end(); iter++) {
              if( *iter=='Y' || *iter=='y')
                Side=1;
              else if(*iter=='N' || *iter=='n')
                Side=0;
              else {
                cout << " Wrong specification for crack_front_sides." << endl;
                exit(1);
              }
              thisTriCrackSidesAtFront.push_back(Side);
            }
          }
          else if(cfsides.length()==0) {
            thisTriCrackSidesAtFront.push_back(0);
            thisTriCrackSidesAtFront.push_back(0);
            thisTriCrackSidesAtFront.push_back(0);
          }
          else {
            cout << " The length of string crack_front_sides for"
                 << " triangles should be 3." << endl;
            exit(1);
          }
          triCrackSidesAtFront[m].push_back(thisTriCrackSidesAtFront);
          thisTriCrackSidesAtFront.clear();
       } // End of triangles

       // 3-- Read in arcs
       for(ProblemSpecP arc_ps=geom_ps->findBlock("arc");
             arc_ps!=0; arc_ps=arc_ps->findNextBlock("arc")) {
          int n;          
          Point p; 
          int cfsID=9999;   // All segments by default
          vector<Point> thisArcPts;

          // Three points on the arc
          arc_ps->require("start_point",p);
          thisArcPts.push_back(p);
          arc_ps->require("middle_point",p);
          thisArcPts.push_back(p);
          arc_ps->require("end_point",p);
          thisArcPts.push_back(p);

          // Resolution on circumference 
          arc_ps->require("resolution_circumference",n);
          arcNCells[m].push_back(n);
          arcs[m].push_back(thisArcPts);
          thisArcPts.clear();
      
          // Crack front segment ID
          arc_ps->get("crack_front_segment_ID",cfsID); 
          arcCrkFrtSegID[m].push_back(cfsID);
       } // End of arc 

       // 4-- Read in ellipses
       for(ProblemSpecP ellipse_ps=geom_ps->findBlock("ellipse");
           ellipse_ps!=0; ellipse_ps=ellipse_ps->findNextBlock("ellipse")) {
          int n;
          Point p;
          int cfsID=9999;  // All segments by default
          vector<Point> thisEllipsePts;

          // Three points on the arc
          ellipse_ps->require("point1_axis1",p);
          thisEllipsePts.push_back(p);
          ellipse_ps->require("point_axis2",p);
          thisEllipsePts.push_back(p);
          ellipse_ps->require("point2_axis1",p);
          thisEllipsePts.push_back(p);

          // Resolution on circumference
          ellipse_ps->require("resolution_circumference",n);
          ellipseNCells[m].push_back(n);
          ellipses[m].push_back(thisEllipsePts);
          thisEllipsePts.clear();

          // Crack front segment ID
          ellipse_ps->get("crack_front_segment_ID",cfsID);
          ellipseCrkFrtSegID[m].push_back(cfsID);
       } // End of ellipses
 
       // 5-- Read in partial ellipses
       for(ProblemSpecP pellipse_ps=geom_ps->findBlock("partial_ellipse");
           pellipse_ps!=0; pellipse_ps=
           pellipse_ps->findNextBlock("partial_ellipse")) {
          int n;
          Point p;
          string Extent;
          int cfsID=9999;  // All segments by default
          vector<Point> thispEllipsePts;

          // Center,two points on major and minor axes
          pellipse_ps->require("center",p);
          thispEllipsePts.push_back(p);
          pellipse_ps->require("point_axis1",p);
          thispEllipsePts.push_back(p);
          pellipse_ps->require("point_axis2",p);
          thispEllipsePts.push_back(p);
          pellipses[m].push_back(thispEllipsePts);
          thispEllipsePts.clear();

          // Extent of the partial ellipse (quarter or half) 
          pellipse_ps->require("extent",Extent);
          pellipseExtent[m].push_back(Extent);

          // Resolution on circumference
          pellipse_ps->require("resolution_circumference",n);
          pellipseNCells[m].push_back(n);

          // Crack front segment ID
          pellipse_ps->require("crack_front_segment_ID",cfsID);              
          pellipseCrkFrtSegID[m].push_back(cfsID);

       } // End of ellipses

    } // End of if crk_ps != 0

    m++; // Next material
  }  // End of loop over materials

  int numMatls=m;  // Total number of materials

#if 1  // Output crack parameters
  int rank,rank_size;
  MPI_Comm_size(mpi_crack_comm, &rank_size);
  MPI_Comm_rank(mpi_crack_comm, &rank);
  if(rank==rank_size-1) { //output from the last rank 
    cout << "*** Crack information output from rank " 
         << rank << " ***" << endl;
    for(m=0; m<numMatls; m++) {
       if(crackType[m]=="NO_CRACK") 
          cout << "\nMaterial " << m << ": no crack exists" << endl;
       else {
          cout << "\nMaterial " << m << ": crack contact type -- " 
               << "\'" << crackType[m] << "\'" << endl;

          if(crackType[m]=="frictional") 
            cout << "            frictional coefficient: " << c_mu[m] << endl;
          else 
            cout << endl;

          if(separateVol[m]<0. || contactVol[m]<0.) 
            cout  << "Check crack contact by displacement criterion" << endl;
          else 
            cout  << "Check crack contact by volume criterion with\n"
                  << "            separate volume = " << separateVol[m]
                  << "\n            contact volume = " << contactVol[m] << endl;
      
          cout <<"\nCrack geometry:" << endl;
          for(int i=0;i<(int)rectangles[m].size();i++) {
             cout << "Rectangle " << i+1 << ": meshed by [" << rectN12[m][i] 
                  << ", " << rectN23[m][i] << ", " << rectN12[m][i]
                  << ", " << rectN23[m][i] << "]" << endl;
             for(int j=0;j<4;j++) 
                cout << "   pt " << j+1 << ": " << rectangles[m][i][j] << endl;
             for(int j=0;j<4;j++) {
                if(rectCrackSidesAtFront[m][i][j]) {
                   int j2=(j+2<5 ? j+2 : 1);
                   cout << "   side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                        << ") located at crack front." << endl;
                }
             }
          } 
          for(int i=0;i<(int)triangles[m].size();i++) {
             cout << "Triangle " << i+1 << ": meshed by [" << triNCells[m][i]
                  << ", " << triNCells[m][i] 
                  << ", " << triNCells[m][i] << "]" << endl;
             for(int j=0;j<3;j++) 
                cout << "   pt " << j+1 << ": " << triangles[m][i][j] << endl;
             for(int j=0;j<3;j++) {
                if(triCrackSidesAtFront[m][i][j]) {
                   int j2=(j+2<4 ? j+2 : 1);
                   cout << "   side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") located at crack front." << endl;
                }
             }
          }
          for(int i=0;i<(int)arcs[m].size();i++) {
             cout << "Arc " << i+1 << ": meshed by " << arcNCells[m][i]
                  << " cells on the circumference.\n"
                  << "   crack front segment ID: " << arcCrkFrtSegID[m][i]
                  << "\n   start, middle and end points of the arc:"  << endl;
             for(int j=0;j<3;j++)
                cout << "   pt " << j+1 << ": " << arcs[m][i][j] << endl;
          }
          for(int i=0;i<(int)ellipses[m].size();i++) {
             cout << "Ellipse " << i+1 << ": meshed by " << ellipseNCells[m][i]
                  << " cells on the circumference.\n"
                  << "   crack front segment ID: " << ellipseCrkFrtSegID[m][i]
                  << endl;
             cout << "   end point on axis1: " << pellipses[m][i][0] << endl;
             cout << "   end point on axis2: " << pellipses[m][i][1] << endl;
             cout << "   another end point on axis1: " << pellipses[m][i][2] 
                  << endl;
          }
          for(int i=0;i<(int)pellipses[m].size();i++) {
             cout << "Partial ellipse " << i+1 << " (" << pellipseExtent[m][i] 
                  << "): meshed by " << pellipseNCells[m][i]
                  << " cells on the circumference.\n"
                  << "   crack front segment ID: " << pellipseCrkFrtSegID[m][i]
                  << endl;
             cout << "   center: " << pellipses[m][i][0] << endl;
             cout << "   end point on axis1: " << pellipses[m][i][1] << endl;
             cout << "   end point on axis2: " << pellipses[m][i][2] << endl;
          }
       } // End of if(crackType...)
    } // End of loop over materials
  } 
#endif 
}

void Crack::addComputesAndRequiresCrackDiscretization(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Do nothing currently 
}

void Crack::CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* /*new_dw*/)
{
  double w;
  int k,i,j,ni,nj,n1,n2,n3;
  int nstart0,nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt,p_1,p_2;
  Point pt1,pt2,pt3,pt4;

  for(int p=0;p<patches->size();p++) { // All ranks 
    const Patch* patch = patches->get(p);

    // Set radius (rJ) of J-path cirlce or number of cells
    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());
    if(rJ<1.e-16)           // No input from input
      rJ=NJ*dx_min;
    else                    // Input radius of J patch circle
      NJ=(int)(rJ/dx_min);

    /* Task 2: Discretize crack plane */
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){ 

       /* Initialize the arries related to cracks 
       */ 
       cx[m].clear();
       cElemNodes[m].clear();
       cElemNorm[m].clear();
       cFrontSegPts[m].clear();
       cFrontSegNorm[m].clear();
       cFrontSegJ[m].clear();
       cFrontSegK[m].clear(); 
       cnumNodes[m]=0;
       cnumElems[m]=0;
       cnumFrontSegs[m]=0;
       cmin[m]=Point(9e16,9e16,9e16);
       cmax[m]=Point(-9e16,-9e16,-9e16); 

       if(crackType[m]=="NO_CRACK") continue;

       /* Task 2a: Discretize quadrilaterals
       */
       nstart0=0;  // Starting node number for each level (in j direction)
       for(k=0; k<(int)rectangles[m].size(); k++) {  // Loop over quadrilaterals
         // Resolutions for the quadrilateral 
         ni=rectN12[m][k];       
         nj=rectN23[m][k]; 
         // Four vertices for the quadrilateral 
         p1=rectangles[m][k][0];   
         p2=rectangles[m][k][1];
         p3=rectangles[m][k][2];
         p4=rectangles[m][k][3];

         // Create temporary arraies
         Point* side2=new Point[2*nj+1];
         Point* side4=new Point[2*nj+1];

         // Generate side-node coordinates
         for(j=0; j<=2*nj; j++) {
           side2[j]=p2+(p3-p2)*(float)j/(2*nj);
           side4[j]=p1+(p4-p1)*(float)j/(2*nj);
         }

         for(j=0; j<=nj; j++) {
           for(i=0; i<=ni; i++) {  
              w=(float)i/(float)ni;
              p_1=side4[2*j];
              p_2=side2[2*j];
              pt=p_1+(p_2-p_1)*w;
              cx[m].push_back(pt);          
           }
           if(j!=nj) {
              for(i=0; i<ni; i++) {
                 w=(float)(2*i+1)/(float)(2*ni);
                 p_1=side4[2*j+1];
                 p_2=side2[2*j+1];
                 pt=p_1+(p_2-p_1)*w;
                 cx[m].push_back(pt);
              }
           }  // End of if j!=nj
         } // End of loop over j

         // Create elements and get normals for quadrilaterals
         for(j=0; j<nj; j++) {
           nstart1=nstart0+(2*ni+1)*j;
           nstart2=nstart1+(ni+1);
           nstart3=nstart2+ni;
           for(i=0; i<ni; i++) {
              /* There are four elements in each sub-rectangle */
              // For the 1st element (n1,n2,n3 three nodes of the element)
              n1=nstart2+i; 
              n2=nstart1+i; 
              n3=nstart1+(i+1);
              cElemNodes[m].push_back(IntVector(n1,n2,n3));
              cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
              // For the 2nd element
              n1=nstart2+i;
              n2=nstart3+i;
              n3=nstart1+i;
              cElemNodes[m].push_back(IntVector(n1,n2,n3));
              cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
              // For the 3rd element
              n1=nstart2+i;
              n2=nstart1+(i+1);
              n3=nstart3+(i+1);
              cElemNodes[m].push_back(IntVector(n1,n2,n3));
              cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
              // For the 4th element 
              n1=nstart2+i;
              n2=nstart3+(i+1);
              n3=nstart3+i;
              cElemNodes[m].push_back(IntVector(n1,n2,n3));
              cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
           }  // End of loop over i
         }  // End of loop over j
         nstart0+=((2*ni+1)*nj+ni+1);  
         delete [] side4;
         delete [] side2;
       } // End of loop over quadrilaterals 

       /* Task 2b: Discretize triangluar segments 
       */
       for(k=0; k<(int)triangles[m].size(); k++) {  // Loop over all triangles
         // Three vertices of the triangle
         p1=triangles[m][k][0];
         p2=triangles[m][k][1];
         p3=triangles[m][k][2];

         // Create temprary arraies
         Point* side12=new Point[triNCells[m][k]+1];
         Point* side13=new Point[triNCells[m][k]+1];

         // Generate node coordinates
         for(j=0; j<=triNCells[m][k]; j++) {
           w=(float)j/(float)triNCells[m][k];
           side12[j]=p1+(p2-p1)*w;
           side13[j]=p1+(p3-p1)*w;
         }
        
         for(j=0; j<=triNCells[m][k]; j++) {
           for(i=0; i<=j; i++) {
             p_1=side12[j];
             p_2=side13[j];
             if(j==0) w=0.0;
             else w=(float)i/(float)j;
             pt=p_1+(p_2-p_1)*w;
             cx[m].push_back(pt);
           } // End of loop over i
         } // End of loop over j
 
         // Generate elements and their normals
         for(j=0; j<triNCells[m][k]; j++) {
           nstart1=nstart0+j*(j+1)/2;
           nstart2=nstart0+(j+1)*(j+2)/2;
           for(i=0; i<j; i++) {
             // Left element
             n1=nstart1+i;
             n2=nstart2+i;
             n3=nstart2+(i+1);
             cElemNodes[m].push_back(IntVector(n1,n2,n3));
             cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
             // Right element
             n1=nstart1+i;
             n2=nstart2+(i+1);
             n3=nstart1+(i+1);
             cElemNodes[m].push_back(IntVector(n1,n2,n3));
             cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
           } // End of loop over i
           n1=nstart0+(j+1)*(j+2)/2-1;
           n2=nstart0+(j+2)*(j+3)/2-2;
           n3=nstart0+(j+2)*(j+3)/2-1;
           cElemNodes[m].push_back(IntVector(n1,n2,n3));
           cElemNorm[m].push_back(TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]));
         } // End of loop over j
         // Add number of nodes in this trianglular segment
         nstart0+=(triNCells[m][k]+1)*(triNCells[m][k]+2)/2;
         delete [] side12;
         delete [] side13;
       } // End of loop over triangles

       /* Task 2c: Define line-segments at crack front for rectangular and 
          triangular crack segments*/
       // Check qradrilaterals
       for(k=0; k<(int)rectangles[m].size(); k++) {  // Loop over quadrilaterals
         for(j=0; j<4; j++) {  // Loop over four sides of the quadrilateral
           if(rectCrackSidesAtFront[m][k][j]==0) continue;
           int j1= (j!=3 ? j+1 : 0);
           pt1=rectangles[m][k][j];
           pt2=rectangles[m][k][j1];
           for(i=0;i<(int)cElemNorm[m].size();i++) { // Loop over elems
             // For sides 3 & 4, check elems reversely
             int istar=( (j==0 || j==1) ? i : (cElemNorm[m].size()-(i+1)) );
             for(int side=0; side<3; side++) { // three sides
               n1=cElemNodes[m][istar].x();
               n2=cElemNodes[m][istar].y();
               n3=cElemNodes[m][istar].z();
               if(side==0) {pt3=cx[m][n1]; pt4=cx[m][n2];}
               else if(side==1) {pt3=cx[m][n2]; pt4=cx[m][n3];}
               else {pt3=cx[m][n3]; pt4=cx[m][n1];}
               if(TwoLinesDuplicate(pt1,pt2,pt3,pt4)) {
                 cFrontSegPts[m].push_back(pt3);
                 cFrontSegPts[m].push_back(pt4);
                 cFrontSegNorm[m].push_back(TriangleNormal(cx[m][n1],
                                                 cx[m][n2],cx[m][n3]));
                 cFrontSegJ[m].push_back(Vector(0.,0.,0.));
                 cFrontSegK[m].push_back(Vector(0.,0.,0.));
               }
             } // End of loop over sides
           } // End of loop over i
         } // End of loop over j
       } // End of loop over k
       // Check triangles
       for(k=0; k<(int)triangles[m].size(); k++) {  // Loop over triangles
         for(j=0; j<3; j++) {  // Loop over three sides of the triangle
           if(triCrackSidesAtFront[m][k][j]==0) continue;
           int j1= (j!=2 ? j+1 : 0);
           pt1=triangles[m][k][j];
           pt2=triangles[m][k][j1];
           for(i=0;i<(int)cElemNorm[m].size();i++) { // Loop over elems
             int istar=( (j==0 || j==1) ? i : (cElemNorm[m].size()-(i+1)) );
             for(int side=0; side<3; side++) { // three sides
               n1=cElemNodes[m][istar].x();
               n2=cElemNodes[m][istar].y();
               n3=cElemNodes[m][istar].z();
               if(side==0) {pt3=cx[m][n1]; pt4=cx[m][n2];}
               else if(side==1) {pt3=cx[m][n2]; pt4=cx[m][n3];}
               else {pt3=cx[m][n3]; pt4=cx[m][n1];}
               if(TwoLinesDuplicate(pt1,pt2,pt3,pt4)) {
                 cFrontSegPts[m].push_back(pt3);
                 cFrontSegPts[m].push_back(pt4);
                 cFrontSegNorm[m].push_back(TriangleNormal(cx[m][n1],
                                                 cx[m][n2],cx[m][n3]));
                 cFrontSegJ[m].push_back(Vector(0.,0.,0.));
                 cFrontSegK[m].push_back(Vector(0.,0.,0.));
               }
             } // End of loop over sides
           } // End of loop over i
         } // End of loop over j
       } // End of loop over k

       /* Task 2d: Discretize arc segments
       */
       for(k=0; k<(int)arcs[m].size(); k++) {  // Loop over all arcs
         // Three points of the arc
         p1=arcs[m][k][0];
         p2=arcs[m][k][1];
         p3=arcs[m][k][2];
         double x1,y1,z1,x2,y2,z2,x3,y3,z3;
         x1=p1.x(); y1=p1.y(); z1=p1.z();
         x2=p2.x(); y2=p2.y(); z2=p2.z();
         x3=p3.x(); y3=p3.y(); z3=p3.z();

         // Find center of the arc
         double a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3;
         a1=2*(x2-x1); b1=2*(y2-y1); c1=2*(z2-z1);
         d1=x1*x1-x2*x2+y1*y1-y2*y2+z1*z1-z2*z2;
         a2=2*(x3-x1); b2=2*(y3-y1); c2=2*(z3-z1);
         d2=x1*x1-x3*x3+y1*y1-y3*y3+z1*z1-z3*z3;
         FindPlaneEquation(p1,p2,p3,a3,b3,c3,d3);
         
         double delt,deltx,delty,deltz;
         delt  = Matrix3(a1,b1,c1,a2,b2,c2,a3,b3,c3).Determinant();
         deltx = Matrix3(-d1,b1,c1,-d2,b2,c2,-d3,b3,c3).Determinant();
         delty = Matrix3(a1,-d1,c1,a2,-d2,c2,a3,-d3,c3).Determinant();
         deltz = Matrix3(a1,b1,-d1,a2,b2,-d2,a3,b3,-d3).Determinant();
         double x0,y0,z0;
         x0=deltx/delt;  y0=delty/delt;  z0=deltz/delt;
         Point origin=Point(x0,y0,z0);
         double radius=sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));
 
         // Define local coordinates
         Vector v1,v2,v3;
         double temp=sqrt(a3*a3+b3*b3+c3*c3);
         v3=Vector(a3/temp,b3/temp,c3/temp);
         v1=TwoPtsDirCos(origin,p1);
         v2=Cross(v3,v1);
         double lx,mx,nx,ly,my,ny;
         lx=v1.x();  mx=v1.y();  nx=v1.z();
         ly=v2.x();  my=v2.y();  ny=v2.z();

         // Angle of the arc
         double angleOfArc;
         double PI=3.141592654;
         double x3prime,y3prime;
         x3prime=lx*(x3-x0)+mx*(y3-y0)+nx*(z3-z0);
         y3prime=ly*(x3-x0)+my*(y3-y0)+ny*(z3-z0);
         double cosTheta=x3prime/radius;
         double sinTheta=y3prime/radius;
         double thetaQ=fabs(asin(y3prime/radius));
         if(sinTheta>=0.) {
           if(cosTheta>=0) angleOfArc=thetaQ;
           else angleOfArc=PI-thetaQ;
         }
         else {
           if(cosTheta<=0.) angleOfArc=PI+thetaQ;
           else angleOfArc=2*PI-thetaQ;
         }

         // Node points
         cx[m].push_back(origin);
         for(j=0;j<=arcNCells[m][k];j++) {  // Loop over points 
           double thetai=angleOfArc*j/arcNCells[m][k];
           double xiprime=radius*cos(thetai);
           double yiprime=radius*sin(thetai);
           double xi=lx*xiprime+ly*yiprime+x0;
           double yi=mx*xiprime+my*yiprime+y0;
           double zi=nx*xiprime+ny*yiprime+z0;
           cx[m].push_back(Point(xi,yi,zi));
         } // End of loop over points
         
         // Crack elements
         for(j=1;j<=arcNCells[m][k];j++) {  // Loop over segs
           n1=nstart0;
           n2=nstart0+j;
           n3=nstart0+(j+1);
           Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
           cElemNodes[m].push_back(IntVector(n1,n2,n3));
           cElemNorm[m].push_back(thisSegNorm);
           // Crack front segments
           if(arcCrkFrtSegID[m][k]==9999 || arcCrkFrtSegID[m][k]==j) {
             cFrontSegPts[m].push_back(cx[m][n2]);
             cFrontSegPts[m].push_back(cx[m][n3]);
             cFrontSegNorm[m].push_back(thisSegNorm);
             cFrontSegJ[m].push_back(Vector(0.,0.,0.));
             cFrontSegK[m].push_back(Vector(0.,0.,0.));    
           }
         }
         nstart0+=arcNCells[m][k]+2;
       } // End of loop over arcs

       /* Task 2e: Discretize elliptic segments
       */
       for(k=0; k<(int)ellipses[m].size(); k++) {
         // Three points of the ellipse
         p1=ellipses[m][k][0];
         p2=ellipses[m][k][1];
         p3=ellipses[m][k][2];
         // Center and half axial lengths of the ellipse
         double x0,y0,z0,a,b;
         Point origin=p3+(p1-p3)*0.5;
         x0=origin.x(); 
         y0=origin.y();  
         z0=origin.z();
         a=(p1-origin).length();
         b=(p2-origin).length();
         // Local coordinates
         Vector v1,v2,v3;
         v1=TwoPtsDirCos(origin,p1);
         v2=TwoPtsDirCos(origin,p2);
         v3=Cross(v1,v2);
         double lx,mx,nx,ly,my,ny;
         lx=v1.x();  mx=v1.y();  nx=v1.z();
         ly=v2.x();  my=v2.y();  ny=v2.z();

         // Crack nodes
         cx[m].push_back(origin);
         for(j=0;j<ellipseNCells[m][k];j++) {  // Loop over points
           double PI=3.141592654;
           double thetai=j*(2*PI)/ellipseNCells[m][k];
           double xiprime=a*cos(thetai);
           double yiprime=b*sin(thetai);
           double xi=lx*xiprime+ly*yiprime+x0;
           double yi=mx*xiprime+my*yiprime+y0;
           double zi=nx*xiprime+ny*yiprime+z0;
           cx[m].push_back(Point(xi,yi,zi));
         } // End of loop over points

         // Crack elements
         for(j=1;j<=ellipseNCells[m][k];j++) {  // Loop over segs
           int j1 = (j==ellipseNCells[m][k]? 1 : j+1);
           n1=nstart0;
           n2=nstart0+j;
           n3=nstart0+j1;
           Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
           cElemNodes[m].push_back(IntVector(n1,n2,n3));
           cElemNorm[m].push_back(thisSegNorm);
           // Crack front segments
           if(ellipseCrkFrtSegID[m][k]==9999 || ellipseCrkFrtSegID[m][k]==j) {
             cFrontSegPts[m].push_back(cx[m][n2]);
             cFrontSegPts[m].push_back(cx[m][n3]);
             cFrontSegNorm[m].push_back(thisSegNorm);
             cFrontSegJ[m].push_back(Vector(0.,0.,0.));
             cFrontSegK[m].push_back(Vector(0.,0.,0.));             
           }
         }
         nstart0+=ellipseNCells[m][k]+1;
       } // End of discretizing ellipses

       // Task 2f: Discretize partial ellipses
       for(k=0; k<(int)pellipses[m].size(); k++) {
         double extent;
         if(pellipseExtent[m][k]=="quarter") extent=0.25;
         if(pellipseExtent[m][k]=="half") extent=0.5;

         // Center, end points on major and minor axes
         Point origin=pellipses[m][k][0];
         Point major_p=pellipses[m][k][1];
         Point minor_p=pellipses[m][k][2];
         double x0,y0,z0,a,b;
         x0=origin.x();
         y0=origin.y();
         z0=origin.z();
         a=(major_p-origin).length();
         b=(minor_p-origin).length();
         // Local coordinates
         Vector v1,v2,v3;
         v1=TwoPtsDirCos(origin,major_p);
         v2=TwoPtsDirCos(origin,minor_p);
         v3=Cross(v1,v2);
         double lx,mx,nx,ly,my,ny;
         lx=v1.x();  mx=v1.y();  nx=v1.z();
         ly=v2.x();  my=v2.y();  ny=v2.z();

         // Crack nodes
         cx[m].push_back(origin);
         for(j=0;j<=pellipseNCells[m][k];j++) {  // Loop over points
           double PI=3.141592654;
           double thetai=j*(2*PI*extent)/pellipseNCells[m][k];
           double xiprime=a*cos(thetai);
           double yiprime=b*sin(thetai);
           double xi=lx*xiprime+ly*yiprime+x0;
           double yi=mx*xiprime+my*yiprime+y0;
           double zi=nx*xiprime+ny*yiprime+z0;
           cx[m].push_back(Point(xi,yi,zi));
         } // End of loop over points

         // Crack elements
         for(j=1;j<=pellipseNCells[m][k];j++) {  // Loop over segs
           n1=nstart0;
           n2=nstart0+j;
           n3=nstart0+j+1;
           Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
           cElemNodes[m].push_back(IntVector(n1,n2,n3));
           cElemNorm[m].push_back(thisSegNorm);
           // Crack front segments
           if(pellipseCrkFrtSegID[m][k]==9999 || pellipseCrkFrtSegID[m][k]==j) {
             cFrontSegPts[m].push_back(cx[m][n2]);
             cFrontSegPts[m].push_back(cx[m][n3]);
             cFrontSegNorm[m].push_back(thisSegNorm);
             cFrontSegJ[m].push_back(Vector(0.,0.,0.));
             cFrontSegK[m].push_back(Vector(0.,0.,0.));
           }
         }
         nstart0+=pellipseNCells[m][k]+2;
       } // End of discretizing partial ellipses

       cnumNodes[m]=cx[m].size();
       cnumElems[m]=cElemNorm[m].size();
       cnumFrontSegs[m]=cFrontSegJ[m].size();

       /* Task 3: Find extent of crack */
       for(i=0; i<cnumNodes[m];i++) {
         cmin[m]=Min(cmin[m],cx[m][i]);
         cmax[m]=Max(cmax[m],cx[m][i]);
       }

#if 1 // Output crack mesh information 
      int rank,rank_size;
      MPI_Comm_size(mpi_crack_comm, &rank_size);
      MPI_Comm_rank(mpi_crack_comm, &rank);
      if(rank==rank_size-1) { // Output from the last rank 
        cout << "\n*** Crack mesh information output from rank "
             << rank << " ***" << endl;
        cout << "MatID: " << m << endl;
        cout << "  Number of crack elements: " << cnumElems[m]
             << "\n  Number of crack nodes: " << cnumNodes[m]
             << "\n  Number of crack front segments: " << cnumFrontSegs[m]
             << endl;
        cout << "  Element nodes and normals (" << cnumElems[m]
             << " elements intotal):" << endl; 
        for(int mp=0; mp<cnumElems[m]; mp++) {
          n1=cElemNodes[m][mp].x();
          n2=cElemNodes[m][mp].y();
          n3=cElemNodes[m][mp].z();
          cout << "     Elem " << mp
               << ": [" << n1 << ", " << n2 << ", " << n3
               << "], norm " << cElemNorm[m][mp] << endl;
        }
        cout << "  Crack nodes coordinates (" << cnumNodes[m]
             << " nodes in total):" << endl; 
        for(int mp=0; mp<cnumNodes[m]; mp++) {
          cout << "     Node " << mp << ": " << cx[m][mp] << endl;
        }
        cout << "  Crack front line-segments (" << cnumFrontSegs[m]
             << " segments in total)" << endl;
        for(int mp=0; mp<cnumFrontSegs[m];mp++) {
          cout << "     Seg " << mp << ": "
               << cFrontSegPts[m][mp*2] << "..." << cFrontSegPts[m][2*mp+1]
               << ", outward normal: " << cFrontSegNorm[m][mp] << endl;
        }
        cout << "\n  Crack extent: " << cmin[m] << "..."
             <<  cmax[m] << endl << endl;
      }// End of if(patch->getID()==0)
#endif
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresParticleVelocityField(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{  
  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, NGN);
  t->computes(lb->gNumPatlsLabel);
  t->computes(lb->GNumPatlsLabel);
  t->computes(lb->GCrackNormLabel);
  t->computes(lb->pgCodeLabel);
}

void Crack::ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int numMatls = d_sharedState->getNumMPMMatls();

    // Detect if doing fracture analysis
    double time = d_sharedState->getElapsedTime();
    FindTimeStepForFractureAnalysis(time);
 
    enum {NO=0,YES};
    enum {SAMESIDE=0,ABOVE_CRACK,BELOW_CRACK};

    Vector dx = patch->dCell(); 
    for(int m=0; m<numMatls; m++) {
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      old_dw->get(px, lb->pXLabel, pset);
      ParticleVariable<Short27> pgCode;
      new_dw->allocateAndPut(pgCode,lb->pgCodeLabel,pset);

      NCVariable<int> gNumPatls;
      NCVariable<int> GNumPatls;
      NCVariable<Vector> GCrackNorm;
      new_dw->allocateAndPut(gNumPatls,  lb->gNumPatlsLabel,  dwi, patch);
      new_dw->allocateAndPut(GNumPatls,  lb->GNumPatlsLabel,  dwi, patch);
      new_dw->allocateAndPut(GCrackNorm, lb->GCrackNormLabel, dwi, patch);
      gNumPatls.initialize(0);
      GNumPatls.initialize(0);
      GCrackNorm.initialize(Vector(0,0,0));

      ParticleSubset* psetWGCs = old_dw->getParticleSubset(dwi, patch, 
                                      Ghost::AroundCells, NGN, lb->pXLabel);
      constParticleVariable<Point> pxWGCs;
      old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

      IntVector ni[MAX_BASIS];

      if(cnumElems[m]==0) {            // For materials with no cracks
        // set pgCode[idx][k]=1
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          for(int k=0; k<d_8or27; k++) pgCode[*iter][k]=1;
        }  
        // Get number of particles around nodes 
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                           itr!=psetWGCs->end();itr++) {
          if(d_8or27==8)
            patch->findCellNodes(pxWGCs[*itr], ni);
          else if(d_8or27==27)
            patch->findCellNodes27(pxWGCs[*itr], ni);
          for(int k=0; k<d_8or27; k++) {
            if(patch->containsNode(ni[k])) 
              gNumPatls[ni[k]]++;
          }
        } //End of loop over partls
      }

      else { // For materials with crack(s)

        //Step 1: Determine if nodes within crack zone 
        Ghost::GhostType  gac = Ghost::AroundCells;
        IntVector g_cmin, g_cmax, cell_idx;
        NCVariable<short> singlevfld;
        new_dw->allocateTemporary(singlevfld, patch, gac, 2*NGN); 
        singlevfld.initialize(0);

        // Get crack extent on grid (g_cmin->g_cmax)
        patch->findCell(cmin[m],cell_idx);
        Point ptmp=patch->nodePosition(cell_idx+IntVector(1,1,1));
        IntVector offset=CellOffset(cmin[m],ptmp,dx);
        g_cmin=cell_idx+IntVector(1-offset.x(),1-offset.y(),1-offset.z());

        patch->findCell(cmax[m],cell_idx);
        ptmp=patch->nodePosition(cell_idx);
        g_cmax=cell_idx+CellOffset(cmax[m],ptmp,dx);

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c=*iter;
          if(c.x()>=g_cmin.x() && c.x()<=g_cmax.x() && c.y()>=g_cmin.y() &&
             c.y()<=g_cmax.y() && c.z()>=g_cmin.z() && c.z()<=g_cmax.z() )
            singlevfld[c]=NO;  // in crack zone
          else
            singlevfld[c]=YES; // in non-crack zone
        }

        /* Step 2: Detect if particle is above, below or in the same side, 
                   count number of particles around nodes */
        NCVariable<int> num0,num1,num2; 
        new_dw->allocateTemporary(num0, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num1, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num2, patch, gac, 2*NGN);
        num0.initialize(0);
        num1.initialize(0);
        num2.initialize(0);
 
        // Generate particle cross code for particles in pset
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          particleIndex idx=*iter;
          if(d_8or27==8)
             patch->findCellNodes(px[idx], ni);
          else if(d_8or27==27)
             patch->findCellNodes27(px[idx], ni);

          for(int k=0; k<d_8or27; k++) {
            if(singlevfld[ni[k]]) {          // for nodes in non-crack zone
              pgCode[idx][k]=0;
              num0[ni[k]]++;
            }
            else {                           // for nodes in crack zone
              // Detect if particles above, below or in same side with nodes 
              short  cross=SAMESIDE;
              Vector norm=Vector(0.,0.,0.);

              // Get node position even if ni[k] beyond this patch
              Point gx=patch->nodePosition(ni[k]);

              for(int i=0; i<cnumElems[m]; i++) {  //loop over crack elements
                //Three vertices of each element
                Point n3,n4,n5;                  
                n3=cx[m][cElemNodes[m][i].x()];
                n4=cx[m][cElemNodes[m][i].y()];
                n5=cx[m][cElemNodes[m][i].z()];

                // If particle and node in same side, continue
                short pPosition = Location(px[idx],gx,n3,n4,n5);
                if(pPosition==SAMESIDE) continue; 

                // Three signed volumes to see if p-g crosses crack
                double v3,v4,v5;
                v3=Volume(gx,n3,n4,px[idx]);
                v4=Volume(gx,n3,n5,px[idx]);
                if(v3*v4>0.) continue;
                v5=Volume(gx,n4,n5,px[idx]);
                if(v3*v5<0.) continue;

                // Particle above crack
                if(pPosition==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) { 
                  if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=ABOVE_CRACK;
                    norm+=cElemNorm[m][i];
                  }
                  else { // no cross
                    cross=SAMESIDE;
                    norm=Vector(0.,0.,0.);
                  }
                }
                // Particle below crack
                if(pPosition==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) { 
                  if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=BELOW_CRACK;
                    norm+=cElemNorm[m][i];
                  }
                  else { // no cross
                    cross=SAMESIDE;
                    norm=Vector(0.,0.,0.);
                  }
                }
              } // End of loop over crack elements

              pgCode[idx][k]=cross;
              if(cross==SAMESIDE)    num0[ni[k]]++;
              if(cross==ABOVE_CRACK) num1[ni[k]]++;
              if(cross==BELOW_CRACK) num2[ni[k]]++;
              if(patch->containsNode(ni[k]) && norm.length()>1.e-16) { 
                norm/=norm.length();
                GCrackNorm[ni[k]]+=norm;
              }
            } // End of if(singlevfld)
          } // End of loop over k
        } // End of loop over particles

        // Step 3: count particles around nodes in GhostCells
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                                     itr!=psetWGCs->end();itr++) {
          particleIndex idx=*itr;
          short handled=NO;
          for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
            if(pxWGCs[idx]==px[*iter]) {
              handled=YES;
              break;
            }
          }

          if(!handled) {               // particles in GhostCells
            if(d_8or27==8)
               patch->findCellNodes(pxWGCs[idx], ni);
            else if(d_8or27==27)
               patch->findCellNodes(pxWGCs[idx], ni);

            for(int k=0; k<d_8or27; k++) {
              Point gx=patch->nodePosition(ni[k]);
              if(singlevfld[ni[k]]) {          // for nodes in non-crack zone
                num0[ni[k]]++;
              }
              else { 
                short  cross=SAMESIDE; 
                Vector norm=Vector(0.,0.,0.);
                for(int i=0; i<cnumElems[m]; i++) {  // Loop over crack elements
                  // Three vertices of each element
                  Point n3,n4,n5;                
                  n3=cx[m][cElemNodes[m][i].x()];
                  n4=cx[m][cElemNodes[m][i].y()];
                  n5=cx[m][cElemNodes[m][i].z()];

                  // If particle and node in same side, continue
                  short pPosition = Location(pxWGCs[idx],gx,n3,n4,n5);
                  if(pPosition==SAMESIDE) continue;

                  // Three signed volumes to see if p-g crosses crack
                  double v3,v4,v5;
                  v3=Volume(gx,n3,n4,pxWGCs[idx]);
                  v4=Volume(gx,n3,n5,pxWGCs[idx]);
                  if(v3*v4>0.) continue;
                  v5=Volume(gx,n4,n5,pxWGCs[idx]);
                  if(v3*v5<0.) continue;

                  // Particle above crack
                  if(pPosition==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) {
                    if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                  (v3==0.||v4==0.||v5==0.) ) ) {
                      cross=ABOVE_CRACK;
                      norm+=cElemNorm[m][i];
                    }
                    else { 
                      cross=SAMESIDE;
                      norm=Vector(0.,0.,0.);
                    }
                  }
                  // Particle below crack
                  if(pPosition==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) {
                    if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                  (v3==0.||v4==0.||v5==0.) ) ) {
                      cross=BELOW_CRACK;
                      norm+=cElemNorm[m][i];
                    } 
                    else {  
                      cross=SAMESIDE;
                      norm=Vector(0.,0.,0.);
                    }
                  }
                } // End of loop over crack elements

                if(cross==SAMESIDE)    num0[ni[k]]++;
                if(cross==ABOVE_CRACK) num1[ni[k]]++;
                if(cross==BELOW_CRACK) num2[ni[k]]++;
                if(patch->containsNode(ni[k]) && norm.length()>1.e-16) {
                  norm/=norm.length();
                  GCrackNorm[ni[k]]+=norm;
                } 
              } // End of if(singlevfld)
            } // End of loop over k
          } // End of if(!handled)
        } // End of loop over particles

        // Step 4: Convert cross codes to field codes (0 to 1 or 2) 
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
           particleIndex idx=*iter;
           if(d_8or27==8)
              patch->findCellNodes(px[idx], ni);
           else if(d_8or27==27)
              patch->findCellNodes27(px[idx], ni);

           for(int k=0; k<d_8or27; k++) {
             if(pgCode[idx][k]==0) {
               if((num1[ni[k]]+num2[ni[k]]==0) || num2[ni[k]]!=0) 
                 pgCode[idx][k]=1;
               else if(num1[ni[k]]!=0) 
                 pgCode[idx][k]=2;
               else {
                 cout << "More than two velocity fields found in " 
                      << "Crack::ParticleVeloccityField for node: " 
                      << ni[k] << endl;
                 exit(1);
               } 
             }
           } // End of loop over k
         } // End of loop patls

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          if(GCrackNorm[c].length()>1.e-16) //unit crack normals
              GCrackNorm[c]/=GCrackNorm[c].length();
                    
          if(num1[c]+num2[c]==0) { // node in non-carck zone
            gNumPatls[c]=num0[c];
            GNumPatls[c]=0;
          }
          else if(num1[c]!=0 && num2[c]!=0) {
            gNumPatls[c]=num1[c];
            GNumPatls[c]=num2[c];
          }
          else {
            if(num1[c]!=0) {
              gNumPatls[c]=num1[c];
              GNumPatls[c]=num0[c];
            }
            if(num2[c]!=0) {
              gNumPatls[c]=num0[c];
              GNumPatls[c]=num2[c];
            }
          }
        } // End of loop over NodeIterator
      } // End of if(numElem[m]!=0)
#if 0 // Output particle velocity field code
      cout << "\n*** Particle velocity field generated "
           << "in Crack::ParticleVelocityField ***" << endl;
      cout << "--patch: " << patch->getID() << endl;
      cout << "--matreial: " << m << endl;
      for(ParticleSubset::iterator iter=pset->begin();
                        iter!=pset->end();iter++) {
        particleIndex idx=*iter;
        cout << "p["<< idx << "]: " << px[idx]<< endl;
        for(int k=0; k<d_8or27; k++) {
          if(pgCode[idx][k]==1)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << endl;
          else if(pgCode[idx][k]==2)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << " ***" << endl;
          else {
             cout << "Unknown particle velocity code in "
                  << "Crack::ParticleVelocityField" << endl;
             exit(1);
          }
        }  // End of loop over nodes (k)
      }  // End of loop over particles
      cout << "\nNumber of particles around nodes:\n" 
           << setw(18) << "node" << setw(20) << "gNumPatls" 
           << setw(20) << "GNumPatls" << endl;
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;
        if(gNumPatls[c]+GNumPatls[c]!=0){ 
        cout << setw(10) << c << setw(20) << gNumPatls[c] 
             << setw(20) << GNumPatls[c] << endl;}
      }
#endif
    } // End of loop numMatls
  } // End of loop patches
}

void Crack::addComputesAndRequiresAdjustCrackContactInterpolated(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  // Data of primary field
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     Ghost::None); 
  t->requires(Task::NewDW, lb->gDisplacementLabel, Ghost::None);

  // Data of additional field
  t->requires(Task::NewDW, lb->GMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel, Ghost::None);

  t->modifies(lb->gVelocityLabel, mss);
  t->modifies(lb->GVelocityLabel, mss);

  t->computes(lb->frictionalWorkLabel);

}

void Crack::AdjustCrackContactInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    enum {NO=0,YES};

    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    double vol0,normVol;
    Vector va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());

    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vcell = dx.x()*dx.y()*dx.z();

    // Need access to all velocity fields at once, primary field
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity(numMatls);

    // Aceess to additional velocity field, for Farcture
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity(numMatls);
    StaticArray<NCVariable<double> >      frictionWork(numMatls);

    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // Data for primary velocity field 
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel,dwi,patch);

      // Data for second velocity field
      new_dw->get(GNumPatls[m],lb->GNumPatlsLabel,  dwi, patch, gnone, 0);
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);
      new_dw->getModifiable(Gvelocity[m],lb->GVelocityLabel,dwi,patch);

      new_dw->allocateAndPut(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
      frictionWork[m].initialize(0.);

      if(crackType[m]=="NO_CRACK") continue;  // no crack in this material
    
      // Loop over nodes to see if there is contact. If yes, adjust velocity field
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;
       
        // Only one velocity field
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue; 
        // Nodes in non-crack-zone
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;  // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
        short Contact=NO;

        if(separateVol[m]<0. || contactVol[m] <0.) { // Use displacement criterion
          //use displacement criterion
          Vector u1=gdisplacement[m][c];
          Vector u2=Gdisplacement[m][c];
          if(Dot((u2-u1),norm) >0. ) {
            Contact=YES;
          }
        }
        else { // use volume criterion
          // Evaluate the nodal saturated volume (vol0) for general cases
          int numCellsWithPatls=0;
          IntVector cellIndex[8];
          patch->findCellsFromNode(c,cellIndex);
          ParticleSubset* psetWGCs= old_dw->getParticleSubset(dwi, patch,
                                 Ghost::AroundCells, NGN, lb->pXLabel);
          constParticleVariable<Point> pxWGCs;
          old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

          short cellWithPatls[8];
          for(int k=0; k<8; k++)  cellWithPatls[k]=0;
          for(ParticleSubset::iterator iter=psetWGCs->begin();
                                     iter!=psetWGCs->end();iter++) {
            particleIndex idx=*iter;
            double xp=pxWGCs[idx].x();
            double yp=pxWGCs[idx].y();
            double zp=pxWGCs[idx].z();
            for(int k=0; k<8; k++) { // Loop over 8 cells around the node
              Point l=patch->nodePosition(cellIndex[k]);
              Point h=patch->nodePosition(cellIndex[k]+IntVector(1,1,1));
              if(xp>l.x() && xp<=h.x() && yp>l.y() && yp<=h.y() &&
                 zp>l.z() && zp<=h.z()) cellWithPatls[k]=1;
            } // End of loop over 8 cells
            short allCellsWithPatls=1;
            for(int k=0; k<8; k++) {
              if(cellWithPatls[k]==0) allCellsWithPatls=0;
            }
            if(allCellsWithPatls) break;
          } // End of loop over patls
          for(int k=0; k<8; k++) numCellsWithPatls+=cellWithPatls[k];
          vol0=(float)numCellsWithPatls/8.*vcell;
    
          normVol=(gvolume[m][c]+Gvolume[m][c])/vol0;
          if(normVol>=contactVol[m] || 
            (normVol>separateVol[m] && Dot((vb-va),norm) > 0. )) {
            Contact=YES;
          }
        }

        if(!Contact) { // No contact  
          gvelocity[m][c]=gvelocity[m][c];
          Gvelocity[m][c]=Gvelocity[m][c];
          frictionWork[m][c] += 0.;
        } 
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Nothing to do with it
            gvelocity[m][c]=gvelocity[m][c];
            Gvelocity[m][c]=Gvelocity[m][c];
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity[m][c]=vc;
            Gvelocity[m][c]=vc;
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="frictional") { // Apply frictional law
            // For velocity field above crack
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>c_mu[m] ) {  // slide
               if(ratioa>0.) mua=c_mu[m];
               if(ratioa<0.) mua=-c_mu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity[m][c]=va+deltva;
               frictionWork[m][c]+=ma*c_mu[m]*dvan*dvan*(fabs(ratioa)-c_mu[m]);
            }
            else {  // stick
               gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }

            // For velocity field below crack
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>c_mu[m]) { // slide
               if(ratiob>0.) mub=c_mu[m];
               if(ratiob<0.) mub=-c_mu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity[m][c]=vb+deltvb;
               frictionWork[m][c]+=mb*c_mu[m]*dvbn*dvbn*(fabs(ratiob)-c_mu[m]);
            }
            else {// stick
               Gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }
          }

          else { // Wrong contact type
            cout << "Unknown crack contact type in subroutine " 
                 << "Crack::AdjustCrackContactInterpolated: " 
                 << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact

      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

void Crack::addComputesAndRequiresAdjustCrackContactIntegrated(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gDisplacementLabel,  Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->gAccelerationLabel,  mss);

  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel,  Ghost::None);
  t->modifies(             lb->GVelocityStarLabel,  mss);
  t->modifies(             lb->GAccelerationLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

}

void Crack::AdjustCrackContactIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    enum {NO=0,YES};

    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    double vol0,normVol;
    Vector aa,ab,va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());
   
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vcell = dx.x()*dx.y()*dx.z();

    // Need access to all velocity fields at once
    // Data of primary field
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
    // Data of additional field 
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      Gacceleration(numMatls);
    // Friction work
    StaticArray<NCVariable<double> >      frictionWork(numMatls);
   
    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // For primary field
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(gacceleration[m],lb->gAccelerationLabel,
                                                         dwi, patch);
      // For additional field
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GNumPatls[m], lb->GNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(Gvelocity_star[m], lb->GVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(Gacceleration[m],lb->GAccelerationLabel,
                                                         dwi, patch);
      new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel,
                                                         dwi, patch);

      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel);

      if(crackType[m]=="NO_CRACK") continue; // No crack(s) in this material

      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;

        // For nodes in non-crack zone, there is no contact, just continue
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue; 
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;   // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity_star[m][c];
        aa=gacceleration[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity_star[m][c];
        ab=Gacceleration[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
        short Contact=NO;

        if(separateVol[m]<0. || contactVol[m] <0.) { // Use displacement criterion
          // Use displacement criterion
          Vector u1=gdisplacement[m][c];
          //+delT*gvelocity_star[m][c];
          Vector u2=Gdisplacement[m][c];
          //+delT*Gvelocity_star[m][c];
          if(Dot((u2-u1),norm) >0. ) {
            Contact=YES;
          } 
        }
        else { // Use volume criterion
          int numCellsWithPatls=0;
          IntVector cellIndex[8];
          patch->findCellsFromNode(c,cellIndex);
          ParticleSubset* psetWGCs= old_dw->getParticleSubset(dwi, patch,
                                 Ghost::AroundCells, NGN, lb->pXLabel);
          constParticleVariable<Point> pxWGCs;
          old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

          short cellWithPatls[8];
          for(int k=0; k<8; k++)  cellWithPatls[k]=0;
          for(ParticleSubset::iterator iter=psetWGCs->begin();
                                     iter!=psetWGCs->end();iter++) {
            particleIndex idx=*iter;
            double xp=pxWGCs[idx].x();
            double yp=pxWGCs[idx].y();
            double zp=pxWGCs[idx].z();
            for(int k=0; k<8; k++) { // Loop over 8 cells around the node
              Point l=patch->nodePosition(cellIndex[k]);
              Point h=patch->nodePosition(cellIndex[k]+IntVector(1,1,1));
              if(xp>l.x() && xp<=h.x() && yp>l.y() && yp<=h.y() &&
                 zp>l.z() && zp<=h.z()) cellWithPatls[k]=1;
            } //End of loop over 8 cells
            short allCellsWithPatls=1;
            for(int k=0; k<8; k++) {
              if(cellWithPatls[k]==0) allCellsWithPatls=0;
            }
            if(allCellsWithPatls) break;
          } // End of loop over patls
          for(int k=0; k<8; k++) numCellsWithPatls+=cellWithPatls[k];
          vol0=(float)numCellsWithPatls/8.*vcell;

          normVol=(gvolume[m][c]+Gvolume[m][c])/vol0;

          if(normVol>=contactVol[m] || 
             (normVol>separateVol[m] && Dot((vb-va),norm) > 0.)) {
            Contact=YES;
          }
        }

        if(!Contact) {// No contact
          gvelocity_star[m][c]=gvelocity_star[m][c];
          gacceleration[m][c]=gacceleration[m][c];
          Gvelocity_star[m][c]=Gvelocity_star[m][c];
          Gacceleration[m][c]=Gacceleration[m][c];
          frictionWork[m][c]+=0.0;
        } 
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Do nothing
            gvelocity_star[m][c]=gvelocity_star[m][c];
            gacceleration[m][c]=gacceleration[m][c];
            Gvelocity_star[m][c]=Gvelocity_star[m][c];
            Gacceleration[m][c]=Gacceleration[m][c];
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity_star[m][c]=vc;
            gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
            Gvelocity_star[m][c]=vc;
            Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="frictional") { // Apply friction law
            // For primary field
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>c_mu[m] ) {  // slide
               if(ratioa>0.) mua= c_mu[m];
               if(ratioa<0.) mua=-c_mu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity_star[m][c]=va+deltva;
               gacceleration[m][c]=aa+deltva/delT;
               frictionWork[m][c]+=ma*c_mu[m]*dvan*dvan*(fabs(ratioa)-c_mu[m]);
            }
            else {   // stick
               gvelocity_star[m][c]=vc;
               gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
            // For additional field
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>c_mu[m]) { // slide
               if(ratiob>0.) mub= c_mu[m];
               if(ratiob<0.) mub=-c_mu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity_star[m][c]=vb+deltvb;
               Gacceleration[m][c]=ab+deltvb/delT;
               frictionWork[m][c]+=mb*c_mu[m]*dvbn*dvbn*(fabs(ratiob)-c_mu[m]);
            }
            else {  // stick
               Gvelocity_star[m][c]=vc;
               Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
          }

          else {
            cout<< "Unknown crack contact type in "
                << "Crack::AdjustCrackContactIntegrated: " 
                << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact
      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

void Crack::addComputesAndRequiresGetNodalSolutions(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  // Required particles' solutions 
  t->requires(Task::NewDW,lb->pMassLabel_preReloc,                gan,NGP);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,              gan,NGP);
  t->requires(Task::NewDW,lb->pDispGradsLabel_preReloc,           gan,NGP);
  t->requires(Task::NewDW,lb->pStrainEnergyDensityLabel_preReloc, gan,NGP);
  t->requires(Task::NewDW,lb->pKineticEnergyDensityLabel,         gan,NGP);

  t->requires(Task::OldDW,lb->pXLabel,                            gan,NGP);
  if(d_8or27==27) t->requires(Task::OldDW, lb->pSizeLabel,        gan,NGP);
  t->requires(Task::NewDW,lb->pgCodeLabel,                        gan,NGP);

  // Required nodal solutions
  t->requires(Task::NewDW,lb->gMassLabel,                         gnone);
  t->requires(Task::NewDW,lb->GMassLabel,                         gnone);

  // The nodal solutions to be calculated
  t->computes(lb->gGridStressLabel);
  t->computes(lb->GGridStressLabel);
  t->computes(lb->gDispGradsLabel);
  t->computes(lb->GDispGradsLabel);
  t->computes(lb->gStrainEnergyDensityLabel);
  t->computes(lb->GStrainEnergyDensityLabel);
  t->computes(lb->gKineticEnergyDensityLabel);
  t->computes(lb->GKineticEnergyDensityLabel);
}

void Crack::GetNodalSolutions(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  /* Compute nodal solutions of stresses, displacement gradients,
     strain energy density and  kinetic energy density by interpolating
     particle's solutions to grid */

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get particle's solutions
      constParticleVariable<Short27> pgCode;
      constParticleVariable<Point>   px;
      constParticleVariable<Vector>  psize;
      constParticleVariable<double>  pmass;
      constParticleVariable<double>  pstrainenergydensity;
      constParticleVariable<double>  pkineticenergydensity;
      constParticleVariable<Matrix3> pstress,pdispgrads;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                          Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);
      old_dw->get(px,                   lb->pXLabel,                   pset);
      if(d_8or27==27) old_dw->get(psize,lb->pSizeLabel,                pset);

      new_dw->get(pmass,                lb->pMassLabel_preReloc,       pset);
      new_dw->get(pstress,              lb->pStressLabel_preReloc,     pset);
      new_dw->get(pdispgrads,           lb->pDispGradsLabel_preReloc,  pset);
      new_dw->get(pstrainenergydensity,
                               lb->pStrainEnergyDensityLabel_preReloc, pset);
      new_dw->get(pgCode,               lb->pgCodeLabel,               pset);

      // Particles' kinetic energy density calculated with updated velocities
      new_dw->get(pkineticenergydensity,lb->pKineticEnergyDensityLabel,pset);

      // Get nodal mass
      constNCVariable<double> gmass, Gmass;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, Ghost::None, 0);

      // Declare nodal variables calculated
      NCVariable<Matrix3> ggridstress,Ggridstress;
      NCVariable<Matrix3> gdispgrads,Gdispgrads;
      NCVariable<double>  gstrainenergydensity,Gstrainenergydensity;
      NCVariable<double>  gkineticenergydensity,Gkineticenergydensity;

      new_dw->allocateAndPut(ggridstress, lb->gGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(Ggridstress, lb->GGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(gdispgrads,  lb->gDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(Gdispgrads,  lb->GDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(gstrainenergydensity,
                                lb->gStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(Gstrainenergydensity,
                                lb->GStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(gkineticenergydensity,
                                lb->gKineticEnergyDensityLabel,dwi,patch);
      new_dw->allocateAndPut(Gkineticenergydensity,
                                lb->GKineticEnergyDensityLabel,dwi,patch);

      ggridstress.initialize(Matrix3(0.));
      Ggridstress.initialize(Matrix3(0.));
      gdispgrads.initialize(Matrix3(0.));
      Gdispgrads.initialize(Matrix3(0.));
      gstrainenergydensity.initialize(0.);
      Gstrainenergydensity.initialize(0.);
      gkineticenergydensity.initialize(0.);
      Gkineticenergydensity.initialize(0.);

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];

      if(calFractParameters || doCrackPropagation) {
        for (ParticleSubset::iterator iter = pset->begin();
                             iter != pset->end(); iter++) {
          particleIndex idx = *iter;
 
          // Get the node indices that surround the cell
          if(d_8or27==8){
            patch->findCellAndWeights(px[idx], ni, S);
          }
          else if(d_8or27==27){
            patch->findCellAndWeights27(px[idx], ni, S, psize[idx]);
          }

          for (int k = 0; k < d_8or27; k++){
            if(patch->containsNode(ni[k])){
              double pmassTimesS=pmass[idx]*S[k];
              if(pgCode[idx][k]==1) { 
                ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
              else if(pgCode[idx][k]==2) {
                Ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                Gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                Gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                Gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
            }
          } // End of loop over k
        } // End of loop over particles

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          // For primary field
          ggridstress[c]           /= gmass[c];
          gdispgrads[c]            /= gmass[c];
          gstrainenergydensity[c]  /= gmass[c];
          gkineticenergydensity[c] /= gmass[c];
          // For additional field
          Ggridstress[c]           /= Gmass[c];
          Gdispgrads[c]            /= Gmass[c];
          Gstrainenergydensity[c]  /= Gmass[c];
          Gkineticenergydensity[c] /= Gmass[c];
        }
      } // End if(calFractParameters || doCrackPropagation)
    }
  }
}

void Crack::addComputesAndRequiresCalculateFractureParameters(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Requires nodal solutions
  int NGC=NJ+NGN+1;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->GNumPatlsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->gDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->GDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->gGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->GGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->gDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->GDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->gStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->GStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->gKineticEnergyDensityLabel,gac,NGC);
  t->requires(Task::NewDW, lb->GKineticEnergyDensityLabel,gac,NGC);
  if(d_8or27==27) 
    t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);

  /*
  Computes J integral for each crack-front segments which are recorded 
  in cFrontSegPts[MNM]. J integral will be stored in 
  cFrontSegJ[MNM], a data member in Crack.h. For both end points 
  of each segment, they have the value of J integral of this segment. 
  J integrals will be converted to stress intensity factors
  for elastic materials.
  */
}

void Crack::CalculateFractureParameters(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int pid=patch->getID();

    // Variables related to MPI
    MPI_Datatype MPI_VECTOR=fun_getTypeDescription((Vector*)0)->getMPIType();
    int rank_size;
    MPI_Comm_size(mpi_crack_comm,&rank_size);

    enum {NO=0,YES};
    short outputJK=YES;

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m=0;m<numMatls;m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      int dwi = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<int>     GnumPatls;
      constNCVariable<Vector>  gdisp,Gdisp;
      constNCVariable<Matrix3> ggridStress,GgridStress;
      constNCVariable<Matrix3> gdispGrads,GdispGrads;
      constNCVariable<double>  gW,GW;
      constNCVariable<double>  gK,GK;

      // Get nodal solutions
      int NGC=NJ+NGN+1; 
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(GnumPatls,  lb->GNumPatlsLabel,     dwi,patch,gac,NGC);
      new_dw->get(gdisp,      lb->gDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(Gdisp,      lb->GDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(ggridStress,lb->gGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(GgridStress,lb->GGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(gdispGrads, lb->gDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(GdispGrads, lb->GDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(gW,lb->gStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(GW,lb->GStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(gK,lb->gKineticEnergyDensityLabel,  dwi,patch,gac,NGC);
      new_dw->get(GK,lb->GKineticEnergyDensityLabel,  dwi,patch,gac,NGC);

      constParticleVariable<Vector> psize;
      if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);

      if(calFractParameters || doCrackPropagation) {
        for(int i=0; i<cnumFrontSegs[m]; i++) { // Loop over crack front segments
          int tag_J=3*i;
          int tag_K=3*i+1;   

          /* Step 1: Define crack front segment coordinates
             v1,v2,v3: direction consies of new axes X',Y' and Z' */
          // Origin of crack-front coordinates, at center of the segment
          Point pt1,pt2,origin;                     
          pt1=cFrontSegPts[m][2*i];
          pt2=cFrontSegPts[m][2*i+1];
          origin=pt1+(pt2-pt1)/2.0;

          if(patch->containsPoint(origin)) { // Segment in the patch
            /* The origin and direction consines of new axes, 
               (v1,v2,v3) a right-hand system 
            */
            double x0,y0,z0;    
            x0=origin.x();  y0=origin.y();  z0=origin.z();

            Vector v1,v2,v3;
            double l1,m1,n1,l2,m2,n2,l3,m3,n3;
            v2=cFrontSegNorm[m][i]; 
            v3=TwoPtsDirCos(pt2,pt1); 
            v1=Cross(v2,v3);         

            l1=v1.x(); m1=v1.y(); n1=v1.z();
            l2=v2.x(); m2=v2.y(); n2=v2.z();
            l3=v3.x(); m3=v3.y(); n3=v3.z();

            // Coordinates transformation matrix from global to local
            Matrix3 T=Matrix3(l1,m1,n1,l2,m2,n2,l3,m3,n3);

            /* Step 2: Find parameters A[14] of J-path circle with equation
               A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0 and
               A10x+A11y+A12z+A13=0 */
            double A[14];    
            FindJPathCircle(origin,v1,v2,v3,A); 
 
            /* Step 3: Find intersection crossPt between J-ptah and crack plane
            */
            Point crossPt;  
            FindIntersectionOfJPathAndCrackPlane(m,rJ,A,crossPt);
 
            // Get coordinates of intersection in local system (xcprime,ycprime)
            double xc,yc,zc,xcprime,ycprime,scprime;
            xc=crossPt.x(); yc=crossPt.y(); zc=crossPt.z();
            xcprime=l1*(xc-x0)+m1*(yc-y0)+n1*(zc-z0);
            ycprime=l2*(xc-x0)+m2*(yc-y0)+n2*(zc-z0);
            scprime=sqrt(xcprime*xcprime+ycprime*ycprime);
   
            /* Step 4: Put integral points in J-path circle and do initialization
            */
            double xprime,yprime,x,y,z;
            double PI=3.141592654;
            int nSegs=16;                       // Number of segments on J-path circle
            Point*   pt = new Point[nSegs+1];   // Integral points
            double*  W  = new double[nSegs+1];  // Strain energy density
            double*  K  = new double[nSegs+1];  // Kinetic energy density
            Matrix3* st = new Matrix3[nSegs+1]; // Stresses in global coordinates
            Matrix3* dg = new Matrix3[nSegs+1]; // Disp grads in global coordinates
            Matrix3* st_prime=new Matrix3[nSegs+1]; // Stresses in local coordinates
            Matrix3* dg_prime=new Matrix3[nSegs+1]; // Disp grads in local coordinates
   
            for(int j=0; j<=nSegs; j++) {       // Loop over points on the circle
              double angle,cosTheta,sinTheta;
              angle=2*PI*(float)j/(float)nSegs;
              cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
              sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
              // Coordinates of integral points in local coordinates
              xprime=rJ*cosTheta;
              yprime=rJ*sinTheta;
              // Coordinates of integral points in global coordinates
              x=l1*xprime+l2*yprime+x0;
              y=m1*xprime+m2*yprime+y0;
              z=n1*xprime+n2*yprime+z0;
              pt[j]       = Point(x,y,z);
              W[j]        = 0.0;
              K[j]        = 0.0;
              st[j]       = Matrix3(0.);
              dg[j]       = Matrix3(0.);
              st_prime[j] = Matrix3(0.);
              dg_prime[j] = Matrix3(0.);
            }

            /* Step 5: Evaluate solutions at integral points in global coordinates
            */
            IntVector ni[MAX_BASIS];
            double S[MAX_BASIS];
            for(int j=0; j<=nSegs; j++) {
              if(d_8or27==8) 
                patch->findCellAndWeights(pt[j],ni,S);
              else if(d_8or27==27)
                patch->findCellAndWeights27(pt[j],ni,S,psize[j]);

              for(int k=0; k<d_8or27; k++) {
                if(GnumPatls[ni[k]]!=0 && j<nSegs/2) {  //below crack
                  W[j]  += GW[ni[k]]          * S[k];
                  K[j]  += GK[ni[k]]          * S[k];
                  st[j] += GgridStress[ni[k]] * S[k];
                  dg[j] += GdispGrads[ni[k]]  * S[k];
                }
                else { //above crack or in non-crack zone
                  W[j]  += gW[ni[k]]          * S[k];
                  K[j]  += gK[ni[k]]          * S[k];
                  st[j] += ggridStress[ni[k]] * S[k];
                  dg[j] += gdispGrads[ni[k]]  * S[k];
                }
              } // End of loop over k
            } // End of loop over j
  
            /* Step 6: Transform the solutions to crack-front coordinates
            */
            for(int j=0; j<=nSegs; j++) {
              for(int i1=1; i1<=3; i1++) {
                for(int j1=1; j1<=3; j1++) {
                  for(int i2=1; i2<=3; i2++) {
                    for(int j2=1; j2<=3; j2++) {
                      st_prime[j](i1,j1) += T(i1,i2)*T(j1,j2)*st[j](i2,j2);
                      dg_prime[j](i1,j1) += T(i1,i2)*T(j1,j2)*dg[j](i2,j2);
                    } 
                  } 
                } // End of loop over j1
              } // End of loop over i1
            } // End of loop over j

            /* Step 7: Get function values at integral points
            */
            double* f_J1 = new double[nSegs+1];
            double* f_J2 = new double[nSegs+1];
            for(int j=0; j<=nSegs; j++) {  
              double angle,cosTheta,sinTheta;
              double t1,t2,t3,dg11,dg21,dg31,dg12,dg22,dg32;
   
              angle=2*PI*(float)j/(float)nSegs;
              cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
              sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
              t1=st_prime[j](1,1)*cosTheta+st_prime[j](1,2)*sinTheta;
              t2=st_prime[j](2,1)*cosTheta+st_prime[j](2,2)*sinTheta;
              t3=st_prime[j](3,1)*cosTheta+st_prime[j](3,2)*sinTheta;
              dg11=dg_prime[j](1,1);
              dg21=dg_prime[j](2,1);
              dg31=dg_prime[j](3,1);
              dg12=dg_prime[j](1,2);
              dg22=dg_prime[j](2,2);
              dg32=dg_prime[j](3,2);
              // Plane strain in local coordinates
              dg31=0.0;
              dg32=0.0;
              f_J1[j]=(W[j]+K[j])*cosTheta-(t1*dg11+t2*dg21+t3*dg31);
              f_J2[j]=(W[j]+K[j])*sinTheta-(t1*dg12+t2*dg22+t3*dg32);
            }
  
            /* Step 8: Get J integral
            */
            double J1=0.,J2=0.; 
            for(int j=0; j<nSegs; j++) {   // Loop over segments
              J1 += f_J1[j] + f_J1[j+1];
              J2 += f_J2[j] + f_J2[j+1];
            } // End of loop over segments
            J1 *= rJ*PI/nSegs;
            J2 *= rJ*PI/nSegs; 
            cFrontSegJ[m][i]=Vector(J1,J2,0.);
   
            /* Step 9: Convert J to K 
            */
            // Task 9a: Find COD near crack tip (point(-d,0,0) in local coordinates)
            double d=rJ/2.;
            double x_d=l1*(-d)+x0;
            double y_d=m1*(-d)+y0;
            double z_d=n1*(-d)+z0;
            Point  p_d=Point(x_d,y_d,z_d);
          
            // Get displacements at point p_d 
            Vector disp_a=Vector(0.);
            Vector disp_b=Vector(0.);
            if(d_8or27==8)
              patch->findCellAndWeights(p_d,ni,S);
            else if(d_8or27==27)
              patch->findCellAndWeights27(p_d,ni,S,psize[0]);
            for(int k=0; k<d_8or27; k++) {
              Matrix3 sum_dg=gdispGrads[ni[k]]+GdispGrads[ni[k]];
              disp_a += gdisp[ni[k]] * S[k];
              disp_b += Gdisp[ni[k]] * S[k];
            }

            // Tranform to local system
            Vector disp_a_prime=Vector(0.);
            Vector disp_b_prime=Vector(0.);
            // Displacements
            for(int i1=0; i1<3; i1++) {
              for(int i2=0; i2<3; i2++) {
                disp_a_prime[i1] += T(i1+1,i2+1)*disp_a[i2];
                disp_b_prime[i1] += T(i1+1,i2+1)*disp_b[i2];
              }
            }
            // Crack opening displacements
            Vector D = disp_a_prime - disp_b_prime;

            // Task 9b: Get crack propagating velocity, currently just set it to zero
            Vector C=Vector(0.,0.,0.);   

            // Calculate SIF and output 
            Vector SIF;
            cm->ConvertJToK(mpm_matl,cFrontSegJ[m][i],C,D,SIF);      
            cFrontSegK[m][i]=SIF; 

            double scaler_K=1.0;
            if(outputJK && m==mS && i==iS) {
              cout << "    (patch " << pid << ", mat " << mS << ", seg " << iS << ")" 
                   << " --- Crack-tip(J_r=" << rJ << "): " << origin 
                   << ", COD: " << D << endl;
              cout << "    J: " << cFrontSegJ[m][i] 
                   << ", K: " << cFrontSegK[m][i]/scaler_K << endl;
            }

            // Output J & K at mS (matID) and iS (segID) in JK.dat
            if(m==mS && i==iS) {
              ofstream outJAndK("JK.dat", ios::app);
              double time=d_sharedState->getElapsedTime();
              outJAndK << setw(3) << mS << setw(3) << iS 
                       << setw(15) << time 
                       << setw(15) << cFrontSegJ[mS][iS].x() 
                       << setw(15) << cFrontSegJ[mS][iS].y()
                       << setw(15) << cFrontSegJ[mS][iS].z() 
                       << setw(15) << cFrontSegK[mS][iS].x() 
                       << setw(15) << cFrontSegK[mS][iS].y()
                       << setw(15) << cFrontSegK[mS][iS].z() << endl;
            }

            //Step 10: Release dynamic arries for this crack front segment
            delete [] pt;
            delete [] W;
            delete [] K;
            delete [] st; 
            delete [] dg;   
            delete [] st_prime;
            delete [] dg_prime;
            delete [] f_J1; 
            delete [] f_J2;
  
            // Task 11: Send J & K to all other ranks
            for(int rank=0; rank<rank_size; rank++) {
              if(rank!=pid) {
                MPI_Send(&cFrontSegJ[m][i],1,MPI_VECTOR,rank,tag_J,mpi_crack_comm);
                MPI_Send(&cFrontSegK[m][i],1,MPI_VECTOR,rank,tag_K,mpi_crack_comm);
              } 
            }
          } // End if patch->containsPoint(origin)
          else {  // All other ranks receive J & K 
            MPI_Status status;
            MPI_Recv(&cFrontSegJ[m][i],1,MPI_VECTOR,MPI_ANY_SOURCE,tag_J,mpi_crack_comm,&status);
            MPI_Recv(&cFrontSegK[m][i],1,MPI_VECTOR,MPI_ANY_SOURCE,tag_K,mpi_crack_comm,&status);
          } // End of if(patch->containsPoint(origin)) {} else
        } // End of loop over crack front segments
      } // End if(calFractParameters || doCrackPropagation)
    } // End of loop over matls
  } // End of loop patches
} 

void Crack::addComputesAndRequiresPropagateCracks(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const 
{ 
    // Nothing to do currently
}

void Crack::PropagateCracks(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++){
    //const Patch* patch = patches->get(p);
    //int pid=patch->getID();

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      // Do nothing currently
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresCrackPointSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Currently do nothing 
}

void Crack::CrackPointSubset(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  // Determine the crack nodes reside in each patch

  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    int pid=patch->getID();

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      cpset[pid][m].clear();
      for(int i=0; i<cnumNodes[m]; i++) {
        if(patch->containsPoint(cx[m][i])) {
          cpset[pid][m].push_back(i);
        } 
      } // End of loop over nodes
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresMoveCracks(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gMassLabel,         gac, 2*NGN);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     gac, 2*NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel, gac, 2*NGN);
  t->requires(Task::NewDW, lb->GMassLabel,         gac, 2*NGN);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     gac, 2*NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel, gac, 2*NGN);

  if(d_8or27==27)
   t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
}

void Crack::MoveCracks(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    int pid=patch->getID();

    enum {NO=0,YES};
    int rank_size;
    MPI_Comm_size(mpi_crack_comm, &rank_size);
    MPI_Datatype MPI_POINT=fun_getTypeDescription((Point*)0)->getMPIType();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );

    // Get the lowest and highest indexes of the whole grid, no offset included
    // need to be modified????
    const Patch* first_patch = patches->get(0);
    IntVector   gridLowIndex = first_patch->getCellLowIndex();
    Point                glp = first_patch->nodePosition(gridLowIndex);

    const Patch* last_patch = patches->get(patches->size()-1);
    IntVector gridHighIndex = last_patch->getCellHighIndex();
    Point               ghp = last_patch->nodePosition(gridHighIndex);
    //cout << "patch: " << pid <<  "glp: " << glp << ", ghp: " << ghp << endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){ // loop over matls    
      if(cnumElems[m]==0) // for materials with no cracks
        continue; 

      /* Get the necessary information
      */
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi=mpm_matl->getDWIndex();
      ParticleSubset* pset=old_dw->getParticleSubset(dwi,patch);
      constParticleVariable<Vector> psize;
      if(d_8or27==27) 
        old_dw->get(psize,lb->pSizeLabel,pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      constNCVariable<int>    gnum,Gnum;
      constNCVariable<Vector> gvelocity_star, Gvelocity_star;
      new_dw->get(gmass,         lb->gMassLabel,        dwi,patch,gac,2*NGN);
      new_dw->get(gnum,          lb->gNumPatlsLabel,    dwi,patch,gac,2*NGN);
      new_dw->get(gvelocity_star,lb->gVelocityStarLabel,dwi,patch,gac,2*NGN);
      new_dw->get(Gmass,         lb->GMassLabel,        dwi,patch,gac,2*NGN);
      new_dw->get(Gnum,          lb->GNumPatlsLabel,    dwi,patch,gac,2*NGN);
      new_dw->get(Gvelocity_star,lb->GVelocityStarLabel,dwi,patch,gac,2*NGN);

      /* Move crack nodes by looping over all crack nodes 
      */
      for(int i=0; i<cnumNodes[m]; i++) {
        int tag_x=3*i+2;

        // Detect if the node in the patch
        short nodeInPatch=NO;
        for(int j=0; j<(int)cpset[pid][m].size(); j++) {
          if(i==cpset[pid][m][j]) {
            nodeInPatch=YES;
            break;
          }
        }

        // If yes, update and send it out to the other ranks
        if(nodeInPatch) { 
          double mg,mG;
          Vector vg,vG;
          Vector vcm = Vector(0.0,0.0,0.0);

          // Step 1: Get element nodes and shape functions          
          IntVector ni[MAX_BASIS];
          double S[MAX_BASIS];
          if(d_8or27==8) 
            patch->findCellAndWeights(cx[m][i], ni, S);
          else if(d_8or27==27) 
            patch->findCellAndWeights27(cx[m][i], ni, S, psize[i]);

          // Step 2: Get center-of-velocity (vcm) 

          // Sum of shape functions from nodes with particle(s) around them
          // This part is necessary for cx located outside the body 
          double sumS=0.0;    
          for(int k =0; k < d_8or27; k++) {
            if(NodeWithinGrid(ni[k],gridLowIndex,gridHighIndex) &&
               (gnum[ni[k]]+Gnum[ni[k]]!=0)) sumS += S[k];
          }

          if(sumS>1.e-6) {   
            for(int k = 0; k < d_8or27; k++) {
              if(NodeWithinGrid(ni[k],gridLowIndex,gridHighIndex) &&
                 (gnum[ni[k]]+Gnum[ni[k]]!=0)) { 
                mg = gmass[ni[k]];
                mG = Gmass[ni[k]];
                vg = gvelocity_star[ni[k]];
                vG = Gvelocity_star[ni[k]];
                vcm += (mg*vg+mG*vG)/(mg+mG)*S[k]/sumS;
              }
            }
          }

          // Step 3: Apply symmetric boundary condition
          for(Patch::FaceType face = Patch::startFace;
               face<=Patch::endFace; face=Patch::nextFace(face)) {
            if(patch->getBCValues(dwi,"Symmetric",face) != 0) {
              if((face==Patch::xminus && fabs(cx[m][i].x()-glp.x())/dx.x()<1.e-3) || 
                 (face==Patch::xplus  && fabs(cx[m][i].x()-ghp.x())/dx.x()<1.e-3) )
              { //cout << "cx[" << m << "," << i << "] on symmetric face x" << endl; 
                vcm[0]=0.;
              }
              if((face==Patch::yminus && fabs(cx[m][i].y()-glp.y())/dx.y()<1.e-3) || 
                  (face==Patch::yplus  && fabs(cx[m][i].y()-ghp.y())/dx.y()<1.e-3) )
              { //cout << "cx[" << m << "," << i << "] on symmetric face y" << endl;
                vcm[1]=0.;
              } 
              if((face==Patch::zminus && fabs(cx[m][i].z()-glp.z())/dx.z()<1.e-3) || 
                 (face==Patch::zplus  && fabs(cx[m][i].z()-ghp.z())/dx.z()<1.e-3) )
              { //cout << "cx[" << m << "," << i << "] on symmetric face z" << endl;
                vcm[2]=0.;
              }
            }
          }

          // Step 4: Move cx with centerofmass velocity and send it out to all ranks
          cx[m][i] += vcm*delT;
          for(int rank=0; rank<rank_size; rank++) {  // loop over all ranks
            if(rank!=pid) MPI_Send(&cx[m][i],1,MPI_POINT,rank,tag_x,mpi_crack_comm);
          }  
        }
        else { // All other processors receive the updated cx 
               // even if they do not know the source of the information
          MPI_Status status;
          MPI_Recv(&cx[m][i],1,MPI_POINT,MPI_ANY_SOURCE,tag_x,mpi_crack_comm,&status);

        } // End of if(nodeInPatch) 

      } // End of loop over crack nodes
    } // End of loop over matls
  } // End of loop over patches
}

short Crack::NodeWithinGrid(const IntVector& ni, const IntVector& low, 
                                                 const IntVector& high)
{
  //low, high--lowest and highest indexes of the grid, no offset included 
  return (ni.x()>=low.x()  && ni.y()>=low.y()  && ni.z()>=low.z() &&
          ni.x()<=high.x() && ni.y()<=high.y() && ni.z()<=high.z()); 
}


void Crack::addComputesAndRequiresUpdateCrackExtentAndNormals(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Currently do nothing
}

void Crack::UpdateCrackExtentAndNormals(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++){
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      // Update crack extent for this material  
      cmin[m]=Point(9.e16,9.e16,9.e16);
      cmax[m]=Point(-9.e16,-9.e16,-9.e16);
      for(int i=0; i<cnumNodes[m]; i++) {
        cmin[m]=Min(cmin[m],cx[m][i]);
        cmax[m]=Max(cmax[m],cx[m][i]);
      } // End of loop over crack points 

      // Update crack element normals
      for(int i=0; i<cnumElems[m]; i++) {
        // n3, n4, n5 three nodes of the element
        int n3=cElemNodes[m][i].x();
        int n4=cElemNodes[m][i].y();
        int n5=cElemNodes[m][i].z();
        cElemNorm[m][i]=TriangleNormal(cx[m][n3],cx[m][n4],cx[m][n5]);
      } // End of loop crack elements
    } // End of loop over matls
  } // End of loop patches
}

// private functions

// Calculate outward normal of a triangle
Vector Crack::TriangleNormal(const Point& p1, 
                     const Point& p2, const Point& p3)
{
  double x21,x31,y21,y31,z21,z31;
  double a,b,c;
  Vector normal;

  x21=p2.x()-p1.x();
  x31=p3.x()-p1.x();
  y21=p2.y()-p1.y();
  y31=p3.y()-p1.y();
  z21=p2.z()-p1.z();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  if(Vector(a,b,c).length() >1.e-16)
     normal=Vector(a,b,c)/Vector(a,b,c).length();
  else
     normal=Vector(a,b,c);
  return normal;
}

// Detect if two points are in same side of a plane
short Crack::Location(const Point& p, const Point& g, 
        const Point& n1, const Point& n2, const Point& n3) 
{
  // p,g -- two points(particle and node)
  // n1,n2,n3 -- three points on the plane
  double x1,y1,z1,x2,y2,z2,x3,y3,z3,xp,yp,zp,xg,yg,zg;
  double x21,y21,z21,x31,y31,z31,a,b,c,d,dp,dg; 
  short cross;
 
  x1=n1.x(); y1=n1.y(); z1=n1.z();
  x2=n2.x(); y2=n2.y(); z2=n2.z();
  x3=n3.x(); y3=n3.y(); z3=n3.z();
  xp=p.x();  yp=p.y();  zp=p.z();
  xg=g.x();  yg=g.y();  zg=g.z();

  x21=x2-x1; y21=y2-y1; z21=z2-z1;
  x31=x3-x1; y31=y3-y1; z31=z3-z1;

  a=y21*z31-z21*y31;
  b=z21*x31-x21*z31;
  c=x21*y31-y21*x31;
  d=-a*x1-b*y1-c*z1;

  dp=a*xp+b*yp+c*zp+d;
  dg=a*xg+b*yg+c*zg+d;

  
  if(fabs(dg)<1.e-16) { // node on crack plane
     if(dp>0.) 
        cross=1;        // p above carck
     else
        cross=2;        // p below crack
  }
  else {                // node not on crack plane
     if(dp*dg>0.) 
        cross=0;        // p, g on same side
     else if(dp>0.) 
        cross=1;        // p above, g below
     else 
        cross=2;        // p below, g above
  }
  return cross;
}

// Compute signed volume of a tetrahedron
double Crack::Volume(const Point& p1, const Point& p2, 
                          const Point& p3, const Point& p)
{ 
   // p1,p2,p3 are three corners on bottom; p vertex
   double vol;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x,y,z;

   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x = p.x(); y = p.y(); z = p.z();

   vol=-(x1-x2)*(y3*z-y*z3)-(x3-x)*(y1*z2-y2*z1)
       +(y1-y2)*(x3*z-x*z3)+(y3-y)*(x1*z2-x2*z1)
       -(z1-z2)*(x3*y-x*y3)-(z3-z)*(x1*y2-x2*y1);

   if(fabs(vol)<1.e-16) return (0.);
   else return(vol);
}
  
IntVector Crack::CellOffset(const Point& p1, const Point& p2, Vector dx)
{
  int nx,ny,nz;
  if(fabs(p1.x()-p2.x())/dx.x()<1e-6) // p1.x()=p2.x()
    nx=NGN-1;
  else
    nx=NGN;
  if(fabs(p1.y()-p2.y())/dx.y()<1e-6) // p1.y()=p2.y()
    ny=NGN-1;
  else
    ny=NGN;
  if(fabs(p1.z()-p2.z())/dx.z()<1e-6) // p1.z()=p2.z()
    nz=NGN-1;
  else
    nz=NGN;

  return IntVector(nx,ny,nz);
}

// Detect if line p3-p4 included in line p1-p2
short Crack::TwoLinesDuplicate(const Point& p1,const Point& p2,
                                 const Point& p3,const Point& p4)
{
   double l12,l31,l32,l41,l42;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4;
   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x4=p4.x(); y4=p4.y(); z4=p4.z();

   l12=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
   l31=sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1));
   l32=sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2));
   l41=sqrt((x4-x1)*(x4-x1)+(y4-y1)*(y4-y1)+(z4-z1)*(z4-z1));
   l42=sqrt((x4-x2)*(x4-x2)+(y4-y2)*(y4-y2)+(z4-z2)*(z4-z2));

   if(fabs(l31+l32-l12)/l12<1.e-3 && fabs(l41+l42-l12)/l12<1.e-3)
     return 1;
   else
     return 0;
}

void Crack::FindJPathCircle(const Point& origin, const Vector& v1,
                    const Vector& v2,const Vector& v3, double A[])
{
   /* Find the parameters A0-A13 of J-patch circle with equation
      A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0
      and A10x+A11y+A12z+A13=0
      where r is radius of the circle */

   double x0,y0,z0;
   double l1,m1,n1,l2,m2,n2,l3,m3,n3;

   x0=origin.x(); y0=origin.y(); z0=origin.z();

   l1=v1.x(); m1=v1.y(); n1=v1.z();
   l2=v2.x(); m2=v2.y(); n2=v2.z();
   l3=v3.x(); m3=v3.y(); n3=v3.z();

   // parameters of the circle
   double term1,term2;
   term1=l1*x0+m1*y0+n1*z0;
   term2=l2*x0+m2*y0+n2*z0;

   A[0]=l1*l1+l2*l2;
   A[1]=m1*m1+m2*m2;
   A[2]=n1*n1+n2*n2;
   A[3]=2*(l1*m1+l2*m2);
   A[4]=2*(l1*n1+l2*n2);
   A[5]=2*(m1*n1+m2*n2);
   A[6]=-2*(l1*term1+l2*term2);
   A[7]=-2*(m1*term1+m2*term2);
   A[8]=-2*(n1*term1+n2*term2);
   A[9]=A[0]*x0*x0+A[1]*y0*y0+A[2]*z0*z0+A[3]*x0*y0+A[4]*x0*z0+A[5]*y0*z0;

   A[10]=l3;
   A[11]=m3; 
   A[12]=n3;
   A[13]=-(l3*x0+m3*y0+n3*z0);
}

void Crack::FindIntersectionOfJPathAndCrackPlane(const int& m,
              const double& radius, const double M[],Point& crossPt)
{
   /* Find intersection between J-path circle and crack plane.
          J-patch circle equations:
      Ax^2+By^2+Cz^2+Dxy+Exz+Fyz+Gx+Hy+Iz+J-r^2=0 and a1x+b1y+c1z+d1=0
          crack plane equation:
      a2x+b2y+c2z+d2=0
          m -- material ID, r -- radius of J-path circle. 
         Parameters of J-path circle stroed in array M.
   */
   double A,B,C,D,E,F,G,H,I,J,a1,b1,c1,d1;
   A=M[0];      a1=M[10];
   B=M[1];      b1=M[11];
   C=M[2];      c1=M[12];
   D=M[3];      d1=M[13];
   E=M[4]; 
   F=M[5]; 
   G=M[6]; 
   H=M[7]; 
   I=M[8]; 
   J=M[9];

   int numCross=0;
   for(int i=0; i<cnumElems[m]; i++) {  // Loop over crack segments
     // Find equation of crack segment: a2x+b2y+c2z+d2=0
     double a2,b2,c2,d2;   // parameters of a 3D plane
     Point pt1,pt2,pt3;    // three vertices of the segment
     pt1=cx[m][cElemNodes[m][i].x()];
     pt2=cx[m][cElemNodes[m][i].y()];
     pt3=cx[m][cElemNodes[m][i].z()];
     FindPlaneEquation(pt1,pt2,pt3,a2,b2,c2,d2);

     /* Define crack-segment coordinates system
        v1,v2,v3 -- dirction cosines of new axes X',Y' and Z'
     */  
     Vector v1,v2,v3;
     double term1 = sqrt(a2*a2+b2*b2+c2*c2);
     v2=Vector(a2/term1,b2/term1,c2/term1);
     v1=TwoPtsDirCos(pt1,pt2);
     v3=Cross(v1,v2);        // right-hand system

     // Get coordinates transform matrix
     double l1,m1,n1,l2,m2,n2,l3,m3,n3;

     l1=v1.x(); m1=v1.y(); n1=v1.z();
     l2=v2.x(); m2=v2.y(); n2=v2.z();
     l3=v3.x(); m3=v2.y(); n3=v3.z();

     /* Find intersection between J-path circle and crack plane
        first combine a1x+b1y+c1z+d1=0 And a2x+b2y+c2z+d2=0, get
        x=p1*z+q1 & y=p2*z+q2 (CASE 1) or 
        x=p1*y+q1 & z=p2*y+q2 (CASE 2) or 
        y=p1*x+q1 & z=p2*y+q2 (CASE 3), depending on the equations
        then combine with equation of the circle, getting the intersection
     */
     int CASE;
     double delt1,delt2,delt3,p1,q1,p2,q2;
     double abar,bbar,cbar,abc;
     double x1,y1,z1,x2,y2,z2;
     Point crossPt1,crossPt2;
 
     delt1=a1*b2-a2*b1;
     delt2=a1*c2-a2*c1;
     delt3=b1*c2-b2*c1;
     if(fabs(delt1)>=fabs(delt2) && fabs(delt1)>=fabs(delt3)) CASE=1;
     if(fabs(delt2)>=fabs(delt1) && fabs(delt2)>=fabs(delt3)) CASE=2;
     if(fabs(delt3)>=fabs(delt1) && fabs(delt3)>=fabs(delt2)) CASE=3;
     switch(CASE) {
       case 1:
         p1=(b1*c2-b2*c1)/delt1;
         q1=(b1*d2-b2*d1)/delt1;
         p2=(a2*c1-a1*c2)/delt1;
         q2=(a2*d1-a1*d2)/delt1;
         abar=p1*p1*A+p2*p2*B+C+p1*p2*D+p1*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*B+(p1*q2+p2*q1)*D+q1*E+q2*F+p1*G+p2*H+I;
         cbar=q1*q1*A+q2*q2*B+q1*q2*D+q1*G+q2*H+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         z1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*z1+q1;
         y1=p2*z1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         z2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*z2+q1;
         y2=p2*z2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 2:
         p1=(b2*c1-b1*c2)/delt2;
         q1=(c1*d2-c2*d1)/delt2;
         p2=(a2*b1-a1*b2)/delt2;
         q2=(a2*d1-a1*d2)/delt2;
         abar=p1*p1*A+B+p2*p2*C+p1*D+p1*p2*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*C+q1*D+(p1*q2+p2*q1)*E+q2*F+p1*G+H+p2*I;
         cbar=q1*q1*A+q2*q2*C+q1*q2*E+q1*G+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         y1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*y1+q1;
         z1=p2*y1+q2;
         crossPt1=Point(x1,y1,z1);
         //the second solution
         y2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*y2+q1;
         z2=p2*y2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 3:
         p1=(a2*c1-a1*c2)/delt3;
         q1=(c1*d2-c2*d1)/delt3;
         p2=(a1*b2-a2*b1)/delt3;
         q2=(b2*d1-b1*d2)/delt3;
         abar=A+p1*p1*B+p2*p2*C+p1*D+p2*E+p1*p2*F;
         bbar=2*p1*q1*B+2*p2*q2*C+q1*D+q2*E+(p1*q2+p2*q1)*F+G+p1*H+p2*I;
         cbar=q1*q1*B+q2*q2*C+q1*q2*F+q1*H+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         x1=0.5*(-bbar+sqrt(abc))/abar;
         y1=p1*x1+q1;
         z1=p2*x1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         x2=0.5*(-bbar-sqrt(abc))/abar;
         y2=p1*x2+q1;
         z2=p2*x2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
     }

     /* Detect if crossPt1 & crossPt2 in the triangular segment.
        transform and rotate the coordinates with the new origin at pt1
        and x'=pt1->pt2, (x',z') in crack plane
     */

     // Get new coordinates of the segment
     double xprime,yprime,zprime;   // Actually, yprime always zero
     Point pt1Prime,pt2Prime,pt3Prime;
     // the first point
     pt1Prime=Point(0.,0.,0.);
     // the second point
     xprime=l1*(pt2.x()-pt1.x())+m1*(pt2.y()-pt1.y())+n1*(pt2.z()-pt1.z());
     yprime=l2*(pt2.x()-pt1.x())+m2*(pt2.y()-pt1.y())+n2*(pt2.z()-pt1.z());
     zprime=l3*(pt2.x()-pt1.x())+m3*(pt2.y()-pt1.y())+n3*(pt2.z()-pt1.z());
     pt2Prime=Point(xprime,yprime,zprime);
     // the third point
     xprime=l1*(pt3.x()-pt1.x())+m1*(pt3.y()-pt1.y())+n1*(pt3.z()-pt1.z());
     yprime=l2*(pt3.x()-pt1.x())+m2*(pt3.y()-pt1.y())+n2*(pt3.z()-pt1.z());
     zprime=l3*(pt3.x()-pt1.x())+m3*(pt3.y()-pt1.y())+n3*(pt3.z()-pt1.z());
     pt3Prime=Point(xprime,yprime,zprime);

     // Get new coordinates of crossPts in local coordinates
     // For the first crossing point
     Point crossPt1Prime;
     xprime=l1*(x1-pt1.x())+m1*(y1-pt1.y())+n1*(z1-pt1.z());
     yprime=l2*(x1-pt1.x())+m2*(y1-pt1.y())+n2*(z1-pt1.z());
     zprime=l3*(x1-pt1.x())+m3*(y1-pt1.y())+n3*(z1-pt1.z());
     crossPt1Prime=Point(xprime,yprime,zprime);
     if(PointInTriangle(crossPt1Prime,pt1Prime,pt2Prime,pt3Prime)) {
       numCross++;
       if(numCross>1 && (crossPt1-crossPt).length()/radius>0.05) {
         cout << " More than one crossings found between J-path"
              << "  and crack plane.\n" 
              << " original crossing1: " << crossPt 
              << ", current crossing1: " << crossPt1 << endl;
         exit(1);
       }
       crossPt=crossPt1;
     }
     // For the second crossing point
     Point crossPt2Prime;
     xprime=l1*(x2-pt1.x())+m1*(y2-pt1.y())+n1*(z2-pt1.z());
     yprime=l2*(x2-pt1.x())+m2*(y2-pt1.y())+n2*(z2-pt1.z());
     zprime=l3*(x2-pt1.x())+m3*(y2-pt1.y())+n3*(z2-pt1.z());
     crossPt2Prime=Point(xprime,yprime,zprime);
     if(PointInTriangle(crossPt2Prime,pt1Prime,pt2Prime,pt3Prime)) {
       numCross++;
       if(numCross>1 && (crossPt2-crossPt).length()/radius>0.05) {
         cout << " More than one crossings found between J-path" 
              << "  and crack plane.\n" 
              << " original crossing2: " << crossPt 
              << ", current crossing2: " << crossPt2 << endl; 
         exit(1);
       }
       crossPt=crossPt2;
     }
   } // End of loop over crack segments

   if(numCross==0) {
     cout << "No crossing between J-path and crack plane found" << endl;
     exit(1);
   }

}

// Direction cossines of two points
Vector Crack::TwoPtsDirCos(const Point& p1,const Point& p2)
{
  double dx,dy,dz,ds;
  dx=p2.x()-p1.x(); 
  dy=p2.y()-p1.y(); 
  dz=p2.z()-p1.z();
  ds=sqrt(dx*dx+dy*dy+dz*dz);
  return Vector(dx/ds, dy/ds, dz/ds);   
}

// Find plane equation by three points
void Crack::FindPlaneEquation(const Point& p1,const Point& p2, 
            const Point& p3, double& a,double& b,double& c,double& d)
{
  // Get a,b,c,d for plane equation ax+by+cz+d=0
  double x21,x31,y21,y31,z21,z31;
  
  x21=p2.x()-p1.x();
  y21=p2.y()-p1.y();
  z21=p2.z()-p1.z();

  x31=p3.x()-p1.x();
  y31=p3.y()-p1.y();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  d=-p1.x()*a-p1.y()*b-p1.z()*c;
}   

// Detect if a point is within a triangle (2D case)
short Crack::PointInTriangle(const Point& p,const Point& pt1,
                           const Point& pt2,const Point& pt3)  
{
  // y=0 for all points

  double x1,z1,x2,z2,x3,z3,x,z;
  double area_p1p2p,area_p2p3p,area_p3p1p,area_p123;

  x1=pt1.x(); z1=pt1.z();
  x2=pt2.x(); z2=pt2.z();
  x3=pt3.x(); z3=pt3.z();
  x =p.x();   z =p.z();

  area_p1p2p=x1*z2+x2*z+x*z1-x1*z-x2*z1-x*z2;
  area_p2p3p=x2*z3+x3*z+x*z2-x2*z-x3*z2-x*z3;
  area_p3p1p=x3*z1+x1*z+x*z3-x3*z-x1*z3-x*z1;

  area_p123=fabs(x1*z2+x2*z3+x3*z1-x1*z3-x2*z1-x3*z2);

  if(fabs(area_p1p2p)/area_p123<1.e-3) area_p1p2p=0.;
  if(fabs(area_p2p3p)/area_p123<1.e-3) area_p2p3p=0.;
  if(fabs(area_p3p1p)/area_p123<1.e-3) area_p3p1p=0.;

  return (area_p1p2p<=0. && area_p2p3p<=0. && area_p3p1p<=0.);
}

// Detect if doing fracture analysis
void Crack::FindTimeStepForFractureAnalysis(double time)
{
  static double timeforsaving=0.0;
  static double timeforpropagation=0.0;

  if(d_calFractParameters=="true") {
    if(time>=timeforsaving) {
      calFractParameters=1;
      timeforsaving+=d_outputInterval;
    }
    else {
     calFractParameters=0;
    }
  }
  else if(d_calFractParameters=="false") {
   calFractParameters=0;
  }
  else if(d_calFractParameters=="every_time_step"){
    calFractParameters=1;
  }

  double propagationInterval=0.0;
  if(d_doCrackPropagation=="true") {
    if(time>=timeforpropagation){
      doCrackPropagation=1;
      timeforpropagation+=propagationInterval;
    }
    else {
      doCrackPropagation=0;
    }
  }
  else if(d_doCrackPropagation=="false") {
    doCrackPropagation=0;
  }
  else if(d_doCrackPropagation=="every_time_step"){
    calFractParameters=1;
  }
}

