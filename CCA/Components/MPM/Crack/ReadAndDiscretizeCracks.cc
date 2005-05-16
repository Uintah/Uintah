/********************************************************************************
    Crack.cc 
    PART ONE: CONSTRUCTOR, DECONSTRUCTOR, READ IN AND DISCRETIZE CRACKS 

    Created by Yajun Guo in 2002-2004.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> 
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;

#define MAX_BASIS 27

Crack::Crack(const ProblemSpecP& ps,SimulationStateP& d_sS,
             Output* d_dataArchiver, MPMLabel* Mlb,MPMFlags* MFlag)
{ 
  MPI_Comm_dup( MPI_COMM_WORLD, & mpi_crack_comm );

  /* Task 1: Initialization of fracture analysis  
  */
  d_sharedState = d_sS;
  dataArchiver  = d_dataArchiver;
  lb     = Mlb;
  flag   = MFlag;
  n8or27 = flag->d_8or27;
  outputJ= false; 

  if(n8or27==8) {NGP=1; NGN=1;}
  else if(n8or27==MAX_BASIS) {NGP=2; NGN=2;}
  
  // Default values of parameters for fracture analysis 
  rdadx=1.;         // Ratio of crack growth to cell-size
  rJ=-1.;           // Radius of J-path circle
  NJ=2;             // J contour size

  // Flag if saving crack geometry for visualization
  saveCrackGeometry=true;
     
  // Flags for fracture analysis
  d_calFractParameters = "false";
  d_doCrackPropagation = "false";
  
  // Flag if using volume-integral in J-integral computation
  useVolumeIntegral=false; 
  // Flag if smoothing crack-front
  smoothCrackFront=false;

  // Initialize boundary type
  for(Patch::FaceType face = Patch::startFace;
       face<=Patch::endFace; face=Patch::nextFace(face)) {
    GridBCType[face]="None";
  }
  
  // Extent of the global grid
  GLP=Point(-9e99,-9e99,-9e99); 
  GHP=Point( 9e99, 9e99, 9e99); 

  // Intervals for calculating JK and doing crack propagation
  calFractParasInterval=0.;
  crackPropInterval=0.;

  // Get .uda directory 
  ProblemSpecP uda_ps = ps->findBlock("DataArchiver");
  uda_ps->get("filebase", udaDir);
  uda_ps->get("save_crack_geometry", saveCrackGeometry);
      
  /* Task 2: Read in MPM parameters related to fracture analysis
  */
  ProblemSpecP mpm_soln_ps = ps->findBlock("MPM");
  if(mpm_soln_ps) {
     mpm_soln_ps->get("calculate_fracture_parameters", d_calFractParameters);
     if(d_calFractParameters!="true" && d_calFractParameters!="false"
	                 && d_calFractParameters!="every_time_step") {
        cout << "'calculate_fracture_parameters' can either be "
	     <<	"'true' or 'false' or 'every_time_step'." 
	     << " Program terminated." << endl;
        exit(1);	
     }
     
     mpm_soln_ps->get("do_crack_propagation", d_doCrackPropagation);
     if(d_doCrackPropagation!="true" && d_doCrackPropagation!="false"
                         && d_doCrackPropagation!="every_time_step") {        
       cout << "'do_crack_propagation' can either be "
            << "'true' or 'false' or 'every_time_step'." 
            << " Program terminated." << endl;
       exit(1);
     }

     mpm_soln_ps->get("use_volume_integral", useVolumeIntegral);
     mpm_soln_ps->get("smooth_crack_front", smoothCrackFront);
     mpm_soln_ps->get("J_radius", rJ);
     mpm_soln_ps->get("dadx",rdadx);
     mpm_soln_ps->get("outputJ",outputJ);
  }

  // Read in extent of the global grid
  ProblemSpecP grid_level_ps = ps->findBlock("Grid")
                           ->findBlock("Level")->findBlock("Box");
  grid_level_ps->get("lower",GLP);
  grid_level_ps->get("upper",GHP);

  // Read in boundary-condition types 
  ProblemSpecP grid_bc_ps = ps->findBlock("Grid")
                           ->findBlock("BoundaryConditions");
  for(ProblemSpecP face_ps = grid_bc_ps->findBlock("Face"); face_ps != 0;
                   face_ps = face_ps->findNextBlock("Face")) {
    map<string,string> values;
    face_ps->getAttributes(values);
    ProblemSpecP bcType_ps = face_ps->findBlock("BCType");
    map<string,string> bc_attr;
    bcType_ps->getAttributes(bc_attr);
    if(values["side"]=="x-")      GridBCType[Patch::xminus]=bc_attr["var"];
    else if(values["side"]=="x+") GridBCType[Patch::xplus] =bc_attr["var"];
    else if(values["side"]=="y-") GridBCType[Patch::yminus]=bc_attr["var"];
    else if(values["side"]=="y+") GridBCType[Patch::yplus] =bc_attr["var"];
    else if(values["side"]=="z-") GridBCType[Patch::zminus]=bc_attr["var"];
    else if(values["side"]=="z+") GridBCType[Patch::zplus] =bc_attr["var"];
  }

  /* Task 3: Allocate memory for crack geometry data
  */
  int numMPMMatls=0;
  ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                   mat_ps=mat_ps->findNextBlock("material") ) numMPMMatls++;
  // physical properties of cracks
  crackType.resize(numMPMMatls);
  cmu.resize(numMPMMatls);
  separateVol.resize(numMPMMatls);
  contactVol.resize(numMPMMatls);
  
  // quadrilateral crack segments
  rectangles.resize(numMPMMatls);
  rectN12.resize(numMPMMatls);
  rectN23.resize(numMPMMatls);
  rectCrackSidesAtFront.resize(numMPMMatls);

  // curved quadrilateral crack segments
  crectangles.resize(numMPMMatls);
  crectNStraightEdges.resize(numMPMMatls);
  crectPtsEdge2.resize(numMPMMatls);
  crectPtsEdge4.resize(numMPMMatls);
  crectCrackSidesAtFront.resize(numMPMMatls);
  
  // triangular crack segments
  triangles.resize(numMPMMatls);
  triNCells.resize(numMPMMatls);
  triCrackSidesAtFront.resize(numMPMMatls);

  // arc crack segments
  arcs.resize(numMPMMatls);
  arcNCells.resize(numMPMMatls);
  arcCrkFrtSegID.resize(numMPMMatls);

  // elliptic or partial alliptic crack segments
  ellipses.resize(numMPMMatls);
  ellipseNCells.resize(numMPMMatls);
  ellipseCrkFrtSegID.resize(numMPMMatls);
  pellipses.resize(numMPMMatls);
  pellipseNCells.resize(numMPMMatls);
  pellipseCrkFrtSegID.resize(numMPMMatls);
  pellipseExtent.resize(numMPMMatls);
   
  cmin.resize(numMPMMatls);
  cmax.resize(numMPMMatls);

  /* Task 4:  Read in crack parameters
  */
  int m=0;  // current mat ID
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material");
         mat_ps!=0; mat_ps=mat_ps->findNextBlock("material") ) {
    ProblemSpecP crk_ps=mat_ps->findBlock("crack");
    if(crk_ps==0) crackType[m]="NO_CRACK";
    if(crk_ps!=0) {
       // Read in crack contact type
       crk_ps->require("type",crackType[m]);

       // Read in parameters of crack contact.
       // Use displacement criterion to check contact
       // if separateVol or contactVol less than zero.
       cmu[m]=0.0;
       separateVol[m]=-1.;
       contactVol[m]=-1.;
       crk_ps->get("separate_volume",separateVol[m]);
       crk_ps->get("contact_volume",contactVol[m]);
       crk_ps->get("mu",cmu[m]);
       if(crackType[m]!="frictional" && crackType[m]!="stick" &&
          crackType[m]!="null") {
          cout << "Unknown crack type: " << crackType[m] << endl;
          exit(1);
       }

       // Initialize the arries related to crack geometries
       // for quadrilateral cracks
       rectangles[m].clear();
       rectN12[m].clear();
       rectN23[m].clear();
       rectCrackSidesAtFront[m].clear();
       // for curved quadrilateral cracks 
       crectangles[m].clear();
       crectNStraightEdges[m].clear();
       crectPtsEdge2[m].clear();
       crectPtsEdge4[m].clear();
       crectCrackSidesAtFront[m].clear();
       // for triangular cracks
       triangles[m].clear();
       triNCells[m].clear();
       triCrackSidesAtFront[m].clear();
       // for arc cracks
       arcs[m].clear();
       arcNCells[m].clear();
       arcCrkFrtSegID[m].clear();
       // for elliptical cracks
       ellipses[m].clear();
       ellipseNCells[m].clear();
       ellipseCrkFrtSegID[m].clear();
       // for partial elliptical cracks
       pellipses[m].clear();
       pellipseExtent[m].clear();
       pellipseNCells[m].clear();
       pellipseCrkFrtSegID[m].clear();

       // Read in parameters of cracks
       ProblemSpecP geom_ps=crk_ps->findBlock("crack_segments");
       ReadRectangularCracks(m,geom_ps);
       ReadCurvedRectangularCracks(m,geom_ps); 
       ReadTriangularCracks(m,geom_ps);
       ReadArcCracks(m,geom_ps);
       ReadEllipticCracks(m,geom_ps);
       ReadPartialEllipticCracks(m,geom_ps);
    } // End of if(crk_ps!=0)
    m++; // Next material
  }  // End of loop over materials

#if 1
  OutputInitialCrackPlane(numMPMMatls);
#endif
}

void Crack::ReadRectangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
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
    short atFront=NO;
    string cfsides("");
    quad_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==4) {
      for(string::const_iterator iter=cfsides.begin();
                        iter!=cfsides.end(); iter++) {
        if(*iter=='Y' || *iter=='y')      atFront=YES;
        else if(*iter=='N' || *iter=='n') atFront=NO;
        else {
          cout << " Wrong specification for crack_front_sides." << endl;
          exit(1);
        }
        thisRectCrackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<4; i++) thisRectCrackSidesAtFront.push_back(NO);
    }
    else {
      cout << " The length of string crack_front_sides for "
           << "quadrilaterals should be 4." << endl;
      exit(1);
    }
    rectCrackSidesAtFront[m].push_back(thisRectCrackSidesAtFront);
    thisRectCrackSidesAtFront.clear();
  } // End of loop over quadrilaterals
}

// Read in parameters of curved quadrilateral segments of crack
void Crack::ReadCurvedRectangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP cquad_ps=geom_ps->findBlock("curved_quad");
	         cquad_ps!=0; cquad_ps=cquad_ps->findNextBlock("curved_quad")) {
    Point p;
    // Four vertices of the curved quadrilateral
    vector<Point> vertices;    
    cquad_ps->require("p1",p);  vertices.push_back(p);
    cquad_ps->require("p2",p);  vertices.push_back(p);
    cquad_ps->require("p3",p);  vertices.push_back(p);
    cquad_ps->require("p4",p);  vertices.push_back(p);
    crectangles[m].push_back(vertices);
    vertices.clear();

    // Resolution on two opposite straight edges
    int n=1;
    cquad_ps->get("resolution_straight_edges",n);
    crectNStraightEdges[m].push_back(n);
    
    // Characteristic points on two opposite cuvered edges
    vector<Point> ptsEdge2,ptsEdge4;                   
    ProblemSpecP edge2_ps=cquad_ps->findBlock("points_curved_edge2"); 
    for(ProblemSpecP pt_ps=edge2_ps->findBlock("point"); pt_ps!=0; 
		     pt_ps=pt_ps->findNextBlock("point")) {  
      pt_ps->get("val",p); 
      ptsEdge2.push_back(p);
    }
    crectPtsEdge2[m].push_back(ptsEdge2);
    ptsEdge2.clear();

    ProblemSpecP edge4_ps=cquad_ps->findBlock("points_curved_edge4");
    for(ProblemSpecP pt_ps=edge4_ps->findBlock("point"); pt_ps!=0;
		     pt_ps=pt_ps->findNextBlock("point")) {
      pt_ps->get("val",p); 
      ptsEdge4.push_back(p);
    }
    crectPtsEdge4[m].push_back(ptsEdge4);
    ptsEdge4.clear();    
    
    // Crack front
    vector<short> crackSidesAtFront;
    short atFront=NO;
    string cfsides("");
    cquad_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==4) {
      for(string::const_iterator iter=cfsides.begin();
                                 iter!=cfsides.end(); iter++) {
        if(*iter=='Y' || *iter=='y')      atFront=YES;
        else if(*iter=='N' || *iter=='n') atFront=NO;
        else {
          cout << " Wrong specification for crack_front_sides." << endl;
          exit(1);
        }
        crackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<4; i++) crackSidesAtFront.push_back(NO);
    }
    else {
      cout << " The length of string crack_front_sides for "
           << "curved quadrilaterals should be 4." << endl;
      exit(1);
    }
    crectCrackSidesAtFront[m].push_back(crackSidesAtFront);
    crackSidesAtFront.clear();
  } // End of loop over cquadrilaterals
}

void Crack::ReadTriangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
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
    short atFront=NO;
    string cfsides("");
    tri_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==3) {
      for(string::const_iterator iter=cfsides.begin();
                        iter!=cfsides.end(); iter++) {
        if( *iter=='Y' || *iter=='n')     atFront=YES;
        else if(*iter=='N' || *iter=='n') atFront=NO;
        else {
          cout << " Wrong specification for crack_front_sides." << endl;
          exit(1);
        }
        thisTriCrackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<3; i++) thisTriCrackSidesAtFront.push_back(NO);
    }
    else {
      cout << " The length of string crack_front_sides for"
           << " triangles should be 3." << endl;
      exit(1);
    }
    triCrackSidesAtFront[m].push_back(thisTriCrackSidesAtFront);
    thisTriCrackSidesAtFront.clear();
  } // End of loop over triangles
}

void Crack::ReadArcCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP arc_ps=geom_ps->findBlock("arc");
       arc_ps!=0; arc_ps=arc_ps->findNextBlock("arc")) {
    int n;
    Point p;
    int cfsID=9999; // All segs are on crack-front by default
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
  } // End of loop over arcs
}

void Crack::ReadEllipticCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP ellipse_ps=geom_ps->findBlock("ellipse");
      ellipse_ps!=0; ellipse_ps=ellipse_ps->findNextBlock("ellipse")) {
    int n;
    Point p;
    int cfsID=9999;  // All segs are on crack-front by default
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
  } // End of loop over ellipses
}

void Crack::ReadPartialEllipticCracks(const int& m,
                        const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP pellipse_ps=geom_ps->findBlock("partial_ellipse");
      pellipse_ps!=0; pellipse_ps=
      pellipse_ps->findNextBlock("partial_ellipse")) {
    int n;
    Point p;
    string Extent;
    int cfsID=9999;  // All segs are on crack-front by default
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
    pellipse_ps->get("crack_front_segment_ID",cfsID);
    pellipseCrkFrtSegID[m].push_back(cfsID);
  } // End of loop over partial ellipses
}

void Crack::OutputInitialCrackPlane(const int& numMatls)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { //output from the first rank
    for(int m=0; m<numMatls; m++) {
      if(crackType[m]=="NO_CRACK")
        cout << "\nMaterial " << m << ": no crack exists" << endl;
      else {
        cout << "\nMaterial " << m << ":\n"
             << "  Crack contact type: " << crackType[m] << endl;
        if(crackType[m]=="frictional")
          cout << "    Frictional coefficient: " << cmu[m] << endl;

        if(crackType[m]!="null") {
          if(separateVol[m]<0. || contactVol[m]<0.)
            cout  << "    Check crack contact by displacement criterion." << endl;
          else {
            cout  << "Check crack contact by volume criterion with\n"
                  << "            separate volume = " << separateVol[m]
                  << "\n            contact volume = " << contactVol[m] << endl;
          }
        }

        cout <<"  Crack geometry:" << endl;
        // quadrilateral cracks
        for(int i=0;i<(int)rectangles[m].size();i++) {
          cout << "    Quadrilateral " << i << ": meshed by [" << rectN12[m][i]
               << ", " << rectN23[m][i] << ", " << rectN12[m][i]
               << ", " << rectN23[m][i] << "]" << endl;
          for(int j=0;j<4;j++)
            cout << "    p" << j+1 << ": " << rectangles[m][i][j] << endl;
          for(int j=0;j<4;j++) {
            if(rectCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<5 ? j+2 : 1);
              cout << "    Edge " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
        }

        // curved quadrilateral cracks
	for(int i=0;i<(int)crectangles[m].size();i++) {
	  cout << "    Curved quadrilateral " << i << ":" << endl;
	  cout << "    Four vertices:" << endl; 
	  // four vertices
	  for(int j=0;j<4;j++) 
	    cout << "      p" << j+1 << ": " << crectangles[m][i][j] << endl;
	  // resolution on straight edges 1 & 3
	  cout << "    Resolution on straight edges (edges p1-p2 and p3-p4):" << crectNStraightEdges[m][i] << endl; 
	  // points on curved egde 2
	  cout << "    Points on curved edge 2 (p2-p3): " << endl;
	  for(int j=0; j< (int)crectPtsEdge2[m][i].size(); j++)
            cout << "      p" << j+1 << ": " << crectPtsEdge2[m][i][j] << endl;
	  // points on curved edge 3
          cout << "    Points on curved edge 4 (p1-p4): " << endl;
          for(int j=0; j< (int)crectPtsEdge4[m][i].size(); j++)
	    cout << "      p" << j+1 << ": " << crectPtsEdge4[m][i][j] << endl; 
          // crack-front sides
	  for(int j=0;j<4;j++) {
	    if(crectCrackSidesAtFront[m][i][j]) {
	      int j2=(j+2<5 ? j+2 : 1);
	      cout << "    Edge " << j+1 << " (p" << j+1 << "-" << "p" << j2
	           << ") is a crack front." << endl;
	    }
	  }
	}	

        // Triangular cracks
        for(int i=0;i<(int)triangles[m].size();i++) {
          cout << "    Triangle " << i << ": meshed by [" << triNCells[m][i]
               << ", " << triNCells[m][i] << ", " << triNCells[m][i]
               << "]" << endl;
          for(int j=0;j<3;j++)
            cout << "    p" << j+1 << ": " << triangles[m][i][j] << endl;
          for(int j=0;j<3;j++) {
            if(triCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<4 ? j+2 : 1);
              cout << "    side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
        }

        // Arc cracks
        for(int i=0;i<(int)arcs[m].size();i++) {
          cout << "    Arc " << i << ": meshed by " << arcNCells[m][i]
               << " cells on the circumference." << endl;
          if(arcCrkFrtSegID[m][i]==9999)
	    cout << "   crack front: on the arc" << endl;
          else
	    cout << "   crack front segment ID: " << arcCrkFrtSegID[m][i] << endl;
          cout << "\n    start, middle and end points of the arc:"  << endl;
          for(int j=0;j<3;j++)
            cout << "    p" << j+1 << ": " << arcs[m][i][j] << endl;
        }

        // Elliptic cracks
        for(int i=0;i<(int)ellipses[m].size();i++) {
          cout << "    Ellipse " << i << ": meshed by " << ellipseNCells[m][i]
               << " cells on the circumference." << endl;
	  if(ellipseCrkFrtSegID[m][i]==9999)
	    cout << "    crack front: on the ellipse circumference" << endl;
          else	  
            cout << "    crack front segment ID: " << ellipseCrkFrtSegID[m][i]
                 << endl;
          cout << "    end point on axis1: " << ellipses[m][i][0] << endl;
          cout << "    end point on axis2: " << ellipses[m][i][1] << endl;
          cout << "    another end point on axis1: " << ellipses[m][i][2]
               << endl;
        }

        // Partial elliptic cracks
        for(int i=0;i<(int)pellipses[m].size();i++) {
          cout << "    Partial ellipse " << i << " (" << pellipseExtent[m][i]
               << "): meshed by " << pellipseNCells[m][i]
               << " cells on the circumference." << endl;
          if(pellipseCrkFrtSegID[m][i]==9999)
            cout << "    crack front: on the ellipse circumference" << endl;
          else
            cout << "    crack front segment ID: " << pellipseCrkFrtSegID[m][i]
                 << endl;
          cout << "    center: " << pellipses[m][i][0] << endl;
          cout << "    end point on axis1: " << pellipses[m][i][1] << endl;
          cout << "    end point on axis2: " << pellipses[m][i][2] << endl;
        }
      } // End of if(crackType...)
    } // End of loop over materials

    // Controlling parameters of crack propagation
    if(d_doCrackPropagation!="false") {
      cout << "  Ratio of crack increment to the cell size (dadx) = "
           << rdadx << "." << endl << endl;
    }
  }
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
  for(int p=0;p<patches->size();p++) { // All ranks
    const Patch* patch = patches->get(p);
       
    int pid,rankSize;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm,&rankSize);

    // Set radius (rJ) of J-path cirlce or number of cells
    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());

    if(rJ<0.) { // No specification in input file
      rJ=NJ*dx_min;
    }  
    else {      // Input radius of J-contour 
      NJ=(int)(rJ/dx_min);
    }  
       
    // Allocate memories for crack mesh data
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    css.resize(numMPMMatls);
    csa.resize(numMPMMatls);
    cx.resize(numMPMMatls);
    ce.resize(numMPMMatls);
    cfSegNodes.resize(numMPMMatls);
    cfSegTime.resize(numMPMMatls);
    cfSegDis.resize(numMPMMatls);
    cfSegVel.resize(numMPMMatls);
    cfSegPreIdx.resize(numMPMMatls);
    cfSegMinIdx.resize(numMPMMatls);
    cfSegMaxIdx.resize(numMPMMatls);
    cfSegPtsT.resize(numMPMMatls);
    cfSegV1.resize(numMPMMatls);
    cfSegV2.resize(numMPMMatls);
    cfSegV3.resize(numMPMMatls);
    cfSegJ.resize(numMPMMatls);
    cfSegK.resize(numMPMMatls);
    cnset.resize(numMPMMatls);
    cfnset.resize(numMPMMatls);
    cfsset.resize(numMPMMatls);

    for(int m = 0; m < numMPMMatls; m++){
      cnset[m].resize(rankSize);
      cfnset[m].resize(rankSize);
      cfsset[m].resize(rankSize);
      // Initialize the arries related to cracks
      cx[m].clear();
      ce[m].clear();
      cfSegNodes[m].clear();
      cmin[m]=Point(9e16,9e16,9e16);
      cmax[m]=Point(-9e16,-9e16,-9e16);

      if(crackType[m]!="NO_CRACK") {
        // Discretize crack plane
        int nstart0=0;  
        DiscretizeRectangularCracks(m,nstart0);
	DiscretizeCurvedRectangularCracks(m,nstart0);
        DiscretizeTriangularCracks(m,nstart0);
        DiscretizeArcCracks(m,nstart0);
        DiscretizeEllipticCracks(m,nstart0);
        DiscretizePartialEllipticCracks(m,nstart0); 

        // Find crack extent
        for(int i=0; i<(int)cx[m].size();i++) {
          cmin[m]=Min(cmin[m],cx[m][i]);
          cmax[m]=Max(cmax[m],cx[m][i]);
        }

        if(d_calFractParameters!="false"||d_doCrackPropagation!="false") {
	  // Initialize crack-front node velocity 
	  cfSegVel[m].resize((int)cfSegNodes[m].size());
	  cfSegTime[m].resize((int)cfSegNodes[m].size());
	  cfSegDis[m].resize((int)cfSegNodes[m].size());
	  for(int i=0; i<(int)cfSegNodes[m].size(); i++) {
	    cfSegVel[m][i]=0.0;
	    cfSegTime[m][i]=0.0;
	    cfSegDis[m][i]=0.0;
	  }

          // Get average length of crack-front segs
	  css[m]=0.;
	  int ncfSegs=(int)cfSegNodes[m].size()/2;
	  for(int i=0; i<ncfSegs; i++) {
	    int n1=cfSegNodes[m][2*i];
	    int n2=cfSegNodes[m][2*i+1];
	    css[m]+=(cx[m][n1]-cx[m][n2]).length();
	  }
	  css[m]/=ncfSegs;
	  
          // Get crack-front-node previous index, and sub-crack extent
          FindCrackFrontNodeIndexes(m);
	   
	  // Get average angle difference of adjacent crack-front segments
	  csa[m]=0.;
	  int count=0; 
	  for(int i=0; i<(int)cfSegNodes[m].size(); i++) {
	    int preIdx=cfSegPreIdx[m][i];
            if(preIdx>0) {
	      Point p =cx[m][cfSegNodes[m][i]];	    
	      Point p1=cx[m][cfSegNodes[m][i-2]];
	      Point p2=cx[m][cfSegNodes[m][i+1]];
	      Vector v1=TwoPtsDirCos(p1,p);
	      Vector v2=TwoPtsDirCos(p,p2);
	      csa[m]+=fabs(acos(Dot(v1,v2)))*180/3.141592654; 
	      count++;
	    }	    
	  }
	  if(count!=0) csa[m]/=count; 
	  else         csa[m]=90; 

          // Get normals of crack plane at crack-front nodes
          if(smoothCrackFront) {
            short smoothSuccessfully=SmoothCrackFrontAndCalculateNormals(m);
            if(!smoothSuccessfully) {
            //  if(pid==0) 
            //    cout << " ! Crack-front normals are obtained "
            //         << "by raw crack-front points." << endl;
            }
          }
          else {
            CalculateCrackFrontNormals(m);
          }
        } // End of if(..) 
         
        // Output crack mesh information
#if 1
        OutputInitialCrackMesh(m);
#endif
      }
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::DiscretizeRectangularCracks(const int& m,int& nstart0)
{
  int k,i,j,ni,nj,n1,n2,n3;
  int nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt;

  for(k=0; k<(int)rectangles[m].size(); k++) { // Loop over quads
    // Resolutions for the quadrilateral
    ni=rectN12[m][k];
    nj=rectN23[m][k];
    // Four vertices for the quad
    p1=rectangles[m][k][0];
    p2=rectangles[m][k][1];
    p3=rectangles[m][k][2];
    p4=rectangles[m][k][3];

    // Nodes on sides p2-p3 and p1-p4
    Point* side23=new Point[2*nj+1];
    Point* side14=new Point[2*nj+1];
    for(j=0; j<=2*nj; j++) {
      side23[j]=p2+(p3-p2)*(float)j/(2*nj);
      side14[j]=p1+(p4-p1)*(float)j/(2*nj);
    }

    // Generate crack points
    for(j=0; j<=nj; j++) {
      for(i=0; i<=ni; i++) {
        pt=side14[2*j]+(side23[2*j]-side14[2*j])*(float)i/ni;
        cx[m].push_back(pt);
      }
      if(j!=nj) {
        for(i=0; i<ni; i++) {
          int jj=2*j+1;
          pt=side14[jj]+(side23[jj]-side14[jj])*(float)(2*i+1)/(2*ni);
          cx[m].push_back(pt);
        }
      }  // End of if j!=nj
    } // End of loop over j

    // Create crack elemss for quad carck
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
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 2nd element
        n1=nstart2+i;
        n2=nstart3+i;
        n3=nstart1+i;
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 3rd element
        n1=nstart2+i;
        n2=nstart1+(i+1);
        n3=nstart3+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 4th element
        n1=nstart2+i;
        n2=nstart3+(i+1);
        n3=nstart3+i;
        ce[m].push_back(IntVector(n1,n2,n3));
      }  // End of loop over i
    }  // End of loop over j
    nstart0+=((2*ni+1)*nj+ni+1);
    delete [] side14;
    delete [] side23;

    // Collect crack-front seg nodes
    int seg0=0;
    for(j=0; j<4; j++) {
      if(!rectCrackSidesAtFront[m][k][j]) {
        seg0=j+1;
        break;
      }
    }
    for(int l=0; l<4; l++) { // Loop over sides of the quad
      j=seg0+l;
      if(j>3) j-=4;
      if(rectCrackSidesAtFront[m][k][j]) {
        int j1 = (j!=3 ? j+1 : 0);
        Point pt1=rectangles[m][k][j];
        Point pt2=rectangles[m][k][j1];
        for(i=0; i<(int)ce[m].size(); i++) {
          int ii=i;
          if(j>1) ii= (int) ce[m].size()-(i+1);
          n1=ce[m][ii].x();
          n2=ce[m][ii].y();
          n3=ce[m][ii].z();
          for(int s=0; s<3; s++) { // Loop over sides of the elem
            int sn=n1,en=n2;
            if(s==1) {sn=n2; en=n3;}
            if(s==2) {sn=n3; en=n1;}
            if(TwoLinesDuplicate(pt1,pt2,cx[m][sn],cx[m][en])) {
              cfSegNodes[m].push_back(sn);
              cfSegNodes[m].push_back(en);
            }
          }
        } // End of loop over i
      }
    } // End of loop over l
  } // End of loop over quads
}

void Crack::DiscretizeCurvedRectangularCracks(const int& m,int& nstart0)
{
  int k,i,j,ni,nj,n1,n2,n3;
  int nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt;

  for(k=0; k<(int)crectangles[m].size(); k++) { // Loop over quads
    // Four vertices for the quad
    p1=crectangles[m][k][0];
    p2=crectangles[m][k][1];
    p3=crectangles[m][k][2];
    p4=crectangles[m][k][3];
			
    // Resolutions on the curved edges (ni), and the straight edges (nj)
    ni=crectNStraightEdges[m][k];
    nj=crectPtsEdge2[m][k].size()+1;
    
    // Nodes on curved edges 2 (p2-p3) & 4 (p0-p4) - "j" direction
    Point* p_s2=new Point[2*nj+1];
    Point* p_s4=new Point[2*nj+1];
    p_s2[0]=p2;   p_s2[2*nj]=p3;
    p_s4[0]=p1;   p_s4[2*nj]=p4;
    for(int l=2; l<2*nj; l+=2) {
      p_s2[l]=crectPtsEdge2[m][k][l/2-1];
      p_s4[l]=crectPtsEdge4[m][k][l/2-1];
    } 	
    for(int l=1; l<2*nj; l+=2) {
      p_s2[l]=p_s2[l-1]+(p_s2[l+1]-p_s2[l-1])/2.;
      p_s4[l]=p_s4[l-1]+(p_s4[l+1]-p_s4[l-1])/2.; 
    }	
    
    // Generate crack points
    for(j=0; j<=nj; j++) {
      for(i=0; i<=ni; i++) { // nodes on lines
        pt=p_s4[2*j]+(p_s2[2*j]-p_s4[2*j])*(float)i/ni;
        cx[m].push_back(pt);
      }     
      if(j!=nj) {
        for(i=0; i<ni; i++) { // nodes at centers 
          int jj=2*j+1;
          pt=p_s4[jj]+(p_s2[jj]-p_s4[jj])*(float)(2*i+1)/(2*ni);
          cx[m].push_back(pt);
        }
      }  // End of if i!=ni
    } // End of loop over i
    delete [] p_s2;
    delete [] p_s4;

    // Generate crack elements
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
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 2nd element
        n1=nstart2+i;
        n2=nstart3+i;
        n3=nstart1+i;
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 3rd element
        n1=nstart2+i;
        n2=nstart1+(i+1);
        n3=nstart3+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 4th element
        n1=nstart2+i;
        n2=nstart3+(i+1);
        n3=nstart3+i;
        ce[m].push_back(IntVector(n1,n2,n3));
      }  // End of loop over j
    }  // End of loop over i
    nstart0+=((2*ni+1)*nj+ni+1);

    // Collect crack-front elements
    int seg0=0; 
    for(j=0; j<4; j++) {
      if(!crectCrackSidesAtFront[m][k][j]) {
        seg0=j+1;
        break;
      }
    }
    for(int l=0; l<4; l++) { // Loop over sides of the quad
      j=seg0+l;   
      if(j>3) j-=4;
      if(crectCrackSidesAtFront[m][k][j]) { // j is the side number of the front  
	// pt1-pt2 is crack-front      
        int j1 = (j!=3 ? j+1 : 0);
        Point pt1=crectangles[m][k][j];
        Point pt2=crectangles[m][k][j1];
        for(i=0; i<(int)ce[m].size(); i++) {
          int ii=i;
          if(j>1) ii= (int) ce[m].size()-(i+1);
          n1=ce[m][ii].x();
          n2=ce[m][ii].y();
          n3=ce[m][ii].z();
          for(int s=0; s<3; s++) { // Loop over sides of the elem
            int sn=n1,en=n2; 
            if(s==1) {sn=n2; en=n3;}
            if(s==2) {sn=n3; en=n1;}
            if(TwoLinesDuplicate(pt1,pt2,cx[m][sn],cx[m][en])) {
              cfSegNodes[m].push_back(sn);
              cfSegNodes[m].push_back(en);
            }
          }
        } // End of loop over i
      }
    } // End of loop over l		  

  } // End of loop over k
}  
    
void Crack::DiscretizeTriangularCracks(const int&m, int& nstart0)
{
  int neq=1;
  int k,i,j;
  int nstart1,nstart2,n1,n2,n3;
  Point p1,p2,p3,pt;

  for(k=0; k<(int)triangles[m].size(); k++) { // Loop over all triangles
    // Three vertices of the triangle
    p1=triangles[m][k][0];
    p2=triangles[m][k][1];
    p3=triangles[m][k][2];

    // Mesh resolution of the triangle
    neq=triNCells[m][k];

    // Create temprary arraies
    Point* side12=new Point[neq+1];
    Point* side13=new Point[neq+1];

    // Generate node coordinates
    for(j=0; j<=neq; j++) {
      side12[j]=p1+(p2-p1)*(float)j/neq;
      side13[j]=p1+(p3-p1)*(float)j/neq;
    }
    for(j=0; j<=neq; j++) {
      for(i=0; i<=j; i++) {
        double w=0.0;
        if(j!=0) w=(float)i/j;
        pt=side12[j]+(side13[j]-side12[j])*w;
        cx[m].push_back(pt);
      } // End of loop over i
    } // End of loop over j

    // Generate crack elems for the triangle
    for(j=0; j<neq; j++) {
      nstart1=nstart0+j*(j+1)/2;
      nstart2=nstart0+(j+1)*(j+2)/2;
      for(i=0; i<j; i++) {
        // Left element
        n1=nstart1+i;
        n2=nstart2+i;
        n3=nstart2+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // Right element
        n1=nstart1+i;
        n2=nstart2+(i+1);
        n3=nstart1+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
      } // End of loop over i
      n1=nstart0+(j+1)*(j+2)/2-1;
      n2=nstart0+(j+2)*(j+3)/2-2;
      n3=nstart0+(j+2)*(j+3)/2-1;
      ce[m].push_back(IntVector(n1,n2,n3));
    } // End of loop over j
    // Accumulate number of crack nodes
    nstart0+=(neq+1)*(neq+2)/2;
    delete [] side12;
    delete [] side13;

    // Collect crack-front-seg nodes
    int seg0=0;
    for(j=0; j<3; j++) {
      if(!triCrackSidesAtFront[m][k][j]) {
        seg0=j+1;
        break;
      }
    }
    for(int l=0; l<3; l++) { // Loop over sides of the triangle
      j=seg0+l;
      if(j>2) j-=3;
      if(triCrackSidesAtFront[m][k][j]) {
        int j1 = (j!=2 ? j+1 : 0);
        Point pt1=triangles[m][k][j];
        Point pt2=triangles[m][k][j1];
        for(i=0; i<(int)ce[m].size(); i++) {
          int ii=i;
          if(j>1) ii= (int) ce[m].size()-(i+1);
          n1=ce[m][ii].x();
          n2=ce[m][ii].y();
          n3=ce[m][ii].z();
          for(int s=0; s<3; s++) { // Loop over sides of the elem
            int sn=n1,en=n2;
            if(s==1) {sn=n2; en=n3;}
            if(s==2) {sn=n3; en=n1;}
            if(TwoLinesDuplicate(pt1,pt2,cx[m][sn],cx[m][en])) {
              cfSegNodes[m].push_back(sn);
              cfSegNodes[m].push_back(en);
            }
          }
        } // End of loop over i
      }
    } // End of loop over l
  } // End of loop over triangles
}

void Crack::DiscretizeArcCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)arcs[m].size(); k++) { // Loop over all arcs
    // Three points of the arc
    Point p1=arcs[m][k][0];
    Point p2=arcs[m][k][1];
    Point p3=arcs[m][k][2];
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
    Vector v31=Cross(v3,v1);
    v2=v31/v31.length();
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
    for(int j=0;j<=arcNCells[m][k];j++) {  // Loop over points
      double thetai=angleOfArc*j/arcNCells[m][k];
      double xiprime=radius*cos(thetai);
      double yiprime=radius*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } // End of loop over points

    // Crack elements
    for(int j=1;j<=arcNCells[m][k];j++) {  // Loop over segs
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+(j+1);
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(arcCrkFrtSegID[m][k]==9999 || arcCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nstart0+=arcNCells[m][k]+2;
  } // End of loop over arcs
}

void Crack::DiscretizeEllipticCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)ellipses[m].size(); k++) {
    // Three points of the ellipse
    Point p1=ellipses[m][k][0];
    Point p2=ellipses[m][k][1];
    Point p3=ellipses[m][k][2];

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
    Vector v12=Cross(v1,v2);
    v3=v12/v12.length();
    double lx,mx,nx,ly,my,ny;
    lx=v1.x();  mx=v1.y();  nx=v1.z();
    ly=v2.x();  my=v2.y();  ny=v2.z();

    // Collect crack nodes
    cx[m].push_back(origin);
    for(int j=0;j<ellipseNCells[m][k];j++) { 
      double PI=3.141592654;
      double thetai=j*(2*PI)/ellipseNCells[m][k];
      double xiprime=a*cos(thetai);
      double yiprime=b*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } 

    // Collect crack elems and crack-front elems
    for(int j=1;j<=ellipseNCells[m][k];j++) { // Loop over segs
      int j1 = (j==ellipseNCells[m][k]? 1 : j+1);
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+j1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(ellipseCrkFrtSegID[m][k]==9999 || 
         ellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nstart0+=ellipseNCells[m][k]+1;
  } // End ofloop over ellipses
}

void Crack::DiscretizePartialEllipticCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)pellipses[m].size(); k++) {
    double extent=0.0;
    if(pellipseExtent[m][k]=="quarter") extent=0.25;
    else if(pellipseExtent[m][k]=="half") extent=0.5;

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
    Vector v12=Cross(v1,v2);
    v3=v12/v12.length();
    double lx,mx,nx,ly,my,ny;
    lx=v1.x();  mx=v1.y();  nx=v1.z();
    ly=v2.x();  my=v2.y();  ny=v2.z();

    // Collect crack nodes
    cx[m].push_back(origin);
    for(int j=0;j<=pellipseNCells[m][k];j++) {
      double PI=3.141592654;
      double thetai=j*(2*PI*extent)/pellipseNCells[m][k];
      double xiprime=a*cos(thetai);
      double yiprime=b*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } 

    // Collect crack elems and crack-front elems
    for(int j=1;j<=pellipseNCells[m][k];j++) {
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+j+1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(pellipseCrkFrtSegID[m][k]==9999 ||
         pellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nstart0+=pellipseNCells[m][k]+2;
  } 
}

void Crack::OutputInitialCrackMesh(const int& m)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { // Output from the first rank
    cout << "\n---Initial Crack mesh---" << endl;
    cout << "MatID: " << m << endl;
    cout << "  Number of crack elems: " << (int)ce[m].size()
         << "\n  Number of crack nodes: " << (int)cx[m].size()
         << "\n  Number of crack-front elems: "
         << (int)cfSegNodes[m].size()/2 << endl;

    cout << "  Crack elements (" << (int)ce[m].size()
         << " elems in total):" << endl;
    for(int i=0; i<(int)ce[m].size(); i++) {
      cout << "     Elem " << i << ": " << ce[m][i] << endl;
    }

    cout << "  Crack nodes (" << (int)cx[m].size()
         << " nodes in total):" << endl;
    for(int i=0; i<(int)cx[m].size(); i++) {
      cout << "     Node " << i << ": " << cx[m][i] << endl;
    }

    cout << "  Crack-front elems and normals (" << (int)cfSegNodes[m].size()/2
         << " elems in total)" << endl;
    cout << "     V1: bi-normal; V2: outer normal; V3: tangential normal." << endl;
    for(int i=0; i<(int)cfSegNodes[m].size();i++) {
      cout << "     Seg " << i/2 << ": node "
           << cfSegNodes[m][i] << cx[m][cfSegNodes[m][i]]
           << ", V1: " << cfSegV1[m][i]
           << ", V2: " << cfSegV2[m][i]
           << ", V3: " << cfSegV3[m][i] << endl;
    }

    cout << "\n  Average length of crack front segs, css[m]="
         << css[m] << endl;
    cout << "\n  Average deviation of crack front segs, csa[m]="
		         << csa[m] << endl;
	

    cout << "\n  Crack extent: " << cmin[m] << "-->"
         <<  cmax[m] << endl << endl;
  }
}

Crack::~Crack()
{
}
