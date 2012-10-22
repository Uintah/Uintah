/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/********************************************************************************
    Crack.cc 
    PART ONE: CONSTRUCTOR, DECONSTRUCTOR, READ IN AND DISCRETIZE CRACKS 

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> 
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Crack/CrackGeometry.h>
#include <CCA/Components/MPM/Crack/CrackGeometryFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

using std::vector;
using std::string;

#define MAX_BASIS 27

Crack::Crack(const ProblemSpecP& ps,SimulationStateP& d_sS,
             Output* d_dataArchiver, MPMLabel* Mlb,MPMFlags* MFlag)
{ 
  MPI_Comm_dup( MPI_COMM_WORLD, & mpi_crack_comm );

  // Task 1: Initialization of fracture analysis  
  
  d_sharedState = d_sS;
  dataArchiver  = d_dataArchiver;
  lb            = Mlb;
  flag          = MFlag;
  n8or27        = flag->d_8or27;
  
  if(n8or27==8) 
    {NGP=1; NGN=1;}
  else if(n8or27==MAX_BASIS) 
    {NGP=2; NGN=2;} 
    
  d_calFractParameters = false;
  d_doCrackPropagation = false;
  useVolumeIntegral    = false; 
  smoothCrackFront     = false;
  saveCrackGeometry    = true;

  rdadx=1.;                     // Ratio of crack incremental to cell-size
  rJ=-1.;                       // Radius of J-integral contour
  NJ=2;                         // J-integral contour size  
  CODOption=0;                  // Calculate COD at a fixed location by default
  
  computeJKInterval=0.;         // Intervals of calculating J & K
  growCrackInterval=0.;         // Interval of doing crack propagation
  
  GLP=Point(-9e99,-9e99,-9e99); // Highest global grid
  GHP=Point( 9e99, 9e99, 9e99); // Lowest global grid
         
  // Initialize boundary type
  for(Patch::FaceType face = Patch::startFace;
                      face<=Patch::endFace; face=Patch::nextFace(face)) {
    GridBCType[face]="None";
  }
  

  // Task 2: Read in MPM parameters related to fracture analysis
  
  ProblemSpecP mpm_soln_ps = ps->findBlock("MPM");
  if(mpm_soln_ps) {
     mpm_soln_ps->get("calculate_fracture_parameters", d_calFractParameters);
     mpm_soln_ps->get("do_crack_propagation", d_doCrackPropagation);
     mpm_soln_ps->get("use_volume_integral", useVolumeIntegral);
     mpm_soln_ps->get("smooth_crack_front", smoothCrackFront);
     mpm_soln_ps->get("J_radius", rJ);
     mpm_soln_ps->get("dadx",rdadx);
     mpm_soln_ps->get("CODOption",CODOption);
  }

  // Get .uda directory 
  ProblemSpecP uda_ps = ps->findBlock("DataArchiver");
  uda_ps->get("filebase", udaDir);
  uda_ps->get("save_crack_geometry", saveCrackGeometry);
  uda_ps->get("computeJKInterval",computeJKInterval);
  uda_ps->get("growCrackInterval",growCrackInterval);
           
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


  // Task 3: Allocate memory for crack geometrical data
 
  int numMPMMatls=0;
  ProblemSpecP mpm_ps = 
    ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                   mat_ps=mat_ps->findNextBlock("material") ) numMPMMatls++;
  // Physical properties of cracks
  stressState.resize(numMPMMatls);
  crackType.resize(numMPMMatls);
  cmu.resize(numMPMMatls);

    // General quad crack segments
  quads.resize(numMPMMatls);
  quadN12.resize(numMPMMatls);
  quadN23.resize(numMPMMatls);
  quadCrackSidesAtFront.resize(numMPMMatls);
  quadRepetition.resize(numMPMMatls);
  quadOffset.resize(numMPMMatls);
    
  // Curved quad crack segments
  cquads.resize(numMPMMatls);
  cquadNStraightSides.resize(numMPMMatls);
  cquadPtsSide2.resize(numMPMMatls);
  cquadPtsSide4.resize(numMPMMatls);
  cquadCrackSidesAtFront.resize(numMPMMatls);
  cquadRepetition.resize(numMPMMatls);
  cquadOffset.resize(numMPMMatls);

  // Rriangular crack segments
  triangles.resize(numMPMMatls);
  triNCells.resize(numMPMMatls);
  triCrackSidesAtFront.resize(numMPMMatls);
  triRepetition.resize(numMPMMatls);
  triOffset.resize(numMPMMatls);

  // Arc crack segments
  arcs.resize(numMPMMatls);
  arcNCells.resize(numMPMMatls);
  arcCrkFrtSegID.resize(numMPMMatls);

  // Elliptic or partial alliptic crack segments
  ellipses.resize(numMPMMatls);
  ellipseNCells.resize(numMPMMatls);
  ellipseCrkFrtSegID.resize(numMPMMatls);
  pellipses.resize(numMPMMatls);
  pellipseNCells.resize(numMPMMatls);
  pellipseCrkFrtSegID.resize(numMPMMatls);
  pellipseExtent.resize(numMPMMatls);
   
  // Crack extent
  cmin.resize(numMPMMatls);
  cmax.resize(numMPMMatls);
  

  // Task 4:  Read in cracks
 
  int m=0;  
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material");
         mat_ps!=0; mat_ps=mat_ps->findNextBlock("material") ) {
    ProblemSpecP crk_ps=mat_ps->findBlock("crack");
    if(crk_ps==0) crackType[m]="NO_CRACK";
    if(crk_ps!=0) {
       // Crack surface contact type, either "friction", "stick" or "null"
       crk_ps->require("type",crackType[m]);
       if(crackType[m]!="friction" && crackType[m]!="stick" && crackType[m]!="null") {
         cout << "Error: unknown crack type: " << crackType[m] << endl;
         exit(1);
       }             

       // Friction coefficient needed for friction contact, zero by default
       cmu[m]=0.0;
       if(crackType[m]=="friction") crk_ps->get("mu",cmu[m]);
        
       // Stress state at crack front
       stressState[m]="planeStress"; 
       crk_ps->get("stress_state",stressState[m]);  

       // Read in crack segments. Presently seven kinds of basic shapes are available.
       // More complicated crack plane can be input by combining the basic shapes.
#if 0
       CrackGeometry* cg = CrackGeometryFactory::create(crk_ps);
       d_crackGeometry.push_back(cg);
#endif

       ProblemSpecP geom_ps=crk_ps->findBlock("crack_segments");
       ReadQuadCracks(m,geom_ps);
       ReadCurvedQuadCracks(m,geom_ps); 
       ReadTriangularCracks(m,geom_ps);
       ReadArcCracks(m,geom_ps);
       ReadEllipticCracks(m,geom_ps);
       ReadPartialEllipticCracks(m,geom_ps);
    } 
    m++;
  }  

  OutputInitialCrackPlane(numMPMMatls);
}

void
Crack::ReadCurvedQuadCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP cquad_ps=geom_ps->findBlock("curved_quad");
                 cquad_ps!=0; cquad_ps=cquad_ps->findNextBlock("curved_quad")) {
           
    // Four vertices of the curved quad
    Point p;
    vector<Point> vertices;    
    cquad_ps->require("p1",p);  vertices.push_back(p);
    cquad_ps->require("p2",p);  vertices.push_back(p);
    cquad_ps->require("p3",p);  vertices.push_back(p);
    cquad_ps->require("p4",p);  vertices.push_back(p);
    cquads[m].push_back(vertices);
    vertices.clear();

    // Mesh resolution on two opposite straight sides
    int n=1;
    cquad_ps->get("resolution_straight_sides",n);
    cquadNStraightSides[m].push_back(n);
    
    // Characteristic points on two opposite cuvered sides
    vector<Point> ptsSide2,ptsSide4;                   
    ProblemSpecP side2_ps=cquad_ps->findBlock("points_curved_side2"); 
    for(ProblemSpecP pt_ps=side2_ps->findBlock("point"); pt_ps!=0; 
                     pt_ps=pt_ps->findNextBlock("point")) {  
      pt_ps->get("val",p); 
      ptsSide2.push_back(p);
    }
    cquadPtsSide2[m].push_back(ptsSide2);

    ProblemSpecP side4_ps=cquad_ps->findBlock("points_curved_side4");
    for(ProblemSpecP pt_ps=side4_ps->findBlock("point"); pt_ps!=0;
                     pt_ps=pt_ps->findNextBlock("point")) {
      pt_ps->get("val",p); 
      ptsSide4.push_back(p);
    }
    cquadPtsSide4[m].push_back(ptsSide4);
    
    if(ptsSide4.size()!=ptsSide2.size()) {
      cout << "Error: The points on curved side 2 and side 4 "
           << "should appear in pairs." << endl;  
    }
    ptsSide2.clear();
    ptsSide4.clear();    
    
    // Crack front
    vector<short> crackSidesAtFront;
    string cfsides("");
    cquad_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==4) {
      for(string::const_iterator iter=cfsides.begin();
                                 iter!=cfsides.end(); iter++) {
         short atFront=NO;
        if(*iter=='Y' || *iter=='y') atFront=YES;
        crackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<4; i++)
        crackSidesAtFront.push_back(NO);
    }
    cquadCrackSidesAtFront[m].push_back(crackSidesAtFront);
    crackSidesAtFront.clear();

    // Repetition information
    n=1; 
    cquad_ps->get("repetition",n);
    cquadRepetition[m].push_back(n);

    Vector offset=Vector(0.,0.,0.);
    if(n>1) cquad_ps->require("offset",offset);
    cquadOffset[m].push_back(offset);
  }
}

void
Crack::ReadQuadCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP quad_ps=geom_ps->findBlock("quad");
      quad_ps!=0; quad_ps=quad_ps->findNextBlock("quad")) {

    // Four vertices (p1-p4) of the quad
    Point p1,p2,p3,p4,p5,p6,p7,p8;
    vector<Point> vertices;
    quad_ps->require("p1",p1);  vertices.push_back(p1);
    quad_ps->require("p2",p2);  vertices.push_back(p2);
    quad_ps->require("p3",p3);  vertices.push_back(p3);
    quad_ps->require("p4",p4);  vertices.push_back(p4);
    
    // Four middle points (p5-p8) of the quad
    if(!quad_ps->get("p5",p5)) p5=p1+0.5*(p2-p1);  vertices.push_back(p5);
    if(!quad_ps->get("p6",p6)) p6=p2+0.5*(p3-p2);  vertices.push_back(p6);
    if(!quad_ps->get("p7",p7)) p7=p3+0.5*(p4-p3);  vertices.push_back(p7);
    if(!quad_ps->get("p8",p8)) p8=p1+0.5*(p4-p1);  vertices.push_back(p8);
    
    quads[m].push_back(vertices);
    vertices.clear();
                  
    // Mesh resolutions 
    int n12=1,n23=1;
    quad_ps->get("resolution_p1_p2",n12);
    quadN12[m].push_back(n12);
    quad_ps->get("resolution_p2_p3",n23);
    quadN23[m].push_back(n23);
                  
    // Crack front
    vector<short> crackSidesAtFront;
    string cfsides("");
    quad_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==4) {
      for(string::const_iterator iter=cfsides.begin();
        iter!=cfsides.end(); iter++) {
        short atFront=NO;
        if(*iter=='Y' || *iter=='y') atFront=YES;
        crackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<4; i++)
      crackSidesAtFront.push_back(NO);
    }
    quadCrackSidesAtFront[m].push_back(crackSidesAtFront);
    crackSidesAtFront.clear();

    // Repetition information
    int n=1; 
    quad_ps->get("repetition",n);
    quadRepetition[m].push_back(n);
    
    Vector offset=Vector(0.,0.,0.);
    if(n>1) quad_ps->require("offset",offset);
    quadOffset[m].push_back(offset);    
  }
}  
                  
void
Crack::ReadTriangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP tri_ps=geom_ps->findBlock("triangle");
       tri_ps!=0; tri_ps=tri_ps->findNextBlock("triangle")) {

    // Three vertices (p1-p3) of the triangle
    Point p1,p2,p3,p4,p5,p6;
    vector<Point> vertices;     
    tri_ps->require("p1",p1);    vertices.push_back(p1);
    tri_ps->require("p2",p2);    vertices.push_back(p2);
    tri_ps->require("p3",p3);    vertices.push_back(p3); 

    // Three middle points (p4-p6) of the triangle
    if(!tri_ps->get("p4",p4)) p4=p1+0.5*(p2-p1);  vertices.push_back(p4);
    if(!tri_ps->get("p5",p5)) p5=p2+0.5*(p3-p2);  vertices.push_back(p5);
    if(!tri_ps->get("p6",p6)) p6=p3+0.5*(p1-p3);  vertices.push_back(p6);
    
    triangles[m].push_back(vertices);
    vertices.clear();

    // Mesh resolution 
    int n=1; 
    tri_ps->get("resolution",n);
    triNCells[m].push_back(n);

    // Crack front
    string cfsides("");
    vector<short> crackSidesAtFront;
    tri_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==3) {
      for(string::const_iterator iter=cfsides.begin();
                        iter!=cfsides.end(); iter++) {
        short atFront=NO;      
        if( *iter=='Y' || *iter=='n') atFront=YES;
        crackSidesAtFront.push_back(atFront);
      }
    }
    else if(cfsides.length()==0) {
      for(int i=0; i<3; i++) crackSidesAtFront.push_back(NO);
    }
    triCrackSidesAtFront[m].push_back(crackSidesAtFront);
    crackSidesAtFront.clear();

    // Repetition information
    n=1;
    tri_ps->get("repetition",n);
    triRepetition[m].push_back(n);
    
    Vector offset=Vector(0.,0.,0.);
    if(n>1) tri_ps->require("offset",offset);
    triOffset[m].push_back(offset);
  }
}

void
Crack::ReadArcCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP arc_ps=geom_ps->findBlock("arc");
                   arc_ps!=0; arc_ps=arc_ps->findNextBlock("arc")) {

    // Three points on the arc
    Point p;    
    vector<Point> thisArcPts;
    arc_ps->require("start_point",p);   thisArcPts.push_back(p);
    arc_ps->require("middle_point",p);  thisArcPts.push_back(p);
    arc_ps->require("end_point",p);     thisArcPts.push_back(p);
    arcs[m].push_back(thisArcPts);
    thisArcPts.clear();    

    // Resolution on circumference
    int n=1;
    arc_ps->require("resolution_circumference",n);
    arcNCells[m].push_back(n);

    // Crack front segment ID, -1 by default which means all segments are crack front
    int cfsID; 
    if(!arc_ps->get("crack_front_segment_ID",cfsID)) cfsID=-1;
    arcCrkFrtSegID[m].push_back(cfsID);
  } 
}

void
Crack::ReadEllipticCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP ellipse_ps=geom_ps->findBlock("ellipse");
      ellipse_ps!=0; ellipse_ps=ellipse_ps->findNextBlock("ellipse")) {

    // Three points on the arc
    Point p; 
    vector<Point> thisEllipsePts; 
    ellipse_ps->require("point1_axis1",p);   thisEllipsePts.push_back(p);
    ellipse_ps->require("point_axis2", p);   thisEllipsePts.push_back(p);
    ellipse_ps->require("point2_axis1",p);   thisEllipsePts.push_back(p);
    ellipses[m].push_back(thisEllipsePts);
    thisEllipsePts.clear();

    // Resolution on circumference
    int n=1; 
    ellipse_ps->require("resolution_circumference",n);
    ellipseNCells[m].push_back(n);

    // Crack front segment ID, -1 by default which means all segments are crack front
    int cfsID;  
    if(!ellipse_ps->get("crack_front_segment_ID",cfsID)) cfsID=-1;
    ellipseCrkFrtSegID[m].push_back(cfsID);
  } 
}

void
Crack::ReadPartialEllipticCracks(const int& m,
                                 const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP pellipse_ps=geom_ps->findBlock("partial_ellipse");
                   pellipse_ps!=0; 
                   pellipse_ps=pellipse_ps->findNextBlock("partial_ellipse")) {

    // Center,two points on major and minor axes
    Point p;
    vector<Point> thispEllipsePts; 
    pellipse_ps->require("center",p);       thispEllipsePts.push_back(p);
    pellipse_ps->require("point_axis1",p);  thispEllipsePts.push_back(p);
    pellipse_ps->require("point_axis2",p);  thispEllipsePts.push_back(p);
    pellipses[m].push_back(thispEllipsePts);
    thispEllipsePts.clear();

    // Extent in degree of the partial ellipse rotating from axis1
    double extent=360.;
    pellipse_ps->get("extent",extent);
    pellipseExtent[m].push_back(extent);

    // Resolution on circumference
    int n=1; 
    pellipse_ps->require("resolution_circumference",n);
    pellipseNCells[m].push_back(n);

    // Crack front segment ID, -1 by default which means all segments are crack front
    int cfsID; 
    if(!pellipse_ps->get("crack_front_segment_ID",cfsID)) cfsID=-1;
    pellipseCrkFrtSegID[m].push_back(cfsID);
  } 
}

void
Crack::OutputInitialCrackPlane(const int& numMatls)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { // output from the first rank
    for(int m=0; m<numMatls; m++) {
      if(crackType[m]=="NO_CRACK")
        cout << "\nMaterial " << m << ": no crack exists" << endl;
      else {
        cout << "\nMaterial " << m << ":\n"
             << "  Crack contact type: " << crackType[m] << endl;
        if(crackType[m]=="friction")
          cout << "    Frictional coefficient: " << cmu[m] << endl;

        cout <<"  Crack geometry:" << endl;
        // general quad cracks
        for(int i=0;i<(int)quads[m].size();i++) {
          cout << "  * Quad " << i+1 << ": meshed by [" << quadN12[m][i]
               << ", " << quadN23[m][i] << ", " << quadN12[m][i]
               << ", " << quadN23[m][i] << "]" << endl;
          for(int j=0;j<8;j++)
            cout << "    p" << j+1 << ": " << quads[m][i][j] << endl;
          for(int j=0;j<4;j++) {
            if(quadCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<5 ? j+2 : 1);
              cout << "    Side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
          // repetition information
          if(quadRepetition[m][i]>1) {
            cout << "    The quad is repeated " << quadRepetition[m][i]
                 << " times with the offset " << quadOffset[m][i] << "." << endl;
          }       
        }

        // curved quad cracks
        for(int i=0;i<(int)cquads[m].size();i++) {
          cout << "  * Curved quad " << i+1 << ":" << endl;
          cout << "    Four vertices:" << endl; 
          // four vertices
          for(int j=0;j<4;j++) 
            cout << "      p" << j+1 << ": " << cquads[m][i][j] << endl;
          // resolution on straight sides 1 & 3
          cout << "    Resolution on straight sides (sides p1-p2 and p3-p4):"
               << cquadNStraightSides[m][i] << endl; 
          // points on curved egde 2
          cout << "    Points on curved side 2 (p2-p3): " << endl;
          for(int j=0; j< (int)cquadPtsSide2[m][i].size(); j++)
            cout << "      p" << j+1 << ": " << cquadPtsSide2[m][i][j] << endl;
          // points on curved side 3
          cout << "    Points on curved side 4 (p1-p4): " << endl;
          for(int j=0; j< (int)cquadPtsSide4[m][i].size(); j++)
            cout << "      p" << j+1 << ": " << cquadPtsSide4[m][i][j] << endl; 
          // crack-front sides
          for(int j=0;j<4;j++) {
            if(cquadCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<5 ? j+2 : 1);
              cout << "    Side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
          // repetition information
          if(cquadRepetition[m][i]>1) {
            cout << "    The quad is repeated " << cquadRepetition[m][i]
                 << " times with the offset " << cquadOffset[m][i] << "." << endl;
          }   
        }       

        // Triangular cracks
        for(int i=0;i<(int)triangles[m].size();i++) {
          cout << "  * Triangle " << i+1 << ": meshed by [" << triNCells[m][i]
               << ", " << triNCells[m][i] << ", " << triNCells[m][i]
               << "]" << endl;
          for(int j=0;j<6;j++)
            cout << "    p" << j+1 << ": " << triangles[m][i][j] << endl;
          for(int j=0;j<3;j++) {
            if(triCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<4 ? j+2 : 1);
              cout << "    side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
          // repetition information
          if(triRepetition[m][i]>1) {
            cout << "    The triangle is repeated " << triRepetition[m][i]
                 << " times with the offset " << triOffset[m][i] << "." << endl;
          }       
        }

        // Arc cracks
        for(int i=0;i<(int)arcs[m].size();i++) {
          cout << "  * Arc " << i+1 << ": meshed by " << arcNCells[m][i]
               << " cells on the circumference." << endl;
          if(arcCrkFrtSegID[m][i]==-1)
            cout << "   crack front: on the arc" << endl;
          else
            cout << "   crack front segment ID: " << arcCrkFrtSegID[m][i] << endl;
          cout << "\n    start, middle and end points of the arc:"  << endl;
          for(int j=0;j<3;j++)
            cout << "    p" << j+1 << ": " << arcs[m][i][j] << endl;
        }

        // Elliptic cracks
        for(int i=0;i<(int)ellipses[m].size();i++) {
          cout << "  * Ellipse " << i+1 << ": meshed by " << ellipseNCells[m][i]
               << " cells on the circumference." << endl;
          if(ellipseCrkFrtSegID[m][i]==-1)
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
          cout << "  * Partial ellipse " << i+1 << " (" << pellipseExtent[m][i]
               << " degree): meshed by " << pellipseNCells[m][i]
               << " cells on the circumference." << endl;
          if(pellipseCrkFrtSegID[m][i]==-1)
            cout << "    crack front: on the ellipse circumference" << endl;
          else
            cout << "    crack front segment ID: " << pellipseCrkFrtSegID[m][i]
                 << endl;
          cout << "    center: " << pellipses[m][i][0] << endl;
          cout << "    end point on axis1: " << pellipses[m][i][1] << endl;
          cout << "    end point on axis2: " << pellipses[m][i][2] << endl;
        }
      } 
    } // End of loop over materials

    // Ratio of crack propagation incremental to cell-size
    if(d_doCrackPropagation) {
      cout << "  Ratio of crack increment to cell size (dadx) = "
           << rdadx << "." << endl << endl;
    }
  }
}

void
Crack::addComputesAndRequiresCrackDiscretization(Task* /*t*/,
                                                 const PatchSet* /*patches*/,
                                                 const MaterialSet* /*matls*/) const
{
  // Do nothing currently
}      

void
Crack::CrackDiscretization(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* /*matls*/,
                           DataWarehouse* /*old_dw*/,
                           DataWarehouse* /*new_dw*/)
{      
  for(int p=0;p<patches->size();p++) { 
    const Patch* patch = patches->get(p);
       
    int pid,rankSize;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm,&rankSize);

    // Set radius (rJ) of J-integral contour or number of cells
    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());

    if(rJ<0.) { // Input NJ, and calculate rJ
      rJ=NJ*dx_min;
    }  
    else {      // Input rJ, and calculate NJ 
      NJ=(int)(rJ/dx_min);
    }  
       
    // Allocate memories for crack mesh
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
      
      // Initialize crack extent
      cmin[m]=Point( 1.e200,  1.e200,  1.e200);
      cmax[m]=Point(-1.e200, -1.e200, -1.e200);

      if(crackType[m]!="NO_CRACK") {
        // Discretize crack segments
        int nnode0=0;  
        DiscretizeQuadCracks(m,nnode0);
        DiscretizeCurvedQuadCracks(m,nnode0);
        DiscretizeTriangularCracks(m,nnode0);
        DiscretizeArcCracks(m,nnode0);
        DiscretizeEllipticCracks(m,nnode0);
        DiscretizePartialEllipticCracks(m,nnode0); 

        // Combine crack segments 
        CombineCrackSegments(m);
        
        // Determine crack extent
        for(int i=0; i<(int)cx[m].size();i++) {
          cmin[m]=Min(cmin[m],cx[m][i]);
          cmax[m]=Max(cmax[m],cx[m][i]);
        }

        // Controlling parameters for fracture parameter calculation and crack propagation
        if(d_calFractParameters ||d_doCrackPropagation) {
          // Initialize parameters of crack-front nodes  
          int num=(int)cfSegNodes[m].size(); 
          cfSegVel[m].resize(num);
          cfSegTime[m].resize(num);
          cfSegDis[m].resize(num);
          for(int i=0; i<num; i++) {
            cfSegVel[m][i]=0.0;
            cfSegTime[m][i]=0.0;
            cfSegDis[m][i]=0.0;
          }

          // Get average length of crack-front segments
          css[m]=0.;
          int ncfSegs=num/2;
          for(int i=0; i<ncfSegs; i++) {
            int n1=cfSegNodes[m][2*i];
            int n2=cfSegNodes[m][2*i+1];
            css[m]+=(cx[m][n1]-cx[m][n2]).length();
          }
          css[m]/=ncfSegs;
          
          // Determine connectivity of crack-front nodes
          FindCrackFrontNodeIndexes(m);
           
          // Get average angle (in degree) of crack-front segments
          csa[m]=0.;
          int count=0; 
          for(int i=0; i<num; i++) {
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
          if(count!=0)
            csa[m]/=count; 
          else
            csa[m]=180; 
        
          // Calculate normals of crack plane at crack-front nodes
          if(smoothCrackFront) {
            if(!SmoothCrackFrontAndCalculateNormals(m))
              CalculateCrackFrontNormals(m);
          }
          else {
            CalculateCrackFrontNormals(m);
          }
          
        } 
#if 0
        OutputInitialCrackMesh(m);
#endif
      }
    } // End of loop over matls
  } 
}

void
Crack::DiscretizeCurvedQuadCracks(const int& m,int& nnode0)
{
  int k,k1,i,j,ni,nj,n1,n2,n3,num;
  int nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt;

  for(k=0; k<(int)cquads[m].size(); k++) {
    Vector offset=cquadOffset[m][k];      
    for(k1=0; k1<(int)cquadRepetition[m][k]; k1++) {       
      // Four vertices of the curved quad
      p1=cquads[m][k][0]+k1*offset;
      p2=cquads[m][k][1]+k1*offset;
      p3=cquads[m][k][2]+k1*offset;
      p4=cquads[m][k][3]+k1*offset;
                        
      // Mesh resolutions on curved sides (ni) & straight sides (nj)
      ni=cquadNStraightSides[m][k];
      nj=(int)(cquadPtsSide2[m][k].size())+1;
    
      // total number of nodes of the quad
      num=(ni+1)*(nj+1)+ni*nj;

      // Flag if node i is on edge j, initialized by NO 
      short** nodeOnEdge = scinew short*[num];
      for(i=0; i<num; i++) nodeOnEdge[i] = scinew short[4];
      for(i=0; i<num; i++) {
        for(j=0; j<4; j++) nodeOnEdge[i][j]=NO;
      }
    
      // Nodes on curved sides 2 (p2-p3) & 4 (p1-p4) - "j" direction
      Point* p_s2=new Point[2*nj+1];
      Point* p_s4=new Point[2*nj+1];
      p_s2[0]=p2;   p_s2[2*nj]=p3;
      p_s4[0]=p1;   p_s4[2*nj]=p4;
      for(int l=2; l<2*nj; l+=2) {
        p_s2[l]=cquadPtsSide2[m][k][l/2-1]+k1*offset;
        p_s4[l]=cquadPtsSide4[m][k][l/2-1]+k1*offset;
      }         
      for(int l=1; l<2*nj; l+=2) {
        p_s2[l]=p_s2[l-1]+(p_s2[l+1]-p_s2[l-1])/2.;
        p_s4[l]=p_s4[l-1]+(p_s4[l+1]-p_s4[l-1])/2.; 
      } 
    
      // Generate crack nodes
      int count=-1; 
      for(j=0; j<=nj; j++) {
        for(i=0; i<=ni; i++) { 
          // Detect edge nodes      
          count++;
          if(j==0)  nodeOnEdge[count][0]=YES;
          if(i==ni) nodeOnEdge[count][1]=YES;
          if(j==nj) nodeOnEdge[count][2]=YES;
          if(i==0)  nodeOnEdge[count][3]=YES;         
          pt=p_s4[2*j]+(p_s2[2*j]-p_s4[2*j])*(float)i/ni;
          cx[m].push_back(pt);
        }     
        if(j!=nj) {
          for(i=0; i<ni; i++) { 
            count++;    
            int jj=2*j+1;
            pt=p_s4[jj]+(p_s2[jj]-p_s4[jj])*(float)(2*i+1)/(2*ni);
            cx[m].push_back(pt);
          }
        }  
      } 
      delete [] p_s2;
      delete [] p_s4;

      // Generate crack elements
      for(j=0; j<nj; j++) {
        nstart1=nnode0+(2*ni+1)*j;
        nstart2=nstart1+(ni+1);
        nstart3=nstart2+ni;
        for(i=0; i<ni; i++) {
          // the 1st element
          n1=nstart2+i;  n2=nstart1+i;  n3=nstart1+(i+1);
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 2nd element
          n1=nstart2+i;  n2=nstart3+i;  n3=nstart1+i;
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 3rd element
          n1=nstart2+i;  n2=nstart1+(i+1);  n3=nstart3+(i+1);
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 4th element
          n1=nstart2+i;  n2=nstart3+(i+1);  n3=nstart3+i;
          ce[m].push_back(IntVector(n1,n2,n3));
        }  // End of loop over j
      }  // End of loop over i

      // Collect crack-front nodes
      for(int j=0; j<4; j++) { // Loop over sides of the quad
        if(cquadCrackSidesAtFront[m][k][j]) { 
          for(i=0; i<(int)ce[m].size(); i++) {
            // three element nodes        
            n1=ce[m][i].x();
            n2=ce[m][i].y();
            n3=ce[m][i].z();
            if(n1<nnode0 || n2<nnode0 || n3<nnode0) continue;
            for(int s=0; s<3; s++) { // Loop over sides of the element
              int sn=n1,en=n2; 
              if(s==1) {sn=n2; en=n3;}
              if(s==2) {sn=n3; en=n1;}
              if(nodeOnEdge[sn-nnode0][j] && nodeOnEdge[en-nnode0][j]) {
                cfSegNodes[m].push_back(sn);
                cfSegNodes[m].push_back(en);
              }
            }
          } // End of loop over i
        }
      } // End of loop over j             
      nnode0+=num;
      delete [] nodeOnEdge;
    }  
  } // End of loop over k
}  
    
void
Crack::DiscretizeQuadCracks(const int& m,int& nnode0)
{
  int k,l,i,j,ni,nj,n1,n2,n3,num;
  int nstart1,nstart2,nstart3;
  double ksi,eta;
  Point pt;

  for(k=0; k<(int)quads[m].size(); k++) {
    for(l=0; l<(int)quadRepetition[m][k]; l++) {    
      // Mesh resolutions of the quad
      ni=quadN12[m][k];
      nj=quadN23[m][k];
    
      // total number of nodes of the quad
      num=(ni+1)*(nj+1)+ni*nj; 

      // Flag if node i is on edge j, initialized by NO 
      short** nodeOnEdge = scinew short*[num];
      for(i=0; i<num; i++) nodeOnEdge[i] = scinew short[4];  
      for(i=0; i<num; i++) {
        for(j=0; j<4; j++) nodeOnEdge[i][j]=NO;
      }     
    
      // Generate crack nodes
      int count=-1; 
      for(j=0; j<=nj; j++) {
        for(i=0; i<=ni; i++) {
          // Detect edge nodes      
          count++;
          if(j==0)  nodeOnEdge[count][0]=YES;
          if(i==ni) nodeOnEdge[count][1]=YES;
          if(j==nj) nodeOnEdge[count][2]=YES;
          if(i==0)  nodeOnEdge[count][3]=YES;
          // Intrinsic coordinates
          ksi=-1.0+(float)(2*i)/ni;
          eta=-1.0+(float)(2*j)/nj;     
          // Global coordinates by interpolation with shape function      
          GetGlobalCoordinatesQuad(m,k,l,ksi,eta,pt);       
          cx[m].push_back(pt);
        }
        if(j!=nj) {
          for(i=0; i<ni; i++) {
            count++;            
            // intrinsic coordinates
            ksi=-1.0+(float)(2*i+1)/ni;
            eta=-1.0+(float)(2*j+1)/nj;   
            // Global coordinates                
            GetGlobalCoordinatesQuad(m,k,l,ksi,eta,pt);
            cx[m].push_back(pt);
          }
        }
      }
  
      // Generate crack elements
      for(j=0; j<nj; j++) {
        nstart1=nnode0+(2*ni+1)*j;
        nstart2=nstart1+(ni+1);
        nstart3=nstart2+ni;
        for(i=0; i<ni; i++) {
          // the 1st element
          n1=nstart2+i;  n2=nstart1+i;  n3=nstart1+(i+1);
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 2nd element
          n1=nstart2+i;  n2=nstart3+i;  n3=nstart1+i;
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 3rd element
          n1=nstart2+i;  n2=nstart1+(i+1);  n3=nstart3+(i+1);
          ce[m].push_back(IntVector(n1,n2,n3));
          // the 4th element
          n1=nstart2+i;  n2=nstart3+(i+1);  n3=nstart3+i;
          ce[m].push_back(IntVector(n1,n2,n3));
        }
      }
  
      // Collect crack-front nodes
      for(int j=0; j<4; j++) { // Loop over sides of the quad
        if(quadCrackSidesAtFront[m][k][j]) {
          for(i=0; i<(int)ce[m].size(); i++) {
            // three element nodes        
            n1=ce[m][i].x();
            n2=ce[m][i].y();
            n3=ce[m][i].z();
            if(n1<nnode0 || n2<nnode0 || n3<nnode0) continue;
            for(int s=0; s<3; s++) { // Loop over sides of the element
              int sn=n1,en=n2;
              if(s==1) {sn=n2; en=n3;}
              if(s==2) {sn=n3; en=n1;}
              if(nodeOnEdge[sn-nnode0][j] && nodeOnEdge[en-nnode0][j]) {
                cfSegNodes[m].push_back(sn);
                cfSegNodes[m].push_back(en);
              }
            }
          } // End of loop over i
        }
      } // End of loop over j
      nnode0+=num;
      for(int i=0;i<num;i++)
        delete nodeOnEdge[i];
      delete [] nodeOnEdge;
    }  
  } // End of loop over quads
}     

void Crack::GetGlobalCoordinatesQuad(const int& m, const int& k, 
        const int& l,const double& x, const double& y, Point& pt)
{
  // (x,y): intrinsic coordinates of point "pt".
         
  // Shape functions of the serendipity eight-noded quadrilateral element
  double sf[8];          
  sf[0]=(1.-x)*(1.-y)*(-1.-x-y)/4.;
  sf[1]=(1.+x)*(1.-y)*(-1.+x-y)/4.;
  sf[2]=(1.+x)*(1.+y)*(-1.+x+y)/4.;
  sf[3]=(1.-x)*(1.+y)*(-1.-x+y)/4.;
  sf[4]=(1.-x*x)*(1.-y)/2.;
  sf[5]=(1.+x)*(1.-y*y)/2.;
  sf[6]=(1.-x*x)*(1.+y)/2.;
  sf[7]=(1.-x)*(1.-y*y)/2.;

  // Global coordinates of (x,y)
  double px=0., py=0., pz=0.; 
  for(int j=0; j<8; j++) {
    px+=sf[j]*(quads[m][k][j].x()+l*quadOffset[m][k].x());
    py+=sf[j]*(quads[m][k][j].y()+l*quadOffset[m][k].y());
    pz+=sf[j]*(quads[m][k][j].z()+l*quadOffset[m][k].z());
  }  
  pt=Point(px,py,pz);
}

void
Crack::DiscretizeTriangularCracks(const int&m, int& nnode0)
{
  int k,l,i,j;
  int neq,num,nstart1,nstart2,n1=0,n2=0,n3=0;
  Point pt;

  for(k=0; k<(int)triangles[m].size(); k++) { 
    for(l=0; l<(int)triRepetition[m][k]; l++) {
      // Mesh resolution of the triangle
      neq=triNCells[m][k];
      
      // total number of nodes of the triangle  
      num=(neq+1)*(neq+2)/2; 

      // Flag if node 'i' is on edge 'j', initialized by NO 
      short** nodeOnEdge = scinew short*[num];
      for(i=0; i<num; i++) nodeOnEdge[i] = scinew short[3];  
      for(i=0; i<num; i++) {
        for(j=0; j<3; j++) nodeOnEdge[i][j]=NO;
      }     
      
      // Generate crack nodes 
      int count=-1; 
      for(j=0; j<=neq; j++) {
        for(i=0; i<=neq-j; i++) {
          // Detect edge nodes
          count++;
          if(j==0)     nodeOnEdge[count][0]=YES;
          if(i+j==neq) nodeOnEdge[count][1]=YES;          
          if(i==0)     nodeOnEdge[count][2]=YES;
          // Intrinsic coordinates
          double ksi=(float)i/neq;
          double eta=(float)j/neq;
          // Global coordinates by interpolation with shape function 
          GetGlobalCoordinatesTriangle(m,k,l,ksi,eta,pt);
          cx[m].push_back(pt);
        } 
      } 

      // Generate crack elements 
      nstart2=nnode0;
      for(j=0; j<neq-1; j++) {
        nstart2+=(neq+1-j);
        nstart1=nstart2-(neq+1-j);
        for(i=0; i<neq-(j+1); i++) {
          // left element
          n1=nstart1+i;  n2=n1+1;  n3=nstart2+i;
          ce[m].push_back(IntVector(n1,n2,n3));
          // right element
          n1=nstart1+(i+1);  n2=nstart2+(i+1);  n3=nstart2+i;
          ce[m].push_back(IntVector(n1,n2,n3));
        } 
        ce[m].push_back(IntVector(n1,n1+1,n2));
      } 
      ce[m].push_back(IntVector(nstart2,nstart2+1,nstart2+2));

      // Collect crack-front nodes
      for(int j=0; j<3; j++) { // Loop over sides of the triangle
        if(triCrackSidesAtFront[m][k][j]) {
          for(i=0; i<(int)ce[m].size(); i++) {
            // three nodes of the element        
            n1=ce[m][i].x();
            n2=ce[m][i].y();
            n3=ce[m][i].z();
            if(n1<nnode0 || n2<nnode0 || n3<nnode0) continue;
            for(int s=0; s<3; s++) { // Loop over sides of the element
              int sn=n1,en=n2;
              if(s==1) {sn=n2; en=n3;}
              if(s==2) {sn=n3; en=n1;}
              if(nodeOnEdge[sn-nnode0][j] && nodeOnEdge[en-nnode0][j]) {
                cfSegNodes[m].push_back(sn);
                cfSegNodes[m].push_back(en);
              }
            }
          } // End of loop over i
        }
      } // End of loop over j      
      nnode0+=num;
      delete [] nodeOnEdge;
    } // End of loop over l
  }
}

void Crack::GetGlobalCoordinatesTriangle(const int& m, const int& k,
                            const int& l,const double& r, const double& s, Point& pt)
{           
  // (r,s): intrinsic coordinates of point "pt".
        
  // Shape functions of the serendipity six-noded triangular element
  double sf[6];
  sf[5]=4.*s*(1.-r-s);  
  sf[4]=4.*r*s;
  sf[3]=4.*r*(1.-r-s);
  sf[2]=s-0.5*(sf[4]+sf[5]);
  sf[1]=r-0.5*(sf[3]+sf[4]);
  sf[0]=(1.-r-s)-0.5*(sf[3]+sf[5]);
        
  // Global coordinates of (r,s)
  double px=0., py=0., pz=0.;
  for(int j=0; j<6; j++) {
    px+=sf[j]*(triangles[m][k][j].x()+l*triOffset[m][k].x());
    py+=sf[j]*(triangles[m][k][j].y()+l*triOffset[m][k].y());
    pz+=sf[j]*(triangles[m][k][j].z()+l*triOffset[m][k].z());
  }
  pt=Point(px,py,pz);
}

void Crack::DiscretizeArcCracks(const int& m, int& nnode0)
{
  for(int k=0; k<(int)arcs[m].size(); k++) { 
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

    // Generate crack nodes
    cx[m].push_back(origin);
    for(int j=0;j<=arcNCells[m][k];j++) {
      double thetai=angleOfArc*j/arcNCells[m][k];
      double xiprime=radius*cos(thetai);
      double yiprime=radius*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } 

    // Generate crack elements
    for(int j=1;j<=arcNCells[m][k];j++) {
      int n1=nnode0;
      int n2=nnode0+j;
      int n3=nnode0+(j+1);
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front nodes
      if(arcCrkFrtSegID[m][k]==-1 || arcCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nnode0+=arcNCells[m][k]+2;
  }
}

void
Crack::DiscretizeEllipticCracks(const int& m, int& nnode0)
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

    // Generate crack nodes
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

    // Generate crack elements
    for(int j=1;j<=ellipseNCells[m][k];j++) { 
      int j1 = (j==ellipseNCells[m][k]? 1 : j+1);
      int n1=nnode0;
      int n2=nnode0+j;
      int n3=nnode0+j1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Collect crack-front nodes
      if(ellipseCrkFrtSegID[m][k]==-1 || 
         ellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nnode0+=ellipseNCells[m][k]+1;
  } 
}

void
Crack::DiscretizePartialEllipticCracks(const int& m, int& nnode0)
{
  for(int k=0; k<(int)pellipses[m].size(); k++) {
    double extent=pellipseExtent[m][k]/360.;

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

    // Generate crack nodes
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

    // Generate crack elements
    for(int j=1;j<=pellipseNCells[m][k];j++) {
      int n1=nnode0;
      int n2=nnode0+j;
      int n3=nnode0+j+1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Collect crack-front nodes
      if(pellipseCrkFrtSegID[m][k]==-1 ||
         pellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
      }
    }
    nnode0+=pellipseNCells[m][k]+2;
  } 
}

void
Crack::CombineCrackSegments(const int& m)
{
  int num=(int)cx[m].size();      
  
  for(int i=0; i<num; i++) {
    for(int j=0; j<i; j++) {
      if(TwoPointsCoincide(cx[m][i], cx[m][j])) {
        ResetCrackNodes(m,i,j);
        ResetCrackElements(m,i,j);
        ResetCrackFrontNodes(m,i,j);
        // After dropping node i, cx[m].size() decreses by 1 
        num--;
        // The components (>i) in cx[m] move back a record.
        // It's tricky, but necessary for multiple coincidences.
        i--;
      } 
    }
  } 

  // Reorder crack-front nodes 
  ReorderCrackFrontNodes(m);

  // Check for errors
  for(int i=0; i<num; i++) {
    for(int j=0; j<num; j++) {
      if(i!=j && TwoPointsCoincide(cx[m][i],cx[m][j])) {
        cout << "Error: duplicate crack nodes are found: cx[" 
             << m << "][" << i << "] = " << "cx[" << m << "][" 
             << j << "]. Program terminated." << endl;
        exit(1);
      }
    }
  }
  for(int i=0; i<(int)ce[m].size(); i++) {
    int n1=ce[m][i].x(), n2=ce[m][i].y(), n3=ce[m][i].z();
    if(n1==n2 || n1==n3 || n2==n3) {
      cout << "Error: crack element ce[m][" << i << "] = " << ce[m][i] 
           << " has two same nodes. Program terminted." << endl;            
      exit(1);
    }  
  } 
}

short
Crack::TwoPointsCoincide(const Point& p1, const Point& p2)
{
  double t=1.e-6;       
  return TwoDoublesEqual( p1.x(), p2.x(), t )  &&
         TwoDoublesEqual( p1.y(), p2.y(), t )  &&
         TwoDoublesEqual( p1.z(), p2.z(), t );
}
        
short
Crack::TwoDoublesEqual(const double& db1, const double& db2, 
                       const double& tolerance)
{
  double ab1=fabs(db1);
  double ab2=fabs(db2);
  double change;
   
  if(db1==db2) return(YES);
  else if(ab1>ab2)
    change=fabs(db1-db2)/ab1;
  else
    change=fabs(db1-db2)/ab2;
   
  // Equal if different by less than 100 ppm
  if(change<tolerance) return(YES);
  else {
    if(ab1<tolerance/100. && ab2<tolerance/100.) return(YES);
    else return(NO);
  }
}
        
void
Crack::ResetCrackNodes(const int& m, const int& n1, const int& /*n2*/)
{
  // If cx[m][n1]=cx[m][n2], drop node n1
                
  int num=(int)cx[m].size();
  Point* tmp = scinew Point[num];    
  for(int i=0; i<num; i++) tmp[i]=cx[m][i];

  cx[m].clear();
  cx[m].resize(num-1);
  for(int i=0; i<num-1; i++) {
    if(i<n1) 
      cx[m][i]=tmp[i];
    else
      cx[m][i]=tmp[i+1];
  }
  delete [] tmp;
}

void
Crack::ResetCrackElements(const int& m, const int& n1, const int& n2)
{
   int num=(int)ce[m].size();
   IntVector* tmp = scinew IntVector[num];
   for(int i=0; i<num; i++) tmp[i]=ce[m][i]; 

   for(int i=0; i<num; i++) { 
     int n=-1;     
     // first node of the element          
     n=tmp[i].x();
     if(n<n1)        ce[m][i][0]=n;
     else if(n==n1)  ce[m][i][0]=n2;
     else            ce[m][i][0]=n-1;
     // second node of the element
     n=tmp[i].y();
     if(n<n1)        ce[m][i][1]=n;
     else if(n==n1)  ce[m][i][1]=n2;
     else            ce[m][i][1]=n-1;
     // third node of the element 
     n=tmp[i].z();
     if(n<n1)        ce[m][i][2]=n;
     else if(n==n1)  ce[m][i][2]=n2;
     else            ce[m][i][2]=n-1;     
   }
   delete [] tmp;
}

void
Crack::ResetCrackFrontNodes(const int& m, const int& n1, const int& n2)
{
   int num=(int)cfSegNodes[m].size();
   int* tmp = scinew int[num];
   for(int i=0; i<num; i++) tmp[i]=cfSegNodes[m][i];

   for(int i=0; i<num; i++) {
     int n=tmp[i];         
     if(n<n1)        cfSegNodes[m][i]=n;
     else if(n==n1)  cfSegNodes[m][i]=n2;
     else            cfSegNodes[m][i]=n-1;
   }   
   delete [] tmp;
}

void
Crack::ReorderCrackFrontNodes(const int& m)
{
  int k1=-1,k2=-1,segs[2];      
  vector<int> tmp;
  
  int num=(int)cfSegNodes[m].size();
  for(int i=0; i<num/2; i++) { 
    // two nodes of the crack-front element       
    k1=cfSegNodes[m][2*i];
    k2=cfSegNodes[m][2*i+1];     
    // element(s) connected by the first node k1          
    FindSegsFromNode(m,k1,segs); 
    if(segs[R]<0) { // a right edge element 
      tmp.push_back(k1);
      tmp.push_back(k2);
      // the rest elements of the sub crack front 
      FindSegsFromNode(m,k2,segs);
      while(segs[L]>=0) { // Node k2 is connected by segs[L] 
        k1=cfSegNodes[m][2*segs[L]];
        k2=cfSegNodes[m][2*segs[L]+1];
        tmp.push_back(k1);
        tmp.push_back(k2);
        FindSegsFromNode(m,k2,segs);
      } 
    } // End of if(segs[R]<0)
  } 

  if((int)tmp.size()==0) { // for enclosed cracks
    // start from the first element       
    k1=cfSegNodes[m][0];
    k2=cfSegNodes[m][1];
    tmp.push_back(k1);
    tmp.push_back(k2);
    // the rest elements of the sub crack front 
    FindSegsFromNode(m,k2,segs);
    while(segs[L]>=0 && (int)tmp.size()<num-1) { 
      k1=cfSegNodes[m][2*segs[L]];
      k2=cfSegNodes[m][2*segs[L]+1];
      tmp.push_back(k1);
      tmp.push_back(k2);
      FindSegsFromNode(m,k2,segs);
    }
  } 

  // Save the reordered crack-front nodes
  for(int i=0; i<num; i++) {
    cfSegNodes[m][i]=tmp[i];
  }
}


// Determine how the crack-font nodes are connected    
void
Crack::FindCrackFrontNodeIndexes(const int& m)
{    
  // The previous node index of a crack-front node (cfSegPreIdx)
  // for which node[i]=node[preIdx] (preIdx<i)
  cfSegPreIdx[m].clear();
  int num=(int)cfSegNodes[m].size();
  cfSegPreIdx[m].resize(num);
  
  for(int i=0; i<num; i++) {
    int preIdx=-1;
    int thisNode=cfSegNodes[m][i];
    for(int j=i-1; j>=0; j--) {
      int preNode=cfSegNodes[m][j];
      if(thisNode==preNode) {
        preIdx=j;
        break;
      } 
    } 
    cfSegPreIdx[m][i]=preIdx;
  } 
  
  // The minimum and maximum node indexes of the crack-front 
  // on which the node resides: cfSegMaxIdx and cfSegMinIdx
  cfSegMaxIdx[m].clear();
  cfSegMinIdx[m].clear();
  cfSegMaxIdx[m].resize(num);
  cfSegMinIdx[m].resize(num);
    
  int maxIdx=-1, minIdx=0;
  for(int i=0; i<num; i++) {
    if(!(i>=minIdx && i<=maxIdx)) { 
      for(int j=((i%2)!=0?i:i+1); j<num; j+=2) {
        if(j==num-1 || (j<num-1 && cfSegNodes[m][j]!=cfSegNodes[m][j+1])) {
          maxIdx=j;
          break;
        }
      }
    }
    cfSegMinIdx[m][i]=minIdx;
    cfSegMaxIdx[m][i]=maxIdx;
    if(i==maxIdx) minIdx=maxIdx+1;
  }
}

// Calculate direction cosines of line p1->p2
Vector Crack::TwoPtsDirCos(const Point& p1,const Point& p2)
{
  Vector v=Vector(0.,0.,0.);

  if(p1!=p2) {
    double l12=(p1-p2).length();
    double dx,dy,dz;
    dx=p2.x()-p1.x();
    dy=p2.y()-p1.y();
    dz=p2.z()-p1.z();
    v=Vector(dx/l12,dy/l12,dz/l12);
  }

  return v;
}

// Smoothe crack front by cubic-spline fit, and then calculate the normals
// Usually, it is not used.
short Crack::SmoothCrackFrontAndCalculateNormals(const int& mm)
{
  int i=-1,l=-1,k=-1;
  int cfNodeSize=(int)cfSegNodes[mm].size();

  // Task 1: Calculate tangential normals at crack-front nodes
  //         by cubic spline fitting, and/or smooth crack front
  
  short  flag=1;       // Smooth successfully
  double ep=1.e-6;     // Tolerance

  cfSegV3[mm].clear();
  cfSegV3[mm].resize(cfNodeSize);

  // Minimum and maximum index of each sub-crack
  int minIdx=-1,maxIdx=-1;
  int minNode=-1,maxNode=-1;
  int numSegs=-1,numPts=-1;
  vector<Point>  pts; // Crack-front point subset of the sub-crack
  vector<Vector> V3;  // Crack-front point tangential vector
  vector<double> dis; // Arc length from the starting point
  vector<int>    idx;

  for(k=0; k<cfNodeSize;k++) {
    // Step a: Collect crack points for current sub-crack
    if(k>maxIdx) { // The next sub-crack
      maxIdx=cfSegMaxIdx[mm][k];
      minIdx=cfSegMinIdx[mm][k];

      // numbers of segments and points of this sub-crack  
      minNode=cfSegNodes[mm][minIdx];
      maxNode=cfSegNodes[mm][maxIdx];
      numSegs=(maxIdx-minIdx+1)/2;
      numPts=numSegs+1;

      // Allocate memories for the sub-crack
      pts.resize(numPts);
      V3.resize(numPts);
      dis.resize(numPts);
      idx.resize(maxIdx+1);
    }

    if(k>=minIdx && k<=maxIdx) { // For the sub-crack
      short preIdx=cfSegPreIdx[mm][k];
      int ki=(k-minIdx+1)/2;
      if(preIdx<0 || preIdx==minIdx) {
        pts[ki]=cx[mm][cfSegNodes[mm][k]];
        // Arc length
        if(k==minIdx) dis[ki]=0.;
        else dis[ki]=dis[ki-1]+(pts[ki]-pts[ki-1]).length();
      }
      idx[k]=ki;
      if(k<maxIdx) continue; // Collect next points
    }

    // Step b: Define how to smooth the sub-crack
    int n=numPts;               // number of points (>=2)
    //int m=(int)(numSegs/2)+2; // number of intervals (>=2)
    int m=2;                    // just two segments 
    int n1=7*m-3;

    // Arries starting from 1
    double* S=new double[n+1];  // arc-length to the first point
    double* X=new double[n+1];  // x indexed from 1
    double* Y=new double[n+1];  // y indexed from 1
    double* Z=new double[n+1];  // z indexed from 1
    for(i=1; i<=n; i++) {
      S[i]=dis[i-1];
      X[i]=pts[i-1].x();
      Y[i]=pts[i-1].y();
      Z[i]=pts[i-1].z();
    }

    int*    g=new int[n+1];     // segID
    int*    j=new int[m+1];     // number of points
    double* s=new double[m+1];  // positions of intervals
    double* ex=new double[n1+1];
    double* ey=new double[n1+1];
    double* ez=new double[n1+1];          

    // Positins of the intervals
    s[1]=S[1]-(S[2]-S[1])/50.;
    for(l=2; l<=m; l++) s[l]=s[1]+(S[n]-s[1])/m*(l-1);    

    // Number of crack-front nodes of each seg & the segs to which
    // the points belongs    
    for(l=1; l<=m; l++) { // Loop over segs
      j[l]=0; // Number of points in the seg
      for(i=1; i<=n; i++) {
        if((l<m  && S[i]>s[l] && S[i]<=s[l+1]) ||
           (l==m && S[i]>s[l] && S[i]<=S[n])) {
          j[l]++; // Number of points in seg l
          g[i]=l; // Seg ID of point i
        }
      }
    }

    // Step c: Smooth the sub-crack points
    if(CubicSpline(n,m,n1,S,X,s,j,ex,ep) &&
       CubicSpline(n,m,n1,S,Y,s,j,ey,ep) &&
       CubicSpline(n,m,n1,S,Z,s,j,ez,ep)) { // Smooth successfully
      for(i=1; i<=n; i++) {
        l=g[i];
        double t=0.,dtdS=0.;
        if(l<m)  {
          t=2*(S[i]-s[l])/(s[l+1]-s[l])-1.;
          dtdS=2./(s[l+1]-s[l]);
        }
        if(l==m) {
          t=2*(S[i]-s[l])/(S[n]-s[l])-1.;
          dtdS=2./(S[n]-s[l]);
        }
    
        double Xv0,Xv1,Xv2,Xv3,Yv0,Yv1,Yv2,Yv3,Zv0,Zv1,Zv2,Zv3;
        Xv0=ex[7*l-6]; Xv1=ex[7*l-5]; Xv2=ex[7*l-4]; Xv3=ex[7*l-3];
        Yv0=ey[7*l-6]; Yv1=ey[7*l-5]; Yv2=ey[7*l-4]; Yv3=ey[7*l-3];
        Zv0=ez[7*l-6]; Zv1=ez[7*l-5]; Zv2=ez[7*l-4]; Zv3=ez[7*l-3];
    
        double t0,t1,t2,t3,t0p,t1p,t2p,t3p;
        t0 =1.; t1 =t;    t2 =2*t*t-1.; t3 =4*t*t*t-3*t;
        t0p=0.; t1p=dtdS; t2p=4*t*dtdS; t3p=(12.*t*t-3.)*dtdS;
    
        V3[i-1].x(Xv1*t1p+Xv2*t2p+Xv3*t3p);
        V3[i-1].y(Yv1*t1p+Yv2*t2p+Yv3*t3p);
        V3[i-1].z(Zv1*t1p+Zv2*t2p+Zv3*t3p);
        pts[i-1].x(Xv0*t0+Xv1*t1+Xv2*t2+Xv3*t3);
        pts[i-1].y(Yv0*t0+Yv1*t1+Yv2*t2+Yv3*t3);
        pts[i-1].z(Zv0*t0+Zv1*t1+Zv2*t2+Zv3*t3);
      }
    }
    else { // Not smooth successfully, use the raw data
      flag=0;
      for(i=0; i<n; i++) {
        Point pt1=(i==0   ? pts[i] : pts[i-1]);
        Point pt2=(i==n-1 ? pts[i] : pts[i+1]);
        V3[i]=TwoPtsDirCos(pt1,pt2);
      }
    }
    
    delete [] g;    
    delete [] j;
    delete [] s;
    delete [] ex;
    delete [] ey;
    delete [] ez;
    delete [] S;
    delete [] X;
    delete [] Y;
    delete [] Z;

    // Step d: Smooth crack-front points and store tangential vectors
    for(i=minIdx;i<=maxIdx;i++) { // Loop over all nodes on the sub-crack 
      int ki=idx[i];
      // Smooth crack-front points
      int ni=cfSegNodes[mm][i];
      cx[mm][ni]=pts[ki];

      // Store tangential vectors
      if(minNode==maxNode && (i==minIdx || i==maxIdx)) {
        // for the first and last points (They coincide) of enclosed cracks
        int k1=idx[minIdx];
        int k2=idx[maxIdx];
        Vector averageV3=(V3[k1]+V3[k2])/2.;
        cfSegV3[mm][i]=-averageV3/averageV3.length();
      }
      else {
        cfSegV3[mm][i]=-V3[ki]/V3[ki].length();
      }
    }
    pts.clear();
    idx.clear();
    dis.clear();
    V3.clear();
  } // End of loop over k
  
  
  // Task 2: Calculate normals of crack plane at crack-front nodes
  cfSegV2[mm].clear();
  cfSegV2[mm].resize(cfNodeSize);
  for(k=0; k<cfNodeSize; k++) {
    int node=cfSegNodes[mm][k];
    int preIdx=cfSegPreIdx[mm][k];

    if(preIdx<0) { // Not operated
      Vector v2T=Vector(0.,0.,0.);
      double totalArea=0.;
      for(i=0; i<(int)ce[mm].size(); i++) { // Loop over crack elems
        // Three nodes of the elems
        int n1=ce[mm][i].x();
        int n2=ce[mm][i].y();
        int n3=ce[mm][i].z();
        if(node==n1 || node==n2 || node==n3) {
          // Three points of the triangle
          Point p1=cx[mm][n1];
          Point p2=cx[mm][n2];
          Point p3=cx[mm][n3];
          // Lengths of sides of the triangle
          double a=(p1-p2).length();
          double b=(p1-p3).length();
          double c=(p2-p3).length();
          // Half of perimeter of the triangle
          double s=(a+b+c)/2.;
          // Area of the triangle
          double thisArea=sqrt(s*(s-a)*(s-b)*(s-c));
          // Normal of the triangle
          Vector thisNorm=TriangleNormal(p1,p2,p3);
          // Area-weighted normal vector
          v2T+=thisNorm*thisArea;
          // Total area of crack plane related to the node
          totalArea+=thisArea;
        }
      } // End of loop over crack elems
      v2T/=totalArea;
      cfSegV2[mm][k]=v2T/v2T.length();
    }
    else { // Calculated
      cfSegV2[mm][k]=cfSegV2[mm][preIdx];
    }
  } // End of loop over crack-front nodes                   

  
  // Task 3: Calculate bi-normals of crack plane at crack-front nodes
  //         and adjust crack-plane normals to make sure the three axes
  //         are perpendicular to each other.
  cfSegV1[mm].clear();
  cfSegV1[mm].resize(cfNodeSize);
  for(k=0; k<cfNodeSize; k++) {
    Vector V1=Cross(cfSegV2[mm][k],cfSegV3[mm][k]);
    cfSegV1[mm][k]=V1/V1.length();
    Vector V2=Cross(cfSegV3[mm][k],cfSegV1[mm][k]);
    cfSegV2[mm][k]=V2/V2.length();
  }

  return flag;
}

short Crack::CubicSpline(const int& n, const int& m, const int& n1,
                         double x[], double y[], double z[],
                         int j[], double e[], const double& ep)
{   
  short flag=1;
  int i,k,n3,l,j1,nk,lk,llk,jj,lly,nnj,mmi,nn,ii,my,jm,ni,nij;
  double h1,h2,xlk,xlk1,a1,a2,a3,a4,t;
                  
  double** f=new double*[n1+1];
  for(i=0; i<n1+1; i++) f[i]=new double[14];
                      
  for(i=1; i<=n1; i++) {
    e[i]=0.;
    for(k=1; k<=13; k++) f[i][k]=0.;
  } 
                        
  n3=0;
  for(l=1; l<=m; l++) {
    if(l<m)
      h1=1./(z[l+1]-z[l]);
    else
      h1=1./(x[n]-z[m]);

    j1=j[l];
    for(k=1; k<=j1; k++) {
      nk=n3+k;
      xlk=2.*(x[nk]-z[l])*h1-1.;
      xlk1=xlk*xlk;
      a1=1.;
      a2=xlk;
      a3=2.*xlk1-1.;
      a4=(4.*xlk1-3.)*xlk;
      e[7*l-6]+=a1*y[nk];
      e[7*l-5]+=a2*y[nk];
      e[7*l-4]+=a3*y[nk];
      e[7*l-3]+=a4*y[nk];
      f[7*l-6][7]+=a1*a1;
      f[7*l-5][7]+=a2*a2;
      f[7*l-4][7]+=a3*a3;
      f[7*l-3][7]+=a4*a4;
      f[7*l-6][8]+=a1*a2;
      f[7*l-5][8]+=a2*a3;
      f[7*l-4][8]+=a3*a4;
      f[7*l-6][9]+=a1*a3;
      f[7*l-5][9]+=a2*a4;
      f[7*l-6][10]+=a1*a4;
    }

    f[7*l-5][6]=f[7*l-6][8];
    f[7*l-4][5]=f[7*l-6][9];
    f[7*l-3][4]=f[7*l-6][10];

    f[7*l-4][6]=f[7*l-5][8];
    f[7*l-3][5]=f[7*l-5][9];

    f[7*l-3][6]=f[7*l-4][8];

    f[7*l-6][4]=-0.5;
    f[7*l-4][2]=-0.5;
    f[7*l-5][3]=0.5;
    f[7*l-3][1]=0.5;
    f[7*l-6][11]=0.5;
    f[7*l-5][10]=0.5;
    f[7*l-4][9]=0.5;
    f[7*l-3][8]=0.5;
    f[7*l-5][4]=-h1;
    f[7*l-5][11]=h1;
    f[7*l-4][3]=4.*h1;
    f[7*l-4][10]=4.*h1;
    f[7*l-4][11]=8.*h1*h1;
    f[7*l-4][4]=-8.*h1*h1;
    f[7*l-3][2]=-9.*h1;
    f[7*l-3][9]=9.*h1;
    f[7*l-3][3]=48.*h1*h1;
    f[7*l-3][10]=48.*h1*h1;

    if(l<=m-1) {
      if(l<m-1)
        h2=1./(z[l+2]-z[l+1]);
      else
        h2=1./(x[n]-z[m]);

      f[7*l-2][3]=1.;
      f[7*l-2][4]=1.;
      f[7*l-2][5]=1.;
      f[7*l-2][6]=1.;
      f[7*l-2][11]=1.;
      f[7*l-2][13]=1.;
      f[7*l-2][10]=-1.;
      f[7*l-2][12]=-1.;
      f[7*l-1][3]=2.*h1;
      f[7*l-1][4]=8.*h1;
      f[7*l-1][5]=18.*h1;
      f[7*l-1][10]=-2.*h2;
      f[7*l-1][11]=8.*h2;
      f[7*l-1][12]=-18.*h2;
      f[7*l][3]=16.*h1*h1;
      f[7*l][4]=96.*h1*h1;
      f[7*l][10]=-16.*h2*h2;
      f[7*l][11]=96.*h2*h2;
    }
    n3+=j[l];
  }

  lk=7;
  llk=lk-1;
  for(jj=1; jj<=llk; jj++) {
    lly=lk-jj;
    nnj=n1+1-jj;
    for(i=1; i<=lly; i++) {
      for(k=2; k<=13; k++) f[jj][k-1]= f[jj][k];
      f[jj][13]=0.;
      mmi=14-i;
      f[nnj][mmi]=0.;
    }
  }

  nn=n1-1;
  for(i=1; i<=nn; i++) {
    k=i;
    ii=i+1;
    for(my=ii; my<=lk; my++) {
      if(fabs(f[my][1])<=fabs(f[k][1])) continue;
      k=my;
    }

    if(k!=i) {
      t=e[i];
      e[i]=e[k];
      e[k]=t;
      for(jj=1; jj<=13; jj++) {
        t=f[i][jj];
        f[i][jj]=f[k][jj];
        f[k][jj]=t;
      }
    }

    if(ep>=fabs(f[i][1])) {
      flag=0;
      return flag; // unsuccessful
    }
    else {
      e[i]/=f[i][1];
      for(jj=2; jj<=13; jj++) f[i][jj]/=f[i][1];

      ii=i+1;
      for(my=ii; my<=lk; my++) {
        t=f[my][1];
        e[my]-=t*e[i];
        for(jj=2; jj<=13; jj++) f[my][jj-1]=f[my][jj]-t*f[i][jj];
        f[my][13]=0.;
      }

      if(lk==n1) continue;
      lk++;
    }
  }

  e[n1]/=f[n1][1];
  jm=2;
  nn=n1-1;
  for(i=1; i<=nn; i++) {
    ni=n1-i;
    for(jj=2; jj<=jm; jj++) {
      nij=ni-1+jj;
      e[ni]-=f[ni][jj]*e[nij];
    }
    if(jm==13) continue;
      jm++;
  }

  return flag;
}

// Calculate crack-front normals by weighted average method,
// which is used by defualt 
void
Crack::CalculateCrackFrontNormals(const int& mm)
{
  // Task 1: Calculate crack-front tangential normals
  int num=cfSegNodes[mm].size();
  cfSegV3[mm].clear();
  cfSegV3[mm].resize(num);
  for(int k=0; k<num; k++) {
    int node=cfSegNodes[mm][k];
    Point pt=cx[mm][node];
        
    int preIdx=cfSegPreIdx[mm][k];
    if(preIdx<0) {// a duplicate node, not operated
      int minIdx=cfSegMinIdx[mm][k];
      int maxIdx=cfSegMaxIdx[mm][k];
      int minNode=cfSegNodes[mm][minIdx];
      int maxNode=cfSegNodes[mm][maxIdx];
        
      // The node to the right of pt 
      int node1=-1;
      if(minNode==maxNode && (k==minIdx || k==maxIdx)) {
        // for the ends of enclosed crack
        node1=cfSegNodes[mm][maxIdx-2];
      }
      else { // for the nodes of non-enclosd cracks or
             // non-end-nodes of enclosed cracks
        int k1=-1;
        if((maxIdx-minIdx+1)/2>2) { // three or more segments
          k1=(k-2)<minIdx+1 ? minIdx+1 : k-2;
        }
        else { // one or two segments
          k1=(k-2)<minIdx ? minIdx : k-2;
        }
        node1=cfSegNodes[mm][k1];
      }
      Point pt1=cx[mm][node1];
       
      // The node to the left of pt
      int node2=-1;
      if(minNode==maxNode && (k==minIdx || k==maxIdx)) {
        // for the ends of enclosed crack
        node2=cfSegNodes[mm][minIdx+2];
      }
      else { // for the nodes of non-enclosd cracks or
             // non-end-nodes of enclosed cracks
        int k2=-1;
        if((maxIdx-minIdx+1)>2) { // Three or more segments
          k2=(k+2)>maxIdx-1 ? maxIdx-1 : k+2;
        }
        else { // one or two segments
          k2=(k+2)>maxIdx ? maxIdx : k+2;
        }
        node2=cfSegNodes[mm][k2];
      }
      Point pt2=cx[mm][node2];
      
      // Weighted tangential vector between pt1->pt->pt2
      double l1=(pt1-pt).length();
      double l2=(pt-pt2).length();
      Vector v1 = (l1==0.? Vector(0.,0.,0.):TwoPtsDirCos(pt1,pt));
      Vector v2 = (l2==0.? Vector(0.,0.,0.):TwoPtsDirCos(pt,pt2));
      Vector v3T=(l1*v1+l2*v2)/(l1+l2);
      cfSegV3[mm][k]=-v3T/v3T.length();
    }
    else { // calculated
      cfSegV3[mm][k]=cfSegV3[mm][preIdx];
    }
  } // End of loop over k
 
  // Reset tangential vectors for edge nodes outside material
  // to the values of the nodes next to them. 
  // This way will eliminate the effect of edge nodes.
  for(int k=0; k<num; k++) {
    int node=cfSegNodes[mm][k];
    int segs[2];
    FindSegsFromNode(mm,node,segs);
    if(segs[R]<0) cfSegV3[mm][k]=cfSegV3[mm][k+1];
    if(segs[L]<0) cfSegV3[mm][k]=cfSegV3[mm][k-1];
  }  

  
  // Task 2: Calculate normals of crack plane at crack-front nodes
  cfSegV2[mm].clear();
  cfSegV2[mm].resize(num);
  for(int k=0; k<num; k++) {
    int node=cfSegNodes[mm][k];

    // Detect if it is an edge node 
    int segs[2];
    FindSegsFromNode(mm,node,segs);
    short edgeNode=NO;
    if(segs[L]<0 || segs[R]<0) edgeNode=YES;
            
    // End information of the sub crack
    int minNode=cfSegNodes[mm][cfSegMinIdx[mm][k]];
    int maxNode=cfSegNodes[mm][cfSegMaxIdx[mm][k]];
            
    int preIdx=cfSegPreIdx[mm][k];
    if(preIdx<0) {// a duplicate node, not operated
      Vector v2T=Vector(0.,0.,0.);
      double totalArea=0.;
      for(int i=0; i<(int)ce[mm].size(); i++) { 
        // Three nodes of the elems
        int n1=ce[mm][i].x();
        int n2=ce[mm][i].y();
        int n3=ce[mm][i].z();
            
        // Detect if the elem is connected to the node
        short elemRelatedToNode=NO;
        if(node==n1 || node==n2 || node==n3) elemRelatedToNode=YES;

        // Detect if the elem is an inner elem
        short innerElem=YES;
        if(minNode!=maxNode &&
        (n1==minNode || n2==minNode || n3==minNode ||
         n1==maxNode || n2==maxNode || n3==maxNode)) innerElem=NO;
        
        // The elem will be used if it is connected to the node AND 
        // if the node is an edge node or the elem is an interior element.
        if(elemRelatedToNode && (innerElem || edgeNode)) {
          // Three points of the triangle
          Point p1=cx[mm][n1];
          Point p2=cx[mm][n2];
          Point p3=cx[mm][n3];
          // Lengths of sides of the triangle
          double a=(p1-p2).length();
          double b=(p1-p3).length();
          double c=(p2-p3).length();
          // Half of perimeter of the triangle
          double s=(a+b+c)/2.;
          // Area of the triangle
          double thisArea=sqrt(s*(s-a)*(s-b)*(s-c));
          // Normal of the triangle
          Vector thisNorm=TriangleNormal(p1,p2,p3);
          // Area-weighted normal vector
          v2T+=thisNorm*thisArea;
          // Total area of crack plane related to the node
          totalArea+=thisArea;
        }
      } // End of loop over crack elems
        
      if(totalArea!=0.) {
        v2T/=totalArea;
      }
      else {
        cout << "Error: divided by zero in calculating outer normal"
             << " at crack front node, cfSegNodes[" << mm << "][" << k 
             << "] = " << cfSegNodes[mm][k] << endl;
        exit(1);
      } 

      cfSegV2[mm][k]=v2T/v2T.length();
    }
    else { // Calculated
      cfSegV2[mm][k]=cfSegV2[mm][preIdx];
    }
  } // End of loop over crack-front nodes

  // Reset normals for edge nodes outside material
  // to the values of the nodes next to them. 
  // This way will eliminate the effect of edge nodes.
  for(int k=0; k<num; k++) {
    int node=cfSegNodes[mm][k];
    int segs[2];
    FindSegsFromNode(mm,node,segs);
    if(segs[R]<0) cfSegV2[mm][k]=cfSegV2[mm][k+1];
    if(segs[L]<0) cfSegV2[mm][k]=cfSegV2[mm][k-1];
  }
  
          
  // Task 3: Calculate bi-normals of crack plane at crack-front nodes
  // and adjust tangential normals to make sure the three axes
  // are perpendicular to each other. Keep V2 unchanged.
  cfSegV1[mm].clear();
  cfSegV1[mm].resize(num);
  for(int k=0; k<num; k++) {
    Vector V1=Cross(cfSegV2[mm][k],cfSegV3[mm][k]);
    cfSegV1[mm][k]=V1/V1.length();
    Vector V3=Cross(cfSegV1[mm][k],cfSegV2[mm][k]);
    cfSegV3[mm][k]=V3/V3.length();
  }
}
      
// Find the segment numbers which are connected by the same node
void
Crack::FindSegsFromNode(const int& m,const int& node, int segs[])
{
  // segs[R] -- the segment on the right of the node
  // segs[L] -- the segment on the left of the node
  segs[R]=segs[L]=-1;

  int ncfSegs=(int)cfSegNodes[m].size()/2;
  for(int j=0; j<ncfSegs; j++) {
    int node0=cfSegNodes[m][2*j];
    int node1=cfSegNodes[m][2*j+1];
    if(node==node1) // the right seg
      segs[R]=j;
    if(node==node0) // the left seg
      segs[L]=j;
  } // End of loop over j

  // See if reasonable
  if(segs[R]<0 && segs[L]<0) {
    cout << "Error: failure to find the crack-front segments for node "
         << node << ". Program terminated." << endl;
    exit(1);
  }
}

// Calculate outer normal of a triangle
Vector Crack::TriangleNormal(const Point& p1, const Point& p2,
                             const Point& p3)
{   
  double x21,x31,y21,y31,z21,z31;
  double a,b,c; 
  Vector norm;
        
  x21=p2.x()-p1.x();
  x31=p3.x()-p1.x();
  y21=p2.y()-p1.y();
  y31=p3.y()-p1.y(); 
  z21=p2.z()-p1.z();
  z31=p3.z()-p1.z();
      
  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
    
  if(Vector(a,b,c).length()>1.e-16)
     norm=Vector(a,b,c)/Vector(a,b,c).length();
  else
     norm=Vector(a,b,c);
        
  return norm;
} 

void
Crack::OutputInitialCrackMesh(const int& m)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { // Output from the first rank
    cout << "\n---Initial Crack mesh---" << endl;
    cout << "MatID: " << m << endl;
    cout << "  Number of crack elements: " << (int)ce[m].size()
         << "\n  Number of crack nodes: " << (int)cx[m].size()
         << "\n  Number of crack-front elements: "
         << (int)cfSegNodes[m].size()/2 << endl;

    cout << "  Crack elements (" << (int)ce[m].size()
         << " elements in total):" << endl;
    for(int i=0; i<(int)ce[m].size(); i++) {
      cout << "     Elem " << i << ": " << ce[m][i] << endl;
    }

    cout << "  Crack nodes (" << (int)cx[m].size()
         << " nodes in total):" << endl;
    for(int i=0; i<(int)cx[m].size(); i++) {
      cout << "     Node " << i << ": " << cx[m][i] << endl;
    }

    cout << "  Crack-front elements and normals (" << (int)cfSegNodes[m].size()/2
         << " elements in total)" << endl;
    cout << "     V1: bi-normal; V2: outer normal; V3: tangential normal." << endl;
    for(int i=0; i<(int)cfSegNodes[m].size();i++) {
      cout << "     Seg " << i/2 << ": node "
           << cfSegNodes[m][i] << cx[m][cfSegNodes[m][i]]
           << ", V1: " << cfSegV1[m][i]
           << ", V2: " << cfSegV2[m][i]
           << ", V3: " << cfSegV3[m][i] << endl;
      if(i%2!=0) cout << endl;
    }

    cout << "  Average length of crack-front segments, css[m]="
         << css[m] << endl;
    cout << "  Average angle of crack-front segments, csa[m]="
         << csa[m] << " degree." << endl;
    cout << "  Crack extent: " << cmin[m] << "-->"
         <<  cmax[m] << endl << endl;
  }
}

Crack::~Crack()
{
}

