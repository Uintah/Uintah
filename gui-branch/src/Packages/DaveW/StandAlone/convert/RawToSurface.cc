#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/SurfaceGeom.h>
#include <Core/Datatypes/FlatAttrib.h>

#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <fstream>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

main(int argc, char **argv) {

  FieldHandle hField = new Field();
  SurfaceGeomHandle hSurface = new SurfaceGeom();
  hSurface->setName("HeartSurface");
  
  string pointFile = "heartPoints.dat";
  string triFile = "heartTri.dat";
  string phiFile = "heartPhi.dat";
  string outputName = "heartField.fld";

  if (argc!=1){
    if (argc!=5){
      cerr << "Format usage:\n <pointsFileName> <triangleFileName> <potentialsFileName> <outputFileName>" << endl;
      return 0;
    }
    
    pointFile.assign(argv[1]);
    triFile.assign(argv[2]);
    phiFile.assign(argv[3]);
    outputName.assign(argv[4]);
  }  
 
  ifstream ptsF(pointFile.c_str());
  ifstream triF(triFile.c_str());
  ifstream phiF(phiFile.c_str());

  if (ptsF.fail() || triF.fail() || phiF.fail()){
    cerr <<" Cann't open one or more input files" << endl;
    return 0;
  }

  double x, y, z;
  int i=0;
 
  // reading in points
  while(ptsF >> x >> y >> z){
    hSurface->d_node.push_back(Point(x, y, z));
    i++;
  }
  
  cout << i  << " points succeseffully read" << endl;

  // reading in triangles
  FaceSimp tmpTri;
  tmpTri.neighbors[0] = tmpTri.neighbors[1] = tmpTri.neighbors[2] = 0;

  i = 0;
  while( triF >> tmpTri.nodes[0] >> tmpTri.nodes[1] >> tmpTri.nodes[2]){
    hSurface->d_face.push_back(tmpTri);
    i++;
  }
  
  cout << i  << " triangles succeseffully read" << endl;

  // finding number of potentials to read
  int np = hSurface->d_node.size();

  // reading in potential values
  LockingHandle<FlatAttrib<double> > hAttrib = new FlatAttrib<double>(np);
  hAttrib->setName("Potentials");
  hAttrib->d_authorName="Alexei Samsonov";
  hAttrib->d_date="January 17, 2001";
  hAttrib->d_orgName="SCI Institute";
  
  i = 0; 
  double tmp;
  while( phiF >> tmp ){  
    hAttrib->set1(i,tmp);
    i++;
  }
  
  cout << i  << " potentials succeseffully read" << endl;

  // filling in the field
  hField->setGeometry(GeomHandle(hSurface.get_rep()));
  hField->addAttribute(AttribHandle(hAttrib.get_rep()));
 
  cout << "atribute are set " << endl;
 
  // saving the field
  TextPiostream stream(outputName.c_str(), Piostream::Write);
  Pio(stream, hField);
  
  return 0;
}    
