/*
 *  GenesisMatrixReader.cc:
 *
 *  Written by:
 *   Martin Cole
 *   Wed Nov 15 15:59:30 MST 2000
 *
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/share/share.h>
#include <fstream>
#include <string>

namespace Butson {
// These should come from a gui...
static char * vectorFile = "/home/cs/mcole/cbutson/vector";
static char * dipoleFile = "/home/cs/mcole/cbutson/dipoles";

using namespace SCIRun;
using namespace std;



class PSECORESHARE GenesisMatrixReader : public Module {

  MatrixOPort* d_outport;
  MatrixHandle d_handle;
  typedef vector<vector<double> > genMat;
  genMat d_mat;
public:
  GenesisMatrixReader(const string& id) : 
    Module("GenesisMatrixReader", id, Source)
  {
    // Create the output data handle and port
    d_outport=scinew MatrixOPort(this, "Output Data", MatrixIPort::Atomic);
    add_oport(d_outport);
  }

  virtual ~GenesisMatrixReader() {}

  virtual void execute() 
  {
    d_mat.clear();
    readVector();
    ifstream dipfs(dipoleFile);

    if (!dipfs) { 
      cerr << "Error could not open file: " << dipoleFile << endl; 
      return;
    }
    vector<double> scaleVec;
    for(;;) {
      double d;
      dipfs >> d; // waste the time val...

      if (dipfs.eof()) { 
	Matrix *mat = getMatrix(scaleVec);
	MatrixHandle matH(mat);
	d_outport->send(matH);
	break; 
      } else {
	scaleVec.clear();
      }
    
      for (unsigned int i = 0; i < d_mat.size(); i++) {
	dipfs >> d;
	scaleVec.push_back(d * 1.0e10);
	//	cout << "i: " << i << " val: " << d * 1.0e10;
      }
      //      cout << endl;
      Matrix *mat = getMatrix(scaleVec);
      MatrixHandle matH(mat);
      d_outport->send_intermediate(matH);
    }
  }

  Matrix* getMatrix(vector<double> &sVec) {
    DenseMatrix *m = scinew DenseMatrix(d_mat.size(), 6);//rows,cols
    genMat::iterator iter = d_mat.begin(); 
    int row = 0;
    while (iter != d_mat.end()) {
      vector<double> &vd = *iter++;
      (*m)[row][0]   = vd[0];
      (*m)[row][1]   = vd[1];
      (*m)[row][2]   = vd[2];
      (*m)[row][3]   = vd[3] * sVec[row]; // scale vector x
      (*m)[row][4]   = vd[4] * sVec[row]; // scale vector y
      (*m)[row++][5] = vd[5] * sVec[row]; // scale vector z
    }
    return m;
  }

  void readVector() {
    ifstream vecfs(vectorFile);

    if (!vecfs) { 
      cerr << "Error could not open file: " << vectorFile << endl; 
      return;
    }
    string buf;
    genMat::iterator iter; 
    int i = 0;
    vecfs >> buf; // waste the name
    while (!vecfs.eof()) {
      d_mat.push_back();
      iter = d_mat.end();
      iter--;
      vector<double> &vd = *iter;
      cout << "dipole " << i++ << ": " << buf << endl;
      double d = 0.0L;

      vecfs >> d;
      cout << "d is: " << d << endl;
      vd.push_back(d);
      vecfs >> d;
      vd.push_back(d);
      vecfs >> d;
      vd.push_back(d);
      vecfs >> d;
      vd.push_back(d);
      vecfs >> d;
      vd.push_back(d);
      vecfs >> d;
      vd.push_back(d);
      vecfs >> buf; // waste the name
    }

//      iter = d_mat.begin(); 
//      while (iter != d_mat.end()) {
//        vector<double> &vd = *iter++;
//        cout << "vd[0] " << vd[0] << " ";
//        cout << "vd[1] " << vd[1] << " ";
//        cout << "vd[2] " << vd[2] << " ";
//        cout << "vd[3] " << vd[3] << " ";
//        cout << "vd[4] " << vd[4] << " ";
//        cout << "vd[5] " << vd[5] << endl;
//      }

  }

  virtual void tcl_command(TCLArgs &args, void *userdata) 
  {
    Module::tcl_command(args, userdata);
  }
};

extern "C" PSECORESHARE Module* make_GenesisMatrixReader(const string& id) {
  return new GenesisMatrixReader(id);
}
} // End namespace Butson
