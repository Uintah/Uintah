/*
 *  ForwardMEG.cc:
 *
 *  Written by:
 *   vanuiter
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE ForwardMEG : public Module {
public:
  ForwardMEG(const string& id);

  virtual ~ForwardMEG();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" RobVSHARE Module* make_ForwardMEG(const string& id) {
  return scinew ForwardMEG(id);
}

ForwardMEG::ForwardMEG(const string& id)
  : Module("ForwardMEG", id, Source, "Magnetic", "RobV")
{
}

ForwardMEG::~ForwardMEG(){
}

void ForwardMEG::execute(){
  
#if 0
// J = (sigma)*E + J(source)


  VectorFieldHandle eField;
  if (!electricFieldP->get(eField)) return;

  MatrixHandle sourceLocsM;
  if (!sourceLocationP->get(sourceLocsM)) return;

  MatrixHandle detectLocsM;
  if (!detectorPtsP->get(detectLocsM)) return;

  /*  timer.clear();
  timer.start();
  cerr << "Starting  "<<"...("<<timer.time()<<")... ";
  */
  
  ugfield = ((VectorFieldUG*)(eField.get_rep()));

  mesh = ugfield->mesh;

  sourceLocations = dynamic_cast<DenseMatrix*>(sourceLocsM.get_rep());

  detectorPts = dynamic_cast<DenseMatrix*>(detectLocsM.get_rep());

  
  Norms = true;
  MatrixHandle detectNormsM;
  if (!detectorNormalsP->get(detectNormsM)) Norms = false;
  else detectNorms = dynamic_cast<DenseMatrix*>(detectNormsM.get_rep());

  currentDensityField = new VectorFieldUG(mesh,VectorFieldUG::ElementValues);

  nelems = mesh->elems.size();
  
  //initialize mesh::get_bounds, so not problems in parallel (NOT NECESSARY!!)
  //  Point min,max;
  //  mesh->get_bounds(min,max);
  // mesh->make_grid(20,20,20,min,max,0);
   //////////////////////////
  
  np=5;//Thread::numProcessors();
  //if (np>4) np/=2;
  cerr << "Number of Processors Used: " << np <<endl;
  Thread::parallel(Parallel<ForwardMEG>(this, &ForwardMEG::parallel1),np, true);
  
  
  //field of J's for each element
  currentDensityFieldMI = new VectorFieldMI(currentDensityField); 
  
  magneticMatrix = new DenseMatrix(detectorPts->nrows(),detectorPts->ncols());

  magnitudeMatrix = new ColumnMatrix(detectorPts->nrows());

  numDetect = detectorPts->nrows();
  Thread::parallel(Parallel<ForwardMEG>(this, &ForwardMEG::parallel2),np, true);

  //for(int l=0; l<64; l++) cerr << (*magnitudeMatrix)[l] <<"\n";

  magneticFieldAtPointsP->send(magneticMatrix);
  
  magnitudeFieldP->send(magnitudeMatrix);


  //  time_t current_time = time(NULL);
  // printf("End Magnetic simulation: %s\n",ctime(&current_time));
  //cerr << "Done! (timer="<<timer.time()<<")\n";
#endif
}

#if 0
void ForwardMEG::parallel2(int proc)
{
  
  int su=proc*numDetect/np;
  int eu=(proc+1)*numDetect/np;

  for (int i=su; i<eu; i++) {

    Vector value;  
    Vector magneticField;

    double x = (*detectorPts)[i][0];
    double y = (*detectorPts)[i][1];
    double z = (*detectorPts)[i][2];
    
    Point  pt (x,y,z);
 
    currentDensityFieldMI->interpolate(pt, magneticField);


    Vector normal;
    
    if (Norms == false) {
      normal = (pt - Point(0,0,0));
      normal.normalize();
    } else {
      double xN = (*detectNorms)[i][0];
      double yN = (*detectNorms)[i][1];
      double zN = (*detectNorms)[i][2];
      normal = Vector(xN,yN,zN);
    }

    // start of B(P) stuff

    int nSources = sourceLocations->nrows();

    (*magneticMatrix)[i][0] = 0;
    (*magneticMatrix)[i][1] = 0;
    (*magneticMatrix)[i][2] = 0;

    for(int j=0; j<nSources;j++) {
 
      double x1 = (*sourceLocations)[j][0];
      double y1 = (*sourceLocations)[j][1];
      double z1 = (*sourceLocations)[j][2];
      double px = (*sourceLocations)[j][3];
      double py = (*sourceLocations)[j][4];
      double pz = (*sourceLocations)[j][5];

      Point  pt2 (x1,y1,z1);
      
      Vector P(px,py,pz);

      Vector radius = pt - pt2; // detector - source

      Vector valuePXR = Cross(P,radius);
      double length = radius.length();
      
      double mu = 1.0;
      value = (valuePXR/(length*length*length))*(mu/(4*M_PI));

      (*magneticMatrix)[i][0] += value.x();
      (*magneticMatrix)[i][1] += value.y();
      (*magneticMatrix)[i][2] += value.z();
    }
    // end of B(P) stuff

    value = Vector((*magneticMatrix)[i][0], (*magneticMatrix)[i][1],(*magneticMatrix)[i][2]);
    
    
    (*magneticMatrix)[i][0] += magneticField.x();
    (*magneticMatrix)[i][1] += magneticField.y();
    (*magneticMatrix)[i][2] += magneticField.z();

 
     Vector m = Vector((*magneticMatrix)[i][0],(*magneticMatrix)[i][1],(*magneticMatrix)[i][2]);

     //use Dot for simulations & length for testing with sphere
     //(*magnitudeMatrix)[i] = m.length();
     (*magnitudeMatrix)[i] = Dot(m,normal);

     /*    
        double angle = Dot(value,magneticField)/(value.length()*magneticField.length());

    double degrees = acos(angle) * 180/PI;
     */
     /*    mutex.lock();
   // cerr << "Source: ("<<X<<","<<Y<<","<<Z<<")"<<"\n"; //New for test
     
          cerr << "DetectorPoint: ("<<(*detectorPts)[i][0]<<","<<(*detectorPts)[i][1]<<","<<(*detectorPts)[i][2]<<")"<<"\n";

	//	cerr<<"Angle: "<<degrees<<"\n";
     
    cerr << "Total Mag: " << (magneticField+value).length() <<"\n";
     cerr << "B(P) Mag: " << value.length() <<"\n";
    cerr << "B(J) Mag: " << magneticField.length() <<"\n";
     	
    cerr << "Total MagneticField: ("<<(*magneticMatrix)[i][0]<<","<<(*magneticMatrix)[i][1]<<","<<(*magneticMatrix)[i][2]<<")\n";
        cerr << "B(P): ("<<value.x()<<","<<value.y()<<","<<value.z()<<")\n";
    cerr << "B(J): ("<<magneticField.x()<<","<<magneticField.y()<<","<<magneticField.z()<<")\n";
	
    cerr << "Bz total (picked up by detector): " << Dot(normal,magneticField+value) <<"\n";
    cerr << "Bz B(P) (primary picked up by detector): " << Dot(normal,value) <<"\n";
    cerr << "Bz B(J) (return picked up by detector): " << Dot(normal,magneticField) <<"\n\n";
    mutex.unlock();
     */
     /*mutex.lock();
     cerr << i << " " << Dot(normal,magneticField) << "\n";
     mutex.unlock();
     */
  }

}

void ForwardMEG::parallel1(int proc)
{
  
  int su=proc*nelems/np;
  int eu=(proc+1)*nelems/np;

  for (int i=su; i<eu; i++) {

    Element* e = mesh->elems[i];

    Array1<double> sigma = mesh->cond_tensors[(e->cond)];
 
    Vector elemField;
    Point centroid = e->centroid();
    ugfield->interpolate(centroid, elemField);
    Vector condElect = mult(sigma,elemField);  //matrix-vector mult 

    currentDensityField->data[i] = condElect;
  }

}

Vector ForwardMEG::mult(Array1<double> matrix, Vector elemField) {

  return(Vector(matrix[0]*elemField.x()+matrix[1]*elemField.y()+matrix[2]*elemField.z(),matrix[1]*elemField.x()+matrix[3]*elemField.y()+matrix[4]*elemField.z(),matrix[2]*elemField.x()+matrix[4]*elemField.y()+matrix[5]*elemField.z()));

}

#endif

void ForwardMEG::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


