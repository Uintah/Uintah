/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/OnTheFlyAnalysis/pointExtract.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StaticArray.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <iostream>
#include <fstream>

#include <cstdio>


/* pointExtract.cc - an AnalysisModule
 * based heavily off Todd's LineExtract module
 * -  Todd Harman
 * -  Steve Brown
 * (C) 2006 University of Utah
 */

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "POINTEXTRACT_DBG_COUT:+" 
static DebugStream cout_doing("POINTEXTRACT_DOING_COUT", false);
static DebugStream cout_dbg("POINTEXTRACT_DBG_COUT", false);
//______________________________________________________________________              
pointExtract::pointExtract(ProblemSpecP& module_spec,
                         SimulationStateP& sharedState,
                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
	cout_dbg << "Constructing pointExtract!"<<endl;
  d_sharedState = sharedState;
  d_prob_spec = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  ps_lb = scinew pointExtractLabel();
}

//__________________________________
pointExtract::~pointExtract()
{
  cout_doing << " Doing: destroying pointExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  delete ps_lb;
  
  // delete each point in our set
  vector<pe_point*>::iterator iter;
  for( iter  = d_points.begin();iter != d_points.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void pointExtract::problemSetup(const ProblemSpecP& prob_spec,
                               GridP& grid,
                               SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tpointExtract" << endl;

  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  
  if(!d_dataArchiver){
    throw InternalError("pointExtract:couldn't get output port", __FILE__, __LINE__);
  }
  
  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();                               
  
  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_StartTime);            
  d_prob_spec->require("timeStop",          d_StopTime);

  //__________________________________
  //  Read in variables label names
  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("pointExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 
  map<string,string> attribute;                    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("pointExtract: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    //__________________________________
    //  Bulletproofing
    // The user must specify the matl for single matl variables
    if ( name == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("pointExtract: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
    }
     d_varLabels.push_back(label);
  }    
  
  //__________________________________
  //  Read in points
  ProblemSpecP points_ps = d_prob_spec->findBlock("points"); 
  if (!points_ps){
    throw ProblemSetupException("\n ERROR:pointExtract: Couldn't find <points> tag \n", __FILE__, __LINE__);    
  }        
  
  //for each point:  
  for (ProblemSpecP point_spec = points_ps->findBlock("line"); point_spec != 0; 
                    point_spec = point_spec->findNextBlock("line")) {
                    
    point_spec->getAttributes(attribute);
    string name = attribute["name"];

/**Thinking out loud:
 * Normally in our problemSpec we don't mix attributes and values on the same node.
 * This is cumbersome, for an input would need to look like this:
 *    <point name="foo"><pointVal>[1.0,2.0,3.0]</pointVal></point>...
 * Why not cheat a little and make it easier like this:
 *    <point name="bar">[4.0,5.0,6.0]</point>...
 * Thus I have stolen the code from ProblemSpec.cc and created a parsePoint function.
 * -SB 2007-02-06
 **/ 
    Point the_point;
    string value = point_spec->getNodeValue();
    parsePoint(value, the_point);
    ostringstream error;
    error << "DEBUG:pointExtract: The value "<<value<<" parsed to point "<<the_point<<endl;
    throw ProblemSetupException(error.str(),__FILE__,__LINE__);
    //__________________________________
    // bullet proofing
    // -every point must have a name
    // -point can't exceed computational domain
    if(name == ""){
      throw ProblemSetupException("\n ERROR:pointExtract: You must name each point <point name=\"something\">\n", 
                                  __FILE__, __LINE__);
    }
    
    //point can't exceed computational domain
    BBox compDomain;
    grid->getSpatialRange(compDomain);

    Point min = compDomain.min();
    Point max = compDomain.max();

    if(the_point.x() < min.x() || the_point.y() < min.y() ||the_point.z() < min.z() ||
       the_point.x() > max.x() || the_point.y() > max.y() ||the_point.z() > max.z() ){
      ostringstream warn;
      warn << "\n ERROR:pointExtract: the point that you've specified " << the_point
           << " is outside of the computational domain. \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    // Start time < stop time
    if(d_StartTime > d_StopTime){
      throw ProblemSetupException("\n ERROR:pointExtract: startTime > stopTime. \n", __FILE__, __LINE__);
    }
    
    // put input variables into the global struct
    pe_point* p = scinew pe_point;
    p->name     = name;
    p->thePt    = the_point;
    d_points.push_back(p); //put it into the vector
  } //end for each point
}



//______________________________________________________________________
void pointExtract::scheduleInitialize(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "pointExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("pointExtract::initialize", 
                  this, &pointExtract::initialize);
  
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void pointExtract::initialize(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse*,
                             DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tpointExtract" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
     
    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), ps_lb->lastWriteTimeLabel);

    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:pointExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void pointExtract::restartInitialize()
{
// need to do something here
//  new_dw->put(max_vartype(0.0), ps_lb->lastWriteTimeLabel);
}

//______________________________________________________________________
void pointExtract::scheduleDoAnalysis(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "pointExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("pointExtract::doAnalysis", 
                   this,&pointExtract::doAnalysis);
                     
  t->requires(Task::OldDW, ps_lb->lastWriteTimeLabel);
  
  Ghost::GhostType gac = Ghost::AroundCells;
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == NULL){
      string name = d_varLabels[i]->getName();
      throw InternalError("pointExtract: scheduleDoAnalysis label not found: " 
                          + name , __FILE__, __LINE__);
    }
    t->requires(Task::NewDW,d_varLabels[i], gac, 1);
  }
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void pointExtract::doAnalysis(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{   
  UintahParallelComponent* DA = dynamic_cast<UintahParallelComponent*>(d_dataArchiver);
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>( DA->getPort("load balancer"));
    
  const Level* level = getLevel(patches);
  
  max_vartype writeTime;
  old_dw->get(writeTime, ps_lb->lastWriteTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    int proc = lb->getPatchwiseProcessorAssignment(patch);
    cout_dbg << Parallel::getMPIRank() << "   working on patch " << patch->getID() << " which is on proc " << proc << endl;
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myrank() && now >= nextWriteTime){
    
     cout_doing << pg->myrank() << " " 
                << "Doing doAnalysis (pointExtract)\t\t\t\tL-"
                << level->getIndex()
                << " patch " << patch->getGridIndex()<< endl;
      //__________________________________
      // loop over each of the variables
      // load them into the data vectors
      vector< constCCVariable<double> > CC_double_data;
      vector< constCCVariable<Vector> > CC_Vector_data;
      constCCVariable<double> q_CC_double;
      constCCVariable<Vector> q_CC_Vector;      
      Ghost::GhostType gac = Ghost::AroundCells;
      int indx = d_matl->getDWIndex();

      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
        
        // bulletproofing
        if(d_varLabels[i] == NULL){
          string name = d_varLabels[i]->getName();
          throw InternalError("pointExtract: analyze label not found: " 
                          + name , __FILE__, __LINE__);
        }
      
        switch( d_varLabels[i]->typeDescription()->getSubType()->getType()){
        case TypeDescription::double_type:
          new_dw->get(q_CC_double, d_varLabels[i], indx, patch, gac, 1);
          CC_double_data.push_back(q_CC_double);
          break;
        case TypeDescription::Vector:
          new_dw->get(q_CC_Vector, d_varLabels[i], indx, patch, gac, 1);
          CC_Vector_data.push_back(q_CC_Vector);
          break;
        default:
          throw InternalError("ERROR:AnalysisModule:lineExtact:Unknown variable type", __FILE__, __LINE__);
        }
      }      

      //__________________________________
      // loop over each line 
      for (unsigned int p =0 ; p < d_points.size(); p++) {
      
        // create the directory structure
        string udaDir = d_dataArchiver->getOutputLocation();
        string dirName = d_points[p]->name;
        string pointPath = udaDir + "/" + dirName;
        
        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        string path = pointPath + "/" + levelIndex;
        
        createDirectory(pointPath, levelIndex);
        
        // find the physical domain and index range
        // associated with this patch
        Point the_pt = d_points[p]->thePt;
        
        Box patchDomain = patch->getExtraBox();
        if(level->getIndex() > 0){ // ignore extra cells on fine patches
          patchDomain = patch->getBox();
        }
        // intersection
        the_pt = Max(patchDomain.lower(), the_pt);
        the_pt = Min(patchDomain.upper(), the_pt);
        
        //indices
        IntVector point_idx;
        patch->findCell(the_pt, point_idx);
        
        // enlarge the index space by 1 except in the main looping direction
        //IntVector one(1,1,1);
        //one[d_lines[l]->loopDir] = 0;
        //end_idx+= one;   

        //__________________________________
        // loop over each point in the line on this patch
        //CellIterator iterLim = CellIterator(start_idx,end_idx);
  
        //for(CellIterator iter=iterLim; !iter.done();iter++) {
          
          //if (!patch->containsCell(*iter))
          //  continue;  // just in case - the point-to-cell logic might throw us off on patch boundaries...
            
          //IntVector c = *iter;
          ostringstream fname;
          fname<<path<<"/i"<< point_idx.x() << "_j" << point_idx.y() << "_k"<< point_idx.z();
          string filename = fname.str();
    
          //create file and write the file header  
          ifstream test(filename.c_str());
          if (!test){
            createFile(filename);
          }

          FILE *fp;
          fp = fopen(filename.c_str(), "a");
          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:pointExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }
          
          // write cell position and time
          Point here = patch->cellPosition(point_idx);
	  ostringstream error;
	  error <<"\nDEBUG: looking at here "<<here.x()<<","<<here.y()<<","<<here.z()
	      <<" and the_point "<<the_pt.x()<<","<<the_pt.y()<<","<<the_pt.z()<<".\n";
	  throw InternalError(error.str(),__FILE__,__LINE__);
	  
          double time = d_dataArchiver->getCurrentTime();
          fprintf(fp,    "%E\t %E\t %E\t %E",here.x(),here.y(),here.z(), time);
          
          // write double variables
          for (unsigned int i=0 ; i <  CC_double_data.size(); i++) {
            fprintf(fp, "\t%16E",CC_double_data[i][point_idx]);            
          }
          
          // write Vector variable
          for (unsigned int i=0 ; i <  CC_Vector_data.size(); i++) {
            fprintf(fp, "\t% 16E \t %16E \t %16E",
                    CC_Vector_data[i][point_idx].x(),
                    CC_Vector_data[i][point_idx].y(),
                    CC_Vector_data[i][point_idx].z() );            
          }
          fprintf(fp,    "\n");
          fclose(fp);
      //  }  // loop over points
      }  // loop over lines 
      lastWriteTime = now;     
    }  // time to write data
     
   new_dw->put(max_vartype(lastWriteTime), ps_lb->lastWriteTimeLabel); 
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void pointExtract::createFile(string& filename)
{ 
  FILE *fp;
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"X_CC \t Y_CC \t Z_CC \t Time"); 
  
  // double variables header
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    VarLabel* vl = d_varLabels[i];
    TypeDescription::Type type =vl->typeDescription()->getSubType()->getType();
    if(type ==  TypeDescription::double_type){
      string name = vl->getName();
      fprintf(fp,"\t %s", name.c_str());
    }
  }
  
  // Vector variables header
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    VarLabel* vl = d_varLabels[i];
    TypeDescription::Type type =vl->typeDescription()->getSubType()->getType();
    if(type ==  TypeDescription::Vector){
      string name = vl->getName();
      fprintf(fp,"\t %s.x \t %s.y \t %s.z", name.c_str(),name.c_str(),name.c_str());
    }
  }
  fprintf(fp,"\n");
  fclose(fp);
  cout << Parallel::getMPIRank() << " pointExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
void pointExtract::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == NULL){
    cout << Parallel::getMPIRank() << "pointExtract:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  }
  closedir(check);
  
  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL){
    cout << "pointExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  }  
  closedir(check);
}

//
// The following are stolen from Core/ProblemSpec/ProblemSpec.cc
//   because I couldn't figure out a better way to use those functions.

//______________________________________________________________________
//

void pointExtract::parsePoint(const string& stringValue, Point &point)
{
    Vector value;

    // Parse out the [num,num,num]
    // Now pull apart the stringValue
    string::size_type i1 = stringValue.find("[");
    string::size_type i2 = stringValue.find_first_of(",");
    string::size_type i3 = stringValue.find_last_of(",");
    string::size_type i4 = stringValue.find("]");
    
    string x_val(stringValue,i1+1,i2-i1-1);
    string y_val(stringValue,i2+1,i3-i2-1);
    string z_val(stringValue,i3+1,i4-i3-1);
    
    checkForInputError(x_val, "double"); 
    checkForInputError(y_val, "double");
    checkForInputError(z_val, "double");
    
    value.x(atof(x_val.c_str()));
    value.y(atof(y_val.c_str()));
    value.z(atof(z_val.c_str()));   
    
    point = Point(value);
   
}


//______________________________________________________________________
//
void
pointExtract::checkForInputError(const string& stringValue, 
                   const string& Int_or_float)
{
  //__________________________________
  //  Make sure stringValue only contains valid characters
  if (Int_or_float != "int") {
    string validChars(" -+.0123456789eE");
    string::size_type  pos = stringValue.find_first_not_of(validChars);
    if (pos != string::npos){
      ostringstream warn;
      warn << "Input file error: I found ("<< stringValue[pos]
           << ") inside of "<< stringValue<< " at position "<< pos <<endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    //__________________________________
    // check for two or more "."
    string::size_type p1 = stringValue.find_first_of(".");    
    string::size_type p2 = stringValue.find_last_of(".");     
    if (p1 != p2){
      ostringstream warn;
      warn << "Input file error: I found two (..) "
           << "inside of "<< stringValue <<endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }  
  if (Int_or_float == "int")  {
    string validChars(" -0123456789");
    string::size_type  pos = stringValue.find_first_not_of(validChars);
    if (pos != string::npos){
      ostringstream warn;
      warn << "Input file error Integer Number: I found ("<< stringValue[pos]
           << ") inside of "<< stringValue<< " at position "<< pos <<endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
} 
