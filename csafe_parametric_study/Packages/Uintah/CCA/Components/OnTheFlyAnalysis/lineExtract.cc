#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/lineExtract.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StaticArray.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <iostream>
#include <fstream>

#include <stdio.h>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "LINEEXTRACT_DBG_COUT:+" 
static DebugStream cout_doing("LINEEXTRACT_DOING_COUT", false);
static DebugStream cout_dbg("LINEEXTRACT_DBG_COUT", false);
//______________________________________________________________________              
lineExtract::lineExtract(ProblemSpecP& module_spec,
                         SimulationStateP& sharedState,
                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState = sharedState;
  d_prob_spec = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  ps_lb = scinew lineExtractLabel();
}

//__________________________________
lineExtract::~lineExtract()
{
  cout_doing << " Doing: destorying lineExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  delete ps_lb;
  
  // delete each line
  vector<line*>::iterator iter;
  for( iter  = d_lines.begin();iter != d_lines.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void lineExtract::problemSetup(const ProblemSpecP& prob_spec,
                               GridP& grid,
                               SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tlineExtract" << endl;

  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  
  if(!d_dataArchiver){
    throw InternalError("lineExtract:couldn't get output port", __FILE__, __LINE__);
  }
  
  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = new MaterialSet();
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
    throw ProblemSetupException("lineExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 
  map<string,string> attribute;                    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("lineExtract: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
     d_varLabels.push_back(label);
  }    
  
  //__________________________________
  //  Read in lines
  ProblemSpecP lines_ps = d_prob_spec->findBlock("lines"); 
  if (!lines_ps){
    throw ProblemSetupException("\n ERROR:lineExtract: Couldn't find <lines> tag \n", __FILE__, __LINE__);    
  }        
             
  for (ProblemSpecP line_spec = lines_ps->findBlock("line"); line_spec != 0; 
                    line_spec = line_spec->findNextBlock("line")) {
                    
    line_spec->getAttributes(attribute);
    string name = attribute["name"];

    Point start, end;
    line_spec->require("startingPt", start);
    line_spec->require("endingPt",   end);
    
    //__________________________________
    // bullet proofing
    // -every line must have a name
    // -line must be parallel to the coordinate system
    // -line can't exceed computational domain
    if(name == ""){
      throw ProblemSetupException("\n ERROR:lineExtract: You must name each line <line name= something>\n", 
                                  __FILE__, __LINE__);
    }
    
    // line must be parallel to the coordinate system
    bool X = (start.x() == end.x());
    bool Y = (start.y() == end.y());  // 2 out of 3 of these must be true
    bool Z = (start.z() == end.z());
    
    bool validLine = false;
    int  loopDir = -9;               // direction to loop over
    
    if( !X && Y && Z){
      validLine = true;
      loopDir = 0;
    }
    if( X && !Y && Z){
      validLine = true;
      loopDir = 1;
    }
    if( X && Y && !Z){
      validLine = true;
      loopDir = 2;
    }
    if(validLine == false){
      ostringstream warn;
      warn << "\n ERROR:lineExtract: the line that you've specified " << start 
           << " " << end << " is not parallel to the coordinate system. \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    //line can't exceed computational domain
    BBox compDomain;
    grid->getSpatialRange(compDomain);

    Point min = compDomain.min();
    Point max = compDomain.max();

    if(start.x() < min.x() || start.y() < min.y() ||start.z() < min.z() ||
       end.x() > max.x()   ||end.y() > max.y()    || end.z() > max.z() ){
      ostringstream warn;
      warn << "\n ERROR:lineExtract: the line that you've specified " << start 
           << " " << end << " begins or ends outside of the computational domain. \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    if(start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
      ostringstream warn;
      warn << "\n ERROR:lineExtract: the line that you've specified " << start 
           << " " << end << " the starting point is > than the ending point \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    // Start time < stop time
    if(d_StartTime > d_StopTime){
      throw ProblemSetupException("\n ERROR:lineExtract: startTime > stopTime. \n", __FILE__, __LINE__);
    }
    
    // put input variables into the global struct
    line* l = scinew line;
    l->name    = name;
    l->startPt = start;
    l->endPt   = end;
    l->loopDir = loopDir;
    d_lines.push_back(l);
  }
}

//______________________________________________________________________
void lineExtract::scheduleInitialize(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "lineExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("lineExtract::initialize", 
                  this, &lineExtract::initialize);
  
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void lineExtract::initialize(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse*,
                             DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tlineExtract" << endl;
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
        warn << "ERROR:lineExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void lineExtract::restartInitialize()
{
// need to do something here
//  new_dw->put(max_vartype(0.0), ps_lb->lastWriteTimeLabel);
}

//______________________________________________________________________
void lineExtract::scheduleDoAnalysis(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "lineExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("lineExtract::doAnalysis", 
                   this,&lineExtract::doAnalysis);
                     
  t->requires(Task::OldDW, ps_lb->lastWriteTimeLabel);
  
  Ghost::GhostType gac = Ghost::AroundCells;
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == NULL){
      string name = d_varLabels[i]->getName();
      throw InternalError("lineExtract: scheduleDoAnalysis label not found: " 
                          + name , __FILE__, __LINE__);
    }
    t->requires(Task::NewDW,d_varLabels[i], gac, 1);
  }
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void lineExtract::doAnalysis(const ProcessorGroup* pg,
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
                << "Doing doAnalysis (lineExtract)\t\t\t\tL-"
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
          throw InternalError("lineExtract: analyze label not found: " 
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
      for (unsigned int l =0 ; l < d_lines.size(); l++) {
      
        // create the directory structure
        string udaDir = d_dataArchiver->getOutputLocation();
        string dirName = d_lines[l]->name;
        string linePath = udaDir + "/" + dirName;
        
        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        string path = linePath + "/" + levelIndex;
        
        createDirectory(linePath, levelIndex);
        
        // find the physical domain and index range
        // associated with this patch
        Point start_pt = d_lines[l]->startPt;
        Point end_pt   = d_lines[l]->endPt;
        
        Box patchDomain = patch->getBox();
        if(level->getIndex() > 0){ // ignore extra cells on fine patches
          patchDomain = patch->getInteriorBox();
        }
        // intersection
        start_pt = Max(patchDomain.lower(), start_pt);
        end_pt   = Min(patchDomain.upper(), end_pt);
        
        //indices
        IntVector start_idx, end_idx;
        patch->findCell(start_pt, start_idx);
        patch->findCell(end_pt,   end_idx);
        
        // enlarge the index space by 1 except in the main looping direction
        IntVector one(1,1,1);
        one[d_lines[l]->loopDir] = 0;
        end_idx+= one;   

        //__________________________________
        // loop over each point in the line on this patch
        CellIterator iterLim = CellIterator(start_idx,end_idx);
  
        for(CellIterator iter=iterLim; !iter.done();iter++) {
          
          if (!patch->containsCell(*iter))
            continue;  // just in case - the point-to-cell logic might throw us off on patch boundaries...
            
          IntVector c = *iter;
          ostringstream fname;
          fname<<path<<"/i"<< c.x() << "_j" << c.y() << "_k"<< c.z();
          string filename = fname.str();
    
          //create file and write the file header  
          ifstream test(filename.c_str());
          if (!test){
            createFile(filename);
          }

          FILE *fp;
          fp = fopen(filename.c_str(), "a");
          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:lineExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }
          
          // write cell position and time
          Point here = patch->cellPosition(c);
          double time = d_dataArchiver->getCurrentTime();
          fprintf(fp,    "%E\t %E\t %E\t %E",here.x(),here.y(),here.z(), time);
          
          // write double variables
          for (unsigned int i=0 ; i <  CC_double_data.size(); i++) {
            fprintf(fp, "\t%16E",CC_double_data[i][c]);            
          }
          
          // write Vector variable
          for (unsigned int i=0 ; i <  CC_Vector_data.size(); i++) {
            fprintf(fp, "\t% 16E \t %16E \t %16E",
                    CC_Vector_data[i][c].x(),
                    CC_Vector_data[i][c].y(),
                    CC_Vector_data[i][c].z() );            
          }
          fprintf(fp,    "\n");
          fclose(fp);
        }  // loop over points
      }  // loop over lines 
      lastWriteTime = now;     
    }  // time to write data
     
   new_dw->put(max_vartype(lastWriteTime), ps_lb->lastWriteTimeLabel); 
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void lineExtract::createFile(string& filename)
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
  cout << Parallel::getMPIRank() << " lineExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
void lineExtract::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == NULL){
    cout << Parallel::getMPIRank() << "lineExtract:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  }
  closedir(check);
  
  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL){
    cout << "lineExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  }  
  closedir(check);
}
