/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/OnTheFlyAnalysis/lineExtract.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MiscMath.h>
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
  
  int numMatls  = d_sharedState->getNumMatls();
  cout << " numMatls  "<< numMatls << endl;
  
  if(!d_dataArchiver){
    throw InternalError("lineExtract:couldn't get output port", __FILE__, __LINE__);
  }
                               
  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_StartTime);            
  d_prob_spec->require("timeStop",          d_StopTime);


  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("lineExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 

  
  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  if(d_prob_spec->findBlock("material") ){
    d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    d_matl = d_sharedState->getMaterial(indx);
  } else {
    d_matl = d_sharedState->getMaterial(0);
  }
  
  int defaultMatl = d_matl->getDWIndex();
  
  //__________________________________
  //  Read in the optional material index from the variables that may be different
  //  from the default index
  vector<int> m;
  m.push_back(defaultMatl);
  d_matl_set = scinew MaterialSet();
  map<string,string> attribute;
    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
   
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }
    
    // bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("lineExtract: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    
    d_varMatl.push_back(matl);
    m.push_back(matl);
  }
  
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  //Construct the matl_set
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  
  //__________________________________
  //  Read in variables label names                
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
    
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("lineExtract: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    //__________________________________
    //  Bulletproofing
    // The user must specify the matl for single matl variables
    if ( name == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("lineExtract: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
    }
    
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();
    
    //__________________________________
    bool throwException = false;  
    
    // only CC, SFCX, SFCY, SFCZ variables
    if(td->getType() != TypeDescription::CCVariable &&
       td->getType() != TypeDescription::SFCXVariable &&
       td->getType() != TypeDescription::SFCYVariable &&
       td->getType() != TypeDescription::SFCZVariable ){
       throwException = true;
    }
    // CC Variables, only Doubles and Vectors 
    if(td->getType() != TypeDescription::CCVariable &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::int_type &&
       subtype->getType() != TypeDescription::Vector  ){
      throwException = true;
    }
    // Face Centered Vars, only Doubles
    if( (td->getType() == TypeDescription::SFCXVariable ||
         td->getType() == TypeDescription::SFCYVariable ||
         td->getType() == TypeDescription::SFCZVariable) &&
         subtype->getType() != TypeDescription::double_type) {
      throwException = true;
    } 
    if(throwException){       
      ostringstream warn;
      warn << "ERROR:AnalysisModule:lineExtact: ("<<label->getName() << " " 
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
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
    double stepSize;
    line_spec->require("startingPt", start);
    line_spec->require("endingPt",   end);
    line_spec->getWithDefault("stepSize", stepSize, 0.0);
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
    l->name     = name;
    l->startPt  = start;
    l->endPt    = end;
    l->loopDir  = loopDir;
    l->stepSize = stepSize;
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
    
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add(d_varMatl[i]);
    matSubSet->addReference();
    
    t->requires(Task::NewDW,d_varLabels[i], matSubSet, gac, 1);
    
    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
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
      vector< constCCVariable<int> >      CC_integer_data;
      vector< constCCVariable<double> >   CC_double_data;
      vector< constCCVariable<Vector> >   CC_Vector_data;
      vector< constSFCXVariable<double> > SFCX_double_data;
      vector< constSFCYVariable<double> > SFCY_double_data;
      vector< constSFCZVariable<double> > SFCZ_double_data;
      
      constCCVariable<int>    q_CC_integer;      
      constCCVariable<double> q_CC_double;
      constCCVariable<Vector> q_CC_Vector;
      
      constSFCXVariable<double> q_SFCX_double;      
      constSFCYVariable<double> q_SFCY_double;
      constSFCZVariable<double> q_SFCZ_double;      
            
      Ghost::GhostType gac = Ghost::AroundCells;
      

      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
        
        // bulletproofing
        if(d_varLabels[i] == NULL){
          string name = d_varLabels[i]->getName();
          throw InternalError("lineExtract: analyze label not found: " 
                          + name , __FILE__, __LINE__);
        }

        const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
        const Uintah::TypeDescription* subtype = td->getSubType();


        int indx = d_varMatl[i];        
        switch(td->getType()){
          case Uintah::TypeDescription::CCVariable:      // CC Variables
            switch(subtype->getType()) {
            
            case Uintah::TypeDescription::double_type:
              new_dw->get(q_CC_double, d_varLabels[i], indx, patch, gac, 1);
              CC_double_data.push_back(q_CC_double);
              break;
             
            case Uintah::TypeDescription::Vector:
              new_dw->get(q_CC_Vector, d_varLabels[i], indx, patch, gac, 1);
              CC_Vector_data.push_back(q_CC_Vector);
              break;
              
            case Uintah::TypeDescription::int_type:
              new_dw->get(q_CC_integer, d_varLabels[i], indx, patch, gac, 1);
              CC_integer_data.push_back(q_CC_integer);
              break; 
            default:
              throw InternalError("LineExtract: invalid data type", __FILE__, __LINE__); 
            }
            break;
          case Uintah::TypeDescription::SFCXVariable:   // SFCX Variables
            new_dw->get(q_SFCX_double, d_varLabels[i], indx, patch, gac, 1);
            SFCX_double_data.push_back(q_SFCX_double);
            break;
          case Uintah::TypeDescription::SFCYVariable:    // SFCY Variables
            new_dw->get(q_SFCY_double, d_varLabels[i], indx, patch, gac, 1);
            SFCY_double_data.push_back(q_SFCY_double);
            break;
          case Uintah::TypeDescription::SFCZVariable:   // SFCZ Variables
            new_dw->get(q_SFCZ_double, d_varLabels[i], indx, patch, gac, 1);
            SFCZ_double_data.push_back(q_SFCZ_double);
            break;
          default:
            ostringstream warn;
            warn << "ERROR:AnalysisModule:lineExtact: ("<<d_varLabels[i]->getName() << " " 
                 << td->getName() << " ) has not been implemented" << endl;
            throw InternalError(warn.str(), __FILE__, __LINE__);
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
        
        double stepSize(d_lines[l]->stepSize);
        Vector dx = patch->dCell();
        double dxDir = dx[d_lines[l]->loopDir];
        double tmp = stepSize/dxDir;
        
        int step = RoundUp(tmp);
        step = Max(step, 1);
        
        Box patchDomain = patch->getExtraBox();
        if(level->getIndex() > 0){ // ignore extra cells on fine patches
          patchDomain = patch->getBox();
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
  
        for(CellIterator iter=iterLim; !iter.done();iter+=step) {
          
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

          PerPatch<FileInfoP> filevar;
          FILE *fp;
          fp = fopen(filename.c_str(), "a");
          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:lineExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }
          
          // write cell position and time
          Point here = patch->cellPosition(c);
          double time = d_dataArchiver->getCurrentTime();
          fprintf(fp,    "%E\t %E\t %E\t %E",here.x(),here.y(),here.z(), time);
         
         
           // WARNING  If you change the order that these are written out you must 
           // also change the order that the header is written
           
          // write CC<int> variables      
          for (unsigned int i=0 ; i <  CC_integer_data.size(); i++) {
            fprintf(fp, "    %i",CC_integer_data[i][c]);            
          }          
          // write CC<double> variables
          for (unsigned int i=0 ; i <  CC_double_data.size(); i++) {
            fprintf(fp, "    %16E",CC_double_data[i][c]);            
          }
          // write CC<Vector> variable
          for (unsigned int i=0 ; i <  CC_Vector_data.size(); i++) {
            fprintf(fp, "    % 16E      %16E      %16E",
                    CC_Vector_data[i][c].x(),
                    CC_Vector_data[i][c].y(),
                    CC_Vector_data[i][c].z() );            
          }         
          // write SFCX<double> variables
          for (unsigned int i=0 ; i <  SFCX_double_data.size(); i++) {
            fprintf(fp, "    %16E",SFCX_double_data[i][c]);            
          }
          // write SFCY<double> variables
          for (unsigned int i=0 ; i <  SFCY_double_data.size(); i++) {
            fprintf(fp, "    %16E",SFCY_double_data[i][c]);            
          }
          // write SFCZ<double> variables
          for (unsigned int i=0 ; i <  SFCZ_double_data.size(); i++) {
            fprintf(fp, "    %16E",SFCZ_double_data[i][c]);            
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
  fprintf(fp,"X_CC      Y_CC      Z_CC      Time"); 
  
  // All CCVariable<int>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::CCVariable && 
       subtype->getType() == TypeDescription::int_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s(%i)", name.c_str(),d_varMatl[i]);
    }
  }
  // All CCVariable<double>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::CCVariable && 
       subtype->getType() == TypeDescription::double_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s(%i)", name.c_str(),d_varMatl[i]);
    }
  }
  // All CCVariable<Vector>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::CCVariable && 
       subtype->getType() == TypeDescription::Vector){
      string name = d_varLabels[i]->getName(); 
      int m = d_varMatl[i];
      fprintf(fp,"     %s(%i).x      %s(%i).y      %s(%i).z", name.c_str(),m,name.c_str(),m,name.c_str(),m);
    }
  }
  // All SFCXVariable<double>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::SFCXVariable && 
       subtype->getType() == TypeDescription::double_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s", name.c_str());
    }
  }
  // All SFCYVariable<double>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::SFCYVariable && 
       subtype->getType() == TypeDescription::double_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s", name.c_str());
    }
  }
  // All SFCZVariable<double>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == TypeDescription::SFCZVariable && 
       subtype->getType() == TypeDescription::double_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s", name.c_str());
    }
  }
  fprintf(fp,"\n");
  fclose(fp);
  cout << Parallel::getMPIRank() << " lineExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
//
void
lineExtract::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << "lineExtract:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  } else {
    closedir(check);
  }
  
  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL ) {
    cout << "lineExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
