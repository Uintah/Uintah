/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/OnTheFlyAnalysis/planeExtract.h>

#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <dirent.h>
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <fstream>

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "PLANEEXTRACT_DBG_COUT:+" 
static DebugStream cout_doing("PLANEEXTRACT_DOING_COUT", false);
static DebugStream cout_dbg("PLANEEXTRACT_DBG_COUT", false);
//______________________________________________________________________              
planeExtract::planeExtract(ProblemSpecP& module_spec,
                           SimulationStateP& sharedState,
                           Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState = sharedState;
  d_prob_spec = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  d_zero_matl = 0;
  ps_lb = scinew planeExtractLabel();
}

//__________________________________
planeExtract::~planeExtract()
{
  cout_doing << " Doing: destorying planeExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
   if(d_zero_matl && d_zero_matl->removeReference()) {
    delete d_zero_matl;
  } 
  
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  VarLabel::destroy(ps_lb->fileVarsStructLabel);
  delete ps_lb;
  
  // delete each plane
  vector<plane*>::iterator iter;
  for( iter  = d_planes.begin();iter != d_planes.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void planeExtract::problemSetup(const ProblemSpecP& prob_spec,
                                const ProblemSpecP& ,
                                GridP& grid,
                                SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tplaneExtract" << endl;
  
  int numMatls  = d_sharedState->getNumMatls();
  if(!d_dataArchiver){
    throw InternalError("planeExtract:couldn't get output port", __FILE__, __LINE__);
  }
                               
  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());       
                                            
  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_startTime);            
  d_prob_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("planeExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 

  
  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  
  const Material* matl = NULL;
  
  if(d_prob_spec->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }
  
  int defaultMatl = matl->getDWIndex();
  
  //__________________________________
  //  Read in the optional material index from the variables that may be different
  //  from the default index
  vector<int> m;
  
  m.push_back(0);            // matl for FileInfo label
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
      throw ProblemSetupException("planeExtract: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
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
      throw ProblemSetupException("planeExtract: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    //__________________________________
    //  Bulletproofing
    // The user must specify the matl for single matl variables
    if ( name == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("planeExtract: You must add (matl='0') to the press_CC plane." , __FILE__, __LINE__);
    }
    
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();
    
    //__________________________________
    bool throwException = false;  
    
    // only CC, SFCX, SFCY, SFCZ variables
    if(td->getType() != TypeDescription::CCVariable   &&
       td->getType() != TypeDescription::SFCXVariable &&
       td->getType() != TypeDescription::SFCYVariable &&
       td->getType() != TypeDescription::SFCZVariable ){
       throwException = true;
    }
    // CC Variables, only Doubles Vectors and Stencil7
    if(td->getType() != TypeDescription::CCVariable       &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::int_type    &&
       subtype->getType() != TypeDescription::Vector      &&
       subtype->getType() != TypeDescription::Stencil7 ){
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
      warn << "ERROR:AnalysisModule:planeExtact: ("<<label->getName() << " " 
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    d_varLabels.push_back(label);
  }    
  
  //__________________________________
  //  Read in planes
  ProblemSpecP planes_ps = d_prob_spec->findBlock("planes"); 
  if (!planes_ps){
    throw ProblemSetupException("\n ERROR:planeExtract: Couldn't find <planes> tag \n", __FILE__, __LINE__);    
  }        
             
  for (ProblemSpecP plane_spec = planes_ps->findBlock("plane"); plane_spec != 0; 
                    plane_spec = plane_spec->findNextBlock("plane")) {
                    
    plane_spec->getAttributes(attribute);
    string name = attribute["name"];

    Point start, end;
    plane_spec->require("startingPt", start);
    plane_spec->require("endingPt",   end);
    //__________________________________
    // bullet proofing
    // -every plane must have a name
    // -plane must be parallel to the coordinate system
    // -plane can't exceed computational domain
    if(name == ""){
      throw ProblemSetupException("\n ERROR:planeExtract: You must name each plane <plane name= something>\n", 
                                  __FILE__, __LINE__);
    }
    
    // plane must be parallel to the coordinate system
    bool X = (start.x() == end.x());
    bool Y = (start.y() == end.y());  // 1 out of 3 of these must be true
    bool Z = (start.z() == end.z());
    
    bool validPlane = false;
    
    if( !X && !Y && Z){               /*  XY plane */
      validPlane = true;
    }
    if( !X && Y && !Z){               /* XZ plane */
      validPlane = true;
    }
    if( X && !Y && !Z){               /* YZ plane */
      validPlane = true;
    }
    if(validPlane == false){
      ostringstream warn;
      warn << "\n ERROR:planeExtract: the plane that you've specified " << start 
           << " " << end << " is not parallel to the coordinate system. \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    //plane can't exceed computational domain
    BBox compDomain;
    grid->getSpatialRange(compDomain);

    Point min = compDomain.min();
    Point max = compDomain.max();

    if(start.x() < min.x() || start.y() < min.y() ||start.z() < min.z() ||
         end.x() > max.x() ||   end.y() > max.y() ||  end.z() > max.z() ){
      ostringstream warn;
      warn << "\n ERROR:planeExtract: the plane that you've specified " << start 
           << " " << end << " begins or ends outside of the computational domain. \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    if( start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
      ostringstream warn;
      warn << "\n ERROR:planeExtract: the plane that you've specified " << start 
           << " " << end << " the starting point is > than the ending point \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    // Start time < stop time
    if(d_startTime > d_stopTime){
      throw ProblemSetupException("\n ERROR:planeExtract: startTime > stopTime. \n", __FILE__, __LINE__);
    }
    
    // put input variables into the global struct
    plane* p = scinew plane;
    p->name     = name;
    p->startPt  = start;
    p->endPt    = end;
    d_planes.push_back(p);
  }
}

//______________________________________________________________________
void planeExtract::scheduleInitialize(SchedulerP& sched,
                                      const LevelP& level)
{
  cout_doing << "planeExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("planeExtract::initialize", 
                  this, &planeExtract::initialize);
  
  t->computes(ps_lb->lastWriteTimeLabel);
 
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void planeExtract::initialize(const ProcessorGroup*, 
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tplaneExtract" << endl;
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
        warn << "ERROR:planeExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void planeExtract::restartInitialize()
{
// need to do something here
//  new_dw->put(max_vartype(0.0), ps_lb->lastWriteTimeLabel);
}

//______________________________________________________________________
void planeExtract::scheduleDoAnalysis(SchedulerP& sched,
                                      const LevelP& level)
{
  cout_doing << "planeExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("planeExtract::doAnalysis", 
                   this,&planeExtract::doAnalysis);
                        
  t->requires(Task::OldDW, ps_lb->lastWriteTimeLabel);
  Ghost::GhostType gac = Ghost::AroundCells;
  
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == NULL){
      string name = d_varLabels[i]->getName();
      throw InternalError("planeExtract: scheduleDoAnalysis label not found: " 
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
void planeExtract::doAnalysis(const ProcessorGroup* pg,
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
  
  if(now < d_startTime || now > d_stopTime){
    return;
  }
  
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
                << "Doing doAnalysis (planeExtract)\t\t\t\tL-"
                << level->getIndex()
                << " patch " << patch->getGridIndex()<< endl;
      //__________________________________
      // loop over each plane 
      for (unsigned int p =0 ; p < d_planes.size(); p++) {
      
        // create the directory structure
        string udaDir = d_dataArchiver->getOutputLocation();
        string dirName = d_planes[p]->name;
        string planePath = udaDir + "/" + dirName;
        
        ostringstream tname;
        tname << "t" << std::setw(5) << std::setfill('0') << d_sharedState->getCurrentTopLevelTimeStep();
        string timestep = tname.str();
        
        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        string path = planePath + "/" + timestep + "/" + levelIndex;
        
        if( d_isDirCreated.count(path) == 0 ){
          createDirectory( planePath, timestep, levelIndex );
          d_isDirCreated.insert( path );
        }
        
        //__________________________________
        // find the physical domain and index range
        // associated with this patch
        Point start_pt = d_planes[p]->startPt;
        Point end_pt   = d_planes[p]->endPt;
        
        
        Box patchDomain = patch->getExtraBox();
        if(level->getIndex() > 0){ // ignore extra cells on fine patches
          patchDomain = patch->getBox();
        }
        // intersection
        start_pt = Max(patchDomain.lower(), start_pt);
        end_pt   = Min(patchDomain.upper(), end_pt);
        
        // indices
        IntVector start_idx, end_idx;
        patch->findCell(start_pt, start_idx);
        patch->findCell(end_pt,   end_idx);
        
        // increase the end_idx by 1 in the out of plane direction
        for(int d=0; d<3; d++){
          if (start_idx[d] == end_idx[d]){
            end_idx[d] += 1;
          }
        }
        
        CellIterator iterLim = CellIterator(start_idx,end_idx);
        
        //__________________________________
        //  Loop over all the variables
        for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
          
          int matl = d_varMatl[i];
          const VarLabel* varLabel = d_varLabels[i];
          string labelName = varLabel->getName();
          
          // bulletproofing
          if(varLabel == NULL){
            throw InternalError("planeExtract: analyze label not found: " 
                            + labelName , __FILE__, __LINE__);
          }

          ostringstream fname;
          fname<< path << "/" << labelName <<"_"<< matl;
          string filename = fname.str();

          //__________________________________
          //  Open the file pointer 
          FILE *fp;
          createFile(filename, varLabel, matl, fp);

          //__________________________________
          //  Now write to the file
          const Uintah::TypeDescription* td = varLabel->typeDescription();
          const Uintah::TypeDescription* subtype = td->getSubType();
           
          switch( td->getType() ){
            case Uintah::TypeDescription::CCVariable:      // CC Variables
              switch( subtype->getType( )) {

              case Uintah::TypeDescription::double_type:
                writeDataD< constCCVariable<double> >(   new_dw, varLabel, matl, patch, iterLim, fp );
                break;

              case Uintah::TypeDescription::Vector:
                writeDataV< constCCVariable<Vector> >(   new_dw, varLabel, matl, patch, iterLim, fp );
                break;

              case Uintah::TypeDescription::int_type:
                writeDataI< constCCVariable<int> >(      new_dw, varLabel, matl, patch, iterLim, fp );
                break;
                 
              case Uintah::TypeDescription::Stencil7:
                writeDataS7< constCCVariable<Stencil7> >(new_dw, varLabel, matl, patch, iterLim, fp );
                break;
              default:
                throw InternalError("planeExtract: invalid data type", __FILE__, __LINE__); 
              }
              break;
            default:
              ostringstream warn;
              warn << "ERROR:AnalysisModule:planeExtact: ("<< labelName << " " 
                   << td->getName() << " ) has not been implemented" << endl;
              throw InternalError(warn.str(), __FILE__, __LINE__);
          }
          fclose(fp);
        }  //loop over variables      
      }  // loop over planes 
      lastWriteTime = now;     
    }  // time to write data
    
    new_dw->put(max_vartype(lastWriteTime), ps_lb->lastWriteTimeLabel); 
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void planeExtract::createFile(const string& filename,
                              const VarLabel* varLabel,
                              const int matl,
                              FILE*& fp )
{
  // if the file already exists then exit.
  ifstream doExists( filename.c_str() );
  if( doExists ){
    fp = fopen(filename.c_str(), "a");
    return;
  }
  
  fp = fopen(filename.c_str(), "w");
  
  if (!fp){
    perror("Error opening file:");
    throw InternalError("\nERROR:dataAnalysisModule:planeExtract:  failed opening file"+filename,__FILE__, __LINE__);
  }
  
  //__________________________________
  //Write out the header
  fprintf(fp,"# X      Y      Z "); 
  
  
  const Uintah::TypeDescription* td = varLabel->typeDescription();
  const Uintah::TypeDescription* subtype = td->getSubType();
  string labelName = varLabel->getName();
  
  switch( subtype->getType( )) {

    case Uintah::TypeDescription::double_type:
      fprintf(fp,"     %s(%i)", labelName.c_str(), matl);
      break;

    case Uintah::TypeDescription::Vector:
      fprintf(fp,"     %s(%i).x      %s(%i).y      %s(%i).z", labelName.c_str(), matl, labelName.c_str(), matl, labelName.c_str(), matl);
      break;

    case Uintah::TypeDescription::int_type:
      fprintf(fp,"     %s(%i)", labelName.c_str(), matl);
      break;

    case Uintah::TypeDescription::Stencil7:
      fprintf(fp,"     %s(%i).n      s      e      w      t      b      p", labelName.c_str(), matl );
      break;
    default:
      throw InternalError("planeExtract: invalid data type", __FILE__, __LINE__); 
  }
  
  fprintf(fp,"\n");
  fflush(fp);
  
  cout << Parallel::getMPIRank() << " planeExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   planeName/LevelIndex
//
void
planeExtract::createDirectory(string& planeName, string& timestep, string& levelIndex)
{
  DIR *check = opendir(planeName.c_str());
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << " planeExtract:Making directory " << planeName << endl;
    MKDIR( planeName.c_str(), 0777 );
  } else {
    closedir(check);
  }

  // timestep
  string path = planeName + "/" + timestep;
  check = opendir( path.c_str() );
  
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << " planeExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
    
  // level index
  path = planeName + "/" + timestep + "/" + levelIndex;
  check = opendir( path.c_str() );
  
  if ( check == NULL ) {
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}

//______________________________________________________________________
template <class Tvar>
void planeExtract::writeDataD( DataWarehouse*  new_dw,
                              const VarLabel* varLabel,
                              const int       indx,
                              const Patch*    patch,
                              CellIterator    iter,
                              FILE*     fp )
{  
  Tvar Q_var;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);
  
  for (;!iter.done();iter++) {
    IntVector c = *iter;
    Point here = patch->cellPosition(c);
                      
    fprintf(fp,    "%16.15E\t %16.15E\t %16.15E\t",here.x(),here.y(),here.z());
    fprintf(fp, "    %16.15E\n",Q_var[c]);
  }  
}

//______________________________________________________________________
template <class Tvar>
void planeExtract::writeDataV( DataWarehouse*  new_dw,
                               const VarLabel* varLabel,
                               const int       indx,
                               const Patch*    patch,
                               CellIterator    iter,
                               FILE*     fp )
{  
  Tvar Q_var;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);
  
  for (;!iter.done();iter++) {
    IntVector c = *iter;
    Point here = patch->cellPosition(c);
                      
    fprintf(fp,    "%16.15E\t %16.15E\t %16.15E\t",here.x(), here.y(), here.z());
    fprintf(fp, "   %16.15E\t %16.15E\t %16.15E\n",Q_var[c].x(), Q_var[c].y(), Q_var[c].z() );
  }  
}
//______________________________________________________________________
template <class Tvar>
void planeExtract::writeDataI( DataWarehouse*  new_dw,
                               const VarLabel* varLabel,
                               const int       indx,
                               const Patch*    patch,
                               CellIterator    iter,
                               FILE*     fp )
{  
  Tvar Q_var;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);
  
  for (;!iter.done();iter++) {
    IntVector c = *iter;
    Point here = patch->cellPosition(c);
                      
    fprintf(fp,    "%16.15E\t %16.15E\t %16.15E\t",here.x(), here.y(), here.z());
    fprintf(fp, "   %i \n",Q_var[c] );
  }  
}
//______________________________________________________________________
template <class Tvar>
void planeExtract::writeDataS7( DataWarehouse*  new_dw,
                               const VarLabel* varLabel,
                               const int       indx,
                               const Patch*    patch,
                               CellIterator    iter,
                               FILE*     fp )
{  
  Tvar Q;
  new_dw->get(Q, varLabel, indx, patch, Ghost::None, 0);
  
  for (;!iter.done();iter++) {
    IntVector c = *iter;
    Point here = patch->cellPosition(c);
    
    fprintf(fp,    "%16.15E\t %16.15E\t %16.15E\t",here.x(), here.y(), here.z());
    fprintf(fp, "   %16.15E\t %16.15E\t %16.15E\t %16.15E\t %16.15E\t %16.15E\t %16.15E \n",
            Q[c].n, Q[c].s, Q[c].e, Q[c].w, Q[c].t, Q[c].b, Q[c].p );
  }  
}
