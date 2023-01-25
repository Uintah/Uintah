/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/lineExtract.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>

#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>
#include <dirent.h>
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
lineExtract::lineExtract(const ProcessorGroup* myworld,
                         const MaterialManagerP materialManager,
                         const ProblemSpecP& module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set = 0;
  d_zero_matl = 0;
  ps_lb = scinew lineExtractLabel();
}

//__________________________________
lineExtract::~lineExtract()
{
  cout_doing << " Doing: destorying lineExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
   if(d_zero_matl && d_zero_matl->removeReference()) {
    delete d_zero_matl;
  }

  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  VarLabel::destroy(ps_lb->fileVarsStructLabel);
  delete ps_lb;

  // delete each line
  vector<line*>::iterator iter;
  for( iter  = d_lines.begin();iter != d_lines.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void lineExtract::problemSetup(const ProblemSpecP& ,
                               const ProblemSpecP& ,
                               GridP& grid,
                               std::vector<std::vector<const VarLabel* > > &PState,
                               std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tlineExtract" << endl;

  int numMatls  = m_materialManager->getNumMatls();

  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime_lineE",
                                            max_vartype::getTypeDescription());

  ps_lb->fileVarsStructLabel   = VarLabel::create("FileInfo_lineExtract",
                                            PerPatch<FileInfoP>::getTypeDescription());

  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", m_analysisFreq);
  m_module_spec->require("timeStart",         d_startTime);
  m_module_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("lineExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }


  // find the material to extract data from.  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  if(m_module_spec->findBlock("material") ){
    d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }
  else if (m_module_spec->findBlock("materialIndex") ){
    int indx;
    m_module_spec->get("materialIndex", indx);
    d_matl = m_materialManager->getMaterial(indx);
  }
  else {
    d_matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = d_matl->getDWIndex();

  //__________________________________
  //  Read in the optional material index from the variables that may be different
  //  from the default index
  vector<int> m;

  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);
  d_matl_set = scinew MaterialSet();
  map<string,string> attribute;

  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {
    var_spec->getAttributes(attribute);

    int matl = defaultMatl;
    if( !attribute["matl"].empty() ) {
      matl = atoi(attribute["matl"].c_str());
    }

    // Bulletproofing
    if( matl < 0 || matl > numMatls ){
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

  // for fileInfo variable
  d_zero_matl = scinew MaterialSubset();
  d_zero_matl->add(0);
  d_zero_matl->addReference();

  //__________________________________
  //  Read in variables label names
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);

    string name = attribute["label"];
    VarLabel* label = VarLabel::find( name );
    if( label == nullptr ){
      throw ProblemSetupException("lineExtract: analyze label not found: " + name , __FILE__, __LINE__);
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
    if(td->getType() != TypeDescription::CCVariable   &&
       td->getType() != TypeDescription::SFCXVariable &&
       td->getType() != TypeDescription::SFCYVariable &&
       td->getType() != TypeDescription::SFCZVariable ){
       throwException = true;
    }
    // CC Variables, only Doubles and Vectors
    if(td->getType() != TypeDescription::CCVariable       &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::int_type    &&
       subtype->getType() != TypeDescription::Vector  ){
      throwException = true;
    }
    // Face Centered Vars, only Doubles
    if( (td->getType() == TypeDescription::SFCXVariable ||
         td->getType() == TypeDescription::SFCYVariable ||
         td->getType() == TypeDescription::SFCZVariable)    &&
        (subtype->getType() != TypeDescription::double_type &&
         subtype->getType() != TypeDescription::Vector) ) {
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
  ProblemSpecP lines_ps = m_module_spec->findBlock("lines");
  if (!lines_ps){
    throw ProblemSetupException("\n ERROR:lineExtract: Couldn't find <lines> tag \n", __FILE__, __LINE__);
  }

  for (ProblemSpecP line_spec = lines_ps->findBlock("line"); line_spec != nullptr; line_spec = line_spec->findNextBlock("line")) {

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
    if(d_startTime > d_stopTime){
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
void lineExtract::scheduleInitialize(SchedulerP   & sched,
                                     const LevelP & level)
{
  cout_doing << "lineExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("lineExtract::initialize",
                  this, &lineExtract::initialize);

  t->computes(ps_lb->lastWriteTimeLabel);
  t->computes(ps_lb->fileVarsStructLabel, d_zero_matl);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void lineExtract::initialize(const ProcessorGroup *,
                             const PatchSubset    * patches,
                             const MaterialSubset *,
                             DataWarehouse        *,
                             DataWarehouse        * new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tlineExtract" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put(max_vartype(tminus), ps_lb->lastWriteTimeLabel);

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo,    ps_lb->fileVarsStructLabel, 0, patch);

    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:lineExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}

//______________________________________________________________________
void lineExtract::scheduleRestartInitialize(SchedulerP   & sched,
                                            const LevelP & level)
{
  scheduleInitialize( sched, level);
}

//______________________________________________________________________
void lineExtract::scheduleDoAnalysis(SchedulerP   & sched,
                                     const LevelP & level)
{
  cout_doing << "lineExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("lineExtract::doAnalysis",
                   this,&lineExtract::doAnalysis);

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_lineExtract", false, false, false, true, true);


  sched_TimeVars( t, level, ps_lb->lastWriteTimeLabel, true );

  t->requires(Task::OldDW, ps_lb->fileVarsStructLabel, d_zero_matl, Ghost::None, 0);

  Ghost::GhostType gac = Ghost::AroundCells;

  //__________________________________
  //
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == nullptr){
      string name = d_varLabels[i]->getName();
      throw InternalError("lineExtract: scheduleDoAnalysis label not found: " + name , __FILE__, __LINE__);
    }

    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add(d_varMatl[i]);
    matSubSet->addReference();

    t->requires(Task::NewDW,d_varLabels[i], matSubSet, gac, 1);

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }

  t->computes(ps_lb->fileVarsStructLabel, d_zero_matl);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void lineExtract::doAnalysis(const ProcessorGroup * pg,
                             const PatchSubset    * patches,
                             const MaterialSubset *,
                             DataWarehouse        * old_dw,
                             DataWarehouse        * new_dw)
{
  const Level* level = getLevel(patches);

  timeVars tv;

  getTimeVars( old_dw, level, ps_lb->lastWriteTimeLabel, tv );
  putTimeVars( new_dw, ps_lb->lastWriteTimeLabel, tv );

  if( tv.isItTime == false ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // open the struct that contains a map of the file pointers.
    // Note: after regridding this may not exist for this patch in the old_dw
    PerPatch<FileInfoP> fileInfo;

    if( old_dw->exists( ps_lb->fileVarsStructLabel, 0, patch ) ){
      old_dw->get(fileInfo, ps_lb->fileVarsStructLabel, 0, patch);
    }else{
      FileInfo* myFileInfo = scinew FileInfo();
      fileInfo.get() = myFileInfo;
    }

    std::map<string, FILE *> myFiles;

    if( fileInfo.get().get_rep() ){
      myFiles = fileInfo.get().get_rep()->files;
    }

    int proc =
      m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);

    cout_dbg << Parallel::getMPIRank() << "   working on patch " << patch->getID() << " which is on proc " << proc << endl;
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myRank() ){

     cout_doing << pg->myRank() << " "
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

      vector< constSFCXVariable<Vector> > SFCX_Vector_data;
      vector< constSFCYVariable<Vector> > SFCY_Vector_data;
      vector< constSFCZVariable<Vector> > SFCZ_Vector_data;

      constCCVariable<int>    q_CC_integer;
      constCCVariable<double> q_CC_double;
      constCCVariable<Vector> q_CC_Vector;

      constSFCXVariable<double> q_SFCX_double;
      constSFCYVariable<double> q_SFCY_double;
      constSFCZVariable<double> q_SFCZ_double;

      constSFCXVariable<Vector> q_SFCX_Vector;
      constSFCYVariable<Vector> q_SFCY_Vector;
      constSFCZVariable<Vector> q_SFCZ_Vector;

      Ghost::GhostType gac = Ghost::AroundCells;


      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {

        // bulletproofing
        if(d_varLabels[i] == nullptr){
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

            switch( subtype->getType() ) {
              case Uintah::TypeDescription::double_type:
                new_dw->get( q_SFCX_double, d_varLabels[i], indx, patch, gac, 1 );
                SFCX_double_data.push_back( q_SFCX_double );
                break;

              case Uintah::TypeDescription::Vector:
                new_dw->get( q_SFCX_Vector, d_varLabels[i], indx, patch, gac, 1 );
                SFCX_Vector_data.push_back( q_SFCX_Vector );
                break;
              default:
                throw InternalError("LineExtract: invalid data type", __FILE__, __LINE__);
            }
            break;
          case Uintah::TypeDescription::SFCYVariable:    // SFCY Variables
            switch( subtype->getType() ) {
              case Uintah::TypeDescription::double_type:
                new_dw->get( q_SFCY_double, d_varLabels[i], indx, patch, gac, 1 );
                SFCY_double_data.push_back( q_SFCY_double );
                break;

              case Uintah::TypeDescription::Vector:
                new_dw->get( q_SFCY_Vector, d_varLabels[i], indx, patch, gac, 1 );
                SFCY_Vector_data.push_back( q_SFCY_Vector );
                break;
              default:
                throw InternalError("LineExtract: invalid data type", __FILE__, __LINE__);
            }
            break;
          case Uintah::TypeDescription::SFCZVariable:   // SFCZ Variables
            switch( subtype->getType() ) {
              case Uintah::TypeDescription::double_type:
                new_dw->get( q_SFCZ_double, d_varLabels[i], indx, patch, gac, 1 );
                SFCZ_double_data.push_back( q_SFCZ_double );
                break;

              case Uintah::TypeDescription::Vector:
                new_dw->get( q_SFCZ_Vector, d_varLabels[i], indx, patch, gac, 1 );
                SFCZ_Vector_data.push_back( q_SFCZ_Vector );
                break;
              default:
                throw InternalError("LineExtract: invalid data type", __FILE__, __LINE__);
            }
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
        string udaDir   = m_output->getOutputLocation();
        string dirName  = d_lines[l]->name;
        string linePath = udaDir + "/" + dirName;

        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        string path       = linePath + "/" + levelIndex;

        if( d_isDirCreated.count(path) == 0){
          createDirectory(linePath, levelIndex);
          d_isDirCreated.insert(path);
        }

        // find the physical domain and index range
        // associated with this patch
        Point start_pt = d_lines[l]->startPt;
        Point end_pt   = d_lines[l]->endPt;

        double stepSize(d_lines[l]->stepSize);
        Vector dx    = patch->dCell();
        double dxDir = dx[d_lines[l]->loopDir];
        double tmp   = stepSize/dxDir;

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

          //__________________________________
          //  Open the file pointer
          //  if it's not in the fileInfo struct then create it
          FILE *fp;

          if( myFiles.count(filename) == 0 ){
            createFile(filename, fp);
            myFiles[filename] = fp;

          } else {
            fp = myFiles[filename];
          }

          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:lineExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }

          // write cell position and time
          Point here = patch->cellPosition(c);
          const int w = d_col_width;
          fprintf(fp,    "%-*E %-*E %-*E %-*E",w, here.x(), w, here.y(), w, here.z(), w, tv.now);

           // WARNING If you change the order that these are written
           // out you must also change the order that the header is
           // written

          // write CC<int> variables
          for (unsigned int i=0 ; i< CC_integer_data.size(); i++) {
            fprintf(fp, "%-*i", w, CC_integer_data[i][c]);
          }

          fprintf_Arrays( fp, c, CC_double_data,   CC_Vector_data);
          fprintf_Arrays( fp, c, SFCX_double_data, SFCX_Vector_data);
          fprintf_Arrays( fp, c, SFCY_double_data, SFCY_Vector_data);
          fprintf_Arrays( fp, c, SFCZ_double_data, SFCZ_Vector_data);

          fprintf(fp,"\n");
          fflush(fp);
        }  // loop over points
      }  // loop over lines
    }  // time to write data

    // put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo, ps_lb->fileVarsStructLabel, 0, patch);
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void lineExtract::createFile(string& filename,  FILE*& fp)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }

   // WARNING If you change the order that these are written
   // out you must also change the order in doAnalysis()

  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"%-*s %-*s %-*s %-*s", d_col_width,"# X_CC",
                                    d_col_width,"Y_CC",
                                    d_col_width,"Z_CC",
                                    d_col_width,"Time [s]");

  printHeader( fp,TypeDescription::CCVariable);
  printHeader( fp,TypeDescription::SFCXVariable);
  printHeader( fp,TypeDescription::SFCYVariable);
  printHeader( fp,TypeDescription::SFCZVariable);

  fprintf(fp,"\n");
  fflush(fp);

  cout << Parallel::getMPIRank() << " lineExtract:Created file " << filename << endl;
}

//______________________________________________________________________
//
void
lineExtract::printHeader( FILE*& fp,
                          const Uintah::TypeDescription::Type myType)
{

  //__________________________________
  // <double/int>
  for (unsigned int i =0 ; i< d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType()      == myType &&
       (subtype->getType() == TypeDescription::double_type ||
        subtype->getType() == TypeDescription::int_type ) ){
      string name = d_varLabels[i]->getName();

      ostringstream colDesc;
      colDesc  <<  std::left << name << "_"<< d_varMatl[i] << setw(d_col_width) << " ";

      string tmp = colDesc.str().substr(0,d_col_width);  // crop the description
      const char* cstr = tmp.c_str();
      fprintf( fp, "%s", cstr );
    }
  }
  //__________________________________
  // <Vector>
  for (unsigned int i =0 ; i< d_varLabels.size(); i++) {
    const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType()      == myType  &&
        subtype->getType() == TypeDescription::Vector ){
      string name = d_varLabels[i]->getName();

      ostringstream colDescX;
      ostringstream colDescY;
      ostringstream colDescZ;
      colDescX << std::left << name << "_"<< d_varMatl[i] << ".x" << setw(d_col_width) << " ";
      colDescY << std::left << name << "_"<< d_varMatl[i] << ".y" << setw(d_col_width) << " ";
      colDescZ << std::left << name << "_"<< d_varMatl[i] << ".z" << setw(d_col_width) << " ";

       // crop the descriptions
      string tmpX = colDescX.str().substr(0,d_col_width);
      string tmpY = colDescY.str().substr(0,d_col_width);
      string tmpZ = colDescZ.str().substr(0,d_col_width);

      const char* cstrX = tmpX.c_str();
      const char* cstrY = tmpY.c_str();
      const char* cstrZ = tmpZ.c_str();
      fprintf( fp, "%s %s %s", cstrX, cstrY, cstrZ );
    }
  }
}

//______________________________________________________________________
//
template< class D, class V >
void lineExtract::fprintf_Arrays( FILE*& fp,
                                  const IntVector& c,
                                  const D& doubleData,
                                  const V& VectorData)
{
   const int w = d_col_width;
   // double variables
   for (unsigned int i=0 ; i< doubleData.size(); i++) {
     fprintf(fp, "%-*E", w, doubleData[i][c]);
   }

   // Vector variable
   for (unsigned int i=0 ; i< VectorData.size(); i++) {
     fprintf(fp, "%-*E %-*E %-*E",
             w, VectorData[i][c].x(),
             w, VectorData[i][c].y(),
             w, VectorData[i][c].z() );
   }
}

//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
//
void
lineExtract::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == nullptr ) {
    cout << Parallel::getMPIRank() << "lineExtract:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  } else {
    closedir(check);
  }

  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == nullptr ) {
    cout << "lineExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
