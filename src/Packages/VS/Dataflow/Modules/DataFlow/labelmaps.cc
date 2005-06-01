/******************************************************************************
 * File: labelmaps.cc
 *
 * Description: C source code for classes to provide an API for Visible Human
 *		segmentation information:  The Master Anatomy Label Map
 *		Spatial Adjacency relations for the anatomical structures
 *		and Bounding Boxes for each anatomical entity.
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 *         DARPA Virtual Soldier Project <http://www.virtualsoldier.net>
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "labelmaps.h"

using namespace std;

/******************************************************************************
 * misc string manip functions 
 ******************************************************************************/

char *space_to_underbar(char *dst, char *src)
{
  char *srcPtr, *dstPtr;
  for(srcPtr = src, dstPtr = dst; *srcPtr != '\0'; srcPtr++, dstPtr++)
  {
    if(*srcPtr == ' ') *dstPtr = '_';
    else *dstPtr = *srcPtr;
  } /* end while(*srcPtr != '\0') */
  /* srcPtr points to '\0', but *dstPtr is uninitialized */
  *dstPtr = '\0';
  return(dst);
} /* end space_to_underbar() */

bool iswhitespace(char *src)
{
  char *srcPtr;

  for(srcPtr = src; *srcPtr != '\0'; srcPtr++)
  {
    if(*srcPtr != ' ' && *srcPtr != '\t') return(false);
  }
  return(true);
}

char *capitalize(char *dst, char *src)
{
  char *srcPtr, *dstPtr;
  srcPtr = src; dstPtr = dst;
  /* skip white space */
  while(*srcPtr != '\0' && (*srcPtr == ' ' || *srcPtr == '\t'))
  { *dstPtr++ = *srcPtr++; }
  if(*srcPtr == '\0')
  {
    *dstPtr = '\0';
    return(dst);
  }
  if( 'a' <= *srcPtr )
  {
      *dstPtr++ = *srcPtr - 32; 
      srcPtr++;
  }
  /* copy the rest of the string */
  while(*srcPtr != '\0')
  {
    *dstPtr++ = *srcPtr++;
  }
  *dstPtr = '\0';
   return(dst);
} /* end capitalize() */

int
countCommas(char *inString)
{
  int count = 0;
  while(*inString != '\0')
  {
    if(*inString++ == ',') count++;
  }
  return count;
} /* end countCommas() */

int
countTabs(char *inString)
{
  int count = 0;
  while(*inString != '\0')
  {
    if(*inString++ == '\t') count++;
  }
  return count;
} /* end countCommas() */

int
splitAtComma(char *srcStr, char **dst0, char **dst1)
{
  char *srcPtr = srcStr;
  int count = 0;

  if(!srcStr || !dst0 || !dst1)
  {
    cerr << "VH_MasterAnatomy::splitAtComma(): NULL string" << endl;
    return(0);
  }
  while(*srcPtr != ',') { srcPtr++; count++; }
#ifdef __APPLE__
  *dst0 = (char *)malloc((count + 1) * sizeof(char));
  bzero(*dst0, count + 1);
  strncpy(*dst0, srcStr, count);
#else
  *dst0 = strndup(srcStr, count);
#endif
  // srcPtr points to ','
  *dst1 = ++srcPtr;
  // remove trailing newline
  while(*srcPtr != '\0' && *srcPtr != '\n') srcPtr++;
  if(*srcPtr == '\n') *srcPtr = '\0';
  return(1);
} /* end splitAtComma() */


char *read_line(char *retbuff, int *bufsize, FILE *infile)
{
char inbuff[VH_FILE_MAXLINE];
int readsize = VH_FILE_MAXLINE;

// read a MAXLINE-byte chunk of the file
char *stat = fgets(inbuff, readsize, infile);
                                                                                
if(stat)
  {
    int readlen = strlen(stat);
    int bufflen = strlen(retbuff);
    bool longline = (readlen == VH_FILE_MAXLINE-1) &&
                         (stat[readlen-1] != '\n');
    bool continued = false;
    if(readlen > 1 && stat[readlen-2] == '\\')
      { // reading continued lines, indicated by '<backslash>-<newline>'
        continued = true;
        readlen -= 2;
      }
    if(longline || continued)
      { // reading a line longer than MAXLINE
        int cumlen = bufflen + readlen + 1;
                                                                                
//      cerr << "\nread_line: ";
                                                                                
        if(cumlen > *bufsize)
          {
//          cerr << "read " << readlen;
//          cerr << " chars into buffer[" << *bufsize << "]";
                                                                                
            // allocate a new buffer for returned string
            *bufsize = *bufsize * 2;
//          retbuff = (char *)realloc( retbuff, *bufsize );
            char *newBuf = new char[*bufsize];
            bcopy(retbuff, newBuf, bufflen+1);
            delete [] retbuff;
            retbuff = newBuf;
            // with new buffsize
          }
//      cerr << "retbuff '" << retbuff << "' inbuff '" << inbuff << "'" << endl;                                                                                
        // append contents of read to return buffer
        strncat(retbuff, inbuff, readlen);
                                                                                
        // continue reading
//      cerr << " read_line calling read_line" << endl;
        retbuff = read_line(retbuff, bufsize, infile);
      }
    else
      { // the total length of the line is less than bufsize
        strcat(retbuff, inbuff);
      }
    return(retbuff);
  }
else
  {
    return((char *)0);
  }
} // end read_line()

/******************************************************************************
 * class VH_MasterAnatomy
 *
 * The mapping between 16-bit integer indices in the Labelmap volume
 * (e.g., Voxel Addresses) to the anatomical names of the segmented
 * tissue types.
 ******************************************************************************/

VH_MasterAnatomy::VH_MasterAnatomy()
{
  anatomyname = new (char *)[VH_LM_NUM_NAMES];
  labelindex = new int[VH_LM_NUM_NAMES];
  num_names = 0;
  maxLabelIndex = 0;
}

VH_MasterAnatomy::~VH_MasterAnatomy()
{
  delete [] anatomyname;
  delete [] labelindex;
}

void
VH_MasterAnatomy::readFile(FILE *infileptr)
{
} // end VH_MasterAnatomy::readFile(FILE *infileptr)


void
VH_MasterAnatomy::readFile(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_MasterAnatomy::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;

  activeFile = string(infilename);
  cerr << "VH_MasterAnatomy::readFile(" << infilename << ")";
  char *indexStr;
  // skip first line
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_MasterAnatomy::readFile(): premature EOF" << endl;
  }
  // clear input buffer line
    strcpy(inLine, "");

  // label 0 is "unknown"
  anatomyname[num_names] = strdup("unknown");
  labelindex[num_names] = 0;
  num_names++;
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form: AnatomyName,Label
      if(!splitAtComma((char *)inLine,
                   &anatomyname[num_names], &indexStr)) break;
      // label indices to not correspond 1-1 to lines in the file
      labelindex[num_names] = atoi(indexStr);
      if(maxLabelIndex < labelindex[num_names])
         maxLabelIndex = labelindex[num_names];
    } // end if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
    num_names++;
    cerr << ".";
  } // end while(read_line(inLine, &buffsize, infile) != 0)

  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
} // end VH_MasterAnatomy::readFile(char *infilename)

char *
VH_MasterAnatomy::get_anatomyname(int targetindex)
{
  if(targetindex < 0)
  {
    cerr << "VH_MasterAnatomy::get_anatomyname(): index " << targetindex;
    cerr << " out of range " << num_names << endl;
    // return "UNKNOWN"
    return(anatomyname[0]);
  }
  int nameIndex;
  for(nameIndex = 0; nameIndex < num_names; nameIndex++)
  {
    if(labelindex[nameIndex] == targetindex) break;
  }
  if(nameIndex >= num_names)
  {
    cerr << "VH_MasterAnatomy::get_anatomyname(): index " << targetindex;
    cerr << " out of range " << maxLabelIndex << endl;
    // return "UNKNOWN"
    return(anatomyname[0]);
  }
  // (else)
  return(anatomyname[nameIndex]);
} // end VH_MasterAnatomy::get_anatomyname(int labelindex)

int
VH_MasterAnatomy::get_labelindex(char *targetname)
{
  for(int index = 0; index < num_names; index++)
  {
    if(anatomyname[index] != 0 && strcmp(anatomyname[index], targetname) == 0)
      return(labelindex[index]);
  }

  return(-1);
} // end VH_MasterAnatomy::get_labelindex(char *anatomyname)

/******************************************************************************
 * class VH_AdjacencyMapping
 *
 * Lists of relations between tissue types by adjacency.  Lists consist of the
 * 16-bit label map tissue indices.
 ******************************************************************************/

VH_AdjacencyMapping::VH_AdjacencyMapping()
{
  rellist = new (int *)[VH_LM_NUM_NAMES];
  numrel = new int[VH_LM_NUM_NAMES];
  num_names = 0;
}

VH_AdjacencyMapping::~VH_AdjacencyMapping()
{
  delete [] rellist;
  delete [] numrel;
}

void
VH_AdjacencyMapping::readFile(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_AdjacencyMapping::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;

  activeFile = string(infilename);
  cerr << "VH_AdjacencyMapping::readFile(" << infilename << ")";
  num_names = 0;
  char *indexStr;
  // skip the first line
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_AdjacencyMapping::readFile(): premature EOF" << endl;
  }
  // clear input buffer line
    strcpy(inLine, "");

  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form: index0,index1,...indexn
      int num_adj = countCommas(inLine);
      numrel[num_names] = num_adj;
   cerr << "line[" << num_names << "], " << strlen(inLine) << " chars, ";
   cerr << num_adj << " relations" << endl;
      rellist[num_names] = new int[num_adj];
      indexStr = strtok(inLine, ",");
      // first index is the MasterAnatomy name index
      if(atoi(indexStr) != num_names)
      {
        cerr << "VH_AdjacencyMapping::readFile(): line index mismatch: ";
        cerr << indexStr << "(" << atoi(indexStr) << ") <=> " << num_names;
        cerr << endl;
      }
      int *intPtr = rellist[num_names];
      bzero(intPtr,num_adj*sizeof(int));
      while((indexStr = strtok(NULL, ",")) != NULL)
      {
        *intPtr++ = atoi(indexStr);
      }
      if(++num_names >= VH_LM_NUM_NAMES)
      {
        cerr << "VH_AdjacencyMapping::readFile(): too many names" << endl;
        return;
      }
      cerr << ".";
    } if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
  } // end while(read_line(inLine, &buffsize, infile) != 0)
  delete [] inLine;
  fclose(infile);
  cerr << "done with reading adjacency mapping file" << endl;
} // end VH_AdjacencyMapping::readFile(char *infilename)

void
VH_AdjacencyMapping::readFile(FILE *infileptr)
{
} // end VH_AdjacencyMapping::readFile(FILE *infileptr)

int *
VH_AdjacencyMapping::adjacent_to(int index)
{
  if(index < num_names) {
     return rellist[index];
  }
  else
  {
     cerr << "VH_AdjacencyMapping::adjacent_to(): index " << index;
     cerr << " out of range " << num_names << endl;
  }
  // return the relation list for "unknown"
  return(rellist[0]);
} /* end VH_AdjacencyMapping::adjacent_to() */

int 
VH_AdjacencyMapping::get_num_rel(int index)
{
  if(index < num_names)
     return numrel[index];
  else
  {
     cerr << "VH_AdjacencyMapping::get_num_rel(): index " << index;
     cerr << " out of range " << num_names << endl;
  }
  // return the number of relations for "unknown"
  return(numrel[0]);
} /* end VH_AdjacencyMapping::get_num_rel() */

/******************************************************************************
 * class VH_AnatomyBoundingBox
 *
 * The spatial extent of a tissue type.  Nodes consist of the full anatomical
 * name and the spatial extrema (xmin, ymin, zmin, xmax, ymax, zmax) of the
 * tissue.  Dimensions are in terms of voxels in the segmented volume.
 ******************************************************************************/

void
VH_AnatomyBoundingBox::append(VH_AnatomyBoundingBox *newNode)
{
  VH_AnatomyBoundingBox *lastNode = blink;
  // append newNode to lastNode
  newNode->flink = lastNode->flink;
  lastNode->flink = newNode;

  newNode->blink = lastNode;
  blink = newNode;
}

void
VH_Anatomy_deleteBBox_node(VH_AnatomyBoundingBox *delNode)
{
  VH_AnatomyBoundingBox *lastNode, *nextNode;
  //  ------------------
  // |   ----    ----   |
  //  ->|    |->|    |--
  //  --|    |<-|    |<-
  // |   ----    ----   |
  //  ------------------
  if(delNode != (VH_AnatomyBoundingBox *)0 &&
     (lastNode = delNode->prev()) != (VH_AnatomyBoundingBox *)0 &&
     (nextNode = delNode->next()) != (VH_AnatomyBoundingBox *)0)
  {
    lastNode->set_next(nextNode);
    nextNode->set_prev(lastNode);
    delete delNode;
  } // (else) empty list
} // end VH_Anatomy_deleteBBox_node()

void
VH_Anatomy_destroyBBox_list(VH_AnatomyBoundingBox *delList)
{
  while(delList != (VH_AnatomyBoundingBox *)0 &&
        delList->next() != delList->prev() &&
        delList->prev() != (VH_AnatomyBoundingBox *)0)
  {
    VH_Anatomy_deleteBBox_node(delList->prev());
  }
  if(delList != (VH_AnatomyBoundingBox *)0)
    delete delList;
} // end VH_Anatomy_destroyBBox_list()

VH_AnatomyBoundingBox *
VH_Anatomy_readBoundingBox_File(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_AnatomyBoundingBox::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return((VH_AnatomyBoundingBox *)0);
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;
  int inLine_cnt = 0;

  cerr << "VH_AnatomyBoundingBox::readFile(" << infilename << ")";
 
  // skip first line
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_AnatomyBoundingBox::readFile(): premature EOF" << endl;
  }
  inLine_cnt++;
  // clear input buffer line
  strcpy(inLine, "");

  // allocate the first Anatomy Name/BoundingBox node
  VH_AnatomyBoundingBox *listRoot = new VH_AnatomyBoundingBox();
  VH_AnatomyBoundingBox *listPtr = listRoot;
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form:
      // Anatomy Name,minX,maxX,minY,maxY,minZ,maxZ,minSlice,maxSlice
      char *linePtr = inLine;
      int charCnt = 0;
      // read free-form, space-separated fields up to first comma
      while(*linePtr++ != ',') charCnt++;
      char *newStr = (char *)malloc((charCnt+1) * sizeof(char));
      if(!newStr)
      {
        cerr << "VH_AnatomyBoundingBox::readFile(): cannot allocate" << endl;
        return listRoot;
      }
      strncpy(newStr, inLine, charCnt);
// Bug correction made April 18, 2005 by R. C. Ward
// Null-terminate newStr
      newStr[charCnt] = '\0';
// End correction on April 18, 2005 by R. C. Ward
      listPtr->set_anatomyname(newStr);
      // debugging...
      cerr << "Debug: anatomyname: " << newStr << endl;

      // linePtr is pointing to ','
// Bug correction on April 10, 2005 by R C Ward
// Comment out the next line to properly read the bounding box coordinates.
//      linePtr++;
// End of correction on April 10, 2005 by R C Ward
//      int minX, maxX, minY, maxY, minZ, maxZ, minSlice, maxSlice;
//      if(sscanf(linePtr, "%d,%d,%d,%d,%d,%d,%d,%d",
//                     &minX, &maxX, &minY, &maxY, &minZ, &maxZ,
//                     &minSlice, &maxSlice
//               ) < 8)
// Modificaiton made April 28, 2005 by R. C. Ward
// Bounding Box coorindates are now floating point
      float minX, maxX, minY, maxY, minZ, maxZ, minSlice, maxSlice;
      if(sscanf(linePtr, "%f,%f,%f,%f,%f,%f,%f,%f",
                     &minX, &maxX, &minY, &maxY, &minZ, &maxZ,
                     &minSlice, &maxSlice
               ) < 8)
      {
        fprintf(stderr,
           "VH_AnatomyBoundingBox::readFile(%s): format error line %d\n",
           infilename, inLine_cnt);
      }
//      if(newStr=="Myocaridal zone 12") {
         cerr << "BB: " <<  minX <<","<< maxX <<","<< minY <<","<<  maxY <<","<<  minZ <<","<<  maxZ <<","<< endl;  
//      }
      listPtr->set_minX(minX); listPtr->set_maxX(maxX);
      listPtr->set_minY(minY); listPtr->set_maxY(maxY);
      listPtr->set_minZ(minZ); listPtr->set_maxZ(maxZ);
      listPtr->set_minSlice(minSlice); listPtr->set_maxSlice(maxSlice);
      // add the new node to the list
      listRoot->append(listPtr);
      // allocate the next Anatomy Name/BoundingBox node
      listPtr = new VH_AnatomyBoundingBox();
    } // end if(strlen(inLine) > 0)
    // clear input buffer line
    strcpy(inLine, "");
    inLine_cnt++;
  } // end while(read_line(inLine, &buffsize, infile) != 0)

  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
  return(listRoot);
} // end VH_AnatomyBoundingBox::readFile(char *infilename)

void
VH_AnatomyBoundingBox::readFile(FILE *infileptr)
{
} // end VH_AnatomyBoundingBox::readFile(FILE *infileptr)

/******************************************************************************
 * Find the boundingBox of a named anatomical entity
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findBoundingBox(VH_AnatomyBoundingBox *list, char *targetname)
{
  VH_AnatomyBoundingBox *listPtr = list;
  cerr << "INSIDE  findBoundingBox targetname: " << targetname << endl;
  while(listPtr->next() != list)
  {
//   cerr << "Anatomy in BB: "<< listPtr->get_anatomyname() << endl;
    if(!strcmp(listPtr->get_anatomyname(), targetname)) break;
    // (else)
    if((listPtr = listPtr->next()) == list) break;
  } // end while(listPtr->flink != (VH_AnatomyBoundingBox *)0)
  return(listPtr);
} // end VH_Anatomy_findBoundingBox()

/******************************************************************************
 * find the largest bounding volume in the segmentation
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findMaxBoundingBox(VH_AnatomyBoundingBox *list)
{
  VH_AnatomyBoundingBox *listPtr = list, *maxBoxPtr = list;
  while(listPtr != (VH_AnatomyBoundingBox *)NULL && listPtr->next() != list)
  {
    if(listPtr->get_minX() <= maxBoxPtr->get_minX() &&
       listPtr->get_minY() <= maxBoxPtr->get_minY() &&
       listPtr->get_minZ() <= maxBoxPtr->get_minZ() &&
       listPtr->get_maxX() >= maxBoxPtr->get_maxX() &&
       listPtr->get_maxY() >= maxBoxPtr->get_maxY() &&
       listPtr->get_maxZ() >= maxBoxPtr->get_maxZ())
    {
      maxBoxPtr = listPtr;
    }
    if((listPtr = listPtr->next()) == list) break;
  } // end while(listPtr->flink != (VH_AnatomyBoundingBox *)0)
  return(maxBoxPtr);
} // end VH_Anatomy_findMaxBoundingBox()
 
/******************************************************************************
 * class VH_injury
 *
 * description: Contains the name of an injured tissue and
 *              iconic geometry to display to indicate the injury extent
 ******************************************************************************/                                                                                

bool
is_injured(char *targetName, vector<VH_injury> &injured_tissue_list)
{
  VH_injury injPtr;
  for(unsigned int i = 0; i < injured_tissue_list.size(); i++)
  {
    injPtr = (VH_injury)injured_tissue_list[i];
    if(string(targetName) == injPtr.anatomyname) return true;
  }
  // (else)
  return false;
} // end is_injured()

bool
is_diagnosis(vector<VH_injury> &injured_tissue_list)
{
  VH_injury injPtr;
  for(unsigned int i = 0; i < injured_tissue_list.size(); i++)
  {
    injPtr = (VH_injury)injured_tissue_list[i];
    if(injPtr.diagnosis != "") return true;
  }
  // (else)
  return false;
} // end is_diagnosis()
                                                                                
string
get_diagnosis(vector<VH_injury> &injured_tissue_list)
{
  VH_injury injPtr;
  for(unsigned int i = 0; i < injured_tissue_list.size(); i++)
  {
    injPtr = (VH_injury)injured_tissue_list[i];
    if(injPtr.diagnosis != "") return injPtr.diagnosis;
  }
  // (else)
  return string("");
} // end get_diagnosis()
 
/******************************************************************************
 * VH_injury::iscomplete()
 *
 * Return whether node is in its completed state or not.
 ******************************************************************************/
bool VH_injury::iscomplete()
{
  bool state = isGeometry == true && timeSet == true && nameSet == true &&
               (((geom_type == "line") &&
                  point0set == true && point1set == true) ||
                ((geom_type == "sphere") && 
                  point0set == true && rad0set == true) ||
                ((geom_type == "cylinder") &&
                  point0set == true && point1set == true &&
                  rad0set == true && rad1set == true) ||
                ((geom_type == "hollow_cylinder") &&
                  point0set == true && point1set == true &&
                  inside_rad0set  == true && inside_rad1set == true &&
                  rad0set == true && rad1set == true)
               );
  return state;
} // end VH_injury::iscomplete()

/******************************************************************************
 * VH_injury::print()
 ******************************************************************************/

void VH_injury::print()
{
  if(isPrimaryInjury) cout << "Primary injury: ";
  else if(isSecondaryInjury) cout << "Secondary injury: ";
  if(isAblate) cout << "Ablated ";
  else if(isStun) cout << "Stunned ";
  cout << "fmaEntity: " << anatomyname;
  cout << " timeStamp: " << timeStamp;
  cout << " time units: " << timeUnit;
  cout << " diagnosis: " << diagnosis;
  cout << " probability: " << probability;
  if(isGeometry)
  {
    cout << " dimEntity: " << geom_type << endl;
    cout << " Axis start point: (";
    cout << axisX0 << ", " << axisY0 << ", " << axisZ0 << ")";
    cout << " Axis end point: (";
    cout << axisX1 << ", " << axisY1 << ", " << axisZ1 << ")";
    cout << " Diameter: " << rad0;
  }
  cout << endl;
} // end VH_injury::print()

/******************************************************************************
 * class VH_HIPvarMap
 *
 * description: Mapping from anatomical names to files containing corresponding
 *              time-varying physiological data channels.
 ******************************************************************************/

VH_HIPvarMap::VH_HIPvarMap()
{
  anatomyname = new (char *)[VH_LM_NUM_NAMES];
  HIPvarFileName = new (char *)[VH_LM_NUM_NAMES];
  num_names = 0;
}
                                                                                
VH_HIPvarMap::~VH_HIPvarMap()
{
  delete [] anatomyname;
  delete [] HIPvarFileName;
}
                                                                                
void
VH_HIPvarMap::readFile(FILE *infileptr)
{
} // end VH_HIPvarMap::readFile(FILE *infileptr)

/******************************************************************************
 * VH_HIPvarMap::readFile()
 ******************************************************************************/
void
VH_HIPvarMap::readFile(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_HIPvarMap::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;
  char *HIPvarFileNamePtr;

  activeFile = string(infilename);
  cerr << "VH_HIPvarMap::readFile(" << infilename << ")";
  
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form: AnatomyName,fileName
      if(!splitAtComma((char *)inLine,
                   &anatomyname[num_names], &HIPvarFileNamePtr)) break;
      HIPvarFileName[num_names] = strdup(HIPvarFileNamePtr);
    } // end if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
    num_names++;
    cerr << ".";
  } // end while(read_line(inLine, &buffsize, infile) != 0)

  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
} // end VH_HIPvarMap::readFile(char *infilename)

char *
VH_HIPvarMap::get_HIPvarFile(char *targetname)
{
  for(int index = 0; index < num_names; index++)
  {
    if(anatomyname[index] != 0 && strcmp(anatomyname[index], targetname) == 0)
      return(HIPvarFileName[index]);
  }

  return((char *)NULL);
} // end VH_VH_HIPvarMap::get_labelindex(char *anatomyname)

VH_hipParam::VH_hipParam(int newColNo, char *newShortName, char *newLongName,
            char *newType, char *newUnit)
{
  col_no = newColNo;
  var_shortName = string(newShortName);
  var_longName  = string(newLongName);
  var_type = string(newType);
  var_unit = string(newUnit);
}

VH_physioMapping::VH_physioMapping(char *newanatomyLump, char *newGE_anatName,
                 char *newEfmaPrefName, int newFMAid)
{
  anatomyLump = string(newanatomyLump);
  GE_anatName = string(newGE_anatName);
  eFMA_prefName = string(newEfmaPrefName);
  fmaID = newFMAid;
}

/******************************************************************************
 * VH_physioMapping::readFile()
 *
 * description: Read a table containing a more detailed listing of HIP
 *              variables corresponding to anatomical structures.
 ******************************************************************************/

void
VH_physioMapping_readFile(char *inFileName,
                          vector<VH_physioMapping *> returnList)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(inFileName, "r")))
  {
    perror("VH_physioMapping_readFile()");
    cerr << "cannot open '" << inFileName << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;

  cerr << "VH_physioMapping_readFile(" << inFileName <<  ")";

  // parse the first line -- header
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_physioMapping_readFile(): premature EOF" << endl;
  }
  int num_cols = countTabs(inLine);

  char anatLump[32], GEanatName[32], eFMAprefName[32], HIP_shortName[32];
  char HIP_longName[32], type[16], unit[16];
  int fmaID, HIPcol;
  if(sscanf(inLine, 
     "%[^\t]\t%[^\t]\t%[^\t]\t%d\t%d\t%[^\t]\t%[^\t]\t%[^\t]\t%[^\t]",
     anatLump, GEanatName, eFMAprefName, &fmaID, &HIPcol, HIP_shortName,
     HIP_longName, type, unit) < 9)
  {
     cerr << "Format error -- line 1 -- header" << endl;
  }
  if(strcmp(anatLump, "Demo anatomical lumps") ||
     strcmp(GEanatName, "GE Anatomy Name") ||
     strcmp(eFMAprefName, "eFMA Preferred Name") ||
     strcmp(HIP_shortName, "HIP var") ||
     strcmp(HIP_longName, "HIP var name") ||
     strcmp(type, "var type") ||
     strcmp(unit, "var unit")
    )
  {
     cerr << "Format error -- line 1 -- header" << endl;
  }
  // clear input buffer line
  strcpy(inLine, "");

  int lineCount = 2;

  VH_physioMapping *newPhysioNode;
  VH_hipParam *newHIPvarNode;
  // parse the remaining lines in the file
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    {
      if(sscanf(inLine,
         "%[^\t]\t%[^\t]\t%[^\t]\t%d\t%d\t%[^\t]\t%[^\t]\t%[^\t]\t%[^\t]",
         anatLump, GEanatName, eFMAprefName, &fmaID, &HIPcol, HIP_shortName,
         HIP_longName, type, unit) < 9)
      {
         cerr << "Format error -- line " << lineCount << endl;
      }
      if(!iswhitespace(anatLump) && !iswhitespace(GEanatName) &&
         !iswhitespace(eFMAprefName))
      { // this is a new physiology mapping
        newPhysioNode =
            new VH_physioMapping(anatLump, GEanatName, eFMAprefName, fmaID);
        returnList.push_back(newPhysioNode);
      }
      else
      { // this is a new HIP parameter for the previous physiology mapping
        newHIPvarNode =
            new VH_hipParam(HIPcol, HIP_shortName, HIP_longName, type, unit);
        newPhysioNode->hip_param.push_back(newHIPvarNode);
      }
    } // end if(strlen(inLine) > 0)
    lineCount++;
  } // end while(read_line(inLine, &buffsize, infile) != 0)
} // end VH_physioMapping_readFile()

