/******************************************************************************
 * File: labelmaps.cc
 *
 * Description: C source code for classes to provide an API for Visible Human
 *		segmentation information:  The Master Anatomy Label Map
 *		Spatial Adjacency relations for the anatomical structures
 *		and Bounding Boxes for each anatmocal entity.
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 *         DARPA Virtual Soldier Project <http://www.virtualsoldier.net>
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <iostream>
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
      labelindex[num_names] = atoi(indexStr);
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
VH_MasterAnatomy::get_anatomyname(int labelindex)
{
  return(anatomyname[labelindex]);
} // end VH_MasterAnatomy::get_anatomyname(int labelindex)

int
VH_MasterAnatomy::get_labelindex(char *targetname)
{
  for(int index = 0; index < num_names; index++)
  {
    if(anatomyname[index] != 0 && strcmp(anatomyname[index], targetname) == 0)
      return(index);
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
      // cerr << "line[" << num_names << "], " << strlen(inLine) << " chars, ";
      // cerr << num_adj << " relations" << endl;
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
      while((indexStr = strtok(NULL, ",")) != NULL)
      {
        *intPtr++ = atoi(indexStr);
      }
      num_names++;
      cerr << ".";
    } if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
  } // end while(read_line(inLine, &buffsize, infile) != 0)
  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
} // end VH_AdjacencyMapping::readFile(char *infilename)

void
VH_AdjacencyMapping::readFile(FILE *infileptr)
{
} // end VH_AdjacencyMapping::readFile(FILE *infileptr)

int *
VH_AdjacencyMapping::adjacent_to(int index)
{
  if(index < num_names)
     return rellist[index];
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
     cerr << "VH_AdjacencyMapping:::get_num_rel(): index " << index;
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
      if(!(listPtr->anatomyname = (char *)malloc((charCnt+1) * sizeof(char))))
      {
        cerr << "VH_AnatomyBoundingBox::readFile(): cannot allocate" << endl;
      }
      strncpy(listPtr->anatomyname, inLine, charCnt);
      // debugging...
      // fprintf(stderr, "anatomyname[%d] = '%s'\n",
      //                 inLine_cnt, listPtr->anatomyname);
      fprintf(stderr, ".");

      // linePtr is pointing to ','
      linePtr++;
      if(sscanf(linePtr, "%d,%d,%d,%d,%d,%d,%d,%d",
                     &(listPtr->minX), &(listPtr->maxX),
                     &(listPtr->minY), &(listPtr->maxY),
                     &(listPtr->minZ), &(listPtr->maxZ),
                     &(listPtr->minSlice), &(listPtr->maxSlice)
               ) < 8)
      {
        fprintf(stderr,
           "VH_AnatomyBoundingBox::readFile(%s): format error line %d\n",
           infilename, inLine_cnt);
      }
      // allocate the next Anatomy Name/BoundingBox node
      listPtr->flink = new VH_AnatomyBoundingBox();
      listPtr = listPtr->flink;
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
 ******************************************************************************/VH_AnatomyBoundingBox *
VH_Anatomy_findBoundingBox(VH_AnatomyBoundingBox *list, char *targetname)
{
  VH_AnatomyBoundingBox *listPtr = list;
  while(listPtr->flink != (VH_AnatomyBoundingBox *)0)
  {
    if(!strcmp(listPtr->anatomyname, targetname)) break;
    // (else)
    if(!(listPtr = listPtr->flink)) break;
  } // end while(listPtr->flink != (VH_AnatomyBoundingBox *)0)
  return(listPtr);
} // end VH_Anatomy_findBoundingBox()

