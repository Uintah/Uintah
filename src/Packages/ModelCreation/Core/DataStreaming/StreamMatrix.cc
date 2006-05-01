/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

/*
 * FILE: timedatafile.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 23 MAR 2005
 */

#include <Packages/ModelCreation/Core/DataStreaming/StreamMatrix.h>
 
#ifdef _WIN32

  // Windows includes
	typedef signed __int64 int64;
	typedef unsigned __int64 uint64;

  #include <stdint.h>
  typedef uint32_t u_int32_t;

  #ifdef HAVE_UNISTD_H
    #include <unistd.h>
    #include <fcntl.h>
  #endif

#else
  
  // Unix includes
	typedef signed long long int64;
	typedef unsigned long long uint64;

  #define __USE_LARGEFILE64

  #ifdef HAVE_UNISTD_H
    #include <unistd.h>
    #include <fcntl.h>
  #endif

  #ifndef O_LARGEFILE
    #define O_LARGEFILE 0
  #endif

#endif


#include <stdio.h>
#include <ctype.h>
#include <sys/stat.h>

#include <Core/OS/Dir.h>
#include <sci_defs/config_defs.h>

using namespace SCIRun;

namespace ModelCreation {


StreamMatrix::StreamMatrix(SCIRun::ProgressReporter* pr) :
  pr_(pr), lineskip_(0), byteskip_(0), elemsize_(0), ntype_(0), dimension_(0),
  start_(0), end_(0), step_(0), subdim_(0), useformatting_(false), swapbytes_(false)
{
}


StreamMatrix::StreamMatrix(SCIRun::ProgressReporter* pr, std::string filename) :
  pr_(pr), ncols_(0), nrows_(0), byteskip_(0), lineskip_(0), elemsize_(0), ntype_(0), dimension_(0),
  start_(0), end_(0), step_(0), subdim_(0), useformatting_(false), swapbytes_(false)
{
  open(filename);
}


StreamMatrix::~StreamMatrix()
{
}


int StreamMatrix::get_numrows()
{
  return(sizes_[0]);
}


int StreamMatrix::get_numcols()
{
  return(sizes_[1]);
}


int StreamMatrix::get_rowspacing()
{
  return(spacings_[0]);
}


int StreamMatrix::get_colspacing()
{
  return(spacings_[1]);
}


std::string StreamMatrix::get_rowkind()
{
  return(kinds_[0]);
}


std::string StreamMatrix::get_colkind()
{
  return(kinds_[1]);
}


std::string StreamMatrix::get_rowunit()
{
  return(units_[0]);
}


std::string StreamMatrix::get_colunit()
{
  return(units_[1]);
}


void StreamMatrix::open(std::string filename)
{
  std::ifstream file;

  int ntries = 0;
  bool success = false;

  // Since the header file maybe being updated
  // This way the reader will try several times during 2
  // seconds to get access to it.
  // It is not the best synchronization, unfortunately the
  // nrrd format is ill designed and does not allow for
  // dynamically adding files by using only a template of a
  // file. Hence the use of this hack.

  while ((ntries < 10)&&(success == false))
  {
    try
    {
      file.open(filename.c_str());
      success = true;
      ntries++;
    }
    catch(...)
    {
      SCIRun::Time::waitFor(0.2);
    }
  }
    
  keyvalue_.clear();
  datafilename_ = "";
  datafilenames_.clear();
  units_.clear();
  kinds_.clear();
  spacings_.clear();
  sizes_.clear();
  coloffset_.clear();
  
  content_ = "";
  encoding_ = "";
  endian_ = "";
  type_ = "";
  unit_ = "";

  lineskip_ = 0;
  byteskip_ = 0;
  elemsize_ = 0;
  ntype_ = 0;
  dimension_ = 0;

  start_ = 0;
  end_ = 0;
  step_ = 0;
  subdim_ = 0;

  useformatting_ = false;
  swapbytes_ = false;
  
  // Start reading the file
  std::string line;

  int numlines = 0; 
     
  while(!file.eof())
  { 
    std::getline(file,line);
    numlines++;
    
    if ((line.size() == 0)&&(datafilename_ == ""))
    {
      // Apparently this is a combined header data file;
      // Stop reading header and telll reader how many lines
      // to skip.
      lineskip_ += numlines;
      datafilename_ = filename;
      break;
    }
    
    std::string::size_type colon = line.find(":");
    if (colon < (line.size()-1))
    {
      std::string keyword = remspaces(line.substr(0,colon));
      std::string attribute = line.substr(colon+1);
      
      if (cmp_nocase(keyword,"encoding") == 0) encoding_ = remspaces(attribute);
      if (cmp_nocase(keyword,"type") == 0) type_ = remspaces(attribute);

      if (cmp_nocase(keyword,"units")== 0)
      {
        std::string::size_type startstr = attribute.find('"');
        while (startstr+1 < attribute.size())
        {
          attribute = attribute.substr(startstr+1);
          std::string::size_type endstr = attribute.find('"');
          if (endstr >= attribute.size()) break;
          std::string unitstr = attribute.substr(0,endstr);
          units_.push_back(unitstr);
          attribute = attribute.substr(endstr+1);
          startstr = attribute.find('"');
        }
      }
   
      if (cmp_nocase(keyword,"kinds")== 0)
      {
        std::string::size_type startstr = attribute.find('"');
        while (startstr+1 < attribute.size())
        {
          attribute = attribute.substr(startstr+1);
          std::string::size_type endstr = attribute.find('"');
          if (endstr >= attribute.size()) break;
          std::string kindstr = attribute.substr(0,endstr);
          kinds_.push_back(kindstr);
          attribute = attribute.substr(endstr+1);
          startstr = attribute.find('"');
        }
      }
      
      if (cmp_nocase(keyword,"spacings")== 0)
      {
        std::istringstream iss(attribute+" ");
        iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
        try
        {
          while(1)
          {
            double spacing;
            iss >> spacing;
            spacings_.push_back(spacing);
          }
        }
        catch(...)
        {
        }
      }

      if (cmp_nocase(keyword,"datafile") == 0) 
      {
        std::string::size_type percent = attribute.find('%');
        if (percent < attribute.size())
        {
           std::istringstream iss(attribute); 
           iss.exceptions(std::ifstream::eofbit | std::istream::failbit | std::istream::badbit);
           start_ = 1;
           end_ = 1;
           step_ = 1;
           subdim_ = 0;
           useformatting_ = true;
           
           try
           {
            iss >> datafilename_;
            iss >> start_;
            iss >> end_;
            iss >> step_;
            iss >> subdim_;
           }
           catch(...)
           {
           }           
        }
        else
        {
            datafilename_ = remspaces(attribute);
        }

        if (datafilename_.size() >= 4)
        {
          if (datafilename_.substr(0,4) == "LIST")
          {
            if(datafilename_.size() > 4)
            {
              std::istringstream iss(datafilename_.substr(4));
              subdim_ = 0;
              iss >> subdim_;
            }
            while(!file.eof())
            { 
              getline(file,line);
              numlines++;
              datafilenames_.push_back(remspaces(line));
            }
          }
        }
      }
      if (cmp_nocase(keyword,"content") == 0) content_ = attribute;
      if (cmp_nocase(keyword,"sampleunits") == 0) unit_ = remspaces(attribute);
      if (cmp_nocase(keyword,"endian") == 0) endian_ = remspaces(attribute);
      if (cmp_nocase(keyword,"dimension")  == 0)
      { 
        std::istringstream iss(attribute);
        iss >> dimension_;
      }
      if (cmp_nocase(keyword,"lineskip") == 0)
      { 
        std::istringstream iss(attribute);
        iss >> lineskip_;
      }

      if (cmp_nocase(keyword,"sizes") == 0)
      { 
        std::istringstream iss(attribute+" ");
        iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
        try
        {
          while(1)
          {
            int size;
            iss >> size;
            sizes_.push_back(size);
          }
        }
        catch(...)
        {
        }
      }
    }
  }

  // Check dimensions
  if (sizes_.size() != 2)
  {
    pr_->error("StreamMatrix: This method only supports 2D matrices");
    return (false);      
  }

  // Correct the subdim values
  if ((subdim_ > 2)&&(subdim_ < 0))
  {
    pr_->error("StreamMatrix: Improper subdim value encountered");
    return (false);    
  }
  if (subdim_ == 0) subdim_ = 1;

  // We only support one of Gordon's types
  // If someone wants more it would be easier to fix teem library
  if ((encoding_ != "raw")&&(encoding_!=""))
  {
    pr_->error("StreamMatrix: Encoding must be 'raw'");
    return (false);    
  }
  encoding_ = "raw";

  gettype(type_,ntype_,elemsize_);
  if (ntype_ == 0)
  {
    pr_->error("StreamMatrix: Unknown encoding encounterd");
    return (false);    
  }

  FILE *datafile;
  if (useformatting_)
  {
    char *buffer = scinew char[datafilename_.size()+40];

    std::string  newfilename;
    datafilenames_.clear();
    
    bool foundend = false;
    
    for (int p=start_;((p<=end_)||(end_ == -1))&&(!foundend);p+=step_)
    {
      ::snprintf(&(buffer[0]),datafilename_.size()+39,datafilename_.c_str(),p);
      buffer[datafilename_.size()+39] = 0;
      newfilename = buffer;

      datafile = ::fopen(newfilename.c_str(),"rb");

      if (datafile == 0)
      {
        std::string::size_type slash = filename.size();
        std::string fn = filename;
        slash = fn.rfind("/");
        if (slash  < filename.size())
        {
          newfilename = filename.substr(0,slash+1) + newfilename;
        }
             
        datafile = ::fopen(newfilename.c_str(),"rb");
        if (datafile == 0)
        {
               
          if ((ncols_ == -1)||(end_ == -1))
          {
            foundend = true;
          }
          else
          {
            pr_->error("StreamMatrix: Could not find/open datafile: "+newfilename);   
            return (false); 
          }
        }
        else
        {
          ::fclose(datafile);
          datafilenames_.push_back(newfilename);
        }
      }
      else
      {
        ::fclose(datafile);
        datafilenames_.push_back(newfilename);
      }
    }
    delete[] buffer;
  }

  if (datafilename_ == "")
  {
    pr_->error("StreamMatrix: No data file specified, separate headers are required");    
    return (false);
  }

  if ((endian_ != "big")&&(endian_ != "little")&&(endian_ != ""))
  {
    pr_->error("StreamMatrix: Unknown endian type encountered");
    return (false); 
  }

  swapbytes_ = false;
  short test = 0x00FF;
  char *testptr = reinterpret_cast<char *>(&test);
  if ((testptr[1])&&(endian_ == "little")) swapbytes_ = true;
  if ((testptr[0])&&(endian_ == "big")) swapbytes_ = true;

  if ((nrows_ < 1)||(ncols_ < -1))
  {
    pr_->error("StreamMatrix: Improper NRRD dimensions: number of columns/rows is smaller then one");  
    return (false);
  }

  if (datafilenames_.size() == 0)
  {
    datafile = ::fopen(datafilename_.c_str(),"rb");
    if (datafile == 0)
    {
      std::string::size_type slash = filename.size();
      std::string fn = filename;
      slash = fn.rfind("/");
      if (slash  < filename.size())
      {
        datafilename_ = filename.substr(0,slash+1) + datafilename_;
      }
      
      datafile = fopen(datafilename_.c_str(),"rb");
      if (datafile == 0)
      {
        pr_->("StreamMatrix: Could not find/open datafile: "+datafilename_);
        return (false);    
      }
    }
    ::fclose(datafile);    
  }  

  int ncolsr =0;
  if (datafilenames_.size() == 0)
  {
    struct stat buf;
    if (LSTAT(datafilename_.c_str(),&buf) < 0)
    {
      pr_->error("StreamMatrix: Could not determine size of datafile");
      return (false);          
    }
    ncolsr = static_cast<int>((buf.st_size)/static_cast<off_t>(nrows_*elemsize_));
  }
  else
  {
    ncolsr = 0;
    std::list<std::string>::iterator p = datafilenames_.begin();
    int q = 0;
    coloffset_.resize(datafilenames_.size()+1);
    for (;p != datafilenames_.end();p++)
    {
      struct stat buf;
      if (LSTAT((*p).c_str(),&buf) < 0)
      {
        pr_->error("StreamMatrix: Could not determine size of datafile");
        return (false);          
      }
      coloffset_[q++] = ncolsr;
      ncolsr += static_cast<int>((buf.st_size)/static_cast<off_t>(nrows_*elemsize_));
    }
    coloffset_[q] = ncolsr;
  }

  if (ncols_ == -1) ncols_ = ncolsr;
  if (ncols_ > ncolsr) ncols_ = ncolsr;

  if (datafilenames_.size() > 0)
  {
    if (byteskip_ != 0)  
    {
      pr_->error("StreamMatrix: Byteskip and data spread out over multiple files is not supported yet");
      return (false);
    }
    if (lineskip_ != 0)
    {
      pr_->error("StreamMatrix: Lineskip and data spread out over multiple files is not supported yet");
      return (false);
    }
  }
}



void StreamMatrix::getcolmatrix(SCIRun::MatrixHandle& mh,SCIRun::MatrixHandle indices)
{
  FILE* datafile;
  int   datafile_uni;

  // Check whether we have an index ort multiple indices
  if (indices.get_rep() == 0)
  {
    pr_->error("StreamMatrix: No indices given for colums");
    return (false);
  }
  
  // Copy the indices into a more accessible vector
  // Get the number of elements (sparse or dense it does not matter)
  std::vector<int> idx(indices->get_data_size());
  double *idataptr =  indices->get_data_pointer();
  
  // Copy and cast the indices to intergers
  for (int p=0; p<idx.size(); p++) idx[p] = static_cast<int>(idataptr[p]);
  
  // Create the output matrix
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(idx.size(),nrows_,);
  if (mat == 0)
  {
    pr_->error("StreamMatrix: Could not allocate matrix");
    return (false);
  }   
  
  // Store the matrix in a handle, so when we fail the memory is freed automatically
  SCIRun::MatrixHandle handle = dynamic_cast<SCIRun::Matrix *>(mat);

  // Get the data pointer where we can store the data
  char* buffer = reinterpret_cast<char *>(mat->get_data_pointer());


  // Loop over all the indices
  int k = 0;
  for (k=0; k< idx.size(); k++)
  {
    int coloffset = idx[k];
     
    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (p=0;p<coloffset_.size();p++) { if((coloffset >= coloffset_[p] )&&(coloffset <coloffset_[p+1])) break;}
      if (p == coloffset_.size()) 
      {
        pr_->error("StreamMatrix: Column index out of range");
        return (false);
      }
      
      // Convert offset to local offset
      coffset -= coloffset_[p];
      fn = datafilenames_[p];
    }
    else
    {
      std::string fn = datafilename_;    
    }
    
    #ifndef HAVE_UNISTD_H 
    
      // Use normal C functions (for files upto 2Gb)
      datafile = ::fopen(fn.c_str(),"rb");
      if (datafile == 0)
      {
        pr_->error("StreamMatrix: Could not find/open datafile");
        return (false);
      }
      
      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::fread(&cbuffer,1,1,datafile) != 1)
            {
              ::fclose(datafile);
              pr_->error("StreamMatrix: Could not read header of datafile"); 
              return (false);
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      if (byteskip_ >= 0)
      {
        if (::fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          ::fclose(datafile);
          pr_->error("StreamMatrix: Could not read datafile"); 
          return (false);
        }
      }
      else
      {
        if (::fseek(datafile,sizes_[0]*sizes_[1]*elemsize_,SEEK_END)!=0)
        {
          ::fclose(datafile);
          pr_->error("StreamMatrix: Could not read datafile"); 
          return (false);
        }
      }
        
      if (::fseek(datafile,elemsize_*sizes_[0]*coffset,SEEK_CUR)!=0)
      {
        ::fclose(datafile);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }    

      if (sizes_[0] != ::fread((void *)buffer,elemsize_,sizes_[0],datafile))
      {
        ::fclose(datafile);
        pr_->error("StreamMatrix: Error reading datafile");
        return (false);
      }
       
      ::fclose(datafile);
      }
    #else
      // Use Unix system for files larger than 2Gb
      
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0)
      {
        pr_->error("StreamMatrix: Could not find/open datafile");
        return (false);
      }
      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              ::close(datafile_uni);
              pr_->error("StreamMatrix: Could not read header of datafile");
              return (false); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
    
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          pr_->error("StreamMatrix: Could not read datafile");
          return (false);
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(sizes_[1])*static_cast<off_t>(sizes_[0])*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          pr_->error("StreamMatrix: Could not read datafile");
          return (false);
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(sizes_[0])*static_cast<off_t>(coffset),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }    

      size_t ret = ::read(datafile_uni,reinterpret_cast<void*>(buffer),static_cast<size_t>(elemsize_*sizes_[0]));
      if (static_cast<size_t>(elemsize_*sizes_[0]) != ret)
      {
        ::close(datafile_uni);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }
     
      ::close(datafile_uni);
    #endif
  
    // Move pointer to next column to be read
    buffer += (sizes_[0]*elemsize_);
  }
  
  // Get the starting pointer again.
  buffer = reinterpret_cast<char *>(mat->get_data_pointer());
  
  // Do the byte swapping and then  unpack the data into doubles
  if (swapbytes_) doswapbytes(reinterpret_cast<void*>(buffer),elemsize_,idx.size()*sizes_[0]);
  if (ntype_ == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]);}

  // We need to trnaspose the matrix (This way we can limit the amount of read operations)
  mh = dynamic_cast<Matrix *>(mat->transpose());

  return (true);
}


void StreamMatrix::getcolmatrix_weights(SCIRun::MatrixHandle& mh,SCIRun::MatrixHandle weights)
{
  FILE* datafile;
  int   datafile_uni;

  // Check whether we have an index ort multiple indices
  if (weights.get_rep() == 0)
  {
    pr_->error("StreamMatrix: No indices given for colums");
    return (false);
  }
  
  SCIRun::SparseRowMatrix *spr_weights = weights->sparse();
  SCIRun::MatrixHandle sprhandle = dynamic_cast<Matrix *>(spr_weights); 
  
  // Copy the indices into a more accessible vector
  // Get the number of elements (sparse or dense it does not matter)
  int    *rows = spr_weights->rows;
  int    *cols  = spr_weights->columns;
  double *vals = spr_weights->a;
  int    nrows = spr_weights->nrows_;
  int    ncols = spr_weights->ncols_;
  int    nnz  = spr_weights->get_data_size(); 
      
  // Create the output matrix
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(nnz,nrows_,);
  if (mat == 0)
  {
    pr_->error("StreamMatrix: Could not allocate matrix");
    return (false);
  }   
  
  // Store the matrix in a handle, so when we fail the memory is freed automatically
  SCIRun::MatrixHandle handle = dynamic_cast<SCIRun::Matrix *>(mat);

  // Get the data pointer where we can store the data
  double* dbuffer = mat->get_data_pointer();
  char* buffer = reinterpret_cast<char *>(mat->get_data_pointer());


  // Loop over all the indices
  
  for (int k=0; k<nnz; k++)
  {
    int coloffset = cols[k];     
    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (p=0;p<coloffset_.size();p++) { if((coloffset >= coloffset_[p] )&&(coloffset <coloffset_[p+1])) break;}
      if (p == coloffset_.size()) 
      {
        pr_->error("StreamMatrix: Column index out of range");
        return (false);
      }
      
      // Convert offset to local offset
      coffset -= coloffset_[p];
      fn = datafilenames_[p];
    }
    else
    {
      std::string fn = datafilename_;    
    }
    
    #ifndef HAVE_UNISTD_H 
    
      // Use normal C functions (for files upto 2Gb)
      datafile = ::fopen(fn.c_str(),"rb");
      if (datafile == 0)
      {
        pr_->error("StreamMatrix: Could not find/open datafile");
        return (false);
      }
      
      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::fread(&cbuffer,1,1,datafile) != 1)
            {
              ::fclose(datafile);
              pr_->error("StreamMatrix: Could not read header of datafile"); 
              return (false);
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      if (byteskip_ >= 0)
      {
        if (::fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          ::fclose(datafile);
          pr_->error("StreamMatrix: Could not read datafile"); 
          return (false);
        }
      }
      else
      {
        if (::fseek(datafile,sizes_[0]*sizes_[1]*elemsize_,SEEK_END)!=0)
        {
          ::fclose(datafile);
          pr_->error("StreamMatrix: Could not read datafile"); 
          return (false);
        }
      }
        
      if (::fseek(datafile,elemsize_*sizes_[0]*coffset,SEEK_CUR)!=0)
      {
        ::fclose(datafile);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }    

      if (sizes_[0] != ::fread((void *)buffer,elemsize_,sizes_[0],datafile))
      {
        ::fclose(datafile);
        pr_->error("StreamMatrix: Error reading datafile");
        return (false);
      }
       
      ::fclose(datafile);
      }
    #else
      // Use Unix system for files larger than 2Gb
      
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0)
      {
        pr_->error("StreamMatrix: Could not find/open datafile");
        return (false);
      }
      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              ::close(datafile_uni);
              pr_->error("StreamMatrix: Could not read header of datafile");
              return (false); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
    
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          pr_->error("StreamMatrix: Could not read datafile");
          return (false);
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(sizes_[1])*static_cast<off_t>(sizes_[0])*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          pr_->error("StreamMatrix: Could not read datafile");
          return (false);
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(sizes_[0])*static_cast<off_t>(coffset),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }    

      size_t ret = ::read(datafile_uni,reinterpret_cast<void*>(buffer),static_cast<size_t>(elemsize_*sizes_[0]));
      if (static_cast<size_t>(elemsize_*sizes_[0]) != ret)
      {
        ::close(datafile_uni);
        pr_->error("StreamMatrix: Improper data file, check number of columns and rows in header file");
        return (false);
      }
     
      ::close(datafile_uni);
    #endif
  
    // Move pointer to next column to be read
    buffer += (sizes_[0]*elemsize_);
  }
  
  // Get the starting pointer again.
  buffer = reinterpret_cast<char *>(mat->get_data_pointer());
  
  // Do the byte swapping and then  unpack the data into doubles
  if (swapbytes_) doswapbytes(reinterpret_cast<void*>(buffer),elemsize_,idx.size()*sizes_[0]);
  if (ntype_ == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(idx.size()*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]);}

  // Fit everything together again
  
  mh = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(sizes_[0],spr_weights->nrows_));

  if (mh.get_rep() == 0)
  {
    pr_->error("StreamMatrix: Could not allocate output matrix");
    return (false);  
  }

  int s = 0;
  double* dataptr = mh->get_data_pointer();
  for (int p=0; p<nrows;p++)
  {
    for (int q = 0; q < sizes_[0], q++) dataptr[q*nrows+p] += 0.0;
    for (int r = rows[p]; r<rows[p+1]; r++)
    {
      for (int q = 0; q < sizes_[0], q++) dataptr[q*nrows+p] += vals[r]*dbuffer[s++];
    }
  }

 return (true);
}













void StreamMatrix::getrowmatrix(SCIRun::MatrixHandle& mh,SCIRun::MatrixHandle indices)
{
  FILE*                   datafile;
  int                     datafile_uni;

  if (indices.get_rep() == 0)
  {
    pr_->error("StreamMatrix: No indices given for colums");
    return (false);
  }
  
  SCIRun::DenseMatrix* imat = indices->dense();
  
  if (imat == 0)
  {
    pr_->error("StreamMatrix: Could not read indices matrix");
    return (false);  
  }

  std::vector<int> idx(imat->get_data_size());
  double *idataptr =  imat->get_data_pointer();
  for (int p=0; p<idx.size(); p++) idx[p] = static_cast<int>(idataptr[p]);
  std::sort(idx.begin(),idx.end());

  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(ncols_,idx.size());
  SCIRun::MatrixHandle handle = dynamic_cast<SCIRun::Matrix *>(mat);
  
  if (mat == 0)
  {
    pr_->error("Could not allocate matrix");
    return (false);
  }
  
  char* buffer = reinterpret_cast<char *>(mat->get_data_pointer());  

  std::vector<int>::iterator it = idx.begin();
  std::vector<int>::iterator it_end = idx.end();

  int cstart= 0;
  int cend = sizes_[1];

  int k = 0;

  while (1)
  {
    std::string fn = datafilename_;

    if (datafilenames_.size() > 0)
    {
      if (datafilenames_.size() == k) break;
      cstart = coloffset_[k];
      cend = coloffset_[k+1];
      fn = datafilenames_[k];
      k++;      
    }
    else
    {
      if (k > 0) break;
      k++;
    }

  #ifndef HAVE_UNISTD_H
    datafile = ::fopen(fn.c_str(),"rb");
    if (datafile == 0) 
    {
      pr_->error("StreamMatrix: Could not find/open datafile");
      return (false);
    }
    
    if (lineskip_ > 0)
    {
       char cbuffer;
       int ln = lineskip_;
       while (ln)
       {
          if(::fread(&cbuffer,1,1,datafile) != 1)
          {
            ::fclose(datafile);
            pr_->error("StreamMatrix: Could not read header of datafile"); 
            return (false);
          }
          if (cbuffer == '\n') ln--;
       }
    }
    

    if (byteskip_ >= 0)
    {
      if (::fseek(datafile,byteskip_,SEEK_CUR)!=0)
      {
        ::fclose(datafile);
        pr_->error("StreamMatrix: Could not read datafile");
        return (false);
      }
    }
    else
    {
      if (fseek(datafile,ncols_*nrows_*elemsize_,SEEK_END)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Could not read datafile");
      }
    }

    
      if (fseek(datafile,elemsize_*rowstart,SEEK_CUR)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (numrows != fread(buffer+(elemsize_*j*numrows),elemsize_,numrows,datafile))
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (fseek(datafile,elemsize_*(nrows_-numrows),SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      fclose(datafile);
    #else
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0) 
      {
        throw TimeDataFileException("Could not find/open datafile");
      }

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              ::close(datafile_uni);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      // Accoring to Gordon's definition
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(ncols_)*static_cast<off_t>(nrows_)*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(rowstart),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (static_cast<size_t>(numrows*elemsize_) != ::read(datafile_uni,buffer+(elemsize_*j*numrows),static_cast<size_t>(elemsize_*numrows)))
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(nrows_-numrows),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      ::close(datafile_uni);  
    #endif
  }
    
  if (swapbytes_) doswapbytes(buffer,elemsize_,numrows*ncols_);

  if (ntype_ == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols_*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  

  mh = dynamic_cast<Matrix *>(mat->transpose());
}

}

