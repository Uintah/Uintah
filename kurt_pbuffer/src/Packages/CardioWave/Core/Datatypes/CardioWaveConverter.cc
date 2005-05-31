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
 * FILE: cardiowaveconverter.h
 * AUTH: Jeroen G Stinstra
 * DATE: 1 FEB 2005
 */
 
#include <Packages/CardioWave/Core/Datatypes/CardioWaveConverter.h>

using namespace std;
using namespace SCIRun;

namespace CardioWave {

CardioWaveConverter::CardioWaveConverter()
{
}


// Function for doing byteswapping when loading a file created on a different platform 
void CardioWaveConverter::swapbytes(void *vbuffer,int elsize,int size)
{
   char temp;
   char *buffer = static_cast<char *>(vbuffer);

   size *= elsize;

   switch(elsize)
   {
      case 0:
      case 1:
         // Do nothing. Element size is 1 byte, so there is nothing to swap
         break;
      case 2:  
        // Do a 2 bytes element byte swap. 
        for(int p=0;p<size;p+=2)
          { temp = buffer[p]; buffer[p] = buffer[p+1]; buffer[p+1] = temp; }
        break;
      case 4:
        // Do a 4 bytes element byte swap.
        for(int p=0;p<size;p+=4)
          { temp = buffer[p]; buffer[p] = buffer[p+3]; buffer[p+3] = temp; 
          temp = buffer[p+1]; buffer[p+1] = buffer[p+2]; buffer[p+2] = temp; }
        break;
      case 8:
        // Do a 8 bytes element byte swap.
        for(int p=0;p<size;p+=8)
          { temp = buffer[p]; buffer[p] = buffer[p+7]; buffer[p+7] = temp; 
          temp = buffer[p+1]; buffer[p+1] = buffer[p+6]; buffer[p+6] = temp; 
          temp = buffer[p+2]; buffer[p+2] = buffer[p+5]; buffer[p+5] = temp; 
          temp = buffer[p+3]; buffer[p+3] = buffer[p+4]; buffer[p+4] = temp; }
            break;
   }  
}

bool CardioWaveConverter::byteswapmachine()
{
    short test = 0x00FF;
    char *ptr;
    // make the compiler happy
    ptr = static_cast<char *>(static_cast<void *>(&test));
    if (ptr[0]) return(true);
    return(false);
}

bool CardioWaveConverter::cwFileTOsciMatrix(std::string filename,MatrixHandle& mh,ProgressReporter *pr)
{
  FILE* fid;
  std::string header(' ',128);
  
  // Make sure that we set a default value
  mh = 0;
  
  // Test whether we have all the proper types
  
  if (sizeof(int64)!=8)
  {
    posterror(pr,"Size of int64 is not 8 bytes");
    return(false);
  }
  
  if (sizeof(int)!=4)
  {
    posterror(pr,"Size of integer is not 4 bytes");
    return(false);
  }

  if (sizeof(double)!=8)
  {
    posterror(pr,"Size of double is not 8 bytes");
    return(false);
  }

  if (sizeof(float)!=4)
  {
    posterror(pr,"Size of float is not 4 bytes");
    return(false);
  }

  if (sizeof(short)!=2)
  {
    posterror(pr,"Size of short is not 2 bytes");
    return(false);
  }  
  
  // Now we know we have a compatible system
  // Try toopen the file
  
  if (!(fid = fopen(filename.c_str(),"rb")))
  {
    // Could not open file some how
    posterror(pr,std::string("Could not open file: ")+filename);
    return(false);
  }
  
  // Each header of CW file is maximal 128 bytes
  if ((fread(static_cast<void*>(&header[0]),1,128,fid)!=128))
  {
    posterror(pr,std::string("Could not open header of CardioWave file: ") + filename);
    fclose(fid);
    return(false);
  }
  
  // We got the header now analyze its contents:
  
  void* buffer;
  
  char filetype = header[0];  // Letter indicating the type of file e.g I B M or S
  char swapbyte = header[1];  // B or L indicating Big endian (B) or Small endian (L)
  
  bool doswapbytes = false;
  if ((swapbyte == 'B')&&(byteswapmachine() == true)) doswapbytes = true;    // We are on a intel architecture 
  if ((swapbyte == 'L')&&(byteswapmachine() == false)) doswapbytes = true;   // We are not on a byteswapping machine
  if ((swapbyte != 'B')&&(swapbyte != 'L'))
  {
    posterror(pr,"Could not read byte encoding order from file: this is not a valid CardioWave file");
    fclose(fid);
    return(false);
  }
  
  // Go over the different types of files
  
  switch (filetype)
  {
    case 'M':
      { // Dense Matrix file
      // Since this one is hardly used we do not have support for it yet
      posterror(pr,"This file type is not yet supported");
      fclose(fid);
      return(false);
      
      }
      break;
    case 'S':
      {
        char eltype = header[2];
        int elsize = 0;
        switch (eltype)
        {
          case '4':
            elsize = 4;
            break;
          case '8':
            elsize = 8;
            break;
          default:
            posterror(pr,"Element size is not supported, data needs to be or 4 bytes (float) or 8 bytes (double)");
            fclose(fid);
            return(false);
        }
        char idxtype = header[3];
        int idxsize = 0;
        switch (idxtype)
        {
          case '4':
            idxsize = 4;
            break;
          default:
            posterror(pr,"Index size is not supported, data needs to be or 4 bytes (int32)");
            fclose(fid);
            return(false);
        }
        
        std::istringstream iss(header.substr(5,122));
        int  nrows; iss >> nrows;     
        int  ncols; iss >> ncols;
        int  bwp;   iss >> bwp;
        int  bwn;   iss >> bwn;
        int  dummy; iss >> dummy;
        int  nz;    iss >> nz;
        
        int *jcoef = scinew int[nz*nrows];
        double *coef = scinew double[nz*nrows];
           
        if ((jcoef==0)||(coef==0))
        {
          posterror(pr,"Could not alloc enough buffer space to convert matrix into scirun format");
          if (jcoef) delete[] jcoef;
          if (coef) delete[] coef;
          fclose(fid);
          return(false);
        }
   
        buffer = reinterpret_cast<void *>(jcoef);
        if ((fread(buffer,1,idxsize*nz*nrows,fid)!=(idxsize*nz*nrows)))
        {
          fclose(fid);
          delete[] jcoef;
          delete[] coef;
          posterror(pr,"Could not read data for column matrix");
          return(false);        
        }    

        if (doswapbytes) swapbytes(buffer,idxsize,nz*nrows);
      
        buffer = reinterpret_cast<void *>(coef);
        if ((fread(buffer,1,elsize*nz*nrows,fid)!=(elsize*nz*nrows)))
        {
          fclose(fid);
          delete[] jcoef;
          delete[] coef;
          posterror(pr,"Could not read data for column matrix");
          return(false);        
        }    

        if (doswapbytes) swapbytes(buffer,elsize,nz*nrows);
             
        fclose(fid);

        if (eltype == '4')
        { // We are reusing memory, hence do it in a reverse order
          double *dbuffer = reinterpret_cast<double *>(buffer);
          float  *fbuffer = reinterpret_cast<float *>(buffer);
          for (int p=((nz*nrows)-1);p>-1;p--)
          { // The compiler normally unroles these kind of loops
            // for better performance 
            dbuffer[p] = static_cast<double>(fbuffer[p]);
          }
        }      
      
        // We now have the data, now get it into a sparse row matrix.
        
        // first count how many spaces we actual need
        int nnz = 0;
        for (int p=0;p<(nz*nrows);p++) if (coef[p] != 0.0) nnz++;
        
      
        int *rr = scinew int[nrows+1];
        int *cc = scinew int[nnz];
        double *d  = scinew double[nnz];
        int *t  = scinew int[nrows];
        
        if ((rr==0)||(cc==0)||(d==0)||(t==0))
        {
          if (rr) delete[] rr;
          if (cc) delete[] cc;
          if (d)  delete[] d;
          if (t)  delete[] t;
          delete[] jcoef;
          delete[] coef;         
          posterror(pr,"Could not allocate enough memory for sparse matrix");
          return(false);
        }
      
        // Clear temp space
        for (int p=0;p<nrows;p++) t[p] = 0;
      
        // count the number of entries per column
        int s=0;
        for (int q=0;q<nz;q++)
          for (int r=0 ;r<nrows;r++,s++)
          {
            if (coef[s] != 0.0) t[r]++;
          }
      
        // fill out the column space thing
        int k = 0;
        rr[0] = 0;
        for (int p=0;p<nrows;p++) { k += t[p]; rr[p+1] = k; } 

        // Clear temp space again
        for (int p=0;p<nrows;p++) t[p] = 0;

        k = 0;
        int m = 0;
        s = 0;
        for (int q=0;q<nz;q++)
          for (int r=0 ;r<nrows;r++,s++)
          {
            if (coef[s] != 0.0)
            {
              k = rr[r]+t[r];
              cc[k] = jcoef[s]-1;
              d[k] = coef[s];
              t[r]++;
            }
          }
        
        double td;
        int tcc;
        
        for (int q=0;q<nrows;q++)
        {
          for (int r=rr[q];r<rr[q+1];r++)
          {
            k = r;
            for (int v=r+1; v<rr[q+1];v++) if (cc[v] < cc[k]) {k = v;} 
            td = d[r];
            tcc = cc[r];
            d[r] = d[k];
            cc[r] = cc[k];
            d[k] = td;
            cc[k] = tcc;
          }
        }
         
        SparseRowMatrix *srm = scinew SparseRowMatrix(nrows,ncols,rr,cc,nnz,d);
  
        if (srm == 0)
        {
          delete[] jcoef;
          delete[] coef;
          delete[] t;
          delete[] rr;
          delete[] cc;
          delete[] d;
          posterror(pr,"Could not allocate a new SparseRowMatrix");
          return(false);
        }

        mh = dynamic_cast<Matrix *>(srm);
        delete[] jcoef;
        delete[] coef;
        delete[] t;
        return(true);
      
      }
    case 'I':
      { // File with integer in the data
        char eltype = header[2];
        int elsize = 0;
        switch (eltype)
        {
          case '4':
            elsize = 4;
            break;
          case '8':
            elsize = 8;
            break;
          default:
            posterror(pr,"Element size is not supported");
            fclose(fid);
            return(false);
        }
        
        std::istringstream iss(header.substr(4,123));
        int  size; iss >> size;
        
        ColumnMatrix* colmat = scinew ColumnMatrix(size);
        if (colmat == 0) 
        { 
          fclose(fid);
          posterror(pr,"Could not create Column Matrix");
          return(false);
        }

        buffer = reinterpret_cast<void *>(colmat->get_data());
        if ((fread(buffer,1,elsize*size,fid)!=(elsize*size)))
        {
          fclose(fid);
          delete[] colmat;
          posterror(pr,"Could not read data for column matrix");
          return(false);        
        } 

        fclose(fid);

        if (doswapbytes) swapbytes(buffer,elsize,size);
        
        if (eltype == '4')
        { // We are reusing memory, hence do it in a reverse order
          double *dbuffer = reinterpret_cast<double *>(buffer);
          int    *ibuffer = reinterpret_cast<int *>(buffer);
          for (int p=(size-1);p>-1;p--)
          { // The compiler normally unroles these kind of loops
            // for better performance 
            dbuffer[p] = static_cast<double>(ibuffer[p]);
          }
        }
        if (eltype == '8')
        {
          double *dbuffer = reinterpret_cast<double *>(buffer);
          int64 *ibuffer = reinterpret_cast<int64 *>(buffer);
          for (int p=0;p<size;p++)
          {
            dbuffer[p] = static_cast<double>(ibuffer[p]);
          }
        }
        
        mh = dynamic_cast<Matrix *>(colmat);
        return(true);
        // Conversion and loading was a success
      }
      break;
    case 'B':
      { // File with integer in the data
        char eltype = header[2];
        int elsize = 0;
        switch (eltype)
        {
          case '1':
            elsize = 1;
            break;
          case '2':
            elsize = 2;
            break;
          default:
            posterror(pr,"Element size is not supported");
            fclose(fid);
            return(false);
        }
        
        std::istringstream iss(header.substr(4,123));
        int  size; iss >> size;
        
        ColumnMatrix* colmat = scinew ColumnMatrix(size);
        if (colmat == 0) 
        { 
          fclose(fid);
          posterror(pr,"Could not create Column Matrix");
          return(false);
        }

        buffer = reinterpret_cast<void *>(colmat->get_data());
        if ((fread(buffer,1,elsize*size,fid)!=(elsize*size)))
        {
          fclose(fid);
          delete[] colmat;
          posterror(pr,"Could not read data for column matrix");
          return(false);        
        } 

        fclose(fid);

        if (doswapbytes) swapbytes(buffer,elsize,size);
        
        if (eltype == '1')
        { // We are reusing memory, hence do it in a reverse order
          double *dbuffer = reinterpret_cast<double *>(buffer);
          char   *bbuffer = reinterpret_cast<char *>(buffer);
          for (int p=(size-1);p>-1;p--)
          { // The compiler normally unroles these kind of loops
            // for better performance 
            dbuffer[p] = static_cast<double>(bbuffer[p]);
          }
        }
        if (eltype == '2')
        { // We are reusing memory, hence do it in a reverse order
          double *dbuffer = reinterpret_cast<double *>(buffer);
          short *sbuffer = reinterpret_cast<short *>(buffer);
          for (int p=(size-1);p>-1;p--)          {
            dbuffer[p] = static_cast<double>(sbuffer[p]);
          }
        }
        
        mh = dynamic_cast<Matrix *>(colmat);
        
        return(true);
        // Conversion and loading was a success
      }
      break;      
    case 'V':
      { // File with integer in the data
        char eltype = header[2];
        int elsize = 0;
        switch (eltype)
        {
          case '4':
            elsize = 4;
            break;
          case '8':
            elsize = 8;
            break;
          default:
            posterror(pr,"Element size is not supported");
            fclose(fid);
            return(false);
        }
        
        std::istringstream iss(header.substr(4,123));
        int  size; iss >> size;
        
        ColumnMatrix* colmat = scinew ColumnMatrix(size);
        if (colmat == 0) 
        { 
          fclose(fid);
          posterror(pr,"Could not create Column Matrix");
          return(false);
        }

        buffer = reinterpret_cast<void *>(colmat->get_data());
        if ((fread(buffer,1,elsize*size,fid)!=(elsize*size)))
        {
          fclose(fid);
          delete[] colmat;
          posterror(pr,"Could not read data for column matrix");
          return(false);        
        } 

        fclose(fid);

        if (doswapbytes) swapbytes(buffer,elsize,size);
        
        if (eltype == '4')
        { // We are reusing memory, hence do it in a reverse order
          double *dbuffer = reinterpret_cast<double *>(buffer);
          float  *fbuffer = reinterpret_cast<float *>(buffer);
          for (int p=(size-1);p>-1;p--)
          { // The compiler normally unroles these kind of loops
            // for better performance 
            dbuffer[p] = static_cast<double>(fbuffer[p]);
          }
        }
        
        mh = dynamic_cast<Matrix *>(colmat);
        return(true);
        // Conversion and loading was a success
      }
      break;      
    default:
      {
        posterror(pr,"Could not read matrix/vector, type identifier unknown");
        fclose(fid);
        return(false);
      }
  }
  
  // never reached
  return(false); 
}

bool CardioWaveConverter::sciMatrixTOcwFile(MatrixHandle mh,std::string filename,ProgressReporter *pr,std::string filetype)
{
  if (filetype == "")
  {
    if (mh->is_sparse()) filetype = "spr"; else filetype = "vec";
  }
  
  if ((filetype != "spr")&&(filetype != "vec")&&(filetype != "ivec")&&(filetype != "bvec"))
  {
    posterror(pr,"Specified file type is not supported");
    return(false);
  }
  
  if (filetype == "bvec")
  {
 
    if (mh->is_sparse())
    {
      posterror(pr,"Data is a sparse matrix and not a column vector as needed to export data");
      return(false);    
    }
         
    double *data = mh->get_data_pointer();
    size_t size =  mh->get_data_size();
    
    if ((data == 0)||(size==0))
    {
      posterror(pr,"No data is stored in matrix, stopping due to empty matrix condition");
      return(false);
    }
    
    if (!((mh->nrows()==1)||(mh->ncols()==1)))
    {
      posterror(pr,"Matrix is no row/column vector");
      return(false);  
    }
     
     
    std::string header(static_cast<size_t>(128),'\0');

    header[0] = 'B';
    header[1] = 'B';
    header[2] = '1'; // We do not yet support byte vectors of size 2
    header[3] = ':';
    header[4] = ' ';
    if (byteswapmachine() == true) header[1] = 'L';
    
    std::ostringstream oss;
    oss << size;
    std::string ssize = oss.str();
    std::cout << "before replace\n";
    header.replace(5,ssize.size(),ssize);
    
    // OK we created a header

    
    char *buffer = scinew char[size];
    
    for (int p=0;p<size;p++) buffer[p] = static_cast<char>(data[p]);

    
    FILE *fid =fopen(filename.c_str(),"w");
    if (fid == 0)
    {
      posterror(pr,"Could not open file");
      delete[] buffer;
      return(false);    
    }
    
    if ( fwrite(static_cast<void *>(&(header[0])),1,128,fid) != 128 )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       delete[] buffer;
       return(false);      
    }

    if ( fwrite(static_cast<void *>(buffer),1,size,fid) != size )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       delete[] buffer;
       return(false);      
    }
    
    delete[] buffer;
    fclose(fid);
    return(true);
  }
  
  if (filetype == "ivec")
  {

    if (mh->is_sparse())
    {
      posterror(pr,"Data is a sparse matrix and not a column vector as needed to export data");
      return(false);    
    }
       
    double *data = mh->get_data_pointer();
    size_t size =  mh->get_data_size();
    
    if ((data == 0)||(size==0))
    {
      posterror(pr,"No data is stored in matrix, stopping due to empty matrix condition");
      return(false);
    }
    
    if (!((mh->nrows()==1)||(mh->ncols()==1)))
    {
      posterror(pr,"Matrix is no row/column vector");
      return(false);  
    }
     
    std::string header(static_cast<size_t>(128),'\0');

    header[0] = 'I';
    header[1] = 'B';
    header[2] = '4'; // We do not yet support byte vectors of size 2
    header[3] = ':';
    header[4] = ' ';
    if (byteswapmachine() == true) header[1] = 'L';
    
    std::ostringstream oss;
    oss << size;
    std::string ssize = oss.str();
    header.replace(5,ssize.size(),ssize);
    
    // OK we created a header
    
    int *buffer = scinew int[size];
    
    for (int p=0;p<size;p++) buffer[p] = static_cast<int>(data[p]);
    
    FILE *fid =fopen(filename.c_str(),"w");
    if (fid == 0)
    {
      posterror(pr,"Could not open file");
      delete[] buffer;
      return(false);    
    }
    
    if ( fwrite(static_cast<void *>(&(header[0])),1,128,fid) != 128 )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       delete[] buffer;
       return(false);      
    }

    if ( fwrite(static_cast<void *>(buffer),4,size,fid) != size )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       delete[] buffer;
       return(false);      
    }
    
    delete[] buffer;
    fclose(fid);
    return(true);
  }
  
  if (filetype == "vec")
  {
   
    if (mh->is_sparse())
    {
      posterror(pr,"Data is a sparse matrix and not a column vector as needed to export data");
      return(false);    
    }
   
    double* data = mh->get_data_pointer();
    size_t size =  mh->get_data_size();
    
    if ((data == 0)||(size==0))
    {
      posterror(pr,"No data is stored in matrix, stopping due to empty matrix condition");
      return(false);
    }
    
    if (!((mh->nrows()==1)||(mh->ncols()==1)))
    {
      posterror(pr,"Matrix is no row/column vector");
      return(false);  
    }
     
    std::string header(static_cast<size_t>(128),'\0');

    header[0] = 'B';
    header[1] = 'B';
    header[2] = '8'; // We do not yet support byte vectors of size 2
    header[3] = ':';
    header[4] = ' ';
    if (byteswapmachine() == true) header[1] = 'L';
    
    std::ostringstream oss;
    oss << size;
    std::string ssize = oss.str();
    header.replace(5,ssize.size(),ssize);
    
    // OK we created a header
  
    FILE *fid =fopen(filename.c_str(),"w");
    if (fid == 0)
    {
      posterror(pr,"Could not open file");
      return(false);    
    }
    
    if ( fwrite(static_cast<void *>(&(header[0])),1,128,fid) != 128 )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       return(false);      
    }

    if ( fwrite(static_cast<void *>(data),8,size,fid) != size )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       return(false);      
    }
    
    fclose(fid);
    return(true);
  }
  
  
  if (filetype == "spr")
  {
    if (!(mh->is_sparse()))
    {
       posterror(pr,"SPR fileformat is for sparse matrices only");
       return(false);       
    }
    
    // Determine bandwith
    
    SparseRowMatrix* sp = dynamic_cast<SparseRowMatrix *>(mh.get_rep()); 
    
    int nz = 1;
    int bwp = 0;
    int bwn = 0;
    
    int *rr = sp->rows;
    int *cc = sp->columns;
    double *d = sp->a;
    int nnz = sp->nnz;
    int nrows = sp->nrows();
    int ncols = sp->ncols();
    
    int diag = 0;
    
    for (int p = 0; p < nrows; p++)
    {
      diag = 1;
      for (int r=rr[p];r<rr[p+1];r++)
      {
        if (cc[r] == p) diag = 0;
        if (bwp < (cc[r]-p)) bwp = cc[r]-p;
        if (bwn > (cc[r]-p)) bwn = cc[r]-p;
      }
      if ((rr[p+1]-rr[p]+diag)>nz) nz = (rr[p+1]-rr[p])+diag;
    }
    
    std::string header(static_cast<size_t>(128),'\0');

    header[0] = 'S';
    header[1] = 'B';
    header[2] = '8'; // We do not yet support byte vectors of size 2
    header[3] = '4';
    header[4] = ':';
    header[5] = ' ';
    if (byteswapmachine() == true) header[1] = 'L';
    
    std::ostringstream oss;
    oss << nrows << " " << ncols << " " << bwp << " " << bwn << " " << "1" << " " << nz;
    
    std::string ssize = oss.str();
    header.replace(6,ssize.size(),ssize);
    
    double *coef = scinew double[nz*nrows];
    int *jcoef = scinew int[nz*nrows];
    
    int s;
    for (int p = 0; p < nrows; p++)
    {
      for (int r = 0; r < nz; r++)
      {
        coef[p+r*nrows] = 0.0;
        jcoef[p+r*nrows] =  p + 1;
      }
    }

    for (int p = 0; p < nrows; p++)
    {
      for (int r = rr[p]; r < rr[p+1]; r++)
      {
        s = cc[r];
        if (s == p)
        {
          coef[p] = d[r];
        }
        else
        {
          for (int t=1; t < nz; t++)
          {
            if (jcoef[p+t*nrows] == p+1)
            {
              jcoef[p+t*nrows] = s+1;
              coef[p+t*nrows] = d[r];
              break;
            }
          }  
        }
      }
    }
    
  // OK we created a header
  
    FILE *fid =fopen(filename.c_str(),"w");
    if (fid == 0)
    {
      posterror(pr,"Could not open file");
      delete[] coef;
      delete[] jcoef;
      return(false);    
    }
    
    if ( fwrite(static_cast<void *>(&(header[0])),1,128,fid) != 128 )
    {
       posterror(pr,"Could not write to file");
       fclose(fid);
       delete[] coef;
       delete[] jcoef;
       return(false);      
    }

    if ( fwrite(static_cast<void *>(jcoef),4,nrows*nz,fid) != nrows*nz )
    {
       posterror(pr,"Could not write to file");
       delete[] coef;
       delete[] jcoef;
       fclose(fid);
       return(false);      
    }

    if ( fwrite(static_cast<void *>(coef),8,nrows*nz,fid) != nrows*nz )
    {
       posterror(pr,"Could not write to file");
       delete[] coef;
       delete[] jcoef;
       fclose(fid);
       return(false);      
    }
    
    fclose(fid);
    return(true);
    
  }
  
  
  
  return(false);
}


void CardioWaveConverter::posterror(ProgressReporter *pr,string msg)
{
  if (pr) pr->error(msg);
}

void CardioWaveConverter::postwarning(ProgressReporter *pr,string msg)
{
  if (pr) pr->warning(msg);
}

}
