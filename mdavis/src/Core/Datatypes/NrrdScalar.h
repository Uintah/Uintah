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
 * FILE: NrrdScalar.h
 * AUTH: Jeroen Stinstra
 * DATE: 29 Nov  2004
 */

#ifndef CORE_DATATYPES_NRRDSCALAR_H
#define CORE_DATATYPES_NRRDSCALAR_H 1
 
#include <Core/Datatypes/NrrdData.h>
#include <string> 
 
namespace SCIRun {

class NrrdScalar {
  public:
    // constructors
    NrrdScalar();
    NrrdScalar(const char T);
    NrrdScalar(const unsigned char T);    
    NrrdScalar(const short T);
    NrrdScalar(const unsigned short T);
    NrrdScalar(const int T);
    NrrdScalar(const unsigned int T);
    NrrdScalar(const float T);
    NrrdScalar(const double T);
    NrrdScalar(std::string str,std::string type="int");

    NrrdScalar(const NrrdScalar& nrrdscalar);
    NrrdScalar(const NrrdDataHandle& handle);
    
    virtual ~NrrdScalar();
    
    bool            set(const char T);
    bool            set(const unsigned char T);
    bool            set(const short T);
    bool            set(const unsigned short T);
    bool            set(const int T);
    bool            set(const unsigned int T);
    bool            set(const float T);
    bool            set(const double T);
    bool            set(std::string str,std::string type="int");

    template<class T> bool get(T& val);
    
    NrrdDataHandle  gethandle();
    
  private:
    NrrdDataHandle nrrdscalar_;

};

inline NrrdDataHandle NrrdScalar::gethandle()
{
    return(nrrdscalar_);
}

template<class T> bool NrrdScalar::get(T& val)
{
    if (nrrdscalar_.get_rep() != 0)
    {
        if (nrrdscalar_->nrrd != 0)
        {
            if ((nrrdscalar_->nrrd->dim > 0)&&(nrrdscalar_->nrrd->axis[0].size > 0))
            {
                if (nrrdscalar_->nrrd->data != 0)
                {
                    switch (nrrdscalar_->nrrd->type)
                    {
                        case nrrdTypeChar:
                        {
                            char* ptr = static_cast<char*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeUChar:
                        {
                            unsigned char* ptr = static_cast<unsigned char*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeShort:
                        {
                            short* ptr = static_cast<short*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeUShort:
                        {
                            unsigned short* ptr = static_cast<unsigned short*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeInt:
                        {
                            int* ptr = static_cast<int*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeUInt:
                        {
                            unsigned int* ptr = static_cast<unsigned int*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeFloat:
                        {
                            float* ptr = static_cast<float*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true);
                        }
                        case nrrdTypeDouble:
                        {
                            double* ptr = static_cast<double*>(nrrdscalar_->nrrd->data);   
                            val = static_cast<T>(*ptr);
                            return(true); 
                        }
                        default:
                            return(false);
                    }
                }
            }
        }
    }
    return(false);
}




} // end namespace 

#endif

