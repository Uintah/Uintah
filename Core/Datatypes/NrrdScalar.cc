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
 * FILE: NrrdScalar.cc
 * AUTH: Jeroen Stinstra
 * DATE: 29 Nov  2004
 */
 
 
// This is a simple implementation to use NrrdData objects as scalars
// The object simply maintains a pointer to a NrrdData object. 
 
#include <Core/Datatypes/NrrdScalar.h>
#include <sstream>

namespace SCIRun {

NrrdScalar::NrrdScalar()
{
    nrrdscalar_ = 0;
}

NrrdScalar::NrrdScalar(const char T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const unsigned char T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const short T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const unsigned short T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const int T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const unsigned int T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const float T)
{
    set(T);
}

NrrdScalar::NrrdScalar(const double T)
{
    set(T);
}

NrrdScalar::NrrdScalar(std::string str,std::string type)
{
    set(str,type);
}


NrrdScalar::NrrdScalar(const NrrdScalar& nrrdscalar)
{
    nrrdscalar_ = nrrdscalar.nrrdscalar_;
}

NrrdScalar::NrrdScalar(const NrrdDataHandle& handle)
{
    nrrdscalar_ = handle;
}


bool NrrdScalar::set(std::string str, std::string type)
{
    if ((type == "char")||(type == "int8"))
    {
        int T;
        std::istringstream iss(str);
        iss >> T;
        return(set(static_cast<char>(T)));
    }
    if ((type == "unsigned char")||(type == "uchar")||(type == "uint8")||(type == "unsigned int8"))
    {
        int T;
        std::istringstream iss(str);
        iss >> T;
        return(set(static_cast<unsigned char>(T)));
    }
    if ((type == "short")||(type == "int16"))
    {
        short T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    if ((type == "unsigned short")||(type == "ushort")||(type == "uint16")||(type == "unsigned int16"))
    {
        unsigned short T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    if ((type == "int")||(type == "int32"))
    {
        int T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    if ((type == "unsigned int")||(type == "uint32")||(type == "unsigned int32")||(type == "uint"))
    {
        unsigned int T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    if ((type == "float")||(type=="single"))
    {
        float T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    if (type == "double")
    {
        double T;
        std::istringstream iss(str);
        iss >> T;
        return(set(T));
    }
    std::cerr << "NrrdScalar: type '" << type << "' has not been implemented" << std::endl;
    return(false);
}


bool NrrdScalar::set(const char T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeChar, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    char *val = (char*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}

bool NrrdScalar::set(const unsigned char T)
{
    nrrdscalar_ = scinew NrrdData();
        
    if (nrrdscalar_.get_rep() == 0) return(false);

    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeUChar, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    unsigned char *val = (unsigned char*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}


bool NrrdScalar::set(const short T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeShort, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    short *val = (short*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(true);
    }
    else
    {
        *val = T;
    }
    return(true);
}

bool NrrdScalar::set(const unsigned short T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeUShort, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    unsigned short *val = (unsigned short*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}

bool NrrdScalar::set(const int T)
{
    nrrdscalar_ = scinew NrrdData();
        
    if (nrrdscalar_.get_rep() == 0) return(false);

    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeInt, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    int *val = (int*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}

bool NrrdScalar::set(const unsigned int T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeUInt, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    unsigned int *val = (unsigned int*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}



bool NrrdScalar::set(const float T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeFloat, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    float *val = (float*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}


bool NrrdScalar::set(const double T)
{
    nrrdscalar_ = scinew NrrdData();
    
    if (nrrdscalar_.get_rep() == 0) return(false);
    
    // Need to add some error checking here
    nrrdAlloc(nrrdscalar_->nrrd, nrrdTypeDouble, 1, 1);
    nrrdAxisInfoSet(nrrdscalar_->nrrd, nrrdAxisInfoLabel, "nrrdscalar");
    nrrdscalar_->nrrd->axis[0].kind = nrrdKindDomain;
    double *val = (double*)nrrdscalar_->nrrd->data;
    if (val == 0)
    {
        nrrdscalar_ = 0;
        return(false);
    }
    else
    {
        *val = T;
    }
    return(true);
}

NrrdScalar::~NrrdScalar()
{
}

} // end namespace
