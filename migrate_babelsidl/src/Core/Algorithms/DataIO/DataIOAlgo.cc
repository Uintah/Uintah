/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of pr_ software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and pr_ permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h> 
#include <Core/OS/Dir.h> // for LSTAT, MKDIR

#include <Core/Algorithms/DataIO/DataIOAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool DataIOAlgo::ReadField(std::string filename, FieldHandle& field, std::string importer)
{  
  if (importer == "")
  {
    Piostream *stream = auto_istream(filename, pr_);
    if (!stream)
    {
      error("Error reading file '" + filename + "'.");
      return (false);
    }
      
    // Read the file
    Pio(*stream, field);
    if (!field.get_rep() || stream->error())
    {
      error("Error reading data from file '" + filename +"'.");
      delete stream;
      return (false);
    }
       
    delete stream;
    return (true);
  }
  else
  {
    FieldIEPluginManager mgr;
    FieldIEPlugin *pl = mgr.get_plugin(importer);
    if (pl)
    {
      field = pl->filereader(pr_, filename.c_str());
      if (field.get_rep()) return (true);
    }
    else
    {
      error("Unknown field reader plug-in");
      return (false);
    }
  }
  return (false);  
}


bool DataIOAlgo::ReadMatrix(std::string filename, MatrixHandle& matrix, std::string importer)
{
  if (importer == "")
  {
    Piostream *stream = auto_istream(filename, pr_);
    if (!stream)
    {
      error("Error reading file '" + filename + "'.");
      return (false);
    }
      
    // Read the file
    Pio(*stream, matrix);
    if (!matrix.get_rep() || stream->error())
    {
      error("Error reading data from file '" + filename +"'.");
      delete stream;
      return (false);
    }
       
    delete stream;
    return (true);
  }
  else
  {
    MatrixIEPluginManager mgr;
    MatrixIEPlugin *pl = mgr.get_plugin(importer);
    if (pl)
    {
      matrix = pl->fileReader_(pr_, filename.c_str());
      if (matrix.get_rep()) return (true);
    }
    else
    {
      error("Unknown matrix reader plug-in");
      return (false);
    }
  }
  return (false);  
}


bool DataIOAlgo::ReadBundle(std::string filename, BundleHandle& bundle, std::string importer)
{
  if (importer != "")
  {
    error("Error no external importers are defined for bundles");
    return (false);
  }
  
  Piostream *stream = auto_istream(filename, pr_);
  if (!stream)
  {
    error("Error reading file '" + filename + "'.");
    return (false);
  }
    
  // Read the file
  Pio(*stream, bundle);
  if (!bundle.get_rep() || stream->error())
  {
    error("Error reading data from file '" + filename +"'.");
    delete stream;
    return (false);
  }
     
  delete stream;
  return (true);
}


bool DataIOAlgo::ReadNrrd(std::string filename, NrrdDataHandle& nrrd, std::string importer)
{
  if (importer != "")
  {
    error("Error no external importers are defined for nrrds");
    return (false);
  }
  
  Piostream *stream = auto_istream(filename, pr_);
  if (!stream)
  {
    error("Error reading file '" + filename + "'.");
    return (false);
  }
    
  // Read the file
  Pio(*stream, nrrd);
  if (!nrrd.get_rep() || stream->error())
  {
    error("Error reading data from file '" + filename +"'.");
    delete stream;
    return (false);
  }
     
  delete stream;
  return (true);
}


bool DataIOAlgo::ReadColorMap(std::string filename, ColorMapHandle& colormap, std::string importer)
{
  if (importer != "")
  {
    error("Error no external importers are defined for colormaps");
    return (false);
  }
  
  Piostream *stream = auto_istream(filename, pr_);
  if (!stream)
  {
    error("Error reading file '" + filename + "'.");
    return (false);
  }
    
  // Read the file
  Pio(*stream, colormap);
  if (!colormap.get_rep() || stream->error())
  {
    error("Error reading data from file '" + filename +"'.");
    delete stream;
    return (false);
  }
     
  delete stream;
  return (true);
}

bool DataIOAlgo::ReadColorMap2(std::string filename, ColorMap2Handle& colormap, std::string importer)
{
  if (importer != "")
  {
    error("Error no external importers are defined for colormaps");
    return (false);
  }
  
  Piostream *stream = auto_istream(filename, pr_);
  if (!stream)
  {
    error("Error reading file '" + filename + "'.");
    return (false);
  }
    
  // Read the file
  Pio(*stream, colormap);
  if (!colormap.get_rep() || stream->error())
  {
    error("Error reading data from file '" + filename +"'.");
    delete stream;
    return (false);
  }
     
  delete stream;
  return (true);
}


bool DataIOAlgo::ReadPath(std::string filename, PathHandle& path, std::string importer)
{
  if (importer != "")
  {
    error("Error no external importers are defined for colormaps");
    return (false);
  }
  
  Piostream *stream = auto_istream(filename, pr_);
  if (!stream)
  {
    error("Error reading file '" + filename + "'.");
    return (false);
  }
    
  // Read the file
  Pio(*stream, path);
  if (!path.get_rep() || stream->error())
  {
    error("Error reading data from file '" + filename +"'.");
    delete stream;
    return (false);
  }
     
  delete stream;
  return (true);
}


bool DataIOAlgo::WriteField(std::string filename, FieldHandle& field, std::string exporter)
{
  if (field.get_rep() == 0) return (false);
  
  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, field);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, field);
    } 
    delete stream;  
  }
  else
  {
    FieldIEPluginManager mgr;
    FieldIEPlugin *pl = mgr.get_plugin(exporter);
    if (pl)
    {
      pl->filewriter(pr_, field, filename.c_str());
      if (field.get_rep()) return (true); 
    }
    else
    {
      error("Could not find requested exporter");
      return (false);
    }
    return (false);
  }
  return (true);
}


bool DataIOAlgo::WriteMatrix(std::string filename, MatrixHandle& matrix, std::string exporter)
{
  if (matrix.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, matrix);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, matrix);
    } 
    delete stream;  
  }
  else
  {
    MatrixIEPluginManager mgr;
    MatrixIEPlugin *pl = mgr.get_plugin(exporter);
    if (pl)
    {
      pl->fileWriter_(pr_, matrix, filename.c_str());
      if (matrix.get_rep()) return (true); 
    }
    else
    {
      error("Could not find requested exporter");
      return (false);
    }
    return (false);
  }
  return (true);
}


bool DataIOAlgo::WriteBundle(std::string filename, BundleHandle& bundle, std::string exporter)
{
  if (bundle.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, bundle);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, bundle);
    } 
    delete stream;  
  }
  else
  {
    error("No exporters are supported for bundles");
    return (false);
  }
  return (true);
}



bool DataIOAlgo::WriteNrrd(std::string filename, NrrdDataHandle& nrrd, std::string exporter)
{
  if (nrrd.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, nrrd);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, nrrd);
    } 
    delete stream;  
  }
  else
  {
    error("No exporters are supported for nrrds");
    return (false);
  }
  return (true);
}


bool DataIOAlgo::WriteColorMap(std::string filename, ColorMapHandle& colormap, std::string exporter)
{
  if (colormap.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, colormap);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, colormap);
    } 
    delete stream;  
  }
  else
  {
    error("No exporters are supported for colormaps");
    return (false);
  }
  return (true);
}

bool DataIOAlgo::WriteColorMap2(std::string filename, ColorMap2Handle& colormap, std::string exporter)
{
  if (colormap.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, colormap);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, colormap);
    } 
    delete stream;  
  }
  else
  {
    error("No exporters are supported for colormaps");
    return (false);
  }
  return (true);
}

bool DataIOAlgo::WritePath(std::string filename, PathHandle& path, std::string exporter)
{
  if (path.get_rep() == 0) return (false);

  if ((exporter == "text")||(exporter == "Text"))
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Text", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, path);
    } 
    delete stream;
  }
  else if (exporter == "")
  {
    Piostream* stream;
    stream = auto_ostream(filename, "Binary", pr_);
    if (stream->error())
    {
      error("Could not open file for writing" + filename);
      return (false);
    }
    else
    {
      // Write the file
      Pio(*stream, path);
    } 
    delete stream;  
  }
  else
  {
    error("No exporters are supported for paths");
    return (false);
  }
  return (true);
}

bool DataIOAlgo::FileExists(std::string filename)
{
  FILE* fp;
  fp = ::fopen (filename.c_str(), "r");
  if (!fp)
  {
    return (false);
  }
  ::fclose(fp);
  
  return (true);
}


bool DataIOAlgo::CreateDir(std::string dirname)
{
  struct stat buf;
  if (::LSTAT(dirname.c_str(),&buf) < 0)
  {
    int exitcode = MKDIR(dirname.c_str(), 0777);
    if (exitcode) return (false);
  }        
  return (true);
}


} // end namespace SCIRunAlgo


