#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

#prints a comma delimited list of arguments' fullnames of a given method
#... to init arguments within a method declaration
def outDefArgs( typeModFunc )
  out = ""
  $args.each { |arg| 
    tipot = arg.type 
    if(typeModFunc != nil) then tipot = typeModFunc.call(arg) end 
    out += tipot+" "+arg.name+"," 
  } 
  out.slice!(out.length-1,out.length)
  return out
end

#prints a semicolon+line delimited list of arguments' fullnames of a given method
#... to init arguments within a function
def semioutDefArgs( typeModFunc )
  out = ""
  $args.each { |arg|
    tipot = arg.type
    if(typeModFunc != nil) then tipot = typeModFunc.call(arg) end
    out += tipot+" "+arg.name+";\n"
  }
  return out
end

#same as outDefArgs but starts with a comma if args exist
def commaoutDefArgs( typeModFunc )
  out = ","
  $args.each { |arg|
    tipot = arg.type
    if(typeModFunc != nil) then tipot = typeModFunc.call(arg) end
    out += tipot+" "+arg.name+","
  }
  out.slice!(out.length-1,out.length)
  return out
end

#prints out a list of arguments of a method, formatted as if calling the method (not fullnames)
#... to call a method
def outCallArgs
  out = ""
  $args.each { |arg|
    out += arg.name+","
  }
  out.slice!(out.length-1,out.length)
  return out
end

#prints a comma delimited list of arguments' fullnames of a given method
#... to init arguments within a method declaration
def outDefMappedArgs( typeModFunc )
  out = ""
  $args.each { |arg|
    tipot = arg.type
    if(typeModFunc != nil) then tipot = typeModFunc.call(arg) end
    out += tipot+" "+arg.name+"->"+arg.mappedType+","
  }
  out.slice!(out.length-1,out.length)
  return out
end

#shorthand functions
def ifNEQPrint( what, to, text)
  if (what != to) then return text end 
end
def ifEQPrint( what, to, text)
  if (what == to) then return text end
end


##########Data Conversion helpers:

#register a data conversion function
def registerConvertFunc(type1, type2, func)
  $convertList.push(ConvertFuncEntry.new(type1,type2,func))
end 

#if any argument map matches a conversion function that has been registered
#then we invoke that function and print out the result 
def emitConvertData
  $args.each { |arg|
    $convertList.each { |ce|
      if((arg.type == ce.typeFrom)&&(arg.mappedType == ce.typeTo)) then
        return ce.func.call(arg)
      end 
    }
  }
end

#(for inout arguments) if the argument matches the argument of a conversion function
#and no mapped argument exists, we allow the other conversion argument to be specified to
#this function. If it matches we call the conversion function.
def typeEmitConvertData(typeTo)
  $args.each { |arg|
    $convertList.each { |ce|
      if((arg.type == ce.typeFrom)&&(arg.mappedType == "")&&(ce.typeTo == typeTo))
        return ce.func.call(arg)
      end
    }
  }
  return ""
end

