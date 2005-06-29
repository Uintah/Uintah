############
#Some bridge specific processing methods and variables
#should go in this file
dotloc = $outfile.index(".",0)
if (dotloc != nil)  
  $makename = $outfile.slice(0,dotloc)
else
  $makename = $outfile
end


# convert type to CCA bindings use
def ccaType( arg )
  type = arg.type 
  case type
    when /array/
      a = type.slice(6,type.length-1)
      t, dim = a.split(/,/)
      if (arg.mode == "in")
        return "const SSIDL::array1<"+t+"&" 
      else
        return "SSIDL::array1<"+t+"&"
      end
    else
      return type
  end
end
                                                                                                         
# convert type to Babel bindings use
def babelType( arg )
  type = arg.type
  case type
    when "string"
      return "char*"
    when "int"
      return "int32_t"
    when /array/
      a = type.slice(6,type.length-1)
      t, dim = a.split(/,/)
      return "sidl_"+t+"__array*"
    when /./
      #object type
      return type.gsub('.','::')
    else
      return type
  end
end

# babel user level types
def userBabelType( arg )
  type = arg.type
  case type
    when /array/
      a = type.slice(6,type.length-1)
      t, dim = a.split(/,/)
      return "sidl::array<"+t+">"
    when /./
      #object type
      return type.gsub('.','::')
    else
      return type
  end
end

