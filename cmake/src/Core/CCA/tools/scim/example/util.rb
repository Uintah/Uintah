
def showArgs
  out = ""
  $args.each { |arg| 
    out += "<i>"+arg.type+"</i> "+arg.name+"," 
  } 
  out.slice!(out.length-1,out.length)
  return out
end


