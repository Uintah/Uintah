#require 'scim'

#creates two separate streams containing based on langc++ tags: one stream is the sidl+mapping directives,
#the other is the C++ code plus the in or out interface mapping directives 
def frontEndPreProcess( text ) 
  cpptext = "" 
  patternstart = "%langC++"
  patternend = "%/langC++"
  intag = "%ininterface"
  intagC = "%/ininterface"
  outtag = "%outinterface"
  outtagC = "%/outinterface"


  left = text.index(patternstart)
  while(left != nil)
    #isolate text that belongs withing the lang tag
    right = text.index(patternend,left)
    lookahead = text.index(patternstart,left+1)
    while(lookahead != nil)&&(right != nil)&&(lookahead < right)
      lookahead = text.index(patternstart,lookahead+1)
      right = text.index(patternend,right+1)
    end
    if(right == nil)
      puts "Missing " + patternend + " in .erb file"
      exit
    end

    while(true)
      s_tag = text.index(intag,left)
      e_tag = text.index(intagC,left)
      if(s_tag == nil)&&(e_tag == nil)
        #no tags = inout
        middlelen = right-(left+patternstart.length)
        cpptext += text.slice(left+patternstart.length,middlelen)
        text[left,right+patternend.length-left] = ""
        break 
      else 
        if(s_tag == nil)
          if(e_tag < right) 
            #tags starts outside of lang statement, ends within it
            middlelen = (e_tag+intagC.length)-(left+patternstart.length)
            cpptext += intag + "\n" + text.slice(left+patternstart.length,middlelen)
            text[left,e_tag+intagC.length-left] = intagC + "\n" + patternstart
          else
            #both tags outside of statement
            middlelen = right-(left+patternstart.length)
            cpptext += intag + "\n" + text.slice(left+patternstart.length,middlelen) + "\n" + intagC +"\n"
            text[left,right+patternend.length-left] = ""
          end
          break
        elsif(e_tag == nil)
          puts "preproc.rb: frontEndPreProcess() error"
          exit
        else
          if(e_tag < s_tag)&&(e_tag < right)
            #tags starts outside of lang statement, ends within it
            middlelen = (e_tag+intagC.length)-(left+patternstart.length)
            cpptext += intag + "\n" + text.slice(left+patternstart.length,middlelen) 
            text[left,e_tag+intagC.length-left] = intagC + "\n" + patternstart 
            break
          elsif(e_tag < s_tag)&&(e_tag > right)
            #both tags outside of statement
            middlelen = right-(left+patternstart.length)
            cpptext += intag + "\n" + text.slice(left+patternstart.length,middlelen) + "\n" + intagC +"\n"
            text[left,right+patternend.length-left] = ""
            e_tag = text.index(intagC,left)
            break
          else
            #both tags inside statement
            middlelen = (e_tag+intagC.length)-(left+patternstart.length)
            cpptext += text.slice(left+patternstart.length,middlelen)
            text[left,e_tag+intagC.length-left] = patternstart 
            right = text.index(patternend,left) 
          end
        end
      end
    end
    
    left = text.index(patternstart,left+1)
  end
  return text,cpptext
end


#replaces loop constructs with actual ruby loops and necesary prep routines
def preProcessParse( text, looptext, patternstart, patternend )
  left = text.index(patternstart)
  while(left != nil)
    right = text.index(patternend,left)
    lookahead = text.index(patternstart,left+1)

    while(lookahead != nil)&&(right != nil)&&(lookahead < right)
      lookahead = text.index(patternstart,lookahead+1)
      right = text.index(patternend,right+1)
    end
                                                                                                          
    if(right == nil)
      puts "Missing " + patternend + " in .erb file"
      exit
    end

    middlelen = right-(left+patternstart.length)
    text[left,right+patternend.length-left]= looptext + text.slice(left+patternstart.length,middlelen) + "<% end%>"
    left = text.index(patternstart,left+1)
  end
  return text
end


def emitPreProcess( template )
  t = File.open(template)
  methodtext = "  <% runtimeCheckMethodLoop%>" +  
               "  <% for j in 0...$map.getMethodMapSize%>" + 
               "  <% loadMethodVars($map,j)%>" 

  porttext = "<% for i in 0...Ir.getMapSize%>" +
             "  <% loadPortVars(i)%>"

  #run preprocessor on method directive 
  text = preProcessParse( t.read, methodtext, "<method>", "</method>" )

  #run preprocessor on port directive
  text = preProcessParse( text, porttext, "<port>", "</port>" )

  return text
end

def emitSrcPreProcess( srcfile )
  s = File.open(srcfile)
  sidlsrc,cppsrc = frontEndPreProcess( s.read )

  if(sidlsrc != "") 
    idlf = File.open(srcfile+".idl","w")
    idlf << sidlsrc << "\n"
    idlf.close
    dosidlparse = true
  end

  if(cppsrc != "") 
    cppf = File.open(srcfile+".cpp","w")
    cppf << cppsrc << "\n"
    cppf.close
    docppparse = true
  end

  return dosidlparse,docppparse 
end
