
$scratchfile = 'scratchfile'

def getArgs
  i = 0
  args = Array.new
  IO.foreach($scratchfile) do |line| 
    args[i] = line.strip.chomp 
    i = i+1
  end 
  return args
end

def ret2sr(lineout)
  aFile = File.new($scratchfile, "w")
  aFile.print(lineout)
  aFile.close
end


