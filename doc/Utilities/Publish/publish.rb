#!/usr/bin/ruby
#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

#
# Update, build, and deliver scirun docs.
#

class Log
  def initialize(obj=nil)
    @log = nil
    if obj == nil
      @log = $stderr
    elsif obj.instance_of?(String)
      @log = File.new(obj, "w")
    elsif obj.kind_of?(IO)
      @log = obj
    else
      raise("Log::initialize: obj must be of String or IO type")
    end
  end
  def write(*args)
    if @log != nil
      @log.write(args)
      @log.flush
    end
  end
end

class ConfError < RuntimeError
  def initialize(msg)
    super("Configuration  error: " + msg)
  end
end

class ConfHash < Hash
  def ConfHash.[](*args)
    super
  end
  def confError(m)
    raise(ConfError, m)
  end
  def missing?(key)
    not has_key?(key)
  end
  def empty?(key)
    self[key].size() == 0
  end
  def errorIfMissing(key)
    confError("Missing \"#{key}\"") if missing?(key)
  end
  def boolean?(key)
    self[key].instance_of?(FalseClass) or self[key].instance_of?(TrueClass)
  end
  def string?(key)
    self[key].instance_of?(String)
  end
  def dest?(d)
    d.instance_of?(Dest)
  end
  def array?(key)
    self[key].instance_of?(Array)
  end
  def errorIfNotBoolean(key)
    confError("Not boolean \"#{key}\"") if not boolean?(key)
  end
  def errorIfNotString(key)
    confError("Not string \"#{key}\"") if not string?(key)
  end
  def errorIfEmpty(key)
    confError("Empty \"#{key}\"") if self[key].size() == 0
  end
  def errorIfNotDest(d)
    confError("Not a Dest \"#{d}\"") if not dest?(d)
  end
  def errorIfNotArray(key)
    confError("Not an array \"#{key}\"") if not array?(key)
  end
end

class Dest < ConfHash
  attr_reader :remote

  User = "user"
  Mach = "mach"
  Dir = "dir"
  Tar = "tar"

  def Dest.[](*args)
    super(*args).init()
  end

  def init()
    self[Mach] = "." if missing?(Mach)
    self[Tar] = "/usr/local/bin/tar" if missing?(Tar)
    self
  end

  def validate
    errorIfNotString(Mach)
    @remote = self[Mach] != "."
    if @remote == true
      errorIfMissing(User)
      errorIfNotString(User)
    end
    errorIfMissing(Dir)
    errorIfNotString(Dir)
    errorIfNotString(Tar)
  end
end

class Configuration < ConfHash
  attr_reader :groupDirs

  LogFile = "logFile"
  Group = "group"
  BuildDir = "buildDir"
  Tree = "treeToPublish"
  Wait = "wait"
  CodeViews = "codeViews"
  Deliver = "deliver"
  Tarball = "tarball"
  Dests = "destinations"
  SSHAgentFile = "sshAgentFile"
  Build = "build"
  ToolsPath = "toolspath"
  ClassPath = "classpath"
  Stylesheet_XSL_HTML = "stylesheet_XSL_HTML"
  Stylesheet_DSSSL_Print = "stylesheet_DSSSL_Print"
  XML_DCL = "XML_DCL"
  Catalog = "catalog"
  Make = "make"
  Update = "update"
  PwdOnly = "pwdOnly"
  Clean = "clean"

  def Configuration.new(file)
    eval("Configuration[#{File.new(file, 'r').read}]").init()
  end

  def Configuration.[](*args)
    super
  end

  def init()
    self[Group] = "BioPSE" if missing?(Group)
    self[BuildDir] = "." if missing?(BuildDir)
    self[BuildDir] = File.expand_path(self[BuildDir])
    self[Wait] = false if missing?(Wait)
    self[CodeViews] = false if missing?(CodeViews)
    self[PwdOnly] = false if missing?(PwdOnly)
    self[Tree] = "SCIRunDocs" if missing?(Tree)
    self[Deliver] = false if missing?(Deliver) or self[PwdOnly] == true
    self[Tarball] = false if missing?(Tarball) or self[PwdOnly] == true
    self[Build] = true if missing?(Build)
    self[ToolsPath] = "" if missing?(ToolsPath)
    self[Make] = "/usr/bin/gnumake" if missing?(Make)
    self[Update] = true if missing?(Update)
    self[LogFile] = $stderr if missing?(LogFile)
    self[Clean] = false if missing?(Clean)
    $log = Log.new(self[LogFile])
    validate()

    if self[PwdOnly] == false
      initializeGroupsDB()
      @groupDirs = @groupsDB[self[Group]]
    end

    if self[Deliver] == true
      self[Dests].each do |d|
	if d.remote
	  ENV["CVS_RSH"] = "ssh"
	  begin
	    File.open(self[SSHAgentFile], "r") do |f|
	      s = f.read
	      ENV['SSH_AUTH_SOCK']=/SSH_AUTH_SOCK=(.*?);/.match(s)[1]
	      ENV['SSH_AGENT_PID']=/SSH_AGENT_PID=(\d+);/.match(s)[1]
	    end
	  rescue
	    confError("Can't get ssh agent info from #{self[SSHAgentFile]}")
	  end
	  break;
	end
      end
    end
    self
  end

  def validate()
    errorIfNotString(Group)
    if not self[Group] =~ /^(BioPSE|SCIRun|Uintah)$/
      confError("\"#{Group}\" must be one of \"BioPSE\", \"SCIRun\", or \"Uintah\"")
    end
    errorIfNotString(BuildDir)
    errorIfEmpty(BuildDir)
    errorIfMissing(Tree)
    errorIfNotString(Tree)
    errorIfNotBoolean(Wait)
    errorIfNotBoolean(CodeViews)
    errorIfNotBoolean(Deliver)
    if self[Deliver] == true
      errorIfMissing(Dests)
      errorIfNotArray(Dests)
      needAgent = false
      self[Dests].each do |d|
	errorIfNotDest(d)
	d.validate()
	needAgent = true if d.remote
      end
      if needAgent
	errorIfMissing(SSHAgentFile)
	errorIfNotString(SSHAgentFile)
	errorIfEmpty(SSHAgentFile)
      end
    end
    errorIfNotBoolean(Tarball)
    errorIfNotBoolean(Build)
    errorIfNotString(ToolsPath)
    errorIfMissing(ClassPath)
    errorIfNotString(ClassPath)
    errorIfMissing(Stylesheet_XSL_HTML)
    errorIfNotString(Stylesheet_XSL_HTML)
    errorIfMissing(Stylesheet_DSSSL_Print)
    errorIfNotString(Stylesheet_DSSSL_Print)
    errorIfMissing(XML_DCL)
    errorIfNotString(XML_DCL)
    errorIfMissing(Catalog)
    errorIfNotString(Catalog)
    errorIfNotString(Make)
    errorIfEmpty(Make)
    errorIfNotBoolean(Update)
    errorIfNotBoolean(Clean)
  end

  def initializeGroupsDB()
    @groupsDB = {}
    @groupsDB["SCIRun"] = ["doc"]
    srcRoot = self[BuildDir] + "/" + self[Tree] + "/src"
    Dir.foreach(srcRoot) do |m|
      @groupsDB["SCIRun"] << "src/#{m}" if not m =~ /^(\.|\.\.|Packages|CVS)$/
    end
    @groupsDB["BioPSE"] = @groupsDB["SCIRun"] + [ "src/Packages/BioPSE",
      "src/Packages/Teem", "src/Packages/MatlabInterface",
      "src/Packages/VDT", "src/Packages/Fusion" ]
    @groupsDB["Uintah"] = @groupsDB["SCIRun"] + [ "src/Packages/Uintah" ]
  end

end

class Docs

  def initialize()
    file = nil
    case ARGV.length
    when 0
      file = "publish.conf"
    when 1
      file = ARGV[0]
    else
      $stderr.print(%Q{Usage: #{File.basename($0)} [config-file]}, "\n")
      exit(1)
    end
    @conf = Configuration.new(file)
#    exit(0)
    @treeRoot = @conf[Configuration::BuildDir] + '/' + @conf[Configuration::Tree]
    @redirect = "2>&1"
  end

  def publish()
    build() if @conf[Configuration::Build] == true
    deliver() if @conf[Configuration::Deliver] == true
  end

  def build()
    ENV["PATH"] = ENV["PATH"] + ":" + @conf[Configuration::ToolsPath]
    ENV["CLASSPATH"] = @conf[Configuration::ClassPath]
    ENV["STYLESHEET_XSL_HTML"] = @conf[Configuration::Stylesheet_XSL_HTML]
    # Next one is for compatibility with 1.10.1 and earlier docs.
    ENV["STYLESHEET_PATH"] = ENV["STYLESHEET_XSL_HTML"]
    ENV["STYLESHEET_DSSSL_PRINT"] = @conf[Configuration::Stylesheet_DSSSL_Print]
    ENV["XML_DCL"] = @conf[Configuration::XML_DCL]
    ENV["CATALOG"] = @conf[Configuration::Catalog]

    pwd = Dir.pwd
    trys = 0
    doclock = File.new("#{@conf[Configuration::BuildDir]}/.doclock", "w+")
    callcc {|$tryAgain|}
    trys += 1
    if doclock.flock(File::LOCK_EX|File::LOCK_NB) == 0
      if @conf[Configuration::Clean] == true
	if @conf[Configuration::PwdOnly] == true
	  clean(Dir.pwd())
	else
	  clean("#{@treeRoot}/doc/")
	end
      end
      if @conf[Configuration::Update] == true
	if @conf[Configuration::PwdOnly] == true
	  updateOne(Dir.pwd())
	else
	  update()
	end
      end
      if @conf[Configuration::PwdOnly] == true
	make(Dir.pwd())
      else
	make("#{@treeRoot}/doc/")
      end
      doclock.flock(File::LOCK_UN)
    elsif @conf[Configuration::Wait]
      $log.write( (trys > 1 ? "." : "Someone else is updating the \
docs.  Waiting...") )
      sleep(10)
      $tryAgain.call
    else
      $log.write("Someone else is updating the docs.  Quiting\n")
    end
    doclock.close
    Dir.chdir(pwd)
  end

  def clean(dir)
    $log.write("Begin clean starting at ", dir, "\n")
    pwd = Dir.pwd()
    Dir.chdir(dir)
    $log.write(`#{@conf[Configuration::Make]} veryclean #{@redirect}`)
    Dir.chdir(pwd)
    $log.write("End clean\n")
  end

  def deliver()
    pwd = Dir.pwd
    Dir.chdir(@conf[Configuration::BuildDir])
    @conf[Configuration::Dests].each do |d|
      deliverOne(d)
    end
    Dir.chdir(pwd)
  end

  # FIXME: Need some file locking on the destination side!
  def deliverOne(dest)
    installScript = <<END_OF_SCRIPT
(cd #{dest[Dest::Dir]}
if #{dest[Dest::Tar]} zxf #{@conf[Configuration::Tree]}.tar.gz;
then 
  if test -d doc 
  then 
    if ! rm -rf doc
    then 
      echo failed to remove old doc
      exit 1
    fi 
  fi 
  if test -d src 
  then 
    if ! rm -rf src 
    then 
      echo failed to remove old src 
      exit 1 
    fi 
  fi 
  if ! mv #{@conf[Configuration::Tree]}/doc . 
  then 
    echo failed to install new doc 
    exit 1 
  fi 
  if ! mv #{@conf[Configuration::Tree]}/src . 
  then 
    echo failed to install new src 
    exit 1 
  fi 
  if ! rm -rf #{@conf[Configuration::Tree]} 
  then 
    echo failed to remove #{@conf[Configuration::Tree]} 
    exit 1 
  fi 
  exit 0   
else 
  echo tar failed 
  exit 1 
fi 
) #{@redirect}
END_OF_SCRIPT

    $log.write("Delivering to ", dest[Dest::Mach], "\n")
    tarball = @treeRoot + "/doc/" + @conf[Configuration::Tree] + ".tar.gz"
    $log.write("Transfering #{tarball}...\n")
    if dest[Dest::Mach] == "."
      $log.write(`cp #{tarball} #{dest[Dest::Dir]} #{@redirect}`)
    else
      $log.write(`scp -p -q #{tarball} #{dest[Dest::User]}@#{dest[Dest::Mach]}:#{dest[Dest::Dir]} #{@redirect}`)
    end
    if $? != 0
      $log.write("Transfer failed. Giving up.\n")
    else
      $log.write("Installing...\n")
      if dest[Dest::Mach] == "."
	$log.write(`#{installScript}`)
      else
	$log.write(`ssh #{dest[Dest::User]}@#{dest[Dest::Mach]} '#{installScript}'`)
      end
      if $? == 0
	$log.write("Finished this delivery.\n")
      else
	$log.write("Installation Failed\n")
      end 
    end
  end

  def updateOne(m)
    pwd = Dir.pwd
    $log.write("Updating ", m, "\n")
    if FileTest.directory?(m)
      Dir.chdir(m)
      $log.write(`cvs update -P -d #{@redirect}`, "\n")
    elsif FileTest.file?(m)
      Dir.chdir(File.dirname(m))
      $log.write(`cvs update #{File.basename(m)} #{@redirect}`, "\n")
    else
      $log.write(m, " doesn't exist - ignoring\n")
    end
    Dir.chdir(pwd)
  end

  def update()
    $log.write("Updating...\n")
    @conf.groupDirs.each do |m|
      updateOne(@treeRoot + "/" + m)
    end
    $log.write("Done Updating\n")
  end

  def make(startDir)
    $log.write("Making docs...\n")
    group = @conf[Configuration::Group].downcase()
    pwd = Dir.pwd()
    Dir.chdir(startDir)
    $log.write(`#{@conf[Configuration::Make]} SRGROUP=#{group} WITH_CV=#{@conf[Configuration::CodeViews] ? "true" : "false"} #{@conf[Configuration::Tarball] ? "tarball" : ""} #{@redirect}`)
    Dir.chdir(pwd)
    $log.write("Done making docs\n")
  end

  private :updateOne, :update, :make, :deliverOne

end

def main
  begin
    tbeg = Time.now
    docs = Docs.new
    docs.publish()
    tend = Time.now
    $log.write("Elapsed time: ", tend - tbeg, "\n")
  rescue
    $log.write($!, "\n")
  end
end

main
