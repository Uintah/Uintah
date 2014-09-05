ifeq ($(MAKE_WHAT),LIB)
  FILENAME:=$(subst lib/,projects/,$(LIBNAME).vcproj)
  $(FILENAME)_CONF:=2
  $(FILENAME)_OUTDIR:=../lib
  $(FILENAME)_NAME:=$(basename $(notdir $(FILENAME)))
else
  FILENAME:=projects/$(subst /,_,$(PROGRAM)).exe.vcproj
  $(FILENAME)_CONF:=1
  $(FILENAME)_OUTDIR:=../$(dir $(PROGRAM))
  $(FILENAME)_NAME:=$(notdir $(PROGRAM)).exe
endif

$(FILENAME)_SRCDIR:=$(SRCDIR)
$(FILENAME)_LIBNAME:=$(subst /,_,$(SRCDIR))

# create a VS-compatible ID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
# ours will be all 0s except the last digits of the first group.  Increment by counting x's
ID_COUNTER ?= x
ID_LESS_TEN?=yes
ID_LESS_HUNDRED?=yes
ID_LESS_THOUSAND?=yes
ID_HEADER ?= "0000000"

ID_BASE:=$(words $(ID_COUNTER))

ifeq ($(ID_LESS_TEN),yes)
  ifeq ("$(ID_BASE)","10")
    ID_LESS_TEN := no
    ID_HEADER := "000000"
  endif
else
ifeq ($(ID_LESS_HUNDRED),yes)
  ifeq ("$(ID_BASE)","100")
    ID_LESS_HUNDRED := no
    ID_HEADER := "00000"
  endif
else
ifeq ($(ID_LESS_THOUSAND),yes)
  ifeq ("$(ID_BASE)","1000")
    ID_LESS_THOUSAND := no
    ID_HEADER := "0000"
  endif
endif
endif
endif

$(FILENAME)_ID:=$(ID_HEADER)$(ID_BASE)"-0000-0000-0000-000000000000"
ID_COUNTER := "$(ID_COUNTER) x"


# in form of tcl.lib libCore_Exceptions.lib (etc.)
$(FILENAME)_LIBS:=$(patsubst -l%, %.lib, $(filter -l%,$(LIBS))) $(patsubst %,lib%.lib,$(subst /,_,$(PSELIBS))) $(filter %.lib,$(LIBS))
$(FILENAME)_PROJ_DEPS:=$(patsubst %,projects/lib%.dll.vcproj,$(subst /,_,$(PSELIBS)))

# remove -L and -LIBPATH, and turn into semi-colon
$(FILENAME)_LIBPATHS:=$(patsubst -LIBPATH:%,%;,$(filter -LIB%,$(LIBS) $(SCI_THIRDPARTY_LIBRARY)))

# turn includes into semicolon-delimited form
VC_INC_PATH:=$(subst ./,./../,$(shell echo $(INCLUDES) | sed 's, -I,;,g' | sed 's,-I,,g'))

ifeq ($(IS_DEBUG),yes)
  CONF_NAME="Debug"
  CONF_OPT=0
  CONF_RTL=2
  CONF_DBG=3
  CONF_MIN_REB="TRUE"
  CONF_LINK_INC=2
  CONF_LINK_OTHER=""
  CONF_RTC=0
else
  CONF_NAME="Release"
  CONF_OPT=2
  CONF_RTL=2
  CONF_DBG=0
  CONF_RTC=0
  CONF_MIN_REB="FALSE"
  CONF_LINK_INC=1
  CONF_LINK_OTHER="OptimizeReferences=\"2\" SEnableCOMDATFolding=\"2\""
endif

# this line is required for TkExtensions...
ifeq ($(FILENAME),projects/libDataflow_TCLThread.dll.vcproj)
  $(FILENAME)_OTHER="IgnoreDefaultLibraryNames=\"libc.lib\""
else
  $(FILENAME)_OTHER=""
endif

ALLVCPROJECTS := $(ALLVCPROJECTS) $(FILENAME)
$(FILENAME)_SRCS := $(SRCS)

# echo -e would produce nice newlines, but some files would have \e characters which kill the file
$(FILENAME):
	@echo "Creating $@"
	@echo "<?xml version=\"1.0\" encoding=\"Windows-1252\"?> \
<VisualStudioProject \
	ProjectType=\"Visual C++\" \
	Version=\"7.10\" \
	Name=\"$(basename $(notdir $@))\" \
	ProjectGUID=\"{$($@_ID)}\" \
	Keyword=\"Win32Proj\"> \
	<Platforms> \
		<Platform \
			Name=\"Win32\"/> \
	</Platforms> \
	<Configurations> \
		<Configuration \
			Name=\"$(CONF_NAME)|Win32\" \
			OutputDirectory=\"$($@_OUTDIR)\" \
			IntermediateDirectory=\"../$($@_SRCDIR)\" \
			ConfigurationType=\"$($@_CONF)\" \
			CharacterSet=\"2\"> \
			<Tool \
				Name=\"VCCLCompilerTool\" \
				Optimization=\"$(CONF_OPT)\" \
				AdditionalIncludeDirectories=\"$(VC_INC_PATH)\" \
				PreprocessorDefinitions=\"_WIN32;_CONSOLE;_USE_MATH_DEFINES;CRT_SECURE_NO_DEPRECATE;BUILD_$($@_LIBNAME)\" \
				MinimalRebuild=\"$(CONF_MIN_REB)\" \
				BasicRuntimeChecks=\"$(CONF_RTC)\" \
				RuntimeLibrary=\"$(CONF_RTL)\" \
				RuntimeTypeInfo=\"TRUE\" \
				UsePrecompiledHeader=\"0\" \
				BufferSecurityCheck=\"FALSE\" \
				WarningLevel=\"1\" \
				Detect64BitPortabilityProblems=\"TRUE\" \
				DebugInformationFormat=\"$(CONF_DBG)\"/> \
			<Tool \
				Name=\"VCLinkerTool\" \
				AdditionalDependencies=\"$($@_LIBS)\" \
				OutputFile=\"`echo '$$\(OutDir)' | sed 's,\\\(,\(,g'`/$($@_NAME)\" \
				LinkIncremental=\"$(CONF_LINK_INC)\" \
				AdditionalLibraryDirectories=\"$($@_LIBPATHS)../lib\" \
				GenerateDebugInformation=\"TRUE\" \
				ProgramDatabaseFile=\"../$($@_SRCDIR)/$($@_NAME).pdb\" \
				SubSystem=\"1\" \
				$(CONF_LINK_OTHER) \
				$($@_OTHER) \
				TargetMachine=\"1\"/> \
		</Configuration> \
	</Configurations> \
	<References> \
	</References> \
	<Files> \
	$(foreach file, $(subst /,\\,$($@_SRCS)), <File RelativePath=\"../../src/$(file)\"> </File>) \
	</Files> \
	<Globals> \
	</Globals> \
</VisualStudioProject> \
" > $@

