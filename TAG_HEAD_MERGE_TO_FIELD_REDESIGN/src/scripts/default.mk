
DEFAULT_TARGET ?= all

default:
	cd $(OBJTOP) && $(MAKE) --print-directory $(DEFAULT_TARGET)

here:
	@echo "Attempting to build in $(HERE), this might not work for all directories..."
	cd $(OBJTOP) && $(MAKE) --print-directory $(HERE_TARGET)

%:
	cd $(OBJTOP) && $(MAKE) --print-directory $@
