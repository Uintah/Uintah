DEFAULT_TARGET ?= all

default:
	cd $(OBJTOP) && $(MAKE) --print-directory $(DEFAULT_TARGET)

%:
	cd $(OBJTOP) && $(MAKE) --print-directory $(DEFAULT_TARGET) $@
