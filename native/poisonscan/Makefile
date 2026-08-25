CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fvisibility=hidden
SRC := ../../src/poisonlab/accel/_c/poisonscan.c
BUILD := ../../src/poisonlab/accel/_bin

UNAME := $(shell uname -s 2>/dev/null || echo Windows)
ifneq (,$(findstring MINGW,$(UNAME)))
  SUFFIX := .dll
else ifneq (,$(findstring MSYS,$(UNAME)))
  SUFFIX := .dll
else ifneq (,$(findstring Windows,$(UNAME)))
  SUFFIX := .dll
else ifeq ($(UNAME),Darwin)
  SUFFIX := .dylib
else
  SUFFIX := .so
  CFLAGS += -fPIC
endif

TARGET := $(BUILD)/libpoisonscan$(SUFFIX)

all: $(TARGET)

$(TARGET): $(SRC) ../../src/poisonlab/accel/_c/poisonscan.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC)
	@echo "built $@"

clean:
	rm -f $(BUILD)/libpoisonscan*
	rm -f $(BUILD)/poisonscan-*

.PHONY: all clean
