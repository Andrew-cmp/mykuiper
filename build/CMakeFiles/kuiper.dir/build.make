# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xiaohou/Desktop/myinfer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiaohou/Desktop/myinfer/build

# Include any dependencies generated for this target.
include CMakeFiles/kuiper.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kuiper.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kuiper.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kuiper.dir/flags.make

CMakeFiles/kuiper.dir/source/data/load_data.cpp.o: CMakeFiles/kuiper.dir/flags.make
CMakeFiles/kuiper.dir/source/data/load_data.cpp.o: ../source/data/load_data.cpp
CMakeFiles/kuiper.dir/source/data/load_data.cpp.o: CMakeFiles/kuiper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kuiper.dir/source/data/load_data.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper.dir/source/data/load_data.cpp.o -MF CMakeFiles/kuiper.dir/source/data/load_data.cpp.o.d -o CMakeFiles/kuiper.dir/source/data/load_data.cpp.o -c /home/xiaohou/Desktop/myinfer/source/data/load_data.cpp

CMakeFiles/kuiper.dir/source/data/load_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper.dir/source/data/load_data.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaohou/Desktop/myinfer/source/data/load_data.cpp > CMakeFiles/kuiper.dir/source/data/load_data.cpp.i

CMakeFiles/kuiper.dir/source/data/load_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper.dir/source/data/load_data.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaohou/Desktop/myinfer/source/data/load_data.cpp -o CMakeFiles/kuiper.dir/source/data/load_data.cpp.s

CMakeFiles/kuiper.dir/source/data/tensor.cpp.o: CMakeFiles/kuiper.dir/flags.make
CMakeFiles/kuiper.dir/source/data/tensor.cpp.o: ../source/data/tensor.cpp
CMakeFiles/kuiper.dir/source/data/tensor.cpp.o: CMakeFiles/kuiper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/kuiper.dir/source/data/tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper.dir/source/data/tensor.cpp.o -MF CMakeFiles/kuiper.dir/source/data/tensor.cpp.o.d -o CMakeFiles/kuiper.dir/source/data/tensor.cpp.o -c /home/xiaohou/Desktop/myinfer/source/data/tensor.cpp

CMakeFiles/kuiper.dir/source/data/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper.dir/source/data/tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaohou/Desktop/myinfer/source/data/tensor.cpp > CMakeFiles/kuiper.dir/source/data/tensor.cpp.i

CMakeFiles/kuiper.dir/source/data/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper.dir/source/data/tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaohou/Desktop/myinfer/source/data/tensor.cpp -o CMakeFiles/kuiper.dir/source/data/tensor.cpp.s

CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o: CMakeFiles/kuiper.dir/flags.make
CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o: ../source/data/tensor_utils.cpp
CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o: CMakeFiles/kuiper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o -MF CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o.d -o CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o -c /home/xiaohou/Desktop/myinfer/source/data/tensor_utils.cpp

CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaohou/Desktop/myinfer/source/data/tensor_utils.cpp > CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.i

CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaohou/Desktop/myinfer/source/data/tensor_utils.cpp -o CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.s

CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o: CMakeFiles/kuiper.dir/flags.make
CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o: ../source/runtime/pnnx/ir.cpp
CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o: CMakeFiles/kuiper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o -MF CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o.d -o CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o -c /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/ir.cpp

CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/ir.cpp > CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.i

CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/ir.cpp -o CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.s

CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o: CMakeFiles/kuiper.dir/flags.make
CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o: ../source/runtime/pnnx/store_zip.cpp
CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o: CMakeFiles/kuiper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o -MF CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o.d -o CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o -c /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/store_zip.cpp

CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/store_zip.cpp > CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.i

CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaohou/Desktop/myinfer/source/runtime/pnnx/store_zip.cpp -o CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.s

# Object files for target kuiper
kuiper_OBJECTS = \
"CMakeFiles/kuiper.dir/source/data/load_data.cpp.o" \
"CMakeFiles/kuiper.dir/source/data/tensor.cpp.o" \
"CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o" \
"CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o" \
"CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o"

# External object files for target kuiper
kuiper_EXTERNAL_OBJECTS =

../lib/libkuiper.so: CMakeFiles/kuiper.dir/source/data/load_data.cpp.o
../lib/libkuiper.so: CMakeFiles/kuiper.dir/source/data/tensor.cpp.o
../lib/libkuiper.so: CMakeFiles/kuiper.dir/source/data/tensor_utils.cpp.o
../lib/libkuiper.so: CMakeFiles/kuiper.dir/source/runtime/pnnx/ir.cpp.o
../lib/libkuiper.so: CMakeFiles/kuiper.dir/source/runtime/pnnx/store_zip.cpp.o
../lib/libkuiper.so: CMakeFiles/kuiper.dir/build.make
../lib/libkuiper.so: /usr/local/lib/libglog.so.0.8.0
../lib/libkuiper.so: /usr/lib/x86_64-linux-gnu/libarmadillo.so
../lib/libkuiper.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
../lib/libkuiper.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
../lib/libkuiper.so: /usr/local/lib/libgflags.so.2.2.2
../lib/libkuiper.so: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
../lib/libkuiper.so: /usr/lib/x86_64-linux-gnu/libpthread.a
../lib/libkuiper.so: CMakeFiles/kuiper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiaohou/Desktop/myinfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library ../lib/libkuiper.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kuiper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kuiper.dir/build: ../lib/libkuiper.so
.PHONY : CMakeFiles/kuiper.dir/build

CMakeFiles/kuiper.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kuiper.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kuiper.dir/clean

CMakeFiles/kuiper.dir/depend:
	cd /home/xiaohou/Desktop/myinfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiaohou/Desktop/myinfer /home/xiaohou/Desktop/myinfer /home/xiaohou/Desktop/myinfer/build /home/xiaohou/Desktop/myinfer/build /home/xiaohou/Desktop/myinfer/build/CMakeFiles/kuiper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kuiper.dir/depend

