# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build

# Include any dependencies generated for this target.
include src/CMakeFiles/graph_skeleton.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/graph_skeleton.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/graph_skeleton.dir/flags.make

src/CMakeFiles/graph_skeleton.dir/init.cc.o: src/CMakeFiles/graph_skeleton.dir/flags.make
src/CMakeFiles/graph_skeleton.dir/init.cc.o: ../src/init.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/graph_skeleton.dir/init.cc.o"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph_skeleton.dir/init.cc.o -c /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/init.cc

src/CMakeFiles/graph_skeleton.dir/init.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph_skeleton.dir/init.cc.i"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/init.cc > CMakeFiles/graph_skeleton.dir/init.cc.i

src/CMakeFiles/graph_skeleton.dir/init.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph_skeleton.dir/init.cc.s"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/init.cc -o CMakeFiles/graph_skeleton.dir/init.cc.s

src/CMakeFiles/graph_skeleton.dir/init.cc.o.requires:

.PHONY : src/CMakeFiles/graph_skeleton.dir/init.cc.o.requires

src/CMakeFiles/graph_skeleton.dir/init.cc.o.provides: src/CMakeFiles/graph_skeleton.dir/init.cc.o.requires
	$(MAKE) -f src/CMakeFiles/graph_skeleton.dir/build.make src/CMakeFiles/graph_skeleton.dir/init.cc.o.provides.build
.PHONY : src/CMakeFiles/graph_skeleton.dir/init.cc.o.provides

src/CMakeFiles/graph_skeleton.dir/init.cc.o.provides.build: src/CMakeFiles/graph_skeleton.dir/init.cc.o


src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o: src/CMakeFiles/graph_skeleton.dir/flags.make
src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o: ../src/skeleton.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph_skeleton.dir/skeleton.cc.o -c /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/skeleton.cc

src/CMakeFiles/graph_skeleton.dir/skeleton.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph_skeleton.dir/skeleton.cc.i"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/skeleton.cc > CMakeFiles/graph_skeleton.dir/skeleton.cc.i

src/CMakeFiles/graph_skeleton.dir/skeleton.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph_skeleton.dir/skeleton.cc.s"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/skeleton.cc -o CMakeFiles/graph_skeleton.dir/skeleton.cc.s

src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.requires:

.PHONY : src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.requires

src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.provides: src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.requires
	$(MAKE) -f src/CMakeFiles/graph_skeleton.dir/build.make src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.provides.build
.PHONY : src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.provides

src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.provides.build: src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o


src/CMakeFiles/graph_skeleton.dir/stat.cc.o: src/CMakeFiles/graph_skeleton.dir/flags.make
src/CMakeFiles/graph_skeleton.dir/stat.cc.o: ../src/stat.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/graph_skeleton.dir/stat.cc.o"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph_skeleton.dir/stat.cc.o -c /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/stat.cc

src/CMakeFiles/graph_skeleton.dir/stat.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph_skeleton.dir/stat.cc.i"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/stat.cc > CMakeFiles/graph_skeleton.dir/stat.cc.i

src/CMakeFiles/graph_skeleton.dir/stat.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph_skeleton.dir/stat.cc.s"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src/stat.cc -o CMakeFiles/graph_skeleton.dir/stat.cc.s

src/CMakeFiles/graph_skeleton.dir/stat.cc.o.requires:

.PHONY : src/CMakeFiles/graph_skeleton.dir/stat.cc.o.requires

src/CMakeFiles/graph_skeleton.dir/stat.cc.o.provides: src/CMakeFiles/graph_skeleton.dir/stat.cc.o.requires
	$(MAKE) -f src/CMakeFiles/graph_skeleton.dir/build.make src/CMakeFiles/graph_skeleton.dir/stat.cc.o.provides.build
.PHONY : src/CMakeFiles/graph_skeleton.dir/stat.cc.o.provides

src/CMakeFiles/graph_skeleton.dir/stat.cc.o.provides.build: src/CMakeFiles/graph_skeleton.dir/stat.cc.o


# Object files for target graph_skeleton
graph_skeleton_OBJECTS = \
"CMakeFiles/graph_skeleton.dir/init.cc.o" \
"CMakeFiles/graph_skeleton.dir/skeleton.cc.o" \
"CMakeFiles/graph_skeleton.dir/stat.cc.o"

# External object files for target graph_skeleton
graph_skeleton_EXTERNAL_OBJECTS =

../graph_skeleton.so: src/CMakeFiles/graph_skeleton.dir/init.cc.o
../graph_skeleton.so: src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o
../graph_skeleton.so: src/CMakeFiles/graph_skeleton.dir/stat.cc.o
../graph_skeleton.so: src/CMakeFiles/graph_skeleton.dir/build.make
../graph_skeleton.so: /usr/lib/x86_64-linux-gnu/libboost_numpy3.so
../graph_skeleton.so: /home/caolinfeng/anaconda3/lib/libpython3.9.so
../graph_skeleton.so: src/CMakeFiles/graph_skeleton.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../../graph_skeleton.so"
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph_skeleton.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/graph_skeleton.dir/build: ../graph_skeleton.so

.PHONY : src/CMakeFiles/graph_skeleton.dir/build

src/CMakeFiles/graph_skeleton.dir/requires: src/CMakeFiles/graph_skeleton.dir/init.cc.o.requires
src/CMakeFiles/graph_skeleton.dir/requires: src/CMakeFiles/graph_skeleton.dir/skeleton.cc.o.requires
src/CMakeFiles/graph_skeleton.dir/requires: src/CMakeFiles/graph_skeleton.dir/stat.cc.o.requires

.PHONY : src/CMakeFiles/graph_skeleton.dir/requires

src/CMakeFiles/graph_skeleton.dir/clean:
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src && $(CMAKE_COMMAND) -P CMakeFiles/graph_skeleton.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/graph_skeleton.dir/clean

src/CMakeFiles/graph_skeleton.dir/depend:
	cd /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/src /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src /home/caolinfeng/clf/xinye/GraphSkeleton/skeleton_compress/build/src/CMakeFiles/graph_skeleton.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/graph_skeleton.dir/depend
