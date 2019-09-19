# ----------------------------------------------------------------------

INCLUDE = -I /usr/local/include/opencv4 
LIBDIR	= -L /usr/local/lib
LIBS	= -lopencv_core -lopencv_highgui -lopencv_ml -lopencv_video \
-lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann \
-lopencv_imgproc -lopencv_face -lopencv_imgcodecs -lopencv_videoio

# ----------------------------------------------------------------------

SOURCE_DIR		= src
#SOURCE_FILES	= $(wildcard $(SOURCE_DIR)/*.cpp) #$(SOUCE_DIR)/$(wildcard *.cpp)
SOURCE_OBJ		= improc.o project_zero.o httpcl.o 
EXECUTABLE_FILE	= pzero 

# ----------------------------------------------------------------------

all:	$(EXECUTABLE_FILE)

# ----------------------------------------------------------------------

$(EXECUTABLE_FILE):		$(SOURCE_OBJ)
	g++ -o $@ $^ $(INCLUDE) $(LIBDIR) $(LIBS)
	rm -rf *.o

# ----------------------------------------------------------------------

improc.o:		$(SOURCE_DIR)/improc.cpp
	g++ -c $^ $(INCLUDE) $(LIBDIR) 

project_zero.o:	$(SOURCE_DIR)/project_zero.cpp
	g++ -c $^ $(INCLUDE) $(LIBDIR)
	
httpcl.o:		$(SOURCE_DIR)/httpcl.cpp 
	g++ -c $^ $(INCLUDE) $(LIBDIR)

# ----------------------------------------------------------------------

clean:
	@rm -rf *.o

# ----------------------------------------------------------------------
