lib_LIBRARIES = libmiblab.a
libmiblab_a_CFLAGS = -g -O2

if HAS_OPENCV

libmiblab_a_CPPFLAGS = -I$(top_srcdir)/include -I$(OPENCV)/include -I$(BOOST)
libmiblab_a_SOURCES = \
	commandline.cpp \
	md5str.cpp \
	rand.cpp \
	statistics.cpp \
	strconv.cpp \
	stringtokenizer.cpp \
	uniqid.cpp \
	feature_selection.cpp \
	normalize.cpp \
	classifier.cpp \
	classifier_shared.cpp \
	bayes.cpp \
	em.cpp \
	knn.cpp	\
	svm.cpp \
	lc.cpp \
	dtree.cpp \
	forest.cpp \
	parzen.cpp \
	lr.cpp

else

libmiblab_a_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
libmiblab_a_SOURCES = \
	commandline.cpp \
	md5str.cpp \
	rand.cpp \
	statistics.cpp \
	strconv.cpp \
	stringtokenizer.cpp \
	uniqid.cpp \
	feature_selection.cpp \
	normalize.cpp \
	svm.cpp \
	lc.cpp \
	lr.cpp

endif
